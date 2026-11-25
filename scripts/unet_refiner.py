# -*- coding: utf-8 -*-
"""
UNet Refiner
用于第二阶段 deterministic residual refinement

输入:
    condition = [input_image, pred_coarse]
    默认:
        input_image: 4通道
        pred_coarse: 1通道
    所以默认输入通道数 = 5

输出:
    pred_residual: 1通道残差
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_num_groups(num_channels: int, max_groups: int = 8) -> int:
    for g in range(min(max_groups, num_channels), 0, -1):
        if num_channels % g == 0:
            return g
    return 1


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(get_num_groups(out_channels), out_channels),
            nn.SiLU(inplace=True),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(get_num_groups(out_channels), out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.1):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(x)
        x = self.conv(x)
        return x


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, dropout: float = 0.1):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_channels + skip_channels, out_channels, dropout=dropout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)

        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)

        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x


class UNetRefiner(nn.Module):
    """
    轻量 U-Net residual refiner

    输入:
        condition: [B, in_channels, H, W]
    输出:
        pred_residual: [B, 1, H, W]
    """
    def __init__(
        self,
        in_channels: int = 5,
        base_channels: int = 32,
        dropout: float = 0.1,
        residual_range: float = 1.0,
        init_output_scale: float = 0.1
    ):
        super().__init__()
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.residual_range = residual_range

        self.inc = DoubleConv(in_channels, base_channels, dropout=dropout)

        self.down1 = DownBlock(base_channels, base_channels * 2, dropout=dropout)
        self.down2 = DownBlock(base_channels * 2, base_channels * 4, dropout=dropout)
        self.down3 = DownBlock(base_channels * 4, base_channels * 8, dropout=dropout)

        self.bottleneck = DoubleConv(base_channels * 8, base_channels * 8, dropout=dropout)

        self.up1 = UpBlock(base_channels * 8, base_channels * 4, base_channels * 4, dropout=dropout)
        self.up2 = UpBlock(base_channels * 4, base_channels * 2, base_channels * 2, dropout=dropout)
        self.up3 = UpBlock(base_channels * 2, base_channels, base_channels, dropout=dropout)

        self.pre_head = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(get_num_groups(base_channels), base_channels),
            nn.SiLU(inplace=True),
        )
        self.out_conv = nn.Conv2d(base_channels, 1, kernel_size=1, bias=True)

        self.output_scale = nn.Parameter(torch.tensor(float(init_output_scale)))

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if getattr(m, "bias", None) is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GroupNorm):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        nn.init.constant_(self.out_conv.weight, 0)
        if self.out_conv.bias is not None:
            nn.init.constant_(self.out_conv.bias, 0)

    def forward(self, condition: torch.Tensor) -> torch.Tensor:
        if torch.isnan(condition).any() or torch.isinf(condition).any():
            condition = torch.where(
                torch.isnan(condition) | torch.isinf(condition),
                torch.zeros_like(condition),
                condition
            )

        x1 = self.inc(condition)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        xb = self.bottleneck(x4)

        x = self.up1(xb, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)

        x = self.pre_head(x)
        residual = self.out_conv(x)

        residual = torch.tanh(residual) * self.residual_range
        residual = residual * self.output_scale

        if torch.isnan(residual).any() or torch.isinf(residual).any():
            residual = torch.where(
                torch.isnan(residual) | torch.isinf(residual),
                torch.zeros_like(residual),
                residual
            )

        return residual

    def refine(self, input_image: torch.Tensor, pred_coarse: torch.Tensor) -> torch.Tensor:
        condition = torch.cat([input_image, pred_coarse], dim=1)
        pred_residual = self(condition)
        pred_refined = torch.clamp(pred_coarse + pred_residual, 0.0, 1.0)
        return pred_refined

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)