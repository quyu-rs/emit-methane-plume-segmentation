# -*- coding: utf-8 -*-
"""
CNN Refiner
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

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_num_groups(num_channels: int, max_groups: int = 8) -> int:
    """
    为 GroupNorm 自动选择可整除的 group 数
    """
    for g in range(min(max_groups, num_channels), 0, -1):
        if num_channels % g == 0:
            return g
    return 1


class ConvNormAct(nn.Module):
    """
    基础卷积块:
    Conv -> GroupNorm -> SiLU -> Dropout
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        dropout: float = 0.1
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False
        )
        self.norm = nn.GroupNorm(get_num_groups(out_channels), out_channels)
        self.act = nn.SiLU(inplace=True)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.dropout(x)
        return x


class ResidualBlock(nn.Module):
    """
    残差卷积块
    """
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.1):
        super().__init__()

        self.block1 = ConvNormAct(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            dropout=dropout
        )
        self.block2 = ConvNormAct(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            dropout=0.0
        )

        if in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = self.block1(x)
        out = self.block2(out)
        return out + identity


class CNNRefiner(nn.Module):
    """
    轻量级 CNN residual refiner

    输入:
        condition: [B, in_channels, H, W]

    输出:
        pred_residual: [B, 1, H, W]
    """
    def __init__(
        self,
        in_channels: int = 5,
        base_channels: int = 64,
        num_blocks: int = 4,
        dropout: float = 0.1,
        residual_range: float = 1.0,
        init_output_scale: float = 0.1
    ):
        super().__init__()
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.num_blocks = num_blocks
        self.residual_range = residual_range

        self.stem = ConvNormAct(
            in_channels=in_channels,
            out_channels=base_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            dropout=dropout
        )

        blocks = []
        for _ in range(num_blocks):
            blocks.append(
                ResidualBlock(
                    in_channels=base_channels,
                    out_channels=base_channels,
                    dropout=dropout
                )
            )
        self.backbone = nn.Sequential(*blocks)

        mid_channels = max(base_channels // 2, 16)
        self.pre_head = nn.Sequential(
            nn.Conv2d(base_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(get_num_groups(mid_channels), mid_channels),
            nn.SiLU(inplace=True),
        )
        self.out_conv = nn.Conv2d(mid_channels, 1, kernel_size=1, bias=True)

        # 可学习缩放，保证初始时更接近 identity refinement
        self.output_scale = nn.Parameter(torch.tensor(float(init_output_scale)))

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GroupNorm):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        # 最后一层 zero-init，初始 residual≈0
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

        x = self.stem(condition)
        x = self.backbone(x)
        x = self.pre_head(x)
        residual = self.out_conv(x)

        # 将残差限制在合理范围，再乘以可学习 scale
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
        """
        便捷接口：直接输出 refined prediction
        """
        condition = torch.cat([input_image, pred_coarse], dim=1)
        pred_residual = self(condition)
        pred_refined = torch.clamp(pred_coarse + pred_residual, 0.0, 1.0)
        return pred_refined

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)