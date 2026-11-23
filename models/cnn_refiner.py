# -*- coding: utf-8 -*-
"""
CNN Refiner
用于第二阶段 deterministic refinement
输入: condition = [input_image, pred_coarse] -> 5通道
输出: 1通道残差 pred_residual
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, out_channels),
            nn.SiLU(inplace=True),
            nn.Dropout2d(dropout),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, out_channels),
            nn.SiLU(inplace=True),
        )

        if in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x) + self.shortcut(x)


class CNNRefiner(nn.Module):
    """
    轻量级 CNN residual refiner
    输入:
        condition: [B, 5, H, W]
    输出:
        pred_residual: [B, 1, H, W]
    """
    def __init__(
        self,
        in_channels: int = 5,
        base_channels: int = 64,
        num_blocks: int = 4,
        dropout: float = 0.1,
        residual_range: float = 0.2
    ):
        super().__init__()
        self.residual_range = residual_range

        layers = [
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, base_channels),
            nn.SiLU(inplace=True),
        ]

        for _ in range(num_blocks):
            layers.append(ConvBlock(base_channels, base_channels, dropout=dropout))

        self.backbone = nn.Sequential(*layers)

        self.head = nn.Sequential(
            nn.Conv2d(base_channels, base_channels // 2, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, base_channels // 2),
            nn.SiLU(inplace=True),
            nn.Conv2d(base_channels // 2, 1, kernel_size=1)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                if getattr(m, "weight", None) is not None:
                    nn.init.constant_(m.weight, 1)
                if getattr(m, "bias", None) is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, condition: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(condition)
        residual = self.head(feat)

        # 与当前 diffusion sample 里的残差范围保持一致，更稳定
        residual = torch.tanh(residual) * self.residual_range
        return residual

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)