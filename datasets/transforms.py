# -*- coding: utf-8 -*-
"""
数据增强变换模块
支持4通道输入和掩码的同步变换
"""
import torch
import torch.nn.functional as F
import random
from typing import Tuple

class Rotate180:
    """180度旋转变换"""

    def __call__(self, input_tensor: torch.Tensor, mask_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        对输入和掩码进行180度旋转
        
        参数:
            input_tensor: 输入张量，形状为 (C, H, W)
            mask_tensor: 掩码张量，形状为 (1, H, W)
        
        返回:
            旋转后的输入和掩码
        """

        input_rotated = torch.rot90(input_tensor, k=2, dims=[1, 2])
        mask_rotated = torch.rot90(mask_tensor, k=2, dims=[1, 2])

        return input_rotated, mask_rotated

class HorizontalFlip:
    """水平翻转变换"""

    def __call__(self, input_tensor: torch.Tensor, mask_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        对输入和掩码进行水平翻转
        
        参数:
            input_tensor: 输入张量，形状为 (C, H, W)
            mask_tensor: 掩码张量，形状为 (1, H, W)
        
        返回:
            翻转后的输入和掩码
        """
        input_flipped = torch.flip(input_tensor, dims=[2])
        mask_flipped = torch.flip(mask_tensor, dims=[2])

        return input_flipped, mask_flipped

class VerticalFlip:
    """垂直翻转变换"""

    def __call__(self, input_tensor: torch.Tensor, mask_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        对输入和掩码进行垂直翻转
        
        参数:
            input_tensor: 输入张量，形状为 (C, H, W)
            mask_tensor: 掩码张量，形状为 (1, H, W)
        
        返回:
            翻转后的输入和掩码
        """
        input_flipped = torch.flip(input_tensor, dims=[1])
        mask_flipped = torch.flip(mask_tensor, dims=[1])

        return input_flipped, mask_flipped

class RandomAugment:
    """随机选择一种数据增强"""

    def __init__(self, transforms=None):
        """
        参数:
            transforms: 变换列表，如果为None则使用默认的三种变换
        """
        if transforms is None:
            transforms = [Rotate180(), HorizontalFlip(), VerticalFlip()]
        self.transforms = transforms

    def __call__(self, input_tensor: torch.Tensor, mask_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        随机选择一种变换并应用
        
        参数:
            input_tensor: 输入张量，形状为 (C, H, W)
            mask_tensor: 掩码张量，形状为 (1, H, W)
        
        返回:
            变换后的输入和掩码
        """
        transform = random.choice(self.transforms)
        return transform(input_tensor, mask_tensor)


class Compose:
    """组合多个变换"""

    def __init__(self, transforms):
        """
        参数:
            transforms: 变换列表
        """
        self.transforms = transforms

    def __call__(self, input_tensor: torch.Tensor, mask_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        依次应用所有变换
        
        参数:
            input_tensor: 输入张量
            mask_tensor: 掩码张量
        
        返回:
            变换后的输入和掩码
        """
        for transform in self.transforms:
            input_tensor, mask_tensor = transform(input_tensor, mask_tensor)

        return input_tensor, mask_tensor



