# -*- coding: utf-8 -*-
"""
Diffusion损失函数实现
用于训练Diffusion模型预测噪声
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiffusionLoss(nn.Module):
    """
    Diffusion损失函数
    使用MSE损失预测噪声
    """
    def __init__(self, reduction: str = 'mean'):
        """
        参数:
            reduction: 损失归约方式，'mean' 或 'sum'
        """
        super().__init__()
        self.reduction = reduction
        self.mse_loss = nn.MSELoss(reduction=reduction)
    
    def forward(
        self,
        predicted_noise: torch.Tensor,
        target_noise: torch.Tensor
    ) -> torch.Tensor:
        """
        计算Diffusion损失
        
        参数:
            predicted_noise: 模型预测的噪声，形状为 (B, 1, H, W)
            target_noise: 真实噪声，形状为 (B, 1, H, W)
        
        返回:
            损失值
        """
        return self.mse_loss(predicted_noise, target_noise)


class SimpleDiffusionLoss(nn.Module):
    """
    简化的Diffusion损失函数
    直接使用MSE，提供更简单的接口
    """
    def __init__(self):
        super().__init__()
    
    def forward(
        self,
        predicted_noise: torch.Tensor,
        target_noise: torch.Tensor
    ) -> torch.Tensor:
        """
        计算MSE损失
        
        参数:
            predicted_noise: 模型预测的噪声，形状为 (B, 1, H, W)
            target_noise: 真实噪声，形状为 (B, 1, H, W)
        
        返回:
            MSE损失值
        """
        return F.mse_loss(predicted_noise, target_noise)



