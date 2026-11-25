# -*- coding: utf-8 -*-
"""
Diffusion Refiner模型实现
用于细化主模型的分割结果
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class TimeEmbedding(nn.Module):
    """
    时间步嵌入层
    将时间步t转换为特征向量
    """
    def __init__(self, dim: int = 128):
        """
        参数:
            dim: 嵌入维度
        """
        super().__init__()
        self.dim = dim
        
        half_dim = dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
        self.register_buffer('emb', emb)
        
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim)
        )
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        参数:
            t: 时间步，形状为 (B,)，值在[0, num_timesteps]
        返回:
            时间嵌入，形状为 (B, dim)
        """
        emb = t[:, None] * self.emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        
        if emb.shape[-1] < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.shape[-1]))
        
        return self.mlp(emb)


class UNetBlock(nn.Module):
    """
    U-Net基础块
    包含卷积、归一化、激活和时间步条件
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        dropout: float = 0.1,
        is_downsample: bool = False,
        is_upsample: bool = False
    ):
        """
        参数:
            in_channels: 输入通道数
            out_channels: 输出通道数
            time_emb_dim: 时间嵌入维度
            dropout: Dropout比率
            is_downsample: 是否下采样
            is_upsample: 是否上采样
        """
        super().__init__()
        self.is_downsample = is_downsample
        self.is_upsample = is_upsample
        
        self.time_proj = nn.Linear(time_emb_dim, out_channels)
        
        if is_downsample:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        elif is_upsample:
            self.conv1 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.act1 = nn.SiLU()
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.act2 = nn.SiLU()
        
        self.dropout = nn.Dropout2d(dropout)
        
        if in_channels == out_channels and not is_downsample and not is_upsample:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        """
        参数:
            x: 输入特征，形状为 (B, in_channels, H, W)
            time_emb: 时间嵌入，形状为 (B, time_emb_dim)
        返回:
            输出特征，形状为 (B, out_channels, H', W')
        """
        residual = self.residual(x)
        
        x = self.conv1(x)
        x = self.norm1(x)
        
        time_emb_proj = self.time_proj(time_emb)
        time_emb_proj = time_emb_proj[:, :, None, None]
        x = x + time_emb_proj
        
        x = self.act1(x)
        x = self.dropout(x)
        
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.act2(x)
        x = self.dropout(x)
        
        if x.shape == residual.shape:
            x = x + residual
        
        return x


class DiffusionRefiner(nn.Module):
    """
    Diffusion Refiner模型
    使用U-Net架构，输入5通道（4通道图像 + 1通道初始掩码），输出1通道残差
    """
    def __init__(
        self,
        in_channels: int = 6,
        base_channels: int = 64,
        time_emb_dim: int = 128,
        num_timesteps: int = 1000,
        dropout: float = 0.1
    ):
        """
        参数:
            in_channels: 输入通道数（5条件 + 1噪声残差 = 6）
            base_channels: 基础通道数
            time_emb_dim: 时间嵌入维度
            num_timesteps: 扩散步数
            dropout: Dropout比率
        """
        super().__init__()
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.time_emb_dim = time_emb_dim
        self.num_timesteps = num_timesteps
        
        self.time_embedding = TimeEmbedding(time_emb_dim)
        
        self.initial_conv = nn.Conv2d(6, base_channels, kernel_size=3, padding=1)
        
        self.encoder = nn.ModuleList([
            UNetBlock(base_channels, base_channels, time_emb_dim, dropout, is_downsample=False),
            UNetBlock(base_channels, base_channels * 2, time_emb_dim, dropout, is_downsample=True),
            UNetBlock(base_channels * 2, base_channels * 4, time_emb_dim, dropout, is_downsample=True),
            UNetBlock(base_channels * 4, base_channels * 8, time_emb_dim, dropout, is_downsample=True),
        ])
        
        self.middle = nn.ModuleList([
            UNetBlock(base_channels * 8, base_channels * 8, time_emb_dim, dropout, is_downsample=False),
            UNetBlock(base_channels * 8, base_channels * 8, time_emb_dim, dropout, is_downsample=False),
        ])
        
        self.decoder = nn.ModuleList([
            UNetBlock(base_channels * 16, base_channels * 4, time_emb_dim, dropout, is_upsample=True),
            UNetBlock(base_channels * 8, base_channels * 2, time_emb_dim, dropout, is_upsample=True),
            UNetBlock(base_channels * 4, base_channels, time_emb_dim, dropout, is_upsample=True),
            UNetBlock(base_channels * 2, base_channels, time_emb_dim, dropout, is_downsample=False),
        ])
        
        self.output_conv = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, 1, kernel_size=1)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.GroupNorm, nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        condition: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向传播
        
        参数:
            x: 噪声残差，形状为 (B, 1, H, W)
            t: 时间步，形状为 (B,)，值在[0, num_timesteps]
            condition: 条件输入（图像+初始掩码），形状为 (B, 5, H, W)
        
        返回:
            预测的噪声，形状为 (B, 1, H, W)
        """

        time_emb = self.time_embedding(t)
        

        if condition is not None:

            x_input = torch.cat([condition, x], dim=1)
        else:

            if x.shape[1] == 1:
                x_input = torch.cat([torch.zeros_like(x).expand(-1, 4, -1, -1), x], dim=1)
            else:
                x_input = x
        

        x = self.initial_conv(x_input)
        

        encoder_features = []
        for block in self.encoder:
            x = block(x, time_emb)
            encoder_features.append(x)
        

        for block in self.middle:
            x = block(x, time_emb)
        

        for i, block in enumerate(self.decoder):

            skip_idx = len(encoder_features) - 1 - i
            if skip_idx >= 0:
                skip = encoder_features[skip_idx]

                if x.shape[2:] != skip.shape[2:]:
                    x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
                x = torch.cat([x, skip], dim=1)
            x = block(x, time_emb)
        

        x = self.output_conv(x)
        
        return x
    
    def count_parameters(self) -> int:
        """统计参数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def sample(
        self,
        condition: torch.Tensor,
        num_steps: int = 20,
        guidance_scale: float = 1.0
    ) -> torch.Tensor:
        """
        采样生成细化结果
        使用DDIM采样（确定性，更稳定）替代DDPM
        
        参数:
            condition: 条件输入（图像+初始掩码），形状为 (B, 5, H, W)
            num_steps: 采样步数
            guidance_scale: 引导强度（暂未使用）
        
        返回:
            细化后的残差，形状为 (B, 1, H, W)
        """
        self.eval()
        batch_size = condition.size(0)
        device = condition.device
        

        if not hasattr(self, 'betas'):
            self._init_noise_schedule(device)
        

        x_t = torch.randn(batch_size, 1, condition.shape[2], condition.shape[3], device=device) * 0.05
        

        num_steps = max(1, min(num_steps, self.num_timesteps))

        timesteps = torch.linspace(self.num_timesteps - 1, 0, num_steps, device=device).long()
        
        with torch.no_grad():
            for i, t in enumerate(timesteps):

                t_idx = t.item() if t.numel() == 1 else int(t)
                t_batch = t.repeat(batch_size)
                

                predicted_noise = self(x_t, t_batch, condition)
                

                alpha_cumprod_t = self.alphas_cumprod[t_idx]
                sqrt_alpha_cumprod_t = torch.sqrt(alpha_cumprod_t)
                sqrt_alpha_cumprod_t = torch.clamp(sqrt_alpha_cumprod_t, min=1e-8)
                sqrt_one_minus_alpha_cumprod_t = torch.sqrt(1.0 - alpha_cumprod_t)
                sqrt_one_minus_alpha_cumprod_t = torch.clamp(sqrt_one_minus_alpha_cumprod_t, min=1e-8)
                

                pred_x0 = (x_t - sqrt_one_minus_alpha_cumprod_t * predicted_noise) / sqrt_alpha_cumprod_t
                
                if i < len(timesteps) - 1:

                    t_prev = timesteps[i + 1]
                    t_prev_idx = t_prev.item() if t_prev.numel() == 1 else int(t_prev)
                    alpha_cumprod_t_prev = self.alphas_cumprod[t_prev_idx]
                    sqrt_alpha_cumprod_t_prev = torch.sqrt(alpha_cumprod_t_prev)
                    sqrt_one_minus_alpha_cumprod_t_prev = torch.sqrt(1.0 - alpha_cumprod_t_prev)
                    sqrt_one_minus_alpha_cumprod_t_prev = torch.clamp(sqrt_one_minus_alpha_cumprod_t_prev, min=1e-8)
                    


                    x_t = sqrt_alpha_cumprod_t_prev * pred_x0 + sqrt_one_minus_alpha_cumprod_t_prev * predicted_noise
                else:

                    x_t = pred_x0
        

        x_t = torch.clamp(x_t, min=-0.2, max=0.2)
        
        return x_t
    
    def _init_noise_schedule(self, device: torch.device):
        """初始化噪声调度（用于采样）"""

        if hasattr(self, 'initial_conv'):
            model_device = next(self.parameters()).device
            device = device if device == model_device else model_device
        
        beta_start = 0.0001
        beta_end = 0.02
        
        self.betas = torch.linspace(beta_start, beta_end, self.num_timesteps, device=device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        

        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)



