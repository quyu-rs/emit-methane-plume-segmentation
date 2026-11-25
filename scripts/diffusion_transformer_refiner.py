# -*- coding: utf-8 -*-
"""
Diffusion Refiner模型实现 - Transformer版本
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class TimeEmbedding(nn.Module):
    def __init__(self, dim: int = 128):
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
        emb = t[:, None] * self.emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if emb.shape[-1] < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.shape[-1]))
        return self.mlp(emb)


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        mlp_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x)
        x = x + residual
        
        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = x + residual
        return x


class DiffusionRefiner(nn.Module):
    """Diffusion Refiner - Transformer版本"""
    def __init__(
        self,
        in_channels: int = 6,
        base_channels: int = 64,
        time_emb_dim: int = 128,
        num_timesteps: int = 1000,
        dropout: float = 0.1,
        patch_size: int = 8,
        num_layers: int = 12,
        num_heads: int = 8
    ):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = base_channels * 8
        self.num_timesteps = num_timesteps
        
        self.time_embedding = TimeEmbedding(time_emb_dim)
        self.time_proj = nn.Linear(time_emb_dim, self.embed_dim)
        
        self.patch_embed = nn.Conv2d(in_channels, self.embed_dim, kernel_size=patch_size, stride=patch_size)
        
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(self.embed_dim, num_heads, 4.0, dropout) for _ in range(num_layers)
        ])
        
        self.unpatch = nn.ConvTranspose2d(self.embed_dim, 1, kernel_size=patch_size, stride=patch_size)
        
        self._initialize_weights()

    def _get_2d_sincos_pos_embed(self, embed_dim: int, grid_size: tuple[int, int], device: torch.device):
        h, w = grid_size
        grid_h = torch.arange(h, device=device, dtype=torch.float32)
        grid_w = torch.arange(w, device=device, dtype=torch.float32)
        grid = torch.meshgrid(grid_h, grid_w, indexing='ij')
        grid = torch.stack(grid, dim=0).reshape(2, -1)
        
        def _get_1d_sincos(pos: torch.Tensor, dim: int):
            omega = torch.arange(dim // 2, device=device, dtype=torch.float32) / (dim // 2)
            omega = 1.0 / (10000.0 ** omega)
            out = torch.einsum('m,d->md', pos, omega)
            return torch.cat([torch.sin(out), torch.cos(out)], dim=1)
        
        emb_h = _get_1d_sincos(grid[0], embed_dim // 2)
        emb_w = _get_1d_sincos(grid[1], embed_dim // 2)
        emb = torch.cat([emb_h, emb_w], dim=-1)
        return emb.unsqueeze(0)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor, t: torch.Tensor, condition: Optional[torch.Tensor] = None) -> torch.Tensor:
        time_emb = self.time_embedding(t)
        
        if condition is not None:
            x_input = torch.cat([condition, x], dim=1)
        else:
            if x.shape[1] == 1:
                x_input = torch.cat([torch.zeros_like(x).expand(-1, 4, -1, -1), x], dim=1)
            else:
                x_input = x
        
        x = self.patch_embed(x_input)
        B, _, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        
        pos_embed = self._get_2d_sincos_pos_embed(self.embed_dim, (h, w), x.device)
        tokens = tokens + pos_embed.to(tokens.dtype)
        
        time_emb_proj = self.time_proj(time_emb).unsqueeze(1)
        tokens = tokens + time_emb_proj
        
        for block in self.transformer_blocks:
            tokens = block(tokens)
        
        tokens = tokens.transpose(1, 2).reshape(B, self.embed_dim, h, w)
        x = self.unpatch(tokens)
        return x

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def sample(self, condition: torch.Tensor, num_steps: int = 20, guidance_scale: float = 1.0) -> torch.Tensor:
        # （与你原来U-Net版本一致的DDIM采样）
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
                t_batch = t.repeat(batch_size)
                predicted_noise = self(x_t, t_batch, condition)
                
                alpha_cumprod_t = self.alphas_cumprod[t.item()]
                sqrt_alpha = torch.sqrt(torch.clamp(alpha_cumprod_t, min=1e-8))
                sqrt_one_minus = torch.sqrt(torch.clamp(1.0 - alpha_cumprod_t, min=1e-8))
                
                pred_x0 = (x_t - sqrt_one_minus * predicted_noise) / sqrt_alpha
                
                if i < len(timesteps) - 1:
                    t_prev = timesteps[i + 1]
                    alpha_prev = self.alphas_cumprod[t_prev.item()]
                    sqrt_alpha_prev = torch.sqrt(torch.clamp(alpha_prev, min=1e-8))
                    sqrt_one_minus_prev = torch.sqrt(torch.clamp(1.0 - alpha_prev, min=1e-8))
                    x_t = sqrt_alpha_prev * pred_x0 + sqrt_one_minus_prev * predicted_noise
                else:
                    x_t = pred_x0
        
        return torch.clamp(x_t, min=-0.2, max=0.2)

    def _init_noise_schedule(self, device: torch.device):
        beta_start = 0.0001
        beta_end = 0.02
        self.betas = torch.linspace(beta_start, beta_end, self.num_timesteps, device=device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)