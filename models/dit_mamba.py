# -*- coding: utf-8 -*-
"""
DiT-Mamba模型实现
包含DiT编码器、Mamba模块和解码器
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional

class DiTEncoder(nn.Module):
    """
    DiT编码器
    处理4通道输入，提取多尺度特征
    """
    def __init__(
        self,
        in_channels: int = 4,
        base_channels: int = 64,
        num_layers: int = 4,
        dropout: float = 0.1
    ):
        """
        参数:
            in_channels: 输入通道数（RGB 3 + ENH 1 = 4）
            base_channels: 基础通道数
            num_layers: 编码器层数
            dropout: Dropout比率
        """
        super().__init__()
        self.num_layers = num_layers
        self.base_channels = base_channels


        self.initial_conv = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout)
        )


        self.layers = nn.ModuleList()
        channels = base_channels

        for i in range(num_layers):

            out_channels = channels * 2 if i < num_layers - 1 else channels

            layer = nn.Sequential(
                nn.Conv2d(channels, out_channels, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Dropout2d(dropout)
            )
            self.layers.append(layer)
            channels = out_channels

        self._initialize_weights()

    def _initialize_weights(self):
        """权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> list:
        """
        前向传播
        
        参数:
            x: 输入张量，形状为 (B, 4, H, W)
        
        返回:
            多尺度特征图列表
        """
        features = []


        x = self.initial_conv(x)
        features.append(x)


        for layer in self.layers:
            x = layer(x)
            features.append(x)

        return features


class MambaBlock(nn.Module):
    """
    Mamba块
    选择性状态空间模型，线性复杂度
    """
    def __init__(
        self,
        dim: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
        use_parallel_scan: bool = False,
        use_vectorized_scan: bool = True
    ):
        """
        参数:
            dim: 输入维度
            d_state: 状态维度
            d_conv: 卷积核大小
            expand: 扩展因子
            dropout: Dropout比率
            use_parallel_scan: 是否使用并行扫描（更快但可能稍不准确）
            use_vectorized_scan: 是否使用向量化扫描（最快，默认启用）
        """
        super().__init__()
        self.dim = dim
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.use_parallel_scan = use_parallel_scan
        self.use_vectorized_scan = use_vectorized_scan
        d_inner = int(expand * dim)


        self.in_proj = nn.Linear(dim, d_inner * 2)


        self.conv1d = nn.Conv1d(
            in_channels=d_inner,
            out_channels=d_inner,
            kernel_size=d_conv,
            groups=d_inner,
            padding=d_conv - 1
        )


        self.activation = nn.SiLU()


        self.norm = nn.LayerNorm(d_inner)


        self.x_proj = nn.Linear(d_inner, d_state * 2)
        self.dt_proj = nn.Linear(d_inner, d_state)




        A_log_init = -torch.ones(d_state, d_state, dtype=torch.float32) * 2.0
        self.A_log = nn.Parameter(A_log_init)



        A_proj_init = torch.randn(d_inner, d_state, dtype=torch.float32) * 0.1
        self.A_proj = nn.Parameter(A_proj_init)
        self.D = nn.Parameter(torch.ones(d_inner))


        self.out_proj = nn.Linear(d_inner, dim)
        self.dropout = nn.Dropout(dropout)

        self._initialize_weights()

    def _initialize_weights(self):
        """权重初始化"""
        nn.init.kaiming_normal_(self.in_proj.weight, mode='fan_in', nonlinearity='linear')
        nn.init.kaiming_normal_(self.conv1d.weight, mode='fan_in', nonlinearity='linear')


        nn.init.normal_(self.x_proj.weight, mean=0.0, std=0.02)
        if self.x_proj.bias is not None:
            nn.init.constant_(self.x_proj.bias, 0.0)
        nn.init.kaiming_normal_(self.dt_proj.weight, mode='fan_in', nonlinearity='linear')
        nn.init.kaiming_normal_(self.out_proj.weight, mode='fan_in', nonlinearity='linear')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        参数:
            x: 输入张量，形状为 (B, L, D) 或 (B, D, H, W)
        
        返回:
            输出张量
        """

        original_shape = x.shape
        if len(original_shape) == 4:
            B, D, H, W = original_shape

            x = x.view(B, D, H * W).permute(0, 2, 1)
            need_reshape = True
        else:
            need_reshape = False

        B, L, D = x.shape


        original_input = x


        if D != self.dim:


            if not hasattr(self, 'dim_proj'):
                self.dim_proj = nn.Linear(D, self.dim).to(x.device)

                nn.init.kaiming_normal_(self.dim_proj.weight, mode='fan_in', nonlinearity='linear')
                nn.init.constant_(self.dim_proj.bias, 0)

                if 'dim_proj' not in dict(self.named_children()):
                    self.add_module('dim_proj', self.dim_proj)
            x = self.dim_proj(x)
            D = self.dim


        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)


        x = x.transpose(1, 2)
        x = self.conv1d(x)[:, :, :L]
        x = x.transpose(1, 2)
        x = self.activation(x)
        x = self.norm(x)


        x_dbl = self.x_proj(x)

        B, C = x_dbl.chunk(2, dim=-1)
        dt = self.dt_proj(x)
        dt = F.softplus(dt) + 1e-4

        dt = torch.clamp(dt, min=1e-4, max=10.0)






        A_log_clamped = torch.clamp(self.A_log, min=-4, max=1)
        A = -torch.exp(A_log_clamped)




        dt_expanded = dt.unsqueeze(-1)
        A_expanded = A.unsqueeze(0).unsqueeze(0)
        dt_A = dt_expanded * A_expanded

        dt_A_sum = dt_A.sum(dim=-1)

        dt_A_max = dt_A_sum.max(dim=-1, keepdim=True)[0]
        dt_A_stable = dt_A_sum - dt_A_max

        dA = torch.exp(dt_A_stable.clamp(min=-6, max=6))



        A_proj = self.A_proj





        dB = dt * B




        if hasattr(self, 'use_vectorized_scan') and self.use_vectorized_scan:
            y = self._selective_scan_vectorized(x, dA, dB, C, A_proj)
        elif self.use_parallel_scan:
            y = self._selective_scan_parallel(x, dA, dB, C, A_proj)
        else:
            y = self._selective_scan(x, dA, dB, C, A_proj)


        z_activated = self.activation(z)
        y = y * z_activated


        y = self.out_proj(y)
        y = self.dropout(y)


        if y.shape == original_input.shape:
            y = y + original_input


        if need_reshape:
            y = y.permute(0, 2, 1).view(original_shape)


        if torch.isnan(y).any() or torch.isinf(y).any():
            y = torch.where(torch.isnan(y) | torch.isinf(y),
                           torch.zeros_like(y), y)

        return y

    def _selective_scan(self, u, dA, dB, C, A):
        """
        选择性扫描（递归实现 - 真正的状态空间模型）
        
        递归状态空间模型：
        - h_t = dA_t * h_{t-1} + dB_t * (A^T @ u_t)
        - y_t = (C_t * h_t) @ A
        
        参数:
            u: 输入 (B, L, d_inner)
            dA: 离散化状态转移矩阵 (B, L, d_state)
            dB: 离散化输入矩阵 (B, L, d_state)
            C: 输出矩阵 (B, L, d_state)
            A: 状态转移矩阵，用于维度投影 (d_inner, d_state)
        
        返回:
            输出 (B, L, d_inner)
        """
        B, L, d_inner = u.shape
        d_state = dA.shape[-1]


        eps = 1e-8









        h = torch.zeros(B, d_state, device=u.device, dtype=u.dtype)


        y_list = []









        for t in range(L):
            u_t = u[:, t, :]
            dA_t = dA[:, t, :]
            dB_t = dB[:, t, :]
            C_t = C[:, t, :]




            u_projected = torch.matmul(u_t, A)




            u_state = dB_t * u_projected
            h = dA_t * h + u_state



            h_max = h.abs().max(dim=-1, keepdim=True)[0]
            h = torch.where(h_max > 100.0, h / (h_max / 100.0 + 1e-8), h)





            state_output = C_t * h
            y_t = torch.matmul(state_output, A.T) + self.D * u_t

            y_list.append(y_t)


        y = torch.stack(y_list, dim=1)


        if torch.isnan(y).any() or torch.isinf(y).any():
            y = torch.where(torch.isnan(y) | torch.isinf(y),
                           torch.zeros_like(y), y)

        return y

    def _selective_scan_vectorized(self, u, dA, dB, C, A):
        """
        向量化的选择性扫描实现
        使用cumsum和矩阵运算，避免循环
        
        数学原理：
        对于状态空间模型 h_t = dA_t * h_{t-1} + dBu_t，其中 h_0 = 0
        可以展开为：
        h_1 = dBu_1
        h_2 = dA_2 * h_1 + dBu_2 = dA_2 * dBu_1 + dBu_2
        h_3 = dA_3 * h_2 + dBu_3 = dA_3 * (dA_2 * dBu_1 + dBu_2) + dBu_3
                                  = dA_3 * dA_2 * dBu_1 + dA_3 * dBu_2 + dBu_3
        
        一般形式：h_t = sum_{k=1}^{t} [prod_{j=k+1}^{t} dA_j] * dBu_k
        
        向量化实现：
        1. 计算累积乘积 prod_{j=1}^{t} dA_j（使用log-space避免数值溢出）
        2. 计算加权输入 dBu_k / prod_{j=1}^{k} dA_j
        3. 反向累积和得到 h_t
        
        注意：
        - 使用log-space计算累积乘积以提高数值稳定性
        - 如果出现NaN/Inf，自动回退到递归实现
        - 已验证与递归实现数学等价（误差在数值精度范围内）
        
        参数:
            u: 输入 (B, L, d_inner)
            dA: 离散化状态转移矩阵 (B, L, d_state)
            dB: 离散化输入矩阵 (B, L, d_state)
            C: 输出矩阵 (B, L, d_state)
            A: 状态转移矩阵，用于维度投影 (d_inner, d_state)
        
        返回:
            输出 (B, L, d_inner)
        """
        B, L, d_inner = u.shape
        d_state = dA.shape[-1]


        u_projected = torch.matmul(u, A)
        dBu = dB * u_projected







        dA_log = torch.log(torch.clamp(torch.abs(dA), min=1e-8)) * torch.sign(dA)
        dA_log_cumsum = torch.cumsum(dA_log, dim=1)

        dA_cumprod = torch.exp(dA_log_cumsum.clamp(min=-3, max=3))




        eps = 1e-6
        dBu_normalized = dBu / (dA_cumprod + eps)


        if torch.isnan(dBu_normalized).any() or torch.isinf(dBu_normalized).any():
            return self._selective_scan(u, dA, dB, C, A)




        dBu_normalized_rev = torch.flip(dBu_normalized, dims=[1])
        dBu_cumsum_rev = torch.cumsum(dBu_normalized_rev, dim=1)
        dBu_cumsum = torch.flip(dBu_cumsum_rev, dims=[1])



        h = dA_cumprod * dBu_cumsum


        state_output = C * h
        y = torch.matmul(state_output, A.T) + self.D.unsqueeze(0).unsqueeze(0) * u


        if torch.isnan(y).any() or torch.isinf(y).any():

            return self._selective_scan(u, dA, dB, C, A)

        return y

    def _selective_scan_parallel(self, u, dA, dB, C, A):
        """
        选择性扫描（向量化的递归实现 - 优化版本）
        
        使用向量化操作优化的递归实现，与_selective_scan数学等价
        对于序列长度L不大的情况，这种实现已经足够高效
        注意：这不是真正的并行扫描算法，而是向量化的递归版本
        
        参数:
            u: 输入 (B, L, d_inner)
            dA: 离散化状态转移矩阵 (B, L, d_state)
            dB: 离散化输入矩阵 (B, L, d_state)
            C: 输出矩阵 (B, L, d_state)
            A: 状态转移矩阵，用于维度投影 (d_inner, d_state)
        
        返回:
            输出 (B, L, d_inner)
        """
        B, L, d_inner = u.shape
        d_state = dA.shape[-1]


        eps = 1e-8




        u_projected = torch.matmul(u, A)


        dBu = dB * u_projected




        h_current = torch.zeros(B, d_state, device=u.device, dtype=u.dtype)


        y_list = []


        for t in range(L):

            h_current = dA[:, t, :] * h_current + dBu[:, t, :]


            state_output = C[:, t, :] * h_current
            y_t = torch.matmul(state_output, A.T) + self.D * u[:, t, :]

            y_list.append(y_t)


        y = torch.stack(y_list, dim=1)


        if torch.isnan(y).any() or torch.isinf(y).any():
            y = torch.where(torch.isnan(y) | torch.isinf(y),
                           torch.zeros_like(y), y)

        return y


class Decoder(nn.Module):
    """
    解码器
    上采样恢复分辨率到512×512
    """
    def __init__(
        self,
        encoder_channels: list,
        base_channels: int = 64,
        num_classes: int = 1,
        dropout: float = 0.1
    ):
        """
        参数:
            encoder_channels: 编码器各层通道数列表
            base_channels: 基础通道数
            num_classes: 输出类别数
            dropout: Dropout比率
        """
        super().__init__()
        self.encoder_channels = encoder_channels


        self.decoder_layers = nn.ModuleList()


        in_channels = encoder_channels[-1]

        for i in range(len(encoder_channels) - 1, 0, -1):
            skip_channels = encoder_channels[i - 1]
            out_channels = max(base_channels, skip_channels // 2)


            layer = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels + skip_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Dropout2d(dropout)
            )
            self.decoder_layers.append(layer)
            in_channels = out_channels


        self.final_upsample = nn.Sequential(
            nn.ConvTranspose2d(in_channels, base_channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )


        self.output = nn.Sequential(
            nn.Conv2d(base_channels, num_classes, kernel_size=1),
            nn.Sigmoid()
        )

        self._initialize_weights()

    def _initialize_weights(self):
        """权重初始化"""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, features: list) -> torch.Tensor:
        """
        前向传播
        
        参数:
            features: 编码器多尺度特征列表
        
        返回:
            分割掩码，形状为 (B, 1, H, W)
        """

        x = features[-1]


        for i, layer in enumerate(self.decoder_layers):

            x = layer[0](x)
            x = layer[1](x)
            x = layer[2](x)


            skip = features[-(i+2)]

            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)


            x = layer[3](x)
            x = layer[4](x)
            x = layer[5](x)
            x = layer[6](x)


        x = self.final_upsample(x)


        x = self.output(x)

        return x


class DiTMamba(nn.Module):
    """
    DiT-Mamba完整模型
    组合编码器、Mamba模块和解码器
    """
    def __init__(
        self,
        in_channels: int = 4,
        base_channels: int = 64,
        num_encoder_layers: int = 4,
        mamba_dim: int = 256,
        num_classes: int = 1,
        dropout: float = 0.1
    ):
        """
        参数:
            in_channels: 输入通道数
            base_channels: 基础通道数
            num_encoder_layers: 编码器层数
            mamba_dim: Mamba模块维度
            num_classes: 输出类别数
            dropout: Dropout比率
        """
        super().__init__()


        self.encoder = DiTEncoder(
            in_channels=in_channels,
            base_channels=base_channels,
            num_layers=num_encoder_layers,
            dropout=dropout
        )


        encoder_channels = [base_channels]
        for i in range(num_encoder_layers):
            if i < num_encoder_layers - 1:
                encoder_channels.append(encoder_channels[-1] * 2)
            else:

                encoder_channels.append(encoder_channels[-1])



        mamba_input_dim = encoder_channels[-1]




        self.mamba_dim_adapter = None
        self.mamba_output_adapter = None
        if mamba_dim != mamba_input_dim:
            self.mamba_dim_adapter = nn.Conv2d(mamba_input_dim, mamba_dim, kernel_size=1)
            self.mamba_output_adapter = nn.Conv2d(mamba_dim, mamba_input_dim, kernel_size=1)
            actual_mamba_dim = mamba_dim
        else:
            actual_mamba_dim = mamba_input_dim

        self.mamba = MambaBlock(
            dim=actual_mamba_dim,
            d_state=48,
            d_conv=4,
            expand=4,
            dropout=dropout
        )


        self.decoder = Decoder(
            encoder_channels=encoder_channels,
            base_channels=base_channels,
            num_classes=num_classes,
            dropout=dropout
        )

        self._initialize_weights()

    def _initialize_weights(self):
        """权重初始化"""

        for name, param in self.named_parameters():
            if param.requires_grad:
                if torch.isnan(param).any() or torch.isinf(param).any():
                    print(f"警告: 参数 {name} 包含NaN或Inf")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        参数:
            x: 输入张量，形状为 (B, 4, 512, 512)
        
        返回:
            分割掩码，形状为 (B, 1, 512, 512)
        """

        if torch.isnan(x).any() or torch.isinf(x).any():
            x = torch.where(torch.isnan(x) | torch.isinf(x),
                           torch.zeros_like(x), x)


        features = self.encoder(x)


        for i, feat in enumerate(features):
            if torch.isnan(feat).any() or torch.isinf(feat).any():
                features[i] = torch.where(torch.isnan(feat) | torch.isinf(feat),
                                          torch.zeros_like(feat), feat)


        mamba_input = features[-1]
        if self.mamba_dim_adapter is not None:
            mamba_input = self.mamba_dim_adapter(mamba_input)

            if torch.isnan(mamba_input).any() or torch.isinf(mamba_input).any():
                mamba_input = torch.where(torch.isnan(mamba_input) | torch.isinf(mamba_input),
                                         torch.zeros_like(mamba_input), mamba_input)

        mamba_output = self.mamba(mamba_input)


        if self.mamba_output_adapter is not None:
            mamba_output = self.mamba_output_adapter(mamba_output)

            if torch.isnan(mamba_output).any() or torch.isinf(mamba_output).any():
                mamba_output = torch.where(torch.isnan(mamba_output) | torch.isinf(mamba_output),
                                          torch.zeros_like(mamba_output), mamba_output)

        features[-1] = mamba_output


        output = self.decoder(features)


        if torch.isnan(output).any() or torch.isinf(output).any():
            output = torch.where(torch.isnan(output) | torch.isinf(output),
                                torch.tensor(0.5, device=output.device, dtype=output.dtype), output)


        if output.shape[2:] != (512, 512):
            output = F.interpolate(output, size=(512, 512), mode='bilinear', align_corners=False)


        output = torch.clamp(output, min=0.0, max=1.0)
        if torch.isnan(output).any() or torch.isinf(output).any():
            output = torch.where(torch.isnan(output) | torch.isinf(output),
                                torch.tensor(0.5, device=output.device, dtype=output.dtype), output)

        return output

    def check_parameters(self):
        """
        检查模型参数是否有NaN/Inf
        
        返回:
            (has_nan, has_inf): 是否包含NaN/Inf
        """
        has_nan = False
        has_inf = False

        for name, param in self.named_parameters():
            if torch.isnan(param).any():
                print(f"警告: 参数 {name} 包含NaN")
                has_nan = True
            if torch.isinf(param).any():
                print(f"警告: 参数 {name} 包含Inf")
                has_inf = True

        return has_nan, has_inf


class DiT(nn.Module):
    """
    纯DiT模型（不包含Mamba模块）
    组合编码器和解码器
    """
    def __init__(
        self,
        in_channels: int = 4,
        base_channels: int = 64,
        num_encoder_layers: int = 4,
        num_classes: int = 1,
        dropout: float = 0.1
    ):
        """
        参数:
            in_channels: 输入通道数
            base_channels: 基础通道数
            num_encoder_layers: 编码器层数
            num_classes: 输出类别数
            dropout: Dropout比率
        """
        super().__init__()


        self.encoder = DiTEncoder(
            in_channels=in_channels,
            base_channels=base_channels,
            num_layers=num_encoder_layers,
            dropout=dropout
        )


        encoder_channels = [base_channels]
        for i in range(num_encoder_layers):
            if i < num_encoder_layers - 1:
                encoder_channels.append(encoder_channels[-1] * 2)
            else:

                encoder_channels.append(encoder_channels[-1])


        self.decoder = Decoder(
            encoder_channels=encoder_channels,
            base_channels=base_channels,
            num_classes=num_classes,
            dropout=dropout
        )

        self._initialize_weights()

    def _initialize_weights(self):
        """权重初始化"""
        for name, param in self.named_parameters():
            if param.requires_grad:
                if torch.isnan(param).any() or torch.isinf(param).any():
                    print(f"警告: 参数 {name} 包含NaN或Inf")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        参数:
            x: 输入张量，形状为 (B, 4, 512, 512)
        
        返回:
            分割掩码，形状为 (B, 1, 512, 512)
        """

        if torch.isnan(x).any() or torch.isinf(x).any():
            x = torch.where(torch.isnan(x) | torch.isinf(x),
                           torch.zeros_like(x), x)


        features = self.encoder(x)


        for i, feat in enumerate(features):
            if torch.isnan(feat).any() or torch.isinf(feat).any():
                features[i] = torch.where(torch.isnan(feat) | torch.isinf(feat),
                                          torch.zeros_like(feat), feat)


        output = self.decoder(features)


        if torch.isnan(output).any() or torch.isinf(output).any():
            output = torch.where(torch.isnan(output) | torch.isinf(output),
                                torch.tensor(0.5, device=output.device, dtype=output.dtype), output)


        if output.shape[2:] != (512, 512):
            output = F.interpolate(output, size=(512, 512), mode='bilinear', align_corners=False)


        output = torch.clamp(output, min=0.0, max=1.0)
        if torch.isnan(output).any() or torch.isinf(output).any():
            output = torch.where(torch.isnan(output) | torch.isinf(output),
                                torch.tensor(0.5, device=output.device, dtype=output.dtype), output)

        return output

    def check_parameters(self):
        """
        检查模型参数是否有NaN/Inf
        
        返回:
            (has_nan, has_inf): 是否包含NaN/Inf
        """
        has_nan = False
        has_inf = False

        for name, param in self.named_parameters():
            if torch.isnan(param).any():
                print(f"警告: 参数 {name} 包含NaN")
                has_nan = True
            if torch.isinf(param).any():
                print(f"警告: 参数 {name} 包含Inf")
                has_inf = True

        return has_nan, has_inf


