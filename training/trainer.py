# -*- coding: utf-8 -*-
"""
训练器实现
包含训练循环、验证循环、防护措施等
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from pathlib import Path
import json
from typing import Dict, Optional, Tuple
import time
from tqdm import tqdm
import logging
from datetime import datetime
import shutil

from training.config import EarlyStopping, WarmupCosineAnnealingLR
from losses.dice_focal_loss import DiceFocalLoss
from metrics.segmentation_metrics import calculate_all_metrics


class Trainer:
    """
    训练器类
    """
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        config: dict,
        checkpoint_dir: str = 'checkpoints',
        log_dir: str = 'logs',
        resume_from: Optional[str] = None,
        model_type: Optional[str] = None
    ):
        """
        参数:
            model: 模型
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            optimizer: 优化器
            criterion: 损失函数
            device: 设备
            config: 训练配置
            checkpoint_dir: 检查点目录
            log_dir: 日志目录
            resume_from: 恢复训练的检查点路径
            model_type: 模型类型，'dit' 或 'ditmamba'，用于生成带类型的checkpoint文件名（可选）
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.config = config
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_dir = Path(log_dir)
        self.model_type = model_type


        self.checkpoint_dir.mkdir(exist_ok=True)
        self.log_dir.mkdir(exist_ok=True)


        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.log_dir / f"training_{timestamp}.log"


        self.logger = logging.getLogger('DiT-Mamba-Training')
        self.logger.setLevel(logging.INFO)


        self.logger.handlers.clear()


        file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)


        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter('%(message)s')
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)

        self.logger.info(f"日志文件: {log_file}")
        self.logger.info("=" * 70)


        self.writer = SummaryWriter(log_dir=str(self.log_dir))


        metrics_config = config.get('metrics', {})
        self.metrics_calc_frequency = metrics_config.get('calc_frequency', 1)


        debug_config = config.get('debug', {})
        self.check_nan_inf_enabled = debug_config.get('check_nan_inf', False)


        self.scheduler = WarmupCosineAnnealingLR(
            optimizer=optimizer,
            warmup_epochs=config.get('lr_scheduler', {}).get('warmup_epochs', 10),
            total_epochs=config.get('base', {}).get('total_epochs', 150),
            warmup_start_lr=config.get('lr_scheduler', {}).get('warmup_start_lr', 1e-6),
            base_lr=config.get('optimizer', {}).get('lr', 1e-4),
            min_lr=config.get('lr_scheduler', {}).get('min_lr', 1e-6),
            cosine_annealing_cycles=config.get('lr_scheduler', {}).get('cosine_annealing_cycles', None)
        )


        self.early_stopping = None
        if config.get('early_stopping', {}).get('enable', True):
            self.early_stopping = EarlyStopping(
                patience=config.get('early_stopping', {}).get('patience', 25),
                min_delta=config.get('early_stopping', {}).get('min_delta', 0.0005),
                mode='max',
                monitor=config.get('early_stopping', {}).get('monitor', 'val_miou'),
                decline_patience=config.get('early_stopping', {}).get('decline_patience', 3),
                decline_threshold=config.get('early_stopping', {}).get('decline_threshold', 0.002)
            )


        self.current_epoch = 0
        self.best_val_miou = 0.0
        self.train_history = []


        self.gradient_accumulation_steps = config.get('base', {}).get('gradient_accumulation_steps', 2)


        self.use_amp = config.get('base', {}).get('mixed_precision', True) and torch.cuda.is_available()
        if self.use_amp:

            self.scaler = torch.cuda.amp.GradScaler(
                init_scale=2.**16,
                growth_factor=2.0,
                backoff_factor=0.5,
                growth_interval=2000
            )
        elif config.get('base', {}).get('mixed_precision', True):
            self.logger.warning("混合精度训练需要GPU，当前使用CPU，已自动禁用混合精度")


        if resume_from:
            self.load_checkpoint(resume_from)

    def check_nan_inf(self, tensor: torch.Tensor, name: str) -> Tuple[bool, bool]:
        """
        检查张量是否包含NaN/Inf
        
        参数:
            tensor: 要检查的张量
            name: 张量名称（用于日志）
        
        返回:
            (has_nan, has_inf)
        """
        has_nan = torch.isnan(tensor).any()
        has_inf = torch.isinf(tensor).any()

        if has_nan:
            self.logger.warning(f"{name} 包含NaN")
        if has_inf:
            self.logger.warning(f"{name} 包含Inf")

        return has_nan.item(), has_inf.item()

    def clip_gradients(self, model: nn.Module, max_norm: float = 1.0):
        """
        梯度裁剪
        
        参数:
            model: 模型
            max_norm: 最大梯度范数
        """
        if self.config.get('gradient', {}).get('clip', True):
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

    def train_epoch(self) -> Dict[str, float]:
        """
        训练一个epoch
        
        返回:
            训练指标字典
        """
        self.model.train()

        total_loss = 0.0
        total_samples = 0
        gradient_norms = []
        skipped_batches = 0


        pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch+1} [Train]')

        self.optimizer.zero_grad()


        accumulated_batches = 0

        for batch_idx, batch in enumerate(pbar):

            inputs = batch['input'].to(self.device)
            targets = batch['mask'].to(self.device)


            if self.check_nan_inf_enabled:
                has_nan, has_inf = self.check_nan_inf(inputs, 'input')
                if has_nan or has_inf:
                    self.logger.warning(f"跳过批次 {batch_idx}: 输入包含NaN/Inf")
                    skipped_batches += 1
                    continue


                has_nan, has_inf = self.check_nan_inf(targets, 'target')
                if has_nan or has_inf:
                    self.logger.warning(f"跳过批次 {batch_idx}: 目标包含NaN/Inf")
                    skipped_batches += 1
                    continue


            try:
                if self.use_amp:
                    with torch.cuda.amp.autocast(enabled=self.use_amp, cache_enabled=True):
                        outputs = self.model(inputs)


                        if self.check_nan_inf_enabled:
                            has_nan, has_inf = self.check_nan_inf(outputs, 'output')
                            if has_nan or has_inf:
                                self.logger.warning(f"跳过批次 {batch_idx}: 输出包含NaN/Inf")
                                skipped_batches += 1
                                continue


                        loss = self.criterion(outputs, targets)


                        if self.check_nan_inf_enabled and (torch.isnan(loss) or torch.isinf(loss)):
                            self.logger.warning(f"跳过批次 {batch_idx}: 损失为NaN/Inf")
                            skipped_batches += 1
                            continue


                        loss = loss / self.gradient_accumulation_steps


                    self.scaler.scale(loss).backward()
                    accumulated_batches += 1
                else:
                    outputs = self.model(inputs)


                    if self.check_nan_inf_enabled:
                        has_nan, has_inf = self.check_nan_inf(outputs, 'output')
                        if has_nan or has_inf:
                            self.logger.warning(f"跳过批次 {batch_idx}: 输出包含NaN/Inf")
                            skipped_batches += 1
                            continue


                    loss = self.criterion(outputs, targets)


                    if self.check_nan_inf_enabled and (torch.isnan(loss) or torch.isinf(loss)):
                        self.logger.warning(f"跳过批次 {batch_idx}: 损失为NaN/Inf")
                        skipped_batches += 1
                        continue


                    loss = loss / self.gradient_accumulation_steps


                    loss.backward()
            except RuntimeError as e:
                if "out of memory" in str(e):
                    torch.cuda.empty_cache()
                    self.logger.error(f"CUDA OOM at batch {batch_idx}, 跳过该批次")
                    skipped_batches += 1

                    if self.use_amp:


                        if accumulated_batches > 0:
                            try:
                                self.scaler.unscale_(self.optimizer)
                                self.optimizer.zero_grad()
                                self.scaler.update()
                            except:

                                self.optimizer.zero_grad()
                        else:

                            self.optimizer.zero_grad()
                        accumulated_batches = 0
                    else:
                        self.optimizer.zero_grad()
                    continue
                else:
                    raise


            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:

                if self.use_amp and accumulated_batches == 0:

                    self.logger.warning(f"批次 {batch_idx}: 梯度累积周期内没有有效的批次，跳过更新")
                    self.optimizer.zero_grad()
                    accumulated_batches = 0
                    continue


                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)

                clip_threshold = self.config.get('gradient', {}).get('clip_threshold', 0.5)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=clip_threshold)


                total_norm = 0.0
                has_nan_grad = False
                has_grad = False
                for name, param in self.model.named_parameters():
                    if param.grad is not None:
                        has_grad = True
                        if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                            self.logger.warning(f"参数 {name} 的梯度包含NaN/Inf，清零")
                            param.grad.zero_()
                            has_nan_grad = True
                        elif any(key in name for key in ['mamba.A_log', 'mamba.A_proj', 'mamba.x_proj', 'mamba.dt_proj', 'mamba.in_proj', 'mamba.out_proj']):




                            mamba_clip_threshold = clip_threshold * 0.3
                            torch.nn.utils.clip_grad_norm_([param], max_norm=mamba_clip_threshold)
                            param_norm = param.grad.data.norm(2)
                            total_norm += param_norm.item() ** 2
                        else:
                            param_norm = param.grad.data.norm(2)
                            total_norm += param_norm.item() ** 2


                if not has_grad:
                    self.logger.warning(f"批次 {batch_idx}: 没有梯度，跳过更新")

                    if self.use_amp and accumulated_batches > 0:

                        try:
                            self.scaler.unscale_(self.optimizer)
                            self.optimizer.zero_grad()
                            self.scaler.update()
                        except:

                            self.optimizer.zero_grad()
                    else:
                        self.optimizer.zero_grad()
                    accumulated_batches = 0
                    continue

                total_norm = total_norm ** (1. / 2)
                gradient_norms.append(total_norm)


                if has_nan_grad:
                    self.logger.warning(f"批次 {batch_idx}: 检测到NaN/Inf梯度，跳过参数更新")


                    has_nan_param = False
                    for name, param in self.model.named_parameters():
                        if torch.isnan(param).any() or torch.isinf(param).any():
                            self.logger.error(f"参数 {name} 本身包含NaN/Inf！")
                            has_nan_param = True

                    if has_nan_param:
                        self.logger.error("模型参数包含NaN/Inf，训练可能已损坏，建议从检查点恢复")


                    if self.use_amp:
                        try:
                            self.optimizer.zero_grad()
                            self.scaler.update()
                        except:

                            self.optimizer.zero_grad()
                    else:
                        self.optimizer.zero_grad()
                    skipped_batches += 1
                    accumulated_batches = 0
                    continue


                if not has_nan_grad:

                    if total_norm > 50.0:
                        self.logger.error(f"严重梯度爆炸: {total_norm:.2f} > 50.0，跳过参数更新")

                        if self.use_amp:
                            try:
                                self.scaler.unscale_(self.optimizer)
                                self.optimizer.zero_grad()

                                self.scaler.update()
                            except:

                                self.optimizer.zero_grad()
                        else:
                            self.optimizer.zero_grad()
                        skipped_batches += 1
                        accumulated_batches = 0
                        continue
                    elif total_norm > 20.0:
                        self.logger.warning(f"梯度范数较大: {total_norm:.2f} > 20.0，使用更激进的裁剪")

                        if self.use_amp:
                            self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=clip_threshold * 0.5)
                    else:


                        if self.use_amp:

                            pass


                    if total_norm <= 50.0:
                        if self.use_amp:
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                        else:
                            self.optimizer.step()



                        with torch.no_grad():
                            for name, param in self.model.named_parameters():
                                if 'mamba.A_log' in name:

                                    param.clamp_(min=-4, max=1)
                                elif 'mamba.A_proj' in name:
                                    param.clamp_(min=-2, max=2)

                self.optimizer.zero_grad()
                accumulated_batches = 0


            total_loss += loss.item() * self.gradient_accumulation_steps
            total_samples += inputs.size(0)


            pbar.set_postfix({
                'loss': f'{loss.item() * self.gradient_accumulation_steps:.4f}',
                'avg_loss': f'{total_loss / (batch_idx + 1 - skipped_batches):.4f}' if (batch_idx + 1 - skipped_batches) > 0 else '0.0000'
            })



        total_batches = len(self.train_loader) - skipped_batches
        if total_batches > 0 and total_batches % self.gradient_accumulation_steps != 0:

            if self.use_amp and accumulated_batches == 0:

                self.logger.warning("最后批次: 梯度累积周期内没有有效的批次，跳过更新")
                self.optimizer.zero_grad()
            else:

                has_grad = False
                for param in self.model.parameters():
                    if param.grad is not None and param.grad.abs().sum() > 0:
                        has_grad = True
                        break

                if has_grad:

                    if self.use_amp:
                        self.scaler.unscale_(self.optimizer)

                    clip_threshold = self.config.get('gradient', {}).get('clip_threshold', 0.5)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=clip_threshold)


                    total_norm = 0.0
                    has_nan_grad = False
                    for name, param in self.model.named_parameters():
                        if param.grad is not None:
                            if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                                self.logger.warning(f"参数 {name} 的梯度包含NaN/Inf，清零")
                                param.grad.zero_()
                                has_nan_grad = True
                            elif any(key in name for key in ['mamba.A_log', 'mamba.A_proj', 'mamba.x_proj', 'mamba.dt_proj', 'mamba.in_proj', 'mamba.out_proj']):




                                mamba_clip_threshold = clip_threshold * 0.3
                                torch.nn.utils.clip_grad_norm_([param], max_norm=mamba_clip_threshold)
                                param_norm = param.grad.data.norm(2)
                                total_norm += param_norm.item() ** 2
                            else:
                                param_norm = param.grad.data.norm(2)
                                total_norm += param_norm.item() ** 2

                    total_norm = total_norm ** (1. / 2)
                    gradient_norms.append(total_norm)


                    if has_nan_grad:
                        self.logger.warning(f"最后批次: 检测到NaN/Inf梯度，跳过参数更新")


                        has_nan_param = False
                        for name, param in self.model.named_parameters():
                            if torch.isnan(param).any() or torch.isinf(param).any():
                                self.logger.error(f"参数 {name} 本身包含NaN/Inf！")
                                has_nan_param = True

                        if has_nan_param:
                            self.logger.error("模型参数包含NaN/Inf，训练可能已损坏，建议从检查点恢复")


                        if self.use_amp:
                            try:
                                self.optimizer.zero_grad()
                                self.scaler.update()
                            except:

                                self.optimizer.zero_grad()
                        else:
                            self.optimizer.zero_grad()
                    else:


                        if total_norm > 50.0:
                            self.logger.error(f"严重梯度爆炸（最后批次）: {total_norm:.2f} > 50.0，跳过参数更新")

                            if self.use_amp:
                                try:
                                    self.scaler.unscale_(self.optimizer)
                                    self.optimizer.zero_grad()

                                    self.scaler.update()
                                except:

                                    self.optimizer.zero_grad()
                            else:
                                self.optimizer.zero_grad()
                        elif total_norm > 20.0:
                            self.logger.warning(f"梯度范数较大（最后批次）: {total_norm:.2f} > 20.0，使用更激进的裁剪")

                            if self.use_amp:
                                self.scaler.unscale_(self.optimizer)
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=clip_threshold * 0.5)


                            if self.use_amp:
                                self.scaler.step(self.optimizer)
                                self.scaler.update()
                            else:
                                self.optimizer.step()



                            with torch.no_grad():
                                for name, param in self.model.named_parameters():
                                    if 'mamba.A_log' in name:
                                        param.clamp_(min=-5, max=2)
                                    elif 'mamba.A_proj' in name:
                                        param.clamp_(min=-2, max=2)
                        else:


                            if self.use_amp:

                                pass


                            if self.use_amp:
                                self.scaler.step(self.optimizer)
                                self.scaler.update()
                            else:
                                self.optimizer.step()



                            with torch.no_grad():
                                for name, param in self.model.named_parameters():
                                    if 'mamba.A_log' in name:
                                        param.clamp_(min=-5, max=2)
                                    elif 'mamba.A_proj' in name:
                                        param.clamp_(min=-2, max=2)
                else:


                    if self.use_amp and accumulated_batches > 0:

                        try:
                            self.scaler.unscale_(self.optimizer)
                            self.optimizer.zero_grad()
                            self.scaler.update()
                        except:

                            self.optimizer.zero_grad()
                    else:
                        self.optimizer.zero_grad()

                self.optimizer.zero_grad()


        avg_loss = total_loss / (len(self.train_loader) - skipped_batches) if (len(self.train_loader) - skipped_batches) > 0 else 0.0
        avg_grad_norm = np.mean(gradient_norms) if gradient_norms else 0.0
        max_grad_norm = np.max(gradient_norms) if gradient_norms else 0.0


        if max_grad_norm > 10.0:
            self.logger.warning(f"梯度范数过大: {max_grad_norm:.2f} > 10.0")

        return {
            'loss': avg_loss,
            'grad_norm_mean': avg_grad_norm,
            'grad_norm_max': max_grad_norm,
            'skipped_batches': skipped_batches
        }

    def validate_epoch(self) -> Dict[str, float]:
        """
        验证一个epoch
        
        返回:
            验证指标字典
        """
        self.model.eval()

        total_loss = 0.0
        all_preds = []
        all_targets = []
        skipped_batches = 0

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f'Epoch {self.current_epoch+1} [Val]')

            for batch_idx, batch in enumerate(pbar):

                inputs = batch['input'].to(self.device)
                targets = batch['mask'].to(self.device)


                if self.check_nan_inf_enabled:
                    has_nan, has_inf = self.check_nan_inf(inputs, 'input')
                    if has_nan or has_inf:
                        self.logger.warning(f"跳过批次 {batch_idx}: 输入包含NaN/Inf")
                        skipped_batches += 1
                        continue


                try:
                    if self.use_amp:
                        with torch.cuda.amp.autocast(enabled=self.use_amp, cache_enabled=True):
                            outputs = self.model(inputs)
                            loss = self.criterion(outputs, targets)
                    else:
                        outputs = self.model(inputs)
                        loss = self.criterion(outputs, targets)


                    if self.check_nan_inf_enabled:
                        has_nan, has_inf = self.check_nan_inf(outputs, 'output')
                        if has_nan or has_inf or torch.isnan(loss) or torch.isinf(loss):
                            self.logger.warning(f"跳过批次 {batch_idx}: 输出或损失包含NaN/Inf")
                            skipped_batches += 1
                            continue
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        torch.cuda.empty_cache()
                        self.logger.error(f"CUDA OOM at validation batch {batch_idx}, 跳过该批次")
                        skipped_batches += 1
                        continue
                    else:
                        raise


                pred_binary = (outputs > 0.5).long().cpu()


                target_processed = targets.clone()
                if target_processed.max() > 1.0:

                    target_processed = (target_processed > 0).long()
                else:

                    target_processed = (target_processed > 0.5).long()

                target_binary = target_processed.cpu()

                all_preds.append(pred_binary)
                all_targets.append(target_binary)

                total_loss += loss.item()

                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'avg_loss': f'{total_loss / (batch_idx + 1 - skipped_batches):.4f}' if (batch_idx + 1 - skipped_batches) > 0 else '0.0000'
                })


        if len(all_preds) > 0:

            should_calc_metrics = (self.current_epoch + 1) % self.metrics_calc_frequency == 0

            if should_calc_metrics:
                try:
                    all_preds = torch.cat(all_preds, dim=0)
                    all_targets = torch.cat(all_targets, dim=0)


                    if all_preds.shape != all_targets.shape:
                        self.logger.error(f"预测和目标形状不匹配: {all_preds.shape} vs {all_targets.shape}")
                        metrics = self._get_default_metrics()
                        metrics['loss'] = total_loss / (len(self.val_loader) - skipped_batches) if (len(self.val_loader) - skipped_batches) > 0 else 0.0
                    else:
                        metrics = calculate_all_metrics(all_preds, all_targets, num_classes=2)
                        metrics['loss'] = total_loss / (len(self.val_loader) - skipped_batches) if (len(self.val_loader) - skipped_batches) > 0 else 0.0
                except Exception as e:
                    self.logger.error(f"计算验证指标失败: {e}", exc_info=True)
                    metrics = self._get_default_metrics()
                    metrics['loss'] = total_loss / (len(self.val_loader) - skipped_batches) if (len(self.val_loader) - skipped_batches) > 0 else 0.0
            else:

                metrics = self._get_default_metrics()
                metrics['loss'] = total_loss / (len(self.val_loader) - skipped_batches) if (len(self.val_loader) - skipped_batches) > 0 else 0.0
        else:
            metrics = self._get_default_metrics()
            metrics['loss'] = 0.0

        return metrics

    def _get_default_metrics(self) -> Dict[str, float]:
        """获取默认指标值（当计算失败时使用）"""
        return {
            'loss': 0.0,
            'miou': 0.0,
            'mprecision': 0.0,
            'recall': 0.0,
            'mpa': 0.0,
            'f1': 0.0,
            'mf1': 0.0
        }

    def check_training_health(self, train_metrics: Dict, val_metrics: Dict, current_lr: float):
        """
        检查训练健康状态
        
        参数:
            train_metrics: 训练指标
            val_metrics: 验证指标
            current_lr: 当前学习率
        """
        warnings = []


        if train_metrics['loss'] < 1e-6:
            warnings.append(f"严重过拟合: 训练损失({train_metrics['loss']:.6f})接近0，建议增加正则化或提前停止")

            if hasattr(self, 'early_stopping') and self.early_stopping is not None:
                self.early_stopping.should_stop = True


        train_val_gap = val_metrics['loss'] - train_metrics['loss']
        if train_val_gap > 0.5:
            warnings.append(f"严重过拟合: 训练损失({train_metrics['loss']:.4f})明显低于验证损失({val_metrics['loss']:.4f})，差距={train_val_gap:.4f}，建议立即停止训练")

            if hasattr(self, 'early_stopping') and self.early_stopping is not None:
                self.early_stopping.should_stop = True
        elif train_val_gap > 0.3:
            warnings.append(f"可能过拟合: 训练损失({train_metrics['loss']:.4f})明显低于验证损失({val_metrics['loss']:.4f})，差距={train_val_gap:.4f}")
        elif train_val_gap > 0.1:
            warnings.append(f"轻微过拟合迹象: 训练损失({train_metrics['loss']:.4f})低于验证损失({val_metrics['loss']:.4f})，差距={train_val_gap:.4f}")


        if len(self.train_history) >= 5:
            recent_train = [h['train_loss'] for h in self.train_history[-5:]]
            recent_val = [h['val_loss'] for h in self.train_history[-5:]]

            train_decreasing = all(recent_train[i] >= recent_train[i+1] for i in range(len(recent_train)-1))
            val_increasing = all(recent_val[i] <= recent_val[i+1] for i in range(len(recent_val)-1))
            if train_decreasing and val_increasing:
                warnings.append("检测到过拟合趋势: 训练损失下降但验证损失上升（最近5个epoch）")


        max_grad_norm = train_metrics.get('grad_norm_max', 0.0)
        if max_grad_norm > 50.0:
            warnings.append(f"严重梯度爆炸: 最大梯度范数={max_grad_norm:.2f} > 50.0")
        elif max_grad_norm > 20.0:
            warnings.append(f"梯度范数较大: 最大梯度范数={max_grad_norm:.2f} > 20.0")


        base_lr = self.config.get('optimizer', {}).get('lr', 1e-4)
        if current_lr < base_lr * 0.01:
            warnings.append(f"学习率过低: {current_lr:.2e} < {base_lr * 0.01:.2e} (初始学习率的1%)")


        if len(self.train_history) >= 10:
            recent_val_losses = [h['val_loss'] for h in self.train_history[-10:]]
            val_loss_range = max(recent_val_losses) - min(recent_val_losses)
            if val_loss_range < 0.001:
                warnings.append(f"验证损失停滞: 最近10个epoch变化范围={val_loss_range:.6f} < 0.001")


        if len(self.train_history) >= 10:
            recent_val_miou = [h['val_miou'] for h in self.train_history[-10:]]
            val_miou_range = max(recent_val_miou) - min(recent_val_miou)
            if val_miou_range < 0.001:
                warnings.append(f"验证MIoU停滞: 最近10个epoch变化范围={val_miou_range:.6f} < 0.001")


        for warning in warnings:
            self.logger.warning(f"训练健康检查: {warning}")

    def save_checkpoint(self, is_best: bool = False):
        """
        保存检查点
        
        参数:
            is_best: 是否为最佳模型
        """
        try:

            try:
                free_space = shutil.disk_usage(self.checkpoint_dir).free
                if free_space < 100 * 1024 * 1024:
                    self.logger.warning(f"磁盘空间不足: {free_space / 1024**2:.2f}MB 可用")

                    self._cleanup_old_checkpoints()
            except Exception as e:
                self.logger.warning(f"检查磁盘空间失败: {e}")

            checkpoint = {
                'epoch': self.current_epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.current_epoch if hasattr(self.scheduler, 'current_epoch') else self.current_epoch,
                'best_val_miou': self.best_val_miou,
                'train_history': self.train_history,
                'config': self.config
            }

            if self.use_amp:
                checkpoint['scaler_state_dict'] = self.scaler.state_dict()


            if self.model_type:
                latest_filename = f'latest-{self.model_type}.pth'
                best_filename = f'best_model-{self.model_type}.pth'
            else:
                latest_filename = 'latest.pth'
                best_filename = 'best_model.pth'


            checkpoint_path = self.checkpoint_dir / latest_filename
            try:
                torch.save(checkpoint, checkpoint_path)
            except OSError as e:
                if "No space left" in str(e) or "磁盘空间" in str(e):
                    self.logger.error("磁盘空间不足，无法保存检查点")

                    self._cleanup_old_checkpoints()
                    torch.save(checkpoint, checkpoint_path)
                else:
                    raise


            if is_best:
                best_path = self.checkpoint_dir / best_filename
                try:
                    torch.save(checkpoint, best_path)
                    self.logger.info(f"保存最佳模型: {best_path}")
                except OSError as e:
                    if "No space left" in str(e) or "磁盘空间" in str(e):
                        self.logger.error("磁盘空间不足，无法保存最佳模型")
                        self._cleanup_old_checkpoints()
                        torch.save(checkpoint, best_path)
                    else:
                        raise
        except Exception as e:
            self.logger.error(f"保存检查点失败: {e}", exc_info=True)
            raise

    def _cleanup_old_checkpoints(self, keep_latest: int = 3):
        """清理旧的检查点文件以释放空间"""
        try:
            checkpoint_files = sorted(self.checkpoint_dir.glob('*.pth'), key=lambda x: x.stat().st_mtime, reverse=True)
            if len(checkpoint_files) > keep_latest:
                for old_file in checkpoint_files[keep_latest:]:
                    try:
                        old_file.unlink()
                        self.logger.info(f"已删除旧检查点: {old_file}")
                    except Exception as e:
                        self.logger.warning(f"删除旧检查点失败 {old_file}: {e}")
        except Exception as e:
            self.logger.warning(f"清理旧检查点失败: {e}")

    def load_checkpoint(self, checkpoint_path: str):
        """
        加载检查点
        
        参数:
            checkpoint_path: 检查点路径
        """
        checkpoint_path_obj = Path(checkpoint_path)
        if not checkpoint_path_obj.exists():
            raise FileNotFoundError(f"检查点文件不存在: {checkpoint_path}")

        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
        except Exception as e:
            self.logger.error(f"加载检查点失败: {e}")
            raise

        try:

            try:
                self.model.load_state_dict(checkpoint['model_state_dict'], strict=True)
            except RuntimeError as e:
                if "Missing key(s)" in str(e) or "Unexpected key(s)" in str(e):
                    self.logger.warning(f"模型架构不完全匹配: {e}")
                    self.logger.warning("尝试非严格模式加载...")
                    self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                else:
                    raise

            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.current_epoch = checkpoint['epoch']
            self.best_val_miou = checkpoint.get('best_val_miou', 0.0)
            self.train_history = checkpoint.get('train_history', [])


            if 'scheduler_state_dict' in checkpoint:
                try:
                    if hasattr(self.scheduler, 'current_epoch'):
                        self.scheduler.current_epoch = checkpoint['scheduler_state_dict']

                        self.scheduler.step(self.scheduler.current_epoch)
                except Exception as e:
                    self.logger.warning(f"恢复调度器状态失败: {e}，将从当前epoch重新开始")

                    for _ in range(self.current_epoch):
                        self.scheduler.step()

            if self.use_amp and 'scaler_state_dict' in checkpoint:
                try:
                    self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
                except Exception as e:
                    self.logger.warning(f"恢复梯度缩放器状态失败: {e}，将重新初始化")
                    self.scaler = torch.cuda.amp.GradScaler()

            self.logger.info(f"从检查点恢复训练: epoch {self.current_epoch}, best_val_miou: {self.best_val_miou:.4f}")
        except Exception as e:
            self.logger.error(f"恢复检查点状态失败: {e}")
            raise

    def train(self):
        """
        主训练循环
        """
        try:
            self.logger.info("=" * 70)
            self.logger.info("开始训练")
            self.logger.info("=" * 70)
            self.logger.info(f"设备: {self.device}")
            self.logger.info(f"总轮数: {self.config.get('base', {}).get('total_epochs', 150)}")
            self.logger.info(f"批次大小: {self.config.get('base', {}).get('batch_size', 4)}")
            self.logger.info(f"梯度累积步数: {self.gradient_accumulation_steps}")
            self.logger.info(f"混合精度: {self.use_amp}")
            self.logger.info("=" * 70)

            for epoch in range(self.current_epoch, self.config.get('base', {}).get('total_epochs', 150)):
                self.current_epoch = epoch


                train_metrics = self.train_epoch()


                val_metrics = self.validate_epoch()


                self.scheduler.step(epoch)
                current_lr = self.scheduler.get_lr()


                metrics = {
                    'epoch': epoch + 1,
                    'train_loss': train_metrics['loss'],
                    'val_loss': val_metrics['loss'],
                    'val_miou': val_metrics['miou'],
                    'val_mprecision': val_metrics['mprecision'],
                    'val_recall': val_metrics['recall'],
                    'val_mpa': val_metrics['mpa'],
                    'lr': current_lr,
                    'grad_norm_mean': train_metrics.get('grad_norm_mean', 0.0),
                    'grad_norm_max': train_metrics.get('grad_norm_max', 0.0)
                }

                self.train_history.append(metrics)


                try:
                    self.writer.add_scalar('Loss/Train', train_metrics['loss'], epoch + 1)
                    self.writer.add_scalar('Loss/Val', val_metrics['loss'], epoch + 1)
                    self.writer.add_scalar('Metrics/Val_MIoU', val_metrics['miou'], epoch + 1)
                    self.writer.add_scalar('Metrics/Val_MPrecision', val_metrics['mprecision'], epoch + 1)
                    self.writer.add_scalar('Metrics/Val_Recall', val_metrics['recall'], epoch + 1)
                    self.writer.add_scalar('Metrics/Val_MPA', val_metrics['mpa'], epoch + 1)
                    self.writer.add_scalar('LearningRate', current_lr, epoch + 1)
                    self.writer.add_scalar('Gradient/Norm_Mean', train_metrics.get('grad_norm_mean', 0.0), epoch + 1)
                    self.writer.add_scalar('Gradient/Norm_Max', train_metrics.get('grad_norm_max', 0.0), epoch + 1)
                except Exception as e:
                    self.logger.warning(f"TensorBoard写入失败: {e}，继续训练")


                self.logger.info(f"\nEpoch {epoch + 1}/{self.config.get('base', {}).get('total_epochs', 150)}")
                self.logger.info(f"  Train Loss: {train_metrics['loss']:.4f}")
                self.logger.info(f"  Val Loss: {val_metrics['loss']:.4f}")
                self.logger.info(f"  Val MIoU: {val_metrics['miou']:.4f}")
                self.logger.info(f"  Val MPrecision: {val_metrics['mprecision']:.4f}")
                self.logger.info(f"  Val Recall: {val_metrics['recall']:.4f}")
                self.logger.info(f"  Val MPA: {val_metrics['mpa']:.4f}")
                self.logger.info(f"  Learning Rate: {current_lr:.2e}")
                self.logger.info(f"  Gradient Norm: {train_metrics.get('grad_norm_mean', 0.0):.4f} (max: {train_metrics.get('grad_norm_max', 0.0):.4f})")


                self.check_training_health(train_metrics, val_metrics, current_lr)



                should_calc_metrics = (self.current_epoch + 1) % self.metrics_calc_frequency == 0
                is_best = False
                if should_calc_metrics and val_metrics['miou'] > 0:
                    is_best = val_metrics['miou'] > self.best_val_miou
                    if is_best:
                        self.best_val_miou = val_metrics['miou']
                        self.logger.info(f"  新的最佳模型! Val MIoU: {self.best_val_miou:.4f}")


                self.save_checkpoint(is_best=is_best)


                try:
                    history_path = self.log_dir / 'training_history.json'

                    serializable_history = []
                    for item in self.train_history:
                        serializable_item = {}
                        for k, v in item.items():
                            if isinstance(v, torch.Tensor):
                                serializable_item[k] = v.item() if v.numel() == 1 else v.tolist()
                            elif isinstance(v, (np.integer, np.floating)):
                                serializable_item[k] = float(v)
                            elif isinstance(v, np.ndarray):
                                serializable_item[k] = v.tolist()
                            else:
                                serializable_item[k] = v
                        serializable_history.append(serializable_item)

                    with open(history_path, 'w', encoding='utf-8') as f:
                        json.dump(serializable_history, f, indent=2, ensure_ascii=False)
                except (TypeError, ValueError, OSError) as e:
                    self.logger.warning(f"保存训练历史失败: {e}")


                if self.early_stopping:


                    if (self.current_epoch + 1) % self.metrics_calc_frequency == 0:
                        if self.early_stopping(val_metrics['miou']):
                            self.logger.info(f"\n早停触发! 最佳Val MIoU: {self.best_val_miou:.4f}")
                            break
        except KeyboardInterrupt:
            self.logger.warning("\n训练被用户中断（Ctrl+C）")
            self.logger.info("正在保存检查点...")
            try:
                self.save_checkpoint(is_best=False)
                self.logger.info("检查点已保存，可以稍后恢复训练")
            except Exception as e:
                self.logger.error(f"保存检查点失败: {e}")
            raise
        except Exception as e:
            self.logger.error(f"训练过程中发生错误: {e}", exc_info=True)
            self.logger.info("正在保存检查点...")
            try:
                self.save_checkpoint(is_best=False)
                self.logger.info("检查点已保存")
            except Exception as save_error:
                self.logger.error(f"保存检查点失败: {save_error}")
            raise
        finally:

            if hasattr(self, 'writer'):
                try:
                    self.writer.close()
                except:
                    pass
            self.logger.info("训练结束，资源已清理")
            self.logger.info(f"最佳Val MIoU: {self.best_val_miou:.4f}")



