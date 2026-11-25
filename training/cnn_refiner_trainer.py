# -*- coding: utf-8 -*-
"""
CNN Refiner训练器
参照 DiffusionTrainer 的训练/验证口径实现
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from typing import Dict, Optional
from tqdm import tqdm
import logging
from datetime import datetime
import shutil

from training.config import EarlyStopping, WarmupCosineAnnealingLR
from metrics.segmentation_metrics import calculate_all_metrics


def dice_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    pred = pred.contiguous()
    target = target.contiguous()

    intersection = (pred * target).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = (2.0 * intersection + eps) / (union + eps)
    return 1.0 - dice.mean()


class RefinerLoss(nn.Module):
    """
    CNN Refiner loss:
        total = dice + bce + lambda_residual * smooth_l1
    """
    def __init__(
        self,
        lambda_residual: float = 0.2,
        dice_weight: float = 1.0,
        bce_weight: float = 1.0
    ):
        super().__init__()
        self.lambda_residual = lambda_residual
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.bce = nn.BCELoss()
        self.smooth_l1 = nn.SmoothL1Loss()

    def forward(
        self,
        pred_refined: torch.Tensor,
        target_mask: torch.Tensor,
        pred_residual: torch.Tensor,
        target_residual: torch.Tensor
    ):
        loss_dice = dice_loss(pred_refined, target_mask)
        loss_bce = self.bce(pred_refined, target_mask)
        loss_res = self.smooth_l1(pred_residual, target_residual)

        total = (
            self.dice_weight * loss_dice
            + self.bce_weight * loss_bce
            + self.lambda_residual * loss_res
        )

        return total, {
            'loss_dice': loss_dice.item(),
            'loss_bce': loss_bce.item(),
            'loss_res': loss_res.item(),
        }


class CNNRefinerTrainer:
    """
    CNN Refiner训练器
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
        resume_from: Optional[str] = None
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.config = config
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_dir = Path(log_dir)

        self.checkpoint_dir.mkdir(exist_ok=True)
        self.log_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.log_dir / f"cnn_refiner_training_{timestamp}.log"

        self.logger = logging.getLogger(f'CNN-Refiner-Training-{timestamp}')
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

        self.writer = SummaryWriter(log_dir=str(self.log_dir / 'tensorboard_cnn_refiner'))

        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_val_miou = 0.0
        self.train_history = []

        self.gradient_accumulation_steps = config.get('base', {}).get('gradient_accumulation_steps', 1)
        self.use_amp = config.get('base', {}).get('mixed_precision', False) and torch.cuda.is_available()
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()

        val_config = config.get('cnn_refiner', {}).get('validation', {})
        self.val_frequency = val_config.get('frequency', 1)
        self.threshold = val_config.get('threshold', 0.5)

        early_stop_config = config.get('early_stopping', {})
        if early_stop_config.get('enable', True):
            monitor = early_stop_config.get('monitor', 'val_miou')
            mode = 'max' if monitor == 'val_miou' else 'min'
            self.early_stopping = EarlyStopping(
                patience=early_stop_config.get('patience', 20),
                min_delta=early_stop_config.get('min_delta', 0.001),
                mode=mode,
                monitor=monitor,
                decline_patience=early_stop_config.get('decline_patience', 5),
                decline_threshold=early_stop_config.get('decline_threshold', 0.005)
            )
        else:
            self.early_stopping = None

        lr_config = config.get('lr_scheduler', {})
        if lr_config.get('strategy') == 'warmup_cosine':
            total_epochs = config.get('cnn_refiner', {}).get('training', {}).get('num_epochs', 100)
            cnn_cfg = config.get('cnn_refiner', {}).get('training', {})
            base_lr = cnn_cfg.get('learning_rate', config.get('optimizer', {}).get('lr', 1e-4))
            self.scheduler = WarmupCosineAnnealingLR(
                optimizer=self.optimizer,
                warmup_epochs=lr_config.get('warmup_epochs', 5),
                total_epochs=total_epochs,
                warmup_start_lr=lr_config.get('warmup_start_lr', 1e-6),
                base_lr=base_lr,
                min_lr=lr_config.get('min_lr', 1e-6),
                cosine_annealing_cycles=lr_config.get('cosine_annealing_cycles', total_epochs)
            )
        else:
            self.scheduler = None

        if resume_from:
            self.load_checkpoint(resume_from)

    def train_epoch(self) -> Dict[str, float]:
        self.model.train()

        total_loss = 0.0
        skipped_batches = 0

        pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch+1} [Train]')
        self.optimizer.zero_grad()
        accumulated_batches = 0

        for batch_idx, batch in enumerate(pbar):
            input_images = batch['input_image'].to(self.device)
            pred_coarse = batch['pred_coarse'].to(self.device)
            residuals = batch['residual'].to(self.device)
            target_masks = batch['target_mask'].to(self.device)

            condition = torch.cat([input_images, pred_coarse], dim=1)

            try:
                if self.use_amp:
                    with torch.cuda.amp.autocast(enabled=True, cache_enabled=True):
                        pred_residual = self.model(condition)
                        final_masks = torch.clamp(pred_coarse + pred_residual, 0.0, 1.0)
                        loss, _ = self.criterion(final_masks, target_masks, pred_residual, residuals)
                        loss = loss / self.gradient_accumulation_steps

                    self.scaler.scale(loss).backward()
                    accumulated_batches += 1
                else:
                    pred_residual = self.model(condition)
                    final_masks = torch.clamp(pred_coarse + pred_residual, 0.0, 1.0)
                    loss, _ = self.criterion(final_masks, target_masks, pred_residual, residuals)
                    loss = loss / self.gradient_accumulation_steps
                    loss.backward()
                    accumulated_batches += 1

                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    if self.use_amp:
                        self.scaler.unscale_(self.optimizer)

                    clip_threshold = self.config.get('gradient', {}).get('clip_threshold', 1.0)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=clip_threshold)

                    if self.use_amp:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()

                    self.optimizer.zero_grad()
                    accumulated_batches = 0

                total_loss += loss.item() * self.gradient_accumulation_steps

                pbar.set_postfix({
                    'loss': f'{loss.item() * self.gradient_accumulation_steps:.4f}',
                    'avg_loss': f'{total_loss / (batch_idx + 1 - skipped_batches):.4f}' if (batch_idx + 1 - skipped_batches) > 0 else '0.0000'
                })

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    torch.cuda.empty_cache()
                    self.logger.error(f"CUDA OOM at batch {batch_idx}, 跳过该批次")
                    skipped_batches += 1
                    if self.use_amp and accumulated_batches > 0:
                        try:
                            self.scaler.unscale_(self.optimizer)
                            self.optimizer.zero_grad()
                            self.scaler.update()
                        except Exception:
                            self.optimizer.zero_grad()
                        accumulated_batches = 0
                    else:
                        self.optimizer.zero_grad()
                    continue
                raise

        if accumulated_batches > 0:
            if self.use_amp:
                self.scaler.unscale_(self.optimizer)
                clip_threshold = self.config.get('gradient', {}).get('clip_threshold', 1.0)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=clip_threshold)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                clip_threshold = self.config.get('gradient', {}).get('clip_threshold', 1.0)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=clip_threshold)
                self.optimizer.step()
            self.optimizer.zero_grad()

        avg_loss = total_loss / (len(self.train_loader) - skipped_batches) if (len(self.train_loader) - skipped_batches) > 0 else 0.0
        return {'loss': avg_loss}

    def validate_epoch(self) -> Dict[str, float]:
        self.model.eval()

        total_loss = 0.0
        all_preds = []
        all_targets = []
        all_baseline_preds = []

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f'Epoch {self.current_epoch+1} [Val]')

            for batch_idx, batch in enumerate(pbar):
                input_images = batch['input_image'].to(self.device)
                pred_coarse = batch['pred_coarse'].to(self.device)
                residuals = batch['residual'].to(self.device)
                target_masks = batch['target_mask'].to(self.device)

                condition = torch.cat([input_images, pred_coarse], dim=1)
                pred_residual = self.model(condition)
                final_masks = torch.clamp(pred_coarse + pred_residual, 0.0, 1.0)

                loss, _ = self.criterion(final_masks, target_masks, pred_residual, residuals)
                total_loss += loss.item()

                pred_binary = (final_masks > self.threshold).long().cpu()
                target_binary = (target_masks > self.threshold).long().cpu()
                baseline_binary = (pred_coarse > self.threshold).long().cpu()

                all_preds.append(pred_binary)
                all_targets.append(target_binary)
                all_baseline_preds.append(baseline_binary)

                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'avg_loss': f'{total_loss / (batch_idx + 1):.4f}'
                })

        if len(all_preds) > 0:
            all_preds = torch.cat(all_preds, dim=0)
            all_targets = torch.cat(all_targets, dim=0)
            all_baseline_preds = torch.cat(all_baseline_preds, dim=0)

            metrics = calculate_all_metrics(all_preds, all_targets, num_classes=2)
            metrics['loss'] = total_loss / len(self.val_loader)

            baseline_metrics = calculate_all_metrics(all_baseline_preds, all_targets, num_classes=2)
            metrics['baseline_miou'] = baseline_metrics['miou']
            metrics['baseline_mprecision'] = baseline_metrics['mprecision']
            metrics['baseline_recall'] = baseline_metrics['recall']
            metrics['baseline_mpa'] = baseline_metrics['mpa']
            metrics['baseline_f1'] = baseline_metrics['f1']
            metrics['baseline_mf1'] = baseline_metrics['mf1']

            metrics['miou_improvement'] = metrics['miou'] - baseline_metrics['miou']
            metrics['mprecision_improvement'] = metrics['mprecision'] - baseline_metrics['mprecision']
            metrics['recall_improvement'] = metrics['recall'] - baseline_metrics['recall']
            metrics['mpa_improvement'] = metrics['mpa'] - baseline_metrics['mpa']
            metrics['f1_improvement'] = metrics['f1'] - baseline_metrics['f1']
            metrics['mf1_improvement'] = metrics['mf1'] - baseline_metrics['mf1']
        else:
            metrics = {
                'loss': 0.0,
                'miou': 0.0, 'mprecision': 0.0, 'recall': 0.0, 'mpa': 0.0, 'f1': 0.0, 'mf1': 0.0,
                'baseline_miou': 0.0, 'baseline_mprecision': 0.0, 'baseline_recall': 0.0,
                'baseline_mpa': 0.0, 'baseline_f1': 0.0, 'baseline_mf1': 0.0,
                'miou_improvement': 0.0, 'mprecision_improvement': 0.0, 'recall_improvement': 0.0,
                'mpa_improvement': 0.0, 'f1_improvement': 0.0, 'mf1_improvement': 0.0
            }

        return metrics

    def save_checkpoint(self, is_best: bool = False):
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_val_miou': self.best_val_miou,
            'train_history': self.train_history,
            'config': self.config
        }

        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        if self.use_amp:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()

        checkpoint_path = self.checkpoint_dir / 'cnn_refiner_latest.pth'
        torch.save(checkpoint, checkpoint_path)

        if is_best:
            best_path = self.checkpoint_dir / 'cnn_refiner_best.pth'
            shutil.copy(checkpoint_path, best_path)
            self.logger.info(f"保存最佳模型: {best_path}")

    def load_checkpoint(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.best_val_miou = checkpoint.get('best_val_miou', 0.0)
        self.train_history = checkpoint.get('train_history', [])

        if 'scheduler_state_dict' in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.scheduler.current_epoch = self.current_epoch

        if 'scaler_state_dict' in checkpoint and self.use_amp:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        self.logger.info(f"从检查点恢复训练: epoch {self.current_epoch}")

    def train(self):
        total_epochs = self.config.get('cnn_refiner', {}).get('training', {}).get('num_epochs', 100)

        self.logger.info("=" * 70)
        self.logger.info("开始CNN Refiner训练")
        self.logger.info("=" * 70)
        self.logger.info(f"总轮数: {total_epochs}")
        self.logger.info(f"训练集样本数: {len(self.train_loader.dataset)}")
        self.logger.info(f"验证集样本数: {len(self.val_loader.dataset)}")
        self.logger.info(f"验证频率: 每{self.val_frequency}个epoch")

        for epoch in range(self.current_epoch, total_epochs):
            self.current_epoch = epoch

            train_metrics = self.train_epoch()
            self.writer.add_scalar('Train/Loss', train_metrics['loss'], epoch)
            self.logger.info(f"Epoch {epoch+1}/{total_epochs} - Train Loss: {train_metrics['loss']:.4f}")

            val_metrics = {}
            if (epoch + 1) % self.val_frequency == 0 or epoch == total_epochs - 1:
                val_metrics = self.validate_epoch()

                for key, value in val_metrics.items():
                    self.writer.add_scalar(f'Val/{key.capitalize()}', value, epoch)

                baseline_miou = val_metrics.get('baseline_miou', 0.0)
                baseline_mprecision = val_metrics.get('baseline_mprecision', 0.0)
                baseline_recall = val_metrics.get('baseline_recall', 0.0)
                baseline_mpa = val_metrics.get('baseline_mpa', 0.0)
                baseline_f1 = val_metrics.get('baseline_f1', 0.0)
                baseline_mf1 = val_metrics.get('baseline_mf1', 0.0)

                cnn_miou = val_metrics.get('miou', 0.0)
                cnn_mprecision = val_metrics.get('mprecision', 0.0)
                cnn_recall = val_metrics.get('recall', 0.0)
                cnn_mpa = val_metrics.get('mpa', 0.0)
                cnn_f1 = val_metrics.get('f1', 0.0)
                cnn_mf1 = val_metrics.get('mf1', 0.0)

                miou_improvement = val_metrics.get('miou_improvement', 0.0)
                mprecision_improvement = val_metrics.get('mprecision_improvement', 0.0)
                recall_improvement = val_metrics.get('recall_improvement', 0.0)
                mpa_improvement = val_metrics.get('mpa_improvement', 0.0)
                f1_improvement = val_metrics.get('f1_improvement', 0.0)
                mf1_improvement = val_metrics.get('mf1_improvement', 0.0)

                self.logger.info(f"Epoch {epoch+1}/{total_epochs} - Val Loss: {val_metrics['loss']:.4f}")
                self.logger.info(
                    f"  基线 (仅pred_coarse) - MIoU: {baseline_miou:.4f}, "
                    f"MPrecision: {baseline_mprecision:.4f}, "
                    f"Recall: {baseline_recall:.4f}, "
                    f"MPA: {baseline_mpa:.4f}, "
                    f"F1: {baseline_f1:.4f}, "
                    f"MF1: {baseline_mf1:.4f}"
                )
                self.logger.info(
                    f"  CNN细化后 - MIoU: {cnn_miou:.4f}, "
                    f"MPrecision: {cnn_mprecision:.4f}, "
                    f"Recall: {cnn_recall:.4f}, "
                    f"MPA: {cnn_mpa:.4f}, "
                    f"F1: {cnn_f1:.4f}, "
                    f"MF1: {cnn_mf1:.4f}"
                )
                self.logger.info(
                    f"  改进量 - MIoU: {miou_improvement:+.4f} ({miou_improvement*100:+.2f}%), "
                    f"MPrecision: {mprecision_improvement:+.4f} ({mprecision_improvement*100:+.2f}%), "
                    f"Recall: {recall_improvement:+.4f} ({recall_improvement*100:+.2f}%), "
                    f"MPA: {mpa_improvement:+.4f} ({mpa_improvement*100:+.2f}%), "
                    f"F1: {f1_improvement:+.4f} ({f1_improvement*100:+.2f}%), "
                    f"MF1: {mf1_improvement:+.4f} ({mf1_improvement*100:+.2f}%)"
                )

                current_miou = val_metrics.get('miou', 0.0)
                is_best = current_miou > self.best_val_miou
                if is_best:
                    self.best_val_miou = current_miou
                    self.best_val_loss = val_metrics['loss']

                self.save_checkpoint(is_best=is_best)

                if self.early_stopping is not None:
                    monitor_metric = val_metrics.get(
                        self.early_stopping.monitor,
                        current_miou if self.early_stopping.monitor == 'val_miou' else val_metrics['loss']
                    )
                    if self.early_stopping(monitor_metric):
                        self.logger.info(f"早停触发，在epoch {epoch+1}停止训练")
                        break

            if self.scheduler is not None:
                self.scheduler.step()
                self.writer.add_scalar('Train/LR', self.scheduler.get_lr(), epoch)

            self.train_history.append({
                'epoch': epoch + 1,
                'train_loss': train_metrics['loss'],
                'val_loss': val_metrics.get('loss', None),
                'val_miou': val_metrics.get('miou', None)
            })

        self.logger.info("=" * 70)
        self.logger.info("训练完成!")
        self.logger.info(f"最佳验证损失: {self.best_val_loss:.4f}")
        self.logger.info(f"最佳验证MIoU: {self.best_val_miou:.4f}")
        self.logger.info("=" * 70)

        self.writer.close()