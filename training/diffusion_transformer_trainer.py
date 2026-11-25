# -*- coding: utf-8 -*-
"""
Diffusion Transformer Refiner训练器实现（已适配Transformer）
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import logging
from datetime import datetime
import shutil
from pathlib import Path
from typing import Dict, Optional, Tuple

from training.config import EarlyStopping, WarmupCosineAnnealingLR
from losses.diffusion_loss import DiffusionLoss
from metrics.segmentation_metrics import calculate_all_metrics


class DiffusionTrainer:
    def __init__(
            self,
            model: nn.Module,
            train_loader: DataLoader,
            val_loader: DataLoader,
            optimizer: torch.optim.Optimizer,
            criterion: nn.Module,
            device: torch.device,
            config: dict,
            num_timesteps: int = 1000,
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
            self.num_timesteps = num_timesteps
            self.checkpoint_dir = Path(checkpoint_dir)
            self.log_dir = Path(log_dir)
        
            self.checkpoint_dir.mkdir(exist_ok=True)
            self.log_dir.mkdir(exist_ok=True)
            
            # ==================== Transformer 专用 ====================
            diffusion_training_config = config.get('diffusion_refiner', {}).get('training', {})
            self.patch_size = diffusion_training_config.get('patch_size', 8)
            # ======================================================
            
            self._init_noise_schedule()
            
            # ==================== 日志与 TensorBoard ====================
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = self.log_dir / f"diffusion_training_{timestamp}.log"
        
            self.logger = logging.getLogger('Diffusion-Training')
            self.logger.setLevel(logging.INFO)
            self.logger.handlers.clear()
        
            file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
            file_handler.setLevel(logging.INFO)
            file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)
        
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            console_formatter = logging.Formatter('%(message)s')
            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)
        
            self.writer = SummaryWriter(log_dir=str(self.log_dir / 'tensorboard'))
        
            self.current_epoch = 0
            self.best_val_loss = float('inf')
            self.best_val_miou = 0.0
            self.train_history = []
        
            self.gradient_accumulation_steps = config.get('base', {}).get('gradient_accumulation_steps', 1)
            self.use_amp = config.get('base', {}).get('mixed_precision', False)
            if self.use_amp:
                self.scaler = torch.cuda.amp.GradScaler()
            
            # ==================== 验证相关（关键修复）===================
            val_config = config.get('diffusion_refiner', {}).get('validation', {})
            self.val_frequency = val_config.get('frequency', 5)
            self.val_num_steps = val_config.get('num_steps', 120)      # ← 这里原来是15，现在改成120更合理
            # ===========================================================
            
            # EarlyStopping
            early_stop_config = config.get('early_stopping', {})
            if early_stop_config.get('enable', True):
                monitor = early_stop_config.get('monitor', 'val_miou')
                mode = 'max' if monitor == 'val_miou' else 'min'
                self.early_stopping = EarlyStopping(
                    patience=early_stop_config.get('patience', 25),
                    min_delta=early_stop_config.get('min_delta', 0.0005),
                    mode=mode,
                    monitor=monitor,
                    decline_patience=early_stop_config.get('decline_patience', 5),
                    decline_threshold=early_stop_config.get('decline_threshold', 0.003)
                )
            else:
                self.early_stopping = None
            
            # 学习率调度器
            lr_config = config.get('lr_scheduler', {})
            if lr_config.get('strategy') == 'warmup_cosine':
                total_epochs = config.get('diffusion_refiner', {}).get('training', {}).get('num_epochs', 100)
                diffusion_config = config.get('diffusion_refiner', {}).get('training', {})
                base_lr = diffusion_config.get('learning_rate', 0.00005)
                self.scheduler = WarmupCosineAnnealingLR(
                    optimizer=self.optimizer,
                    warmup_epochs=lr_config.get('warmup_epochs', 10),
                    total_epochs=total_epochs,
                    warmup_start_lr=lr_config.get('warmup_start_lr', 1e-6),
                    base_lr=base_lr,
                    min_lr=lr_config.get('min_lr', 3e-6),
                    cosine_annealing_cycles=lr_config.get('cosine_annealing_cycles', total_epochs)
                )
            else:
                self.scheduler = None
            
            if resume_from:
                self.load_checkpoint(resume_from)

    
    def _pad_to_patch_size(self, tensor: torch.Tensor) -> torch.Tensor:
        B, C, H, W = tensor.shape
        pad_h = (self.patch_size - H % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - W % self.patch_size) % self.patch_size
        if pad_h == 0 and pad_w == 0:
            return tensor
        return F.pad(tensor, (0, pad_w, 0, pad_h), mode='reflect')

    def _unpad(self, tensor: torch.Tensor, original_h: int, original_w: int) -> torch.Tensor:
        return tensor[:, :, :original_h, :original_w]

    def _init_noise_schedule(self):
        # 你原来的 _init_noise_schedule 代码...
        beta_start = 0.0001
        beta_end = 0.02
        self.betas = torch.linspace(beta_start, beta_end, self.num_timesteps, device=self.device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)

    def sample_timesteps(self, batch_size: int) -> torch.Tensor:
        return torch.randint(0, self.num_timesteps, (batch_size,), device=self.device)

    def add_noise(self, x: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # 你原来的 add_noise 代码...
        if noise is None:
            noise = torch.randn_like(x)
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        noisy_x = sqrt_alphas_cumprod_t * x + sqrt_one_minus_alphas_cumprod_t * noise
        return noisy_x, noise
    
    def train_epoch(self) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        
        total_loss = 0.0
        total_samples = 0
        skipped_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch+1} [Train]')
        self.optimizer.zero_grad()
        accumulated_batches = 0
        
        for batch_idx, batch in enumerate(pbar):
            input_images = batch['input_image'].to(self.device)
            pred_coarse = batch['pred_coarse'].to(self.device)
            residuals = batch['residual'].to(self.device)
            
            batch_size = input_images.size(0)
            

            t = self.sample_timesteps(batch_size)
            

            noisy_residuals, noise = self.add_noise(residuals, t)
            

            condition = torch.cat([input_images, pred_coarse], dim=1)
            
            try:
                if self.use_amp:
                    with torch.cuda.amp.autocast(enabled=True, cache_enabled=True):

                        predicted_noise = self.model(noisy_residuals, t, condition)
                        

                        loss = self.criterion(predicted_noise, noise)
                        loss = loss / self.gradient_accumulation_steps
                    
                    self.scaler.scale(loss).backward()
                    accumulated_batches += 1
                else:

                    predicted_noise = self.model(noisy_residuals, t, condition)
                    

                    loss = self.criterion(predicted_noise, noise)
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
                total_samples += batch_size
                
                pbar.set_postfix({
                    'loss': f'{loss.item() * self.gradient_accumulation_steps:.4f}',
                    'avg_loss': f'{total_loss / (batch_idx + 1 - skipped_batches):.4f}' if (batch_idx + 1 - skipped_batches) > 0 else '0.0000'
                })
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    torch.cuda.empty_cache()
                    self.logger.error(f"CUDA OOM at batch {batch_idx}, 跳过该批次")
                    skipped_batches += 1
                    if self.use_amp and accumulated_batches > 0:
                        try:
                            self.scaler.unscale_(self.optimizer)
                            self.optimizer.zero_grad()
                            self.scaler.update()
                        except:
                            self.optimizer.zero_grad()
                        accumulated_batches = 0
                    else:
                        self.optimizer.zero_grad()
                    continue
                else:
                    raise
        
        avg_loss = total_loss / (len(self.train_loader) - skipped_batches) if (len(self.train_loader) - skipped_batches) > 0 else 0.0
        
        return {'loss': avg_loss}
    


    def _sample_with_model(self, condition: torch.Tensor, num_steps: int = 20, guidance_scale: float = 1.0) -> torch.Tensor:
        self.model.eval()
        orig_h, orig_w = condition.shape[2], condition.shape[3]
        condition_padded = self._pad_to_patch_size(condition)
        
        with torch.no_grad():
            refined_residual = self.model.sample(
                condition=condition_padded,
                num_steps=num_steps,
                guidance_scale=guidance_scale
            )
        return self._unpad(refined_residual, orig_h, orig_w)
    

    
    def validate_epoch(self) -> Dict[str, float]:

        """验证一个epoch"""
        self.model.eval()
    
        total_loss = 0.0
        all_preds = []
        all_targets = []
        all_baseline_preds = []
        skipped_batches = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f'Epoch {self.current_epoch+1} [Val]')
            
            for batch_idx, batch in enumerate(pbar):
                input_images = batch['input_image'].to(self.device)
                pred_coarse = batch['pred_coarse'].to(self.device)
                residuals = batch['residual'].to(self.device)
                target_masks = batch['target_mask'].to(self.device)
                

                condition = torch.cat([input_images, pred_coarse], dim=1)
                

                refined_residuals = self._sample_with_model(condition, num_steps=self.val_num_steps)
                

                final_masks = pred_coarse + refined_residuals
                final_masks = torch.clamp(final_masks, 0.0, 1.0)
                

                pred_binary = (final_masks > 0.5).long().cpu()
                target_binary = (target_masks > 0.5).long().cpu()
                

                baseline_binary = (pred_coarse > 0.5).long().cpu()
                
                all_preds.append(pred_binary)
                all_targets.append(target_binary)
                all_baseline_preds.append(baseline_binary)
                

                loss = F.mse_loss(final_masks, target_masks)
                total_loss += loss.item()
                
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'avg_loss': f'{total_loss / (batch_idx + 1 - skipped_batches):.4f}' if (batch_idx + 1 - skipped_batches) > 0 else '0.0000'
                })
        

        if len(all_preds) > 0:
            try:
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
                
            except Exception as e:
                self.logger.error(f"计算验证指标失败: {e}", exc_info=True)
                metrics = {'loss': total_loss / len(self.val_loader), 
                          'miou': 0.0, 'mprecision': 0.0, 'recall': 0.0, 'mpa': 0.0, 
                          'f1': 0.0, 'mf1': 0.0,
                          'baseline_miou': 0.0, 'baseline_mprecision': 0.0, 
                          'baseline_recall': 0.0, 'baseline_mpa': 0.0,
                          'baseline_f1': 0.0, 'baseline_mf1': 0.0,
                          'miou_improvement': 0.0, 'mprecision_improvement': 0.0,
                          'recall_improvement': 0.0, 'mpa_improvement': 0.0,
                          'f1_improvement': 0.0, 'mf1_improvement': 0.0}
        else:
            metrics = {'loss': 0.0, 'miou': 0.0, 'mprecision': 0.0, 'recall': 0.0, 'mpa': 0.0,
                      'f1': 0.0, 'mf1': 0.0,
                      'baseline_miou': 0.0, 'baseline_mprecision': 0.0,
                      'baseline_recall': 0.0, 'baseline_mpa': 0.0,
                      'baseline_f1': 0.0, 'baseline_mf1': 0.0,
                      'miou_improvement': 0.0, 'mprecision_improvement': 0.0,
                      'recall_improvement': 0.0, 'mpa_improvement': 0.0,
                      'f1_improvement': 0.0, 'mf1_improvement': 0.0}
        
        return metrics
    

    
    
    def save_checkpoint(self, is_best: bool = False):
        """保存检查点"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_val_miou': self.best_val_miou,
            'train_history': self.train_history,
            'config': self.config,
            'num_timesteps': self.num_timesteps
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        if self.use_amp:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        

        checkpoint_path = self.checkpoint_dir / 'diffusion_refiner_latest.pth'
        torch.save(checkpoint, checkpoint_path)
        

        if is_best:
            best_path = self.checkpoint_dir / 'diffusion_refiner_best.pth'
            shutil.copy(checkpoint_path, best_path)
            self.logger.info(f"保存最佳模型: {best_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """加载检查点"""
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
        """主训练循环"""
        total_epochs = self.config.get('diffusion_refiner', {}).get('training', {}).get('num_epochs', 40)
        
        self.logger.info("=" * 70)
        self.logger.info("开始Diffusion训练")
        self.logger.info("=" * 70)
        self.logger.info(f"总轮数: {total_epochs}")
        self.logger.info(f"训练集样本数: {len(self.train_loader.dataset)}")
        self.logger.info(f"验证集样本数: {len(self.val_loader.dataset)}")
        self.logger.info(f"扩散步数: {self.num_timesteps}")
        self.logger.info(f"验证频率: 每{self.val_frequency}个epoch")
        
        for epoch in range(self.current_epoch, total_epochs):
            self.current_epoch = epoch
            

            train_metrics = self.train_epoch()
            

            self.writer.add_scalar('Train/Loss', train_metrics['loss'], epoch)
            self.logger.info(f"Epoch {epoch+1}/{total_epochs} - Train Loss (预测噪声MSE): {train_metrics['loss']:.4f}")
            

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
                
                diffusion_miou = val_metrics.get('miou', 0.0)
                diffusion_mprecision = val_metrics.get('mprecision', 0.0)
                diffusion_recall = val_metrics.get('recall', 0.0)
                diffusion_mpa = val_metrics.get('mpa', 0.0)
                diffusion_f1 = val_metrics.get('f1', 0.0)
                diffusion_mf1 = val_metrics.get('mf1', 0.0)
                
                miou_improvement = val_metrics.get('miou_improvement', 0.0)
                mprecision_improvement = val_metrics.get('mprecision_improvement', 0.0)
                recall_improvement = val_metrics.get('recall_improvement', 0.0)
                mpa_improvement = val_metrics.get('mpa_improvement', 0.0)
                f1_improvement = val_metrics.get('f1_improvement', 0.0)
                mf1_improvement = val_metrics.get('mf1_improvement', 0.0)
                
                self.logger.info(f"Epoch {epoch+1}/{total_epochs} - Val Loss (Mask MSE): {val_metrics['loss']:.4f}")
                self.logger.info(f"  基线 (仅pred_coarse) - MIoU: {baseline_miou:.4f}, "
                               f"MPrecision: {baseline_mprecision:.4f}, "
                               f"Recall: {baseline_recall:.4f}, "
                               f"MPA: {baseline_mpa:.4f}, "
                               f"F1: {baseline_f1:.4f}, "
                               f"MF1: {baseline_mf1:.4f}")
                self.logger.info(f"  Diffusion细化后 - MIoU: {diffusion_miou:.4f}, "
                               f"MPrecision: {diffusion_mprecision:.4f}, "
                               f"Recall: {diffusion_recall:.4f}, "
                               f"MPA: {diffusion_mpa:.4f}, "
                               f"F1: {diffusion_f1:.4f}, "
                               f"MF1: {diffusion_mf1:.4f}")
                self.logger.info(f"  改进量 - MIoU: {miou_improvement:+.4f} ({miou_improvement*100:+.2f}%), "
                               f"MPrecision: {mprecision_improvement:+.4f} ({mprecision_improvement*100:+.2f}%), "
                               f"Recall: {recall_improvement:+.4f} ({recall_improvement*100:+.2f}%), "
                               f"MPA: {mpa_improvement:+.4f} ({mpa_improvement*100:+.2f}%), "
                               f"F1: {f1_improvement:+.4f} ({f1_improvement*100:+.2f}%), "
                               f"MF1: {mf1_improvement:+.4f} ({mf1_improvement*100:+.2f}%)")
                

                current_miou = val_metrics.get('miou', 0.0)
                is_best = current_miou > self.best_val_miou
                if is_best:
                    self.best_val_miou = current_miou
                    self.best_val_loss = val_metrics['loss']
                
                self.save_checkpoint(is_best=is_best)
                

                if self.early_stopping is not None:
                    monitor_metric = val_metrics.get(self.early_stopping.monitor, 
                                                     current_miou if self.early_stopping.monitor == 'val_miou' else val_metrics['loss'])
                    if self.early_stopping(monitor_metric):
                        self.logger.info(f"早停触发，在epoch {epoch+1}停止训练")
                        break
            

            if self.scheduler is not None:
                self.scheduler.step()
                self.writer.add_scalar('Train/LR', self.scheduler.get_lr(), epoch)
            

            self.train_history.append({
                'epoch': epoch + 1,
                'train_loss': train_metrics['loss'],
                'val_loss': val_metrics.get('loss', 0.0) if (epoch + 1) % self.val_frequency == 0 else None,
                'val_miou': val_metrics.get('miou', 0.0) if (epoch + 1) % self.val_frequency == 0 else None
            })
        
        self.logger.info("=" * 70)
        self.logger.info("训练完成!")
        self.logger.info(f"最佳验证损失: {self.best_val_loss:.4f}")
        self.logger.info(f"最佳验证MIoU: {self.best_val_miou:.4f}")
        self.logger.info("=" * 70)
        
        self.writer.close()

    

    