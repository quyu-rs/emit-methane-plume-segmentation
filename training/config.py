# -*- coding: utf-8 -*-
"""
训练配置模块
包含学习率调度器、配置管理等
"""
import torch
import torch.optim.lr_scheduler as lr_scheduler
from typing import Dict, Any, Optional
import yaml
from pathlib import Path

class WarmupCosineAnnealingLR:
    """
    预热+余弦退火学习率调度器
    """
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int = 10,
        total_epochs: int = 150,
        warmup_start_lr: float = 1e-6,
        base_lr: float = 1e-4,
        min_lr: float = 1e-6,
        cosine_annealing_cycles: Optional[int] = None
    ):
        """
        参数:
            optimizer: 优化器
            warmup_epochs: 预热轮数
            total_epochs: 总轮数
            warmup_start_lr: 预热起始学习率
            base_lr: 基础学习率（预热后）
            min_lr: 最小学习率
            cosine_annealing_cycles: 余弦退火周期数，如果为None则使用total_epochs - warmup_epochs
        """
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.warmup_start_lr = warmup_start_lr
        self.base_lr = base_lr
        self.min_lr = min_lr

        if cosine_annealing_cycles is None:
            self.cosine_annealing_cycles = total_epochs - warmup_epochs
        else:
            self.cosine_annealing_cycles = cosine_annealing_cycles
        self.current_epoch = 0

    def step(self, epoch: Optional[int] = None):
        """
        更新学习率
        
        参数:
            epoch: 当前epoch，如果为None则使用内部计数器
        """
        if epoch is not None:
            self.current_epoch = epoch
        else:
            self.current_epoch += 1

        if self.current_epoch < self.warmup_epochs:

            lr = self.warmup_start_lr + (self.base_lr - self.warmup_start_lr) * \
                 (self.current_epoch / self.warmup_epochs)
        else:

            cosine_epoch = self.current_epoch - self.warmup_epochs
            progress = cosine_epoch / self.cosine_annealing_cycles

            progress = min(progress, 1.0)
            lr = self.min_lr + (self.base_lr - self.min_lr) * \
                 (1 + torch.cos(torch.tensor(progress * 3.141592653589793))) / 2


        lr = max(lr, self.min_lr)


        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr



        if lr < self.base_lr * 0.01:
            import logging
            logger = logging.getLogger('DiT-Mamba-Training')
            if not logger.handlers:
                print(f"警告: 学习率过低: {lr:.2e} < {self.base_lr * 0.01:.2e} (初始学习率的1%)")
                print("建议: 检查学习率调度器配置，或考虑使用学习率重启策略")
            else:
                logger.warning(f"学习率过低: {lr:.2e} < {self.base_lr * 0.01:.2e} (初始学习率的1%)")
                logger.warning("建议: 检查学习率调度器配置，或考虑使用学习率重启策略")

    def get_lr(self) -> float:
        """获取当前学习率"""
        return self.optimizer.param_groups[0]['lr']
    
    def state_dict(self) -> Dict[str, Any]:
        """
        返回调度器的状态字典
        用于保存检查点
        """
        return {
            'warmup_epochs': self.warmup_epochs,
            'total_epochs': self.total_epochs,
            'warmup_start_lr': self.warmup_start_lr,
            'base_lr': self.base_lr,
            'min_lr': self.min_lr,
            'cosine_annealing_cycles': self.cosine_annealing_cycles,
            'current_epoch': self.current_epoch
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """
        从状态字典加载调度器状态
        用于恢复检查点
        
        参数:
            state_dict: 状态字典
        """
        self.warmup_epochs = state_dict.get('warmup_epochs', self.warmup_epochs)
        self.total_epochs = state_dict.get('total_epochs', self.total_epochs)
        self.warmup_start_lr = state_dict.get('warmup_start_lr', self.warmup_start_lr)
        self.base_lr = state_dict.get('base_lr', self.base_lr)
        self.min_lr = state_dict.get('min_lr', self.min_lr)
        self.cosine_annealing_cycles = state_dict.get('cosine_annealing_cycles', self.cosine_annealing_cycles)
        self.current_epoch = state_dict.get('current_epoch', self.current_epoch)


class EarlyStopping:
    """
    早停机制
    支持两种停止条件：
    1. 传统patience机制：连续N个epoch没有改善
    2. 下降趋势检测：连续N个epoch显著下降（检测过拟合）
    """
    def __init__(
        self,
        patience: int = 25,
        min_delta: float = 0.0005,
        mode: str = 'max',
        monitor: str = 'val_miou',
        decline_patience: int = 3,
        decline_threshold: float = 0.002
    ):
        """
        参数:
            patience: 耐心值（多少个epoch没有改善就停止）
            min_delta: 最小改善量
            mode: 'max'表示越大越好，'min'表示越小越好
            monitor: 监控的指标名称
            decline_patience: 连续下降几个epoch就停止（检测过拟合）
            decline_threshold: 下降阈值，超过此值认为是显著下降
        """
        self.patience = patience
        self.min_delta = min_delta
        self.decline_patience = decline_patience
        self.decline_threshold = decline_threshold
        self.mode = mode
        self.monitor = monitor
        self.counter = 0
        self.decline_counter = 0
        self.best_score = None
        self.previous_score = None
        self.early_stop = False
        self.should_stop = False

    def __call__(self, score: float) -> bool:
        """
        检查是否应该早停
        
        参数:
            score: 当前指标值
        
        返回:
            是否应该早停
        """

        if self.should_stop:
            self.early_stop = True
            return True

        if self.best_score is None:
            self.best_score = score
            self.previous_score = score
            return False

        if self.mode == 'max':

            if self.previous_score is not None:
                decline = self.previous_score - score
                if decline > self.decline_threshold:

                    self.decline_counter += 1
                else:

                    self.decline_counter = 0


            if self.decline_counter >= self.decline_patience:
                self.early_stop = True
                return True


            if score > self.best_score + self.min_delta:
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
                    return True

            self.previous_score = score
        else:

            if self.previous_score is not None:
                increase = score - self.previous_score
                if increase > self.decline_threshold:

                    self.decline_counter += 1
                else:
                    self.decline_counter = 0

            if self.decline_counter >= self.decline_patience:
                self.early_stop = True
                return True

            if score < self.best_score - self.min_delta:
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
                    return True

            self.previous_score = score

        return self.early_stop


class TrainingConfig:
    """
    训练配置类
    """
    def __init__(self, config_path: Optional[str] = None):
        """
        参数:
            config_path: 配置文件路径（YAML格式）
        """

        self.default_config = {
            'base': {
                'total_epochs': 150,
                'batch_size': 4,
                'gradient_accumulation_steps': 2,
                'effective_batch_size': 8,
                'mixed_precision': True
            },
            'optimizer': {
                'type': 'Adam',
                'lr': 1e-4,
                'weight_decay': 1e-4,
                'beta1': 0.9,
                'beta2': 0.999,
                'eps': 1e-8
            },
            'lr_scheduler': {
                'strategy': 'warmup_cosine',
                'warmup_epochs': 10,
                'warmup_start_lr': 1e-6,
                'min_lr': 1e-6,
                'cosine_annealing_cycles': 140
            },
            'gradient': {
                'clip': True,
                'clip_threshold': 1.0,
                'clip_method': 'L2'
            },
            'regularization': {
                'dropout': 0.1,
                'weight_decay': 1e-4
            },
            'early_stopping': {
                'enable': True,
                'patience': 25,
                'min_delta': 0.0005,
                'monitor': 'val_miou',
                'decline_patience': 3,
                'decline_threshold': 0.002
            },
            'loss': {
                'dice_weight': 1.0,
                'focal_weight': 1.0,
                'focal_alpha': 0.365,
                'focal_gamma': 2.0,
                'class_weights': [0.73, 1.27]
            },
            'data': {
                'num_workers': 4,
                'pin_memory': True,
                'prefetch_factor': 2
            }
        }


        if config_path and Path(config_path).exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                user_config = yaml.safe_load(f)
            self.config = self._merge_config(self.default_config, user_config)
        else:
            self.config = self.default_config

    def _merge_config(self, default: dict, user: dict) -> dict:
        """合并配置"""
        result = default.copy()
        for key, value in user.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_config(result[key], value)
            else:
                result[key] = value
        return result

    def get(self, key: str, default: Any = None) -> Any:
        """获取配置值"""
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value

    def save(self, path: str):
        """保存配置到文件"""
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)



