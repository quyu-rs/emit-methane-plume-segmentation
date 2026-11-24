# -*- coding: utf-8 -*-
"""
UNet Refiner训练脚本
复用第二阶段预处理数据（diffusion_preprocessed_data.pt）
"""

import sys
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
import argparse
import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.unet_refiner import UNetRefiner
from training.unet_refiner_trainer import UNetRefinerTrainer, RefinerLoss
from training.config import TrainingConfig


class RefinerDataset(Dataset):
    """
    第二阶段通用数据集
    从预处理数据加载
    """
    def __init__(self, preprocessed_data: list):
        self.data = preprocessed_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        def to_tensor(x):
            if isinstance(x, torch.Tensor):
                return x.float()
            elif isinstance(x, np.ndarray):
                return torch.from_numpy(x).float()
            else:
                return torch.tensor(x, dtype=torch.float32)

        return {
            'input_image': to_tensor(item['input_image']),
            'pred_coarse': to_tensor(item['pred_coarse']),
            'residual': to_tensor(item['residual']),
            'target_mask': to_tensor(item['target_mask']),
            'name': item['name']
        }


def create_data_loaders(preprocessed_data: dict, config: dict):
    train_data = preprocessed_data.get('train', [])
    val_data = preprocessed_data.get('val', [])

    train_dataset = RefinerDataset(train_data)
    val_dataset = RefinerDataset(val_data)

    train_config = config.get('unet_refiner', {}).get('training', {})
    data_config = config.get('data', {})

    batch_size = int(train_config.get('batch_size', data_config.get('batch_size', 8)))
    num_workers = int(data_config.get('num_workers', 1))
    pin_memory = bool(data_config.get('pin_memory', True))

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    return train_loader, val_loader


def main():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass
        print("已清理 CUDA 缓存")

    parser = argparse.ArgumentParser(description='训练UNet Refiner')
    parser.add_argument('--preprocessed_data', type=str, required=True, help='预处理数据文件路径')
    parser.add_argument('--config', type=str, default='config-unet.yaml', help='配置文件路径')
    parser.add_argument('--resume', type=str, default=None, help='恢复训练的检查点路径')
    args = parser.parse_args()

    print("=" * 70)
    print("UNet Refiner训练")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

    config_path = project_root / args.config
    if not config_path.exists():
        print(f"错误: 配置文件不存在: {config_path}")
        return

    config = TrainingConfig(str(config_path))

    preprocessed_path = Path(args.preprocessed_data)
    if not preprocessed_path.is_absolute():
        preprocessed_path = project_root / args.preprocessed_data

    if not preprocessed_path.exists():
        print(f"错误: 预处理数据文件不存在: {preprocessed_path}")
        return

    print(f"\n加载预处理数据: {preprocessed_path}")
    preprocessed_data = torch.load(preprocessed_path, map_location='cpu')
    print(f"训练集: {len(preprocessed_data.get('train', []))}个样本")
    print(f"验证集: {len(preprocessed_data.get('val', []))}个样本")

    print("\n创建数据加载器...")
    train_loader, val_loader = create_data_loaders(preprocessed_data, config.config)

    print("\n创建UNet模型...")
    model_cfg = config.get('unet_refiner', {}).get('model', {})
    model = UNetRefiner(
        in_channels=int(model_cfg.get('in_channels', 5)),
        base_channels=int(model_cfg.get('base_channels', 32)),
        residual_range=float(model_cfg.get('residual_range', 1.0)),
        init_output_scale=float(model_cfg.get('init_output_scale', 0.1)),
        dropout=float(config.get('regularization', {}).get('dropout', 0.1))
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params / 1e6:.2f}M")

    loss_cfg = config.get('unet_refiner', {}).get('loss', {})
    criterion = RefinerLoss(
        lambda_residual=float(loss_cfg.get('lambda_residual', 0.2)),
        dice_weight=float(loss_cfg.get('dice_weight', 1.0)),
        bce_weight=float(loss_cfg.get('bce_weight', 1.0))
    )

    optimizer_config = config.get('optimizer', {})
    unet_train_cfg = config.get('unet_refiner', {}).get('training', {})
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(unet_train_cfg.get('learning_rate', optimizer_config.get('lr', 1e-4))),
        weight_decay=float(optimizer_config.get('weight_decay', 1e-4)),
        betas=(
            float(optimizer_config.get('beta1', 0.9)),
            float(optimizer_config.get('beta2', 0.999))
        ),
        eps=float(optimizer_config.get('eps', 1e-8))
    )

    checkpoint_dir = config.get('unet_refiner', {}).get('checkpoint_dir', None)
    log_dir = config.get('unet_refiner', {}).get('log_dir', None)

    if checkpoint_dir is None:
        checkpoint_dir = str(project_root / 'checkpoints')
    elif not Path(checkpoint_dir).is_absolute():
        checkpoint_dir = str(project_root / checkpoint_dir)

    if log_dir is None:
        log_dir = str(project_root / 'logs')
    elif not Path(log_dir).is_absolute():
        log_dir = str(project_root / log_dir)

    print("\n创建训练器...")
    trainer = UNetRefinerTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        config=config.config,
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir,
        resume_from=args.resume
    )

    print("\n开始训练...")
    trainer.train()

    print("\n训练完成!")


if __name__ == '__main__':
    main()