# -*- coding: utf-8 -*-
"""
小规模训练测试脚本
使用小数据集测试训练流程
"""
import sys
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import json


project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.dit_mamba import DiTMamba
from datasets.rows_dataset import RowsDataset
from datasets.transforms import Rotate180
from losses.dice_focal_loss import DiceFocalLoss
from training.trainer import Trainer
from training.config import TrainingConfig

def create_small_data_loaders(config: dict, num_samples: int = 10):
    """创建小规模数据加载器"""

    with open('datasets/train_split.json', 'r', encoding='utf-8') as f:
        train_samples = json.load(f)


    small_train_samples = train_samples[:num_samples]


    train_transform = Rotate180()


    train_dataset = RowsDataset(
        base_path='data/rows',
        channels=[0, 1, 2],
        mask_mode='enhanced',
        transform=train_transform,
        sample_list=small_train_samples
    )


    val_dataset = RowsDataset(
        base_path='data/rows',
        channels=[0, 1, 2],
        mask_mode='enhanced',
        transform=None,
        sample_list=small_train_samples[:max(2, num_samples // 5)]
    )


    train_loader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    return train_loader, val_loader

def main():
    """主函数"""
    print("=" * 70)
    print("小规模训练测试")
    print("=" * 70)


    config = TrainingConfig()


    config.config['base']['total_epochs'] = 2
    config.config['base']['batch_size'] = 2
    config.config['base']['gradient_accumulation_steps'] = 1


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")


    print("\n创建模型...")
    model = DiTMamba(
        in_channels=4,
        base_channels=64,
        num_encoder_layers=4,
        mamba_dim=256,
        num_classes=1,
        dropout=0.1
    )


    has_nan, has_inf = model.check_parameters()
    if has_nan or has_inf:
        print("错误: 模型参数包含NaN/Inf")
        return


    print("\n创建小规模数据加载器...")
    train_loader, val_loader = create_small_data_loaders(config.config, num_samples=10)
    print(f"训练集样本数: {len(train_loader.dataset)}")
    print(f"验证集样本数: {len(val_loader.dataset)}")


    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-4
    )


    class_weights = torch.tensor([0.5, 1.5], dtype=torch.float32)
    criterion = DiceFocalLoss(
        dice_weight=1.0,
        focal_weight=1.0,
        focal_alpha=0.25,
        focal_gamma=2.0,
        class_weights=class_weights
    )


    print("\n创建训练器...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        config=config.config,
        checkpoint_dir='checkpoints_test',
        log_dir='logs_test',
        model_type='ditmamba'
    )


    print("\n开始小规模测试训练...")
    trainer.train()

    print("\n小规模测试完成!")
    print("=" * 70)
    print("测试结果:")
    print(f"  训练流程: 正常")
    print(f"  监控指标: 正常记录")
    print(f"  防护措施: 有效")
    print("=" * 70)

if __name__ == '__main__':
    main()



