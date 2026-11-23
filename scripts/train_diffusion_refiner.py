# -*- coding: utf-8 -*-
"""
Diffusion Refiner训练脚本
"""
import sys
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import argparse
from tqdm import tqdm
import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.diffusion_refiner import DiffusionRefiner
from training.diffusion_trainer import DiffusionTrainer
from training.config import TrainingConfig
from losses.diffusion_loss import DiffusionLoss


class DiffusionDataset(Dataset):
    """
    Diffusion训练数据集
    从预处理数据加载
    """
    def __init__(self, preprocessed_data: list):
        """
        参数:
            preprocessed_data: 预处理数据列表
        """
        self.data = preprocessed_data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        def to_tensor(x):
            if isinstance(x, torch.Tensor):
                return x
            elif isinstance(x, np.ndarray):
                return torch.from_numpy(x)
            else:
                return torch.tensor(x)
        
        return {
            'input_image': to_tensor(item['input_image']),
            'pred_coarse': to_tensor(item['pred_coarse']),
            'residual': to_tensor(item['residual']),
            'target_mask': to_tensor(item['target_mask']),
            'name': item['name']
        }


def create_data_loaders(preprocessed_data: dict, config: dict):
    """
    创建数据加载器
    
    参数:
        preprocessed_data: 预处理数据字典
        config: 配置字典
    """
    train_data = preprocessed_data.get('train', [])
    val_data = preprocessed_data.get('val', [])
    
    train_dataset = DiffusionDataset(train_data)
    val_dataset = DiffusionDataset(val_data)
    
    train_config = config.get('diffusion_refiner', {}).get('training', {})
    data_config = config.get('data', {})
    
    batch_size = train_config.get('batch_size', data_config.get('batch_size', 32))
    num_workers = data_config.get('num_workers', 1)
    pin_memory = data_config.get('pin_memory', True)
    
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
    """主函数"""

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        print("已清理 CUDA 缓存")
    
    parser = argparse.ArgumentParser(description='训练Diffusion Refiner')
    parser.add_argument('--preprocessed_data', type=str, required=True,
                       help='预处理数据文件路径')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='配置文件路径')
    parser.add_argument('--resume', type=str, default=None,
                       help='恢复训练的检查点路径')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Diffusion Refiner训练")
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
    

    print(f"\n加载预处理数据: {args.preprocessed_data}")
    preprocessed_data = torch.load(args.preprocessed_data, map_location='cpu')
    print(f"训练集: {len(preprocessed_data.get('train', []))}个样本")
    print(f"验证集: {len(preprocessed_data.get('val', []))}个样本")
    

    print("\n创建数据加载器...")
    train_loader, val_loader = create_data_loaders(preprocessed_data, config.config)
    

    print("\n创建Diffusion模型...")
    diffusion_config = config.get('diffusion_refiner', {}).get('training', {})
    num_timesteps = diffusion_config.get('num_timesteps', 1000)
    
    model = DiffusionRefiner(
        in_channels=6,
        base_channels=64,
        time_emb_dim=128,
        num_timesteps=num_timesteps,
        dropout=config.get('regularization', {}).get('dropout', 0.1)
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params / 1e6:.2f}M")
    

    criterion = DiffusionLoss()
    

    optimizer_config = config.get('optimizer', {})
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=diffusion_config.get('learning_rate', optimizer_config.get('lr', 1e-4)),
        weight_decay=optimizer_config.get('weight_decay', 1e-4),
        betas=(
            optimizer_config.get('beta1', 0.9),
            optimizer_config.get('beta2', 0.999)
        ),
        eps=optimizer_config.get('eps', 1e-8)
    )
    

    print("\n创建训练器...")
    trainer = DiffusionTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        config=config.config,
        num_timesteps=num_timesteps,
        checkpoint_dir=str(project_root / 'checkpoints'),
        log_dir=str(project_root / 'logs'),
        resume_from=args.resume
    )
    

    print("\n开始训练...")
    trainer.train()
    
    print("\n训练完成!")


if __name__ == '__main__':
    main()



