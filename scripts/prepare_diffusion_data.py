# -*- coding: utf-8 -*-
"""
数据预处理脚本
使用主模型生成初始预测和残差，用于Diffusion训练
"""
import sys
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import json
import argparse
from tqdm import tqdm

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.dit_mamba import DiTMamba, DiT
from datasets.rows_dataset import RowsDataset
from datasets.dataset_adapter import DatasetAdapter
from training.config import TrainingConfig


def load_main_model(checkpoint_path: str, device: torch.device, config_path: str = None, model_type: str = 'ditmamba'):
    """
    加载主模型
    
    参数:
        checkpoint_path: 检查点路径
        device: 设备
        config_path: 配置文件路径（可选）
        model_type: 模型类型，'dit' 或 'ditmamba'
    """
    print(f"加载主模型: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if 'config' in checkpoint and checkpoint['config'] is not None:
        model_config = checkpoint['config'].get('model', {})
        base_channels = model_config.get('base_channels', 64)
        num_encoder_layers = model_config.get('num_encoder_layers', 4)
        mamba_dim = model_config.get('mamba_dim', 256)
        dropout = checkpoint['config'].get('regularization', {}).get('dropout', 0.1)
        print(f"从checkpoint读取模型配置: base_channels={base_channels}, layers={num_encoder_layers}, mamba_dim={mamba_dim}")
    elif config_path and Path(config_path).exists():
        config = TrainingConfig(config_path)
        model_config = config.get('model', {})
        base_channels = model_config.get('base_channels', 64)
        num_encoder_layers = model_config.get('num_encoder_layers', 4)
        mamba_dim = model_config.get('mamba_dim', 256)
        dropout = config.get('regularization', {}).get('dropout', 0.1)
        print(f"从配置文件读取模型配置: base_channels={base_channels}, layers={num_encoder_layers}, mamba_dim={mamba_dim}")
    else:
        base_channels = 64
        num_encoder_layers = 4
        mamba_dim = 256
        dropout = 0.1
        print("警告: 使用默认模型参数，可能与checkpoint不匹配")
    
    if model_type == 'dit':
        model = DiT(
            in_channels=4,
            base_channels=base_channels,
            num_encoder_layers=num_encoder_layers,
            num_classes=1,
            dropout=dropout
        )
    else:
        model = DiTMamba(
            in_channels=4,
            base_channels=base_channels,
            num_encoder_layers=num_encoder_layers,
            mamba_dim=mamba_dim,
            num_classes=1,
            dropout=dropout
        )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"模型加载成功 (epoch {checkpoint.get('epoch', 'unknown')})")
    print(f"最佳Val MIoU: {checkpoint.get('best_val_miou', 0.0):.4f}")
    
    return model


def create_data_loaders(config: dict, device: torch.device):
    """创建数据加载器（复用train.py的逻辑）"""
    data_config = config.get('data', {})
    if data_config is None or not isinstance(data_config, dict):
        data_config = {}
    
    use_dataset_adapter = data_config.get('use_dataset_adapter', False)
    dataset_class = DatasetAdapter if use_dataset_adapter else RowsDataset
    
    if use_dataset_adapter:
        base_path = str(project_root / 'data/dataset')
        channels = 0
        
        train_split_adapter_path = project_root / 'datasets/dataset_adapter_train_split.json'
        val_split_adapter_path = project_root / 'datasets/dataset_adapter_val_split.json'
        train_split_path = project_root / 'datasets/train_split.json'
        val_split_path = project_root / 'datasets/val_split.json'
        
        train_samples = None
        val_samples = None
        
        if train_split_adapter_path.exists():
            with open(train_split_adapter_path, 'r', encoding='utf-8') as f:
                train_samples = json.load(f)
        elif train_split_path.exists():
            with open(train_split_path, 'r', encoding='utf-8') as f:
                train_samples = json.load(f)
        
        if val_split_adapter_path.exists():
            with open(val_split_adapter_path, 'r', encoding='utf-8') as f:
                val_samples = json.load(f)
        elif val_split_path.exists():
            with open(val_split_path, 'r', encoding='utf-8') as f:
                val_samples = json.load(f)
    else:
        with open(project_root / 'datasets/train_split.json', 'r', encoding='utf-8') as f:
            train_samples = json.load(f)
        with open(project_root / 'datasets/val_split.json', 'r', encoding='utf-8') as f:
            val_samples = json.load(f)
        
        base_path = str(project_root / 'data/rows')
        channels = [0, 1, 2]
    

    train_dataset = dataset_class(
        base_path=base_path,
        channels=channels,
        mask_mode='enhanced',
        transform=None,
        sample_list=train_samples
    )
    

    val_dataset = dataset_class(
        base_path=base_path,
        channels=channels,
        mask_mode='enhanced',
        transform=None,
        sample_list=val_samples
    )
    

    batch_size = data_config.get('batch_size', 32)
    num_workers = data_config.get('num_workers', 1)
    pin_memory = data_config.get('pin_memory', True)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
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


def prepare_data(model: nn.Module, data_loader: DataLoader, device: torch.device, split_name: str):
    """
    预处理数据，生成初始预测和残差
    
    参数:
        model: 主模型
        data_loader: 数据加载器
        device: 设备
        split_name: 数据集名称（'train' 或 'val'）
    
    返回:
        预处理数据列表
    """
    print(f"\n处理{split_name}集...")
    
    preprocessed_data = []
    
    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(data_loader, desc=f'处理{split_name}集')):
            inputs = batch['input'].to(device)
            targets = batch['mask'].to(device)
            names = batch['name']
            

            targets_normalized = targets.float()
            if targets_normalized.max() > 1.0:
                targets_normalized = targets_normalized / 255.0
            targets_normalized = torch.clamp(targets_normalized, min=0.0, max=1.0)
            

            pred_coarse = model(inputs)
            pred_coarse = torch.clamp(pred_coarse, min=0.0, max=1.0)
            

            residuals = targets_normalized - pred_coarse
            

            batch_size = inputs.size(0)
            for i in range(batch_size):
                preprocessed_data.append({
                    'name': names[i],
                    'input_image': inputs[i].cpu(),
                    'pred_coarse': pred_coarse[i].cpu(),
                    'residual': residuals[i].cpu(),
                    'target_mask': targets_normalized[i].cpu()
                })
    
    print(f"{split_name}集处理完成，共{len(preprocessed_data)}个样本")
    
    return preprocessed_data


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='准备Diffusion训练数据')
    parser.add_argument('--main_model', type=str, required=True,
                       help='主模型检查点路径')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='配置文件路径')
    parser.add_argument('--model_type', type=str, default='ditmamba',
                       choices=['dit', 'ditmamba'],
                       help='模型类型')
    parser.add_argument('--output', type=str, default='diffusion_preprocessed_data.pt',
                       help='输出文件路径')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Diffusion数据预处理")
    print("=" * 70)
    

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    

    config_path = project_root / args.config
    if not config_path.exists():
        print(f"错误: 配置文件不存在: {config_path}")
        return
    
    config = TrainingConfig(str(config_path))
    

    main_model = load_main_model(
        args.main_model,
        device,
        config_path=str(config_path),
        model_type=args.model_type
    )
    

    print("\n创建数据加载器...")
    train_loader, val_loader = create_data_loaders(config.config, device)
    print(f"训练集样本数: {len(train_loader.dataset)}")
    print(f"验证集样本数: {len(val_loader.dataset)}")
    

    train_data = prepare_data(main_model, train_loader, device, 'train')
    val_data = prepare_data(main_model, val_loader, device, 'val')
    

    output_path = project_root / args.output
    output_data = {
        'train': train_data,
        'val': val_data,
        'config': config.config,
        'main_model_path': args.main_model,
        'model_type': args.model_type
    }
    
    torch.save(output_data, output_path)
    print(f"\n预处理数据已保存到: {output_path}")
    print(f"训练集: {len(train_data)}个样本")
    print(f"验证集: {len(val_data)}个样本")
    
    print("\n数据预处理完成!")


if __name__ == '__main__':
    main()



