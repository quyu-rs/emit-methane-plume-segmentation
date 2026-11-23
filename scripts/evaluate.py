# -*- coding: utf-8 -*-
"""
模型评估脚本
在测试集上评估模型，计算所有指标，生成评估报告
"""
import sys
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import json
import numpy as np
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # 设置非交互式后端，避免GUI问题
import matplotlib.pyplot as plt
from PIL import Image


project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.dit_mamba import DiTMamba, DiT
from models.diffusion_refiner import DiffusionRefiner
from datasets.rows_dataset import RowsDataset
from datasets.dataset_adapter import DatasetAdapter
from metrics.segmentation_metrics import calculate_all_metrics
from training.config import TrainingConfig
from typing import Optional

def load_model(checkpoint_path: str, device: torch.device, config_path: Optional[str] = None):
    """加载模型
    
    参数:
        checkpoint_path: 检查点文件路径
        device: 设备
        config_path: 可选的配置文件路径（当checkpoint没有config时使用）
    """
    print(f"加载模型: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # 检测checkpoint中的模型类型
    checkpoint_keys = set(checkpoint['model_state_dict'].keys())
    has_mamba_keys = any('mamba' in key.lower() for key in checkpoint_keys)
    
    if has_mamba_keys:
        model_type = 'ditmamba'
        print("检测到模型类型: DiT-Mamba (包含Mamba模块)")
    else:
        model_type = 'dit'
        print("检测到模型类型: DiT (不包含Mamba模块)")

    # 优先从checkpoint读取配置
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
        print("建议: 使用--config参数指定配置文件，或确保checkpoint包含config字段")

    # 根据检测到的类型创建对应的模型
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

    # 尝试加载，如果失败则使用strict=False
    try:
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    except RuntimeError as e:
        print(f"警告: 严格加载失败: {e}")
        print("尝试使用strict=False加载（允许部分键缺失）...")
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        if missing_keys:
            print(f"缺失的键数量: {len(missing_keys)}")
            if len(missing_keys) <= 10:
                print(f"缺失的键: {missing_keys}")
            else:
                print(f"缺失的键（前10个）: {missing_keys[:10]}")
        if unexpected_keys:
            print(f"意外的键数量: {len(unexpected_keys)}")
            if len(unexpected_keys) <= 10:
                print(f"意外的键: {unexpected_keys}")
            else:
                print(f"意外的键（前10个）: {unexpected_keys[:10]}")
    
    model.to(device)
    model.eval()

    print(f"模型加载成功 (epoch {checkpoint.get('epoch', 'unknown')})")
    print(f"最佳Val MIoU: {checkpoint.get('best_val_miou', 0.0):.4f}")

    return model

def load_diffusion_model(checkpoint_path: str, device: torch.device, config: dict):
    """
    加载Diffusion模型
    
    参数:
        checkpoint_path: Diffusion模型检查点路径
        device: 设备
        config: 配置字典
    
    返回:
        Diffusion模型
    """
    print(f"加载Diffusion模型: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    diffusion_config = config.get('diffusion_refiner', {}).get('training', {})
    num_timesteps = checkpoint.get('num_timesteps', diffusion_config.get('num_timesteps', 1000))
    dropout = config.get('regularization', {}).get('dropout', 0.1)
    
    model = DiffusionRefiner(
        in_channels=6,
        base_channels=64,
        time_emb_dim=128,
        num_timesteps=num_timesteps,
        dropout=dropout
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"Diffusion模型加载成功 (epoch {checkpoint.get('epoch', 'unknown')})")
    
    return model


def evaluate_model(model: nn.Module, test_loader: DataLoader, device: torch.device, 
                   diffusion_model: Optional[nn.Module] = None, config: Optional[dict] = None):
    """评估模型"""
    print("\n" + "=" * 70)
    print("开始评估")
    print("=" * 70)

    all_preds = []
    all_targets = []
    all_names = []
    total_loss = 0.0

    criterion = nn.BCELoss()
    

    use_diffusion = diffusion_model is not None
    if use_diffusion:
        if config is not None:
            inference_config = config.get('diffusion_refiner', {}).get('inference', {})
            num_steps = inference_config.get('num_steps', 20)
        else:
            num_steps = 20
        print(f"使用Diffusion细化，采样步数: {num_steps}")

    with torch.no_grad():
        pbar = tqdm(test_loader, desc='评估中')

        for batch in pbar:
            inputs = batch['input'].to(device)
            targets = batch['mask'].to(device)
            names = batch['name']


            outputs = model(inputs)
            

            if use_diffusion:

                condition = torch.cat([inputs, outputs], dim=1)
                

                refined_residuals = diffusion_model.sample(condition, num_steps=num_steps)
                

                outputs = outputs + refined_residuals
                outputs = torch.clamp(outputs, 0.0, 1.0)


            targets_normalized = targets.float()
            if targets_normalized.max() > 1.0:
                targets_normalized = targets_normalized / 255.0
            targets_normalized = torch.clamp(targets_normalized, min=0.0, max=1.0)


            loss = criterion(outputs, targets_normalized)
            total_loss += loss.item()


            pred_binary = (outputs > 0.5).long().cpu()

            if targets.max() > 1.0:
                target_binary = (targets > 0).long().cpu()
            else:
                target_binary = (targets > 0.5).long().cpu()

            all_preds.append(pred_binary)
            all_targets.append(target_binary)
            all_names.extend(names)


    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)


    metrics = calculate_all_metrics(all_preds, all_targets, num_classes=2)
    metrics['loss'] = total_loss / len(test_loader)

    return metrics, all_preds, all_targets, all_names

def visualize_results(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    output_dir: str = 'results/visualizations',
    num_samples: int = 10,
    diffusion_model: Optional[nn.Module] = None,
    config: Optional[dict] = None
):
    """可视化分割结果"""
    print("\n" + "=" * 70)
    print("生成可视化结果")
    print("=" * 70)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    model.eval()
    use_diffusion = diffusion_model is not None
    if use_diffusion:
        if config is not None:
            inference_config = config.get('diffusion_refiner', {}).get('inference', {})
            num_steps = inference_config.get('num_steps', 20)
        else:
            num_steps = 20

    with torch.no_grad():
        for idx, batch in enumerate(test_loader):
            if idx >= num_samples:
                break

            inputs = batch['input'].to(device)
            targets = batch['mask'].to(device)
            names = batch['name']


            outputs = model(inputs)
            

            if use_diffusion:
                condition = torch.cat([inputs, outputs], dim=1)
                refined_residuals = diffusion_model.sample(condition, num_steps=num_steps)
                outputs = outputs + refined_residuals
                outputs = torch.clamp(outputs, 0.0, 1.0)
            
            pred_binary = (outputs > 0.5).float().cpu()


            for i in range(inputs.size(0)):
                name = names[i]


                input_rgb = inputs[i, :3].cpu().permute(1, 2, 0).numpy()
                input_rgb = (input_rgb / 255.0).clip(0, 1)


                target_mask = targets[i, 0].cpu().numpy()


                pred_mask = pred_binary[i, 0].numpy()


                fig, axes = plt.subplots(1, 4, figsize=(20, 5))

                axes[0].imshow(input_rgb)
                axes[0].set_title('Input (RGB)')
                axes[0].axis('off')

                axes[1].imshow(target_mask, cmap='gray')
                axes[1].set_title('Ground Truth')
                axes[1].axis('off')

                axes[2].imshow(pred_mask, cmap='gray')
                axes[2].set_title('Prediction')
                axes[2].axis('off')


                overlay = input_rgb.copy()
                overlay[pred_mask > 0.5] = [1, 0, 0]
                axes[3].imshow(overlay)
                axes[3].set_title('Overlay')
                axes[3].axis('off')

                plt.tight_layout()
                plt.savefig(output_path / f'{name}.png', dpi=150, bbox_inches='tight')
                plt.close()

    print(f"可视化结果保存到: {output_path}")

def generate_report(metrics: dict, output_path: str = 'evaluation_report.json'):
    """生成评估报告"""
    print("\n" + "=" * 70)
    print("生成评估报告")
    print("=" * 70)


    targets = {
        'miou': 0.70,
        'mprecision': 0.72,
        'recall': 0.65,
        'mpa': 0.78,
        'f1': 0.68,
        'mf1': 0.68
    }


    report = {
        'metrics': metrics,
        'targets': targets,
        'comparison': {
            'miou': {
                'actual': metrics['miou'],
                'target': targets['miou'],
                'meet_target': metrics['miou'] >= targets['miou']
            },
            'mprecision': {
                'actual': metrics['mprecision'],
                'target': targets['mprecision'],
                'meet_target': metrics['mprecision'] >= targets['mprecision']
            },
            'recall': {
                'actual': metrics['recall'],
                'target': targets['recall'],
                'meet_target': metrics['recall'] >= targets['recall']
            },
            'mpa': {
                'actual': metrics['mpa'],
                'target': targets['mpa'],
                'meet_target': metrics['mpa'] >= targets['mpa']
            },
            'f1': {
                'actual': metrics.get('f1', 0.0),
                'target': targets['f1'],
                'meet_target': metrics.get('f1', 0.0) >= targets['f1']
            },
            'mf1': {
                'actual': metrics.get('mf1', 0.0),
                'target': targets['mf1'],
                'meet_target': metrics.get('mf1', 0.0) >= targets['mf1']
            }
        },
        'summary': {
            'all_meet_target': all([
                metrics['miou'] >= targets['miou'],
                metrics['mprecision'] >= targets['mprecision'],
                metrics['recall'] >= targets['recall'],
                metrics['mpa'] >= targets['mpa'],
                metrics.get('f1', 0.0) >= targets['f1'],
                metrics.get('mf1', 0.0) >= targets['mf1']
            ])
        }
    }


    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)


    print("\n评估结果:")
    print(f"  MIoU: {metrics['miou']:.4f} (目标: >={targets['miou']:.2f}) {'[PASS]' if metrics['miou'] >= targets['miou'] else '[FAIL]'}")
    print(f"  MPrecision: {metrics['mprecision']:.4f} (目标: >={targets['mprecision']:.2f}) {'[PASS]' if metrics['mprecision'] >= targets['mprecision'] else '[FAIL]'}")
    print(f"  Recall: {metrics['recall']:.4f} (目标: >={targets['recall']:.2f}) {'[PASS]' if metrics['recall'] >= targets['recall'] else '[FAIL]'}")
    print(f"  MPA: {metrics['mpa']:.4f} (目标: >={targets['mpa']:.2f}) {'[PASS]' if metrics['mpa'] >= targets['mpa'] else '[FAIL]'}")
    print(f"  F1: {metrics.get('f1', 0.0):.4f} (目标: >={targets['f1']:.2f}) {'[PASS]' if metrics.get('f1', 0.0) >= targets['f1'] else '[FAIL]'}")
    print(f"  MF1: {metrics.get('mf1', 0.0):.4f} (目标: >={targets['mf1']:.2f}) {'[PASS]' if metrics.get('mf1', 0.0) >= targets['mf1'] else '[FAIL]'}")
    print(f"  Loss: {metrics['loss']:.4f}")

    print(f"\n报告保存到: {output_path}")

    return report

def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='模型评估')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pth',
                       help='模型检查点路径')
    parser.add_argument('--config', type=str, default=None,
                       help='配置文件路径（当checkpoint没有config时使用）')
    parser.add_argument('--visualize', action='store_true',
                       help='是否生成可视化结果')
    parser.add_argument('--num_vis', type=int, default=10,
                       help='可视化样本数量')
    parser.add_argument('--use_diffusion', action='store_true',
                       help='是否使用Diffusion细化')
    parser.add_argument('--diffusion_checkpoint', type=str, default=None,
                       help='Diffusion模型检查点路径')

    args = parser.parse_args()

    print("=" * 70)
    print("DiT-Mamba 模型评估")
    print("=" * 70)


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")


    if not Path(args.checkpoint).exists():
        print(f"错误: 检查点文件不存在: {args.checkpoint}")
        print("请先训练模型或指定正确的检查点路径")
        return


    config_path = args.config
    if config_path is None:
        default_config = project_root / 'config.yaml'
        if default_config.exists():
            config_path = str(default_config)
            print(f"使用默认配置文件: {config_path}")

    model = load_model(args.checkpoint, device, config_path=config_path)
    

    if config_path and Path(config_path).exists():
        config_obj = TrainingConfig(config_path)
        config_dict = config_obj.config
    else:
        config_dict = {}
    

    diffusion_model = None
    if args.use_diffusion:
        diffusion_checkpoint = args.diffusion_checkpoint
        if diffusion_checkpoint is None:

            diffusion_checkpoint = config_dict.get('diffusion_refiner', {}).get('model_path')
        
        if diffusion_checkpoint and Path(diffusion_checkpoint).exists():
            diffusion_model = load_diffusion_model(diffusion_checkpoint, device, config_dict)
        else:
            print(f"警告: Diffusion模型路径不存在: {diffusion_checkpoint}")
            print("将不使用Diffusion细化")
            args.use_diffusion = False


    checkpoint = torch.load(args.checkpoint, map_location=device)
    use_dataset_adapter = False
    base_path = 'data/rows'
    channels = [0, 1, 2]


    if 'config' in checkpoint and checkpoint['config'] is not None:
        data_config = checkpoint['config'].get('data', {})
        use_dataset_adapter = data_config.get('use_dataset_adapter', False)
        if use_dataset_adapter:
            base_path = 'data/dataset'
            channels = 0
    elif config_path and Path(config_path).exists():
        config = TrainingConfig(config_path)
        data_config = config.get('data', {})
        use_dataset_adapter = data_config.get('use_dataset_adapter', False)
        if use_dataset_adapter:
            base_path = 'data/dataset'
            channels = 0


    print("\n创建测试数据加载器...")


    if use_dataset_adapter:
        test_split_path = project_root / 'datasets/dataset_adapter_test_split.json'
        if not test_split_path.exists():
            test_split_path = project_root / 'datasets/test_split.json'
    else:
        test_split_path = project_root / 'datasets/test_split.json'

    if test_split_path.exists():
        with open(test_split_path, 'r', encoding='utf-8') as f:
            test_samples = json.load(f)
    else:
        print(f"警告: 测试集划分文件不存在: {test_split_path}")
        test_samples = None


    dataset_class = DatasetAdapter if use_dataset_adapter else RowsDataset

    test_dataset = dataset_class(
        base_path=base_path,
        channels=channels,
        mask_mode='enhanced',
        transform=None,
        sample_list=test_samples
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    print(f"测试集样本数: {len(test_dataset)}")


    metrics, all_preds, all_targets, all_names = evaluate_model(
        model, test_loader, device, 
        diffusion_model=diffusion_model if args.use_diffusion else None,
        config=config_dict if args.use_diffusion else None
    )


    report = generate_report(metrics)


    if args.visualize:
        visualize_results(
            model, test_loader, device, num_samples=args.num_vis,
            diffusion_model=diffusion_model if args.use_diffusion else None,
            config=config_dict if args.use_diffusion else None
        )

    print("\n" + "=" * 70)
    print("评估完成!")
    print("=" * 70)

if __name__ == '__main__':
    main()



