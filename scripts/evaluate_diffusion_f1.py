# -*- coding: utf-8 -*-
"""
Diffusion模型F1 Score评估脚本
重新评估已训练的Diffusion模型，计算包括F1 Score在内的所有指标
"""
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import argparse
from tqdm import tqdm
import numpy as np
import json
from datetime import datetime

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.diffusion_refiner import DiffusionRefiner
from training.config import TrainingConfig
from metrics.segmentation_metrics import calculate_all_metrics


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


class DiffusionEvaluator:
    """
    Diffusion模型评估器
    """
    def __init__(self, model, device, num_timesteps=1000):
        """
        参数:
            model: Diffusion模型
            device: 设备
            num_timesteps: 扩散步数
        """
        self.model = model.to(device)
        self.device = device
        self.num_timesteps = num_timesteps
        self._init_noise_schedule()
    
    def _init_noise_schedule(self):
        """初始化噪声调度"""
        beta_start = 0.0001
        beta_end = 0.02
        
        self.betas = torch.linspace(beta_start, beta_end, self.num_timesteps, device=self.device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
    
    def sample_with_model(
        self,
        condition: torch.Tensor,
        num_steps: int = 20
    ) -> torch.Tensor:
        """
        采样生成细化结果
        使用DDIM采样
        
        参数:
            condition: 条件输入（图像+初始掩码），形状为 (B, 5, H, W)
            num_steps: 采样步数
        
        返回:
            细化后的残差，形状为 (B, 1, H, W)
        """
        self.model.eval()
        batch_size = condition.size(0)
        

        x_t = torch.randn(batch_size, 1, condition.shape[2], condition.shape[3], device=self.device) * 0.03
        

        num_steps = max(1, min(num_steps, self.num_timesteps))

        timesteps = torch.linspace(self.num_timesteps - 1, 0, num_steps, device=self.device).long()
        
        with torch.no_grad():
            for i, t in enumerate(timesteps):

                t_idx = t.item() if t.numel() == 1 else int(t)
                t_batch = t.repeat(batch_size)
                

                predicted_noise = self.model(x_t, t_batch, condition)
                

                alpha_cumprod_t = self.alphas_cumprod[t_idx]
                sqrt_alpha_cumprod_t = torch.sqrt(alpha_cumprod_t)
                sqrt_alpha_cumprod_t = torch.clamp(sqrt_alpha_cumprod_t, min=1e-8)
                sqrt_one_minus_alpha_cumprod_t = torch.sqrt(1.0 - alpha_cumprod_t)
                sqrt_one_minus_alpha_cumprod_t = torch.clamp(sqrt_one_minus_alpha_cumprod_t, min=1e-8)
                

                pred_x0 = (x_t - sqrt_one_minus_alpha_cumprod_t * predicted_noise) / sqrt_alpha_cumprod_t
                
                if i < len(timesteps) - 1:

                    t_prev = timesteps[i + 1]
                    t_prev_idx = t_prev.item() if t_prev.numel() == 1 else int(t_prev)
                    alpha_cumprod_t_prev = self.alphas_cumprod[t_prev_idx]
                    sqrt_alpha_cumprod_t_prev = torch.sqrt(alpha_cumprod_t_prev)
                    sqrt_one_minus_alpha_cumprod_t_prev = torch.sqrt(1.0 - alpha_cumprod_t_prev)
                    sqrt_one_minus_alpha_cumprod_t_prev = torch.clamp(sqrt_one_minus_alpha_cumprod_t_prev, min=1e-8)
                    
                    x_t = sqrt_alpha_cumprod_t_prev * pred_x0 + sqrt_one_minus_alpha_cumprod_t_prev * predicted_noise
                else:

                    x_t = pred_x0
        

        x_t = torch.clamp(x_t, min=-0.15, max=0.15)
        
        return x_t
    
    def evaluate(self, val_loader, num_steps=15):
        """
        评估模型
        
        参数:
            val_loader: 验证数据加载器
            num_steps: 采样步数
        
        返回:
            评估指标字典
        """
        self.model.eval()
        
        all_preds = []
        all_targets = []
        all_baseline_preds = []
        total_loss = 0.0
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc='评估中')
            
            for batch in pbar:
                input_images = batch['input_image'].to(self.device)
                pred_coarse = batch['pred_coarse'].to(self.device)
                residuals = batch['residual'].to(self.device)
                target_masks = batch['target_mask'].to(self.device)
                

                condition = torch.cat([input_images, pred_coarse], dim=1)
                

                refined_residuals = self.sample_with_model(condition, num_steps=num_steps)
                

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
        

        if len(all_preds) > 0:
            all_preds = torch.cat(all_preds, dim=0)
            all_targets = torch.cat(all_targets, dim=0)
            all_baseline_preds = torch.cat(all_baseline_preds, dim=0)
            

            metrics = calculate_all_metrics(all_preds, all_targets, num_classes=2)
            metrics['loss'] = total_loss / len(val_loader)
            

            baseline_metrics = calculate_all_metrics(all_baseline_preds, all_targets, num_classes=2)
            

            for key in ['miou', 'mprecision', 'recall', 'mpa', 'f1', 'mf1']:
                metrics[f'baseline_{key}'] = baseline_metrics[key]
            

            for key in ['miou', 'mprecision', 'recall', 'mpa', 'f1', 'mf1']:
                metrics[f'{key}_improvement'] = metrics[key] - baseline_metrics[key]
        
        return metrics


def load_diffusion_model(checkpoint_path: str, device: torch.device, config: dict):
    """
    加载Diffusion模型
    
    参数:
        checkpoint_path: Diffusion模型检查点路径
        device: 设备
        config: 配置字典
    
    返回:
        Diffusion模型和num_timesteps
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
    print(f"扩散步数: {num_timesteps}")
    
    return model, num_timesteps


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='评估Diffusion模型并计算F1 Score')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Diffusion模型检查点路径')
    parser.add_argument('--preprocessed_data', type=str, required=True,
                       help='预处理数据文件路径')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='配置文件路径')
    parser.add_argument('--num_steps', type=int, default=15,
                       help='采样步数（默认15）')
    parser.add_argument('--output', type=str, default=None,
                       help='输出JSON文件路径（可选）')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Diffusion模型F1 Score评估")
    print("=" * 70)
    

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    

    config_path = project_root / args.config
    if not config_path.exists():
        print(f"错误: 配置文件不存在: {config_path}")
        return
    
    config = TrainingConfig(str(config_path))
    

    print(f"\n加载预处理数据: {args.preprocessed_data}")
    preprocessed_data = torch.load(args.preprocessed_data, map_location='cpu')
    val_data = preprocessed_data.get('val', [])
    print(f"验证集: {len(val_data)}个样本")
    
    if len(val_data) == 0:
        print("错误: 验证集为空")
        return
    

    val_dataset = DiffusionDataset(val_data)
    data_config = config.get('data', {})
    batch_size = data_config.get('batch_size', 32)
    num_workers = data_config.get('num_workers', 1)
    pin_memory = data_config.get('pin_memory', True)
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    

    model, num_timesteps = load_diffusion_model(args.checkpoint, device, config.config)
    

    evaluator = DiffusionEvaluator(model, device, num_timesteps)
    

    print(f"\n开始评估（采样步数: {args.num_steps}）...")
    metrics = evaluator.evaluate(val_loader, num_steps=args.num_steps)
    

    print("\n" + "=" * 70)
    print("评估结果")
    print("=" * 70)
    print(f"\n验证损失 (Mask MSE): {metrics['loss']:.6f}")
    
    print("\n基线指标 (仅pred_coarse):")
    print(f"  MIoU:      {metrics['baseline_miou']:.4f}")
    print(f"  MPrecision: {metrics['baseline_mprecision']:.4f}")
    print(f"  Recall:    {metrics['baseline_recall']:.4f}")
    print(f"  MPA:       {metrics['baseline_mpa']:.4f}")
    print(f"  F1:        {metrics['baseline_f1']:.4f}")
    print(f"  MF1:       {metrics['baseline_mf1']:.4f}")
    
    print("\nDiffusion细化后指标:")
    print(f"  MIoU:      {metrics['miou']:.4f}")
    print(f"  MPrecision: {metrics['mprecision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  MPA:       {metrics['mpa']:.4f}")
    print(f"  F1:        {metrics['f1']:.4f}")
    print(f"  MF1:       {metrics['mf1']:.4f}")
    
    print("\n改进量:")
    print(f"  MIoU:      {metrics['miou_improvement']:+.4f} ({metrics['miou_improvement']*100:+.2f}%)")
    print(f"  MPrecision: {metrics['mprecision_improvement']:+.4f} ({metrics['mprecision_improvement']*100:+.2f}%)")
    print(f"  Recall:    {metrics['recall_improvement']:+.4f} ({metrics['recall_improvement']*100:+.2f}%)")
    print(f"  MPA:       {metrics['mpa_improvement']:+.4f} ({metrics['mpa_improvement']*100:+.2f}%)")
    print(f"  F1:        {metrics['f1_improvement']:+.4f} ({metrics['f1_improvement']*100:+.2f}%)")
    print(f"  MF1:       {metrics['mf1_improvement']:+.4f} ({metrics['mf1_improvement']*100:+.2f}%)")
    
    print("=" * 70)
    

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        result = {
            'checkpoint': str(args.checkpoint),
            'preprocessed_data': str(args.preprocessed_data),
            'num_steps': args.num_steps,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'metrics': metrics
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        print(f"\n结果已保存到: {output_path}")
    else:

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = project_root / 'checkpoints' / f'diffusion_f1_evaluation_{timestamp}.json'
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        result = {
            'checkpoint': str(args.checkpoint),
            'preprocessed_data': str(args.preprocessed_data),
            'num_steps': args.num_steps,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'metrics': metrics
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        print(f"\n结果已保存到: {output_path}")


if __name__ == '__main__':
    main()



