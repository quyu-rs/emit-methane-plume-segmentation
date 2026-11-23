# -*- coding: utf-8 -*-
"""
测试不同阈值对预测结果的影响
用于诊断小目标检测问题
"""
import sys
from pathlib import Path
import torch
import numpy as np
import json
import argparse


project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.generate_comparison_figure import (
    load_model,
    check_sample_has_plume,
    predict_with_model,
    load_raw_enh,
    load_rgb_image,
    load_label_mask,
    calculate_iou_numpy
)


def test_thresholds_for_sample(
    model,
    sample_name: str,
    base_path: Path,
    device: torch.device,
    thresholds: list = [0.2, 0.3, 0.4, 0.5, 0.6],
    enh_x0: float = 81.15,
    enh_x1: float = 62.18
):
    """
    测试不同阈值对单个样本的预测结果
    
    参数:
        model: 训练好的模型
        sample_name: 样本名称
        base_path: 数据根目录
        device: 计算设备
        thresholds: 要测试的阈值列表
        enh_x0: ENH归一化参数x0
        enh_x1: ENH归一化参数x1
    """
    print(f"\n{'='*70}")
    print(f"测试样本: {sample_name}")
    print(f"{'='*70}")


    channel_id = 0
    enh_path = base_path / 'enh' / f"{sample_name}_{channel_id}.tif"
    rgb_path = base_path / 'rgb' / f"{sample_name}_{channel_id}.tif"
    plm_path = base_path / 'plm' / f"{sample_name}_{channel_id}.tif"


    if not all([enh_path.exists(), rgb_path.exists(), plm_path.exists()]):
        print(f"错误: 样本文件不存在")
        return None


    enh_raw = load_raw_enh(enh_path)
    rgb = load_rgb_image(rgb_path)
    label = load_label_mask(plm_path)


    enh_normalized = ((enh_raw - enh_x0) / enh_x1 * 255).clip(0, 255).astype(np.uint8)


    print("\n进行模型推理...")
    _, _, pred_prob, prob_stats_base = predict_with_model(
        model, rgb, enh_normalized, enh_raw, device, enh_x0, enh_x1, threshold=0.1
    )

    print(f"\n预测概率统计（使用阈值0.1进行推理）:")
    print(f"  最大值: {prob_stats_base['max']:.4f}")
    print(f"  最小值: {prob_stats_base['min']:.4f}")
    print(f"  平均值: {prob_stats_base['mean']:.4f}")
    print(f"  标准差: {prob_stats_base['std']:.4f}")


    label_pixels = int((label > 0).sum())
    total_pixels = int(label.size)
    print(f"\n标签信息:")
    print(f"  有羽流的像素: {label_pixels}/{total_pixels} ({100*label_pixels/total_pixels:.2f}%)")


    print(f"\n{'='*70}")
    print(f"测试不同阈值的结果:")
    print(f"{'='*70}")
    print(f"{'阈值':<8} {'IoU':<8} {'预测像素数':<15} {'预测像素占比':<15} {'最大概率':<10}")
    print(f"{'-'*70}")

    results = []
    for threshold in thresholds:

        pred_mask = (pred_prob > threshold).astype(np.float32)


        iou = calculate_iou_numpy(pred_mask, label)


        pred_pixels = int(pred_mask.sum())
        pred_ratio = 100 * pred_pixels / total_pixels


        if label_pixels > 0:
            label_region_probs = pred_prob[label > 0]
            max_prob_in_label = float(label_region_probs.max()) if len(label_region_probs) > 0 else 0.0
        else:
            max_prob_in_label = 0.0

        print(f"{threshold:<8.2f} {iou:<8.4f} {pred_pixels:<15} {pred_ratio:<15.2f}% {max_prob_in_label:<10.4f}")

        results.append({
            'threshold': threshold,
            'iou': iou,
            'pred_pixels': pred_pixels,
            'pred_ratio': pred_ratio,
            'max_prob_in_label': max_prob_in_label
        })


    best_iou = max(results, key=lambda x: x['iou'])
    print(f"\n最佳IoU: {best_iou['iou']:.4f} (阈值: {best_iou['threshold']:.2f})")


    print(f"\n{'='*70}")
    print(f"诊断建议:")
    print(f"{'='*70}")

    if prob_stats_base['max'] < 0.5:
        print(f"[警告] 最大预测概率 ({prob_stats_base['max']:.4f}) < 0.5")
        print(f"   建议: 使用更低的阈值 (< 0.5) 或改进模型")

    if label_pixels > 0 and best_iou['max_prob_in_label'] < 0.5:
        print(f"[警告] 标签区域内的最大预测概率 ({best_iou['max_prob_in_label']:.4f}) < 0.5")
        print(f"   建议: 模型对真实羽流区域的置信度较低，可能需要:")
        print(f"         - 降低阈值到 {best_iou['threshold']:.2f} 或更低")
        print(f"         - 改进模型训练（增加小目标样本、调整损失函数）")

    if best_iou['iou'] < 0.3:
        print(f"[警告] 最佳IoU ({best_iou['iou']:.4f}) < 0.3")
        print(f"   建议: 模型检测效果较差，可能需要:")
        print(f"         - 检查数据质量")
        print(f"         - 重新训练模型")
        print(f"         - 使用数据增强")

    if best_iou['iou'] > 0.5:
        print(f"[OK] 最佳阈值 {best_iou['threshold']:.2f} 下IoU ({best_iou['iou']:.4f}) > 0.5，效果良好")

    return results


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='测试不同阈值对预测结果的影响',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:

  python scripts/test_thresholds.py --checkpoint checkpoints/best_model.pth \\
      --sample 20230205T171255_neg_001


  python scripts/test_thresholds.py --checkpoint checkpoints/best_model.pth \\
      --samples 20230205T171255_neg_001 20240929T113148_neg_001


  python scripts/test_thresholds.py --checkpoint checkpoints/best_model.pth \\
      --sample 20230205T171255_neg_001 --thresholds 0.1 0.2 0.3 0.4 0.5
        """
    )
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pth',
                       help='模型检查点路径')
    parser.add_argument('--sample', type=str, default=None,
                       help='单个样本名称')
    parser.add_argument('--samples', type=str, nargs='+', default=None,
                       help='多个样本名称列表')
    parser.add_argument('--data_path', type=str, default='data/rows',
                       help='数据根目录')
    parser.add_argument('--thresholds', type=float, nargs='+',
                       default=[0.2, 0.3, 0.4, 0.5, 0.6],
                       help='要测试的阈值列表，默认 [0.2, 0.3, 0.4, 0.5, 0.6]')

    args = parser.parse_args()


    if args.sample:
        sample_names = [args.sample]
    elif args.samples:
        sample_names = args.samples
    else:

        print("未指定样本，从测试集选择正样本进行测试...")
        with open('datasets/test_split.json', 'r', encoding='utf-8') as f:
            test_samples = json.load(f)

        base_path = Path(args.data_path)
        pos_samples = [s for s in test_samples if check_sample_has_plume(s, base_path)]

        if len(pos_samples) >= 5:
            import random
            random.seed(42)
            sample_names = random.sample(pos_samples, 5)
            print(f"随机选择5个正样本: {sample_names}")
        else:
            sample_names = pos_samples[:min(5, len(pos_samples))]
            print(f"选择所有可用正样本: {sample_names}")


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")


    if not Path(args.checkpoint).exists():
        print(f"错误: 检查点文件不存在: {args.checkpoint}")
        return

    model = load_model(args.checkpoint, device)


    base_path = Path(args.data_path)
    all_results = {}

    for sample_name in sample_names:
        results = test_thresholds_for_sample(
            model, sample_name, base_path, device,
            thresholds=args.thresholds
        )
        if results:
            all_results[sample_name] = results


    if len(all_results) > 1:
        print(f"\n{'='*70}")
        print(f"总结:")
        print(f"{'='*70}")
        for sample_name, results in all_results.items():
            best = max(results, key=lambda x: x['iou'])
            print(f"{sample_name}: 最佳IoU {best['iou']:.4f} (阈值 {best['threshold']:.2f})")


if __name__ == '__main__':
    main()



