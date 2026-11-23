# -*- coding: utf-8 -*-
"""
重新评估数据集质量和分布
用于数据补充后的重新检查
"""
import sys
from pathlib import Path
import json
import numpy as np
import rasterio
from collections import defaultdict


project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from datasets.rows_dataset import RowsDataset
from datasets.split_dataset import get_all_samples, split_dataset
from scripts.comprehensive_training_check import comprehensive_training_check

def analyze_data_distribution_detailed(base_path='data/rows'):
    """详细分析数据分布"""
    print("=" * 70)
    print("详细数据分布分析")
    print("=" * 70)

    base = Path(base_path)
    plm_dir = base / 'plm'


    all_samples = get_all_samples(base_path)


    channel_stats = defaultdict(lambda: {'pos': 0, 'neg': 0, 'total': 0})

    for sample_name, is_positive in all_samples.items():

        for channel_id in [0, 1, 2]:
            plm_file = plm_dir / f"{sample_name}_{channel_id}.tif"
            if plm_file.exists():
                channel_stats[channel_id]['total'] += 1
                if is_positive:
                    channel_stats[channel_id]['pos'] += 1
                else:
                    channel_stats[channel_id]['neg'] += 1

    print("\n按通道统计:")
    for channel_id in sorted(channel_stats.keys()):
        stats = channel_stats[channel_id]
        pos_ratio = stats['pos'] / stats['total'] * 100 if stats['total'] > 0 else 0
        print(f"  通道 {channel_id}:")
        print(f"    总样本: {stats['total']}")
        print(f"    正样本: {stats['pos']} ({pos_ratio:.1f}%)")
        print(f"    负样本: {stats['neg']} ({100-pos_ratio:.1f}%)")


    total_samples = len(all_samples)
    pos_samples = sum(1 for is_pos in all_samples.values() if is_pos)
    neg_samples = total_samples - pos_samples

    print(f"\n总体统计（唯一样本）:")
    print(f"  总样本数: {total_samples}")
    print(f"  正样本: {pos_samples} ({pos_samples/total_samples*100:.1f}%)")
    print(f"  负样本: {neg_samples} ({neg_samples/total_samples*100:.1f}%)")


    total_with_channels = sum(stats['total'] for stats in channel_stats.values())
    pos_with_channels = sum(stats['pos'] for stats in channel_stats.values())
    neg_with_channels = sum(stats['neg'] for stats in channel_stats.values())

    print(f"\n多通道扩展后统计:")
    print(f"  总样本数: {total_with_channels}")
    print(f"  正样本: {pos_with_channels} ({pos_with_channels/total_with_channels*100:.1f}%)")
    print(f"  负样本: {neg_with_channels} ({neg_with_channels/total_with_channels*100:.1f}%)")


    total_with_aug = total_with_channels * 2
    print(f"\n包含数据增强（180度旋转）后:")
    print(f"  总样本数: {total_with_aug}")

    return {
        'unique_samples': {
            'total': total_samples,
            'pos': pos_samples,
            'neg': neg_samples,
            'pos_ratio': pos_samples / total_samples if total_samples > 0 else 0
        },
        'multi_channel': {
            'total': total_with_channels,
            'pos': pos_with_channels,
            'neg': neg_with_channels,
            'pos_ratio': pos_with_channels / total_with_channels if total_with_channels > 0 else 0
        },
        'with_augmentation': {
            'total': total_with_aug
        },
        'by_channel': {
            str(ch): {
                'total': stats['total'],
                'pos': stats['pos'],
                'neg': stats['neg'],
                'pos_ratio': stats['pos'] / stats['total'] if stats['total'] > 0 else 0
            }
            for ch, stats in channel_stats.items()
        }
    }

def check_data_completeness_detailed(base_path='data/rows'):
    """详细检查数据完整性"""
    print("=" * 70)
    print("详细数据完整性检查")
    print("=" * 70)

    base = Path(base_path)
    rgb_dir = base / 'rgb'
    enh_dir = base / 'enh'
    plm_dir = base / 'plm'


    rgb_files = sorted(list(rgb_dir.glob('*.tif'))) if rgb_dir.exists() else []
    enh_files = sorted(list(enh_dir.glob('*.tif'))) if enh_dir.exists() else []
    plm_files = sorted(list(plm_dir.glob('*.tif'))) if plm_dir.exists() else []

    print(f"RGB文件: {len(rgb_files)}")
    print(f"ENH文件: {len(enh_files)}")
    print(f"PLM文件: {len(plm_files)}")


    channel_counts = defaultdict(lambda: {'rgb': 0, 'enh': 0, 'plm': 0})

    for rgb_file in rgb_files:
        if rgb_file.name.endswith('_0.tif'):
            channel_counts[0]['rgb'] += 1
        elif rgb_file.name.endswith('_1.tif'):
            channel_counts[1]['rgb'] += 1
        elif rgb_file.name.endswith('_2.tif'):
            channel_counts[2]['rgb'] += 1

    for enh_file in enh_files:
        if enh_file.name.endswith('_0.tif'):
            channel_counts[0]['enh'] += 1
        elif enh_file.name.endswith('_1.tif'):
            channel_counts[1]['enh'] += 1
        elif enh_file.name.endswith('_2.tif'):
            channel_counts[2]['enh'] += 1

    for plm_file in plm_files:
        if plm_file.name.endswith('_0.tif'):
            channel_counts[0]['plm'] += 1
        elif plm_file.name.endswith('_1.tif'):
            channel_counts[1]['plm'] += 1
        elif plm_file.name.endswith('_2.tif'):
            channel_counts[2]['plm'] += 1

    print("\n按通道统计文件数:")
    for channel_id in sorted(channel_counts.keys()):
        counts = channel_counts[channel_id]
        print(f"  通道 {channel_id}: RGB={counts['rgb']}, ENH={counts['enh']}, PLM={counts['plm']}")
        if counts['rgb'] != counts['enh'] or counts['rgb'] != counts['plm']:
            print(f"    [WARN] 通道 {channel_id} 文件数不一致!")


    print("\n检查文件对应关系...")
    missing_pairs = []


    unique_samples = set()
    for rgb_file in rgb_files:
        if rgb_file.name.endswith('_0.tif'):
            sample_name = rgb_file.stem.rsplit('_', 1)[0]
            unique_samples.add(sample_name)

    for sample_name in sorted(unique_samples):
        for channel_id in [0, 1, 2]:
            rgb_file = rgb_dir / f"{sample_name}_{channel_id}.tif"
            enh_file = enh_dir / f"{sample_name}_{channel_id}.tif"
            plm_file = plm_dir / f"{sample_name}_{channel_id}.tif"

            if not rgb_file.exists() or not enh_file.exists() or not plm_file.exists():
                missing_pairs.append({
                    'sample': sample_name,
                    'channel': channel_id,
                    'missing_rgb': not rgb_file.exists(),
                    'missing_enh': not enh_file.exists(),
                    'missing_plm': not plm_file.exists()
                })

    if len(missing_pairs) > 0:
        print(f"[WARN] 发现 {len(missing_pairs)} 个缺失文件对")
        for item in missing_pairs[:10]:
            print(f"  {item['sample']}_通道{item['channel']}: RGB={not item['missing_rgb']}, ENH={not item['missing_enh']}, PLM={not item['missing_plm']}")
    else:
        print("[OK] 所有文件对应关系完整")

    return {
        'total_files': {
            'rgb': len(rgb_files),
            'enh': len(enh_files),
            'plm': len(plm_files)
        },
        'by_channel': {
            str(ch): counts for ch, counts in channel_counts.items()
        },
        'unique_samples': len(unique_samples),
        'missing_pairs': len(missing_pairs)
    }

def main():
    """主函数"""
    print("=" * 70)
    print("数据集重新评估（数据补充后）")
    print("=" * 70)

    base_path = 'data/rows'


    completeness = check_data_completeness_detailed(base_path)


    distribution = analyze_data_distribution_detailed(base_path)


    print("\n" + "=" * 70)
    print("运行全面训练检查")
    print("=" * 70)
    comprehensive_results = comprehensive_training_check(base_path)


    print("\n" + "=" * 70)
    print("测试多通道数据加载")
    print("=" * 70)

    try:
        dataset = RowsDataset(
            base_path=base_path,
            channels=[0, 1, 2],
            mask_mode='enhanced'
        )
        print(f"[OK] 多通道数据集加载成功: {len(dataset)} 个样本")


        from datasets.transforms import Rotate180
        dataset_aug = RowsDataset(
            base_path=base_path,
            channels=[0, 1, 2],
            mask_mode='enhanced',
            transform=Rotate180()
        )
        print(f"[OK] 包含数据增强的数据集: {len(dataset_aug)} 个样本")
        print(f"[OK] 预期样本数（多通道+旋转）: {len(dataset) * 2}")

    except Exception as e:
        print(f"[FAIL] 数据加载测试失败: {e}")
        import traceback
        traceback.print_exc()


    print("\n" + "=" * 70)
    print("数据集评估汇总")
    print("=" * 70)

    summary = {
        'completeness': completeness,
        'distribution': distribution,
        'comprehensive_check': comprehensive_results,
        'data_expansion': {
            'unique_samples': distribution['unique_samples']['total'],
            'multi_channel_samples': distribution['multi_channel']['total'],
            'with_augmentation_samples': distribution['with_augmentation']['total'],
            'expansion_ratio': {
                'multi_channel': distribution['multi_channel']['total'] / distribution['unique_samples']['total'] if distribution['unique_samples']['total'] > 0 else 0,
                'with_augmentation': distribution['with_augmentation']['total'] / distribution['unique_samples']['total'] if distribution['unique_samples']['total'] > 0 else 0
            }
        }
    }


    report_path = 'data_reevaluation_report.json'
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n评估报告已保存到: {report_path}")


    print("\n" + "=" * 70)
    print("关键统计信息")
    print("=" * 70)
    print(f"唯一样本数: {distribution['unique_samples']['total']}")
    print(f"  正样本: {distribution['unique_samples']['pos']} ({distribution['unique_samples']['pos_ratio']*100:.1f}%)")
    print(f"  负样本: {distribution['unique_samples']['neg']} ({1-distribution['unique_samples']['pos_ratio']:.1f}%)")
    print(f"\n多通道扩展后: {distribution['multi_channel']['total']} 个样本")
    print(f"包含数据增强后: {distribution['with_augmentation']['total']} 个样本")
    print(f"\n数据扩展倍数:")
    print(f"  多通道: {summary['data_expansion']['expansion_ratio']['multi_channel']:.1f}x")
    print(f"  数据增强: {summary['data_expansion']['expansion_ratio']['with_augmentation']:.1f}x")

    print("\n" + "=" * 70)
    print("建议下一步操作:")
    print("=" * 70)
    print("1. 重新运行数据集划分: python datasets/split_dataset.py")
    print("2. 验证数据加载器: python scripts/test_dataloader.py")
    print("3. 开始训练: python scripts/train.py")
    print("=" * 70)

if __name__ == '__main__':
    main()



