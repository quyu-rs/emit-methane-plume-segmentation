# -*- coding: utf-8 -*-
"""
验证和优化ENH归一化参数
检查data/dataset格式的ENH数据归一化后的分布
"""
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import json


project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from datasets.dataset_adapter import DatasetAdapter

def analyze_enh_distribution(base_path='data/dataset', num_samples=100):
    """分析ENH数据分布"""
    print("=" * 70)
    print("ENH归一化参数验证")
    print("=" * 70)


    dataset = DatasetAdapter(
        base_path=base_path,
        channels=0,
        mask_mode='binary',
        normalize_enh=False
    )

    if len(dataset) == 0:
        print("错误: 数据集为空")
        return None

    enh_values = []
    enh_normalized_values = []

    num_check = min(num_samples, len(dataset))
    print(f"\n分析前 {num_check} 个样本（共 {len(dataset)} 个）...")

    for i in range(num_check):
        if (i + 1) % 20 == 0:
            print(f"  处理进度: {i + 1}/{num_check}")

        sample = dataset[i]

        enh_raw = sample['input'][3].numpy()
        enh_values.extend(enh_raw.flatten().tolist())


        enh_x0, enh_x1 = 81.15, 62.18
        enh_norm = (enh_raw - enh_x0) / enh_x1 * 255
        enh_norm = np.clip(enh_norm, 0, 255)
        enh_normalized_values.extend(enh_norm.flatten().tolist())

    enh_values = np.array(enh_values)
    enh_normalized_values = np.array(enh_normalized_values)

    print(f"\n原始ENH数据统计:")
    print(f"  最小值: {enh_values.min():.2f}")
    print(f"  最大值: {enh_values.max():.2f}")
    print(f"  平均值: {enh_values.mean():.2f}")
    print(f"  标准差: {enh_values.std():.2f}")
    print(f"  中位数: {np.median(enh_values):.2f}")
    print(f"  25%分位数: {np.percentile(enh_values, 25):.2f}")
    print(f"  75%分位数: {np.percentile(enh_values, 75):.2f}")

    print(f"\n归一化后ENH数据统计 (当前参数: enh_x0={81.15}, enh_x1={62.18}):")
    print(f"  最小值: {enh_normalized_values.min():.2f}")
    print(f"  最大值: {enh_normalized_values.max():.2f}")
    print(f"  平均值: {enh_normalized_values.mean():.2f}")
    print(f"  标准差: {enh_normalized_values.std():.2f}")
    print(f"  中位数: {np.median(enh_normalized_values):.2f}")


    out_of_range_low = np.sum(enh_normalized_values < 0)
    out_of_range_high = np.sum(enh_normalized_values > 255)
    out_of_range = out_of_range_low + out_of_range_high
    print(f"\n超出[0,255]范围的像素数:")
    print(f"  小于0: {out_of_range_low} ({out_of_range_low/len(enh_normalized_values)*100:.2f}%)")
    print(f"  大于255: {out_of_range_high} ({out_of_range_high/len(enh_normalized_values)*100:.2f}%)")
    print(f"  总计: {out_of_range} ({out_of_range/len(enh_normalized_values)*100:.2f}%)")


    enh_min, enh_max = enh_values.min(), enh_values.max()
    suggested_x0 = enh_min
    suggested_x1 = (enh_max - enh_min) / 255.0 if (enh_max - enh_min) > 0 else 1.0

    print(f"\n建议的归一化参数（基于Min-Max）:")
    print(f"  enh_x0: {suggested_x0:.2f}")
    print(f"  enh_x1: {suggested_x1:.2f}")


    enh_suggested = (enh_values - suggested_x0) / suggested_x1 * 255
    enh_suggested = np.clip(enh_suggested, 0, 255)

    print(f"\n使用建议参数归一化后:")
    print(f"  最小值: {enh_suggested.min():.2f}")
    print(f"  最大值: {enh_suggested.max():.2f}")
    print(f"  平均值: {enh_suggested.mean():.2f}")
    print(f"  标准差: {enh_suggested.std():.2f}")


    enh_mean = enh_values.mean()
    enh_std = enh_values.std()

    robust_min = max(enh_min, enh_mean - 3 * enh_std)
    robust_max = min(enh_max, enh_mean + 3 * enh_std)
    robust_x0 = robust_min
    robust_x1 = (robust_max - robust_min) / 255.0 if (robust_max - robust_min) > 0 else 1.0

    print(f"\n稳健归一化参数（基于均值±3标准差）:")
    print(f"  enh_x0: {robust_x0:.2f}")
    print(f"  enh_x1: {robust_x1:.2f}")


    enh_robust = (enh_values - robust_x0) / robust_x1 * 255
    enh_robust = np.clip(enh_robust, 0, 255)

    print(f"\n使用稳健参数归一化后:")
    print(f"  最小值: {enh_robust.min():.2f}")
    print(f"  最大值: {enh_robust.max():.2f}")
    print(f"  平均值: {enh_robust.mean():.2f}")

    results = {
        'original': {
            'min': float(enh_values.min()),
            'max': float(enh_values.max()),
            'mean': float(enh_values.mean()),
            'std': float(enh_values.std()),
            'median': float(np.median(enh_values)),
            'p25': float(np.percentile(enh_values, 25)),
            'p75': float(np.percentile(enh_values, 75))
        },
        'normalized_current': {
            'min': float(enh_normalized_values.min()),
            'max': float(enh_normalized_values.max()),
            'mean': float(enh_normalized_values.mean()),
            'std': float(enh_normalized_values.std()),
            'median': float(np.median(enh_normalized_values)),
            'out_of_range': int(out_of_range),
            'out_of_range_pct': float(out_of_range/len(enh_normalized_values)*100)
        },
        'suggested_params_minmax': {
            'enh_x0': float(suggested_x0),
            'enh_x1': float(suggested_x1)
        },
        'suggested_params_robust': {
            'enh_x0': float(robust_x0),
            'enh_x1': float(robust_x1)
        },
        'current_params': {
            'enh_x0': 81.15,
            'enh_x1': 62.18
        }
    }


    output_file = project_root / 'enh_normalization_analysis.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n分析结果已保存到: {output_file}")

    return results

def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='验证ENH归一化参数')
    parser.add_argument('--base_path', type=str, default='data/dataset',
                       help='数据集根目录路径')
    parser.add_argument('--num_samples', type=int, default=100,
                       help='分析的样本数量（0表示分析全部）')

    args = parser.parse_args()

    num_samples = args.num_samples if args.num_samples > 0 else 0

    results = analyze_enh_distribution(args.base_path, num_samples)

    if results:
        print("\n" + "=" * 70)
        print("建议:")
        print("=" * 70)

        out_of_range_pct = results['normalized_current']['out_of_range_pct']

        if out_of_range_pct > 5:
            print(f"1. [重要] 当前归一化参数导致 {out_of_range_pct:.2f}% 的像素超出[0,255]范围")
            print("   建议使用稳健归一化参数或Min-Max参数")
            print(f"   稳健参数: enh_x0={results['suggested_params_robust']['enh_x0']:.2f}, "
                  f"enh_x1={results['suggested_params_robust']['enh_x1']:.2f}")
        else:
            print("1. [OK] 当前归一化参数工作良好，可以继续使用")

        print("\n2. 如果使用建议的参数，可以在DatasetAdapter初始化时传入:")
        print("   dataset = DatasetAdapter(")
        print("       base_path='data/dataset',")
        print(f"       enh_x0={results['suggested_params_robust']['enh_x0']:.2f},")
        print(f"       enh_x1={results['suggested_params_robust']['enh_x1']:.2f}")
        print("   )")

        print("\n3. 或者在训练脚本的配置文件中设置ENH归一化参数")
        print("=" * 70)

if __name__ == '__main__':
    main()



