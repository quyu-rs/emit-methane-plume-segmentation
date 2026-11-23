# -*- coding: utf-8 -*-
"""
查找 case2 和 case6 对应的样本名称
根据羽流大小特征来匹配
"""
import sys
from pathlib import Path
import numpy as np
try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.generate_figure_from_dataset import check_sample_quality_fast, detect_data_format


def find_samples_by_plume_size(base_path: Path, data_format: str, num_samples: int = 6):
    """根据羽流大小查找样本"""
    print(f"扫描 {base_path} 目录...")
    

    if data_format == 'dataset_convert':
        rgb_dir = base_path / 'rgb'
        if not rgb_dir.exists():
            print(f"错误: RGB目录不存在: {rgb_dir}")
            return []
        
        all_files = list(rgb_dir.glob('*.tif'))
        sample_names = list(set([f.stem.rsplit('_', 1)[0] for f in all_files]))
    else:
        rgb_dir = base_path / 'rgb'
        if not rgb_dir.exists():
            print(f"错误: RGB目录不存在: {rgb_dir}")
            return []
        
        all_files = list(rgb_dir.glob('*.jpg'))
        sample_names = list(set([f.stem for f in all_files]))
    
    print(f"找到 {len(sample_names)} 个样本，检查质量...")
    

    valid_samples = []
    for idx, sample_name in enumerate(sample_names):
        if (idx + 1) % 500 == 0:
            print(f"  处理进度: {idx + 1}/{len(sample_names)}")
        
        quality = check_sample_quality_fast(sample_name, base_path, data_format)
        if quality.get('is_valid', False) and quality.get('has_plume', False):
            valid_samples.append((sample_name, quality))
    
    print(f"找到 {len(valid_samples)} 个正样本")
    

    valid_samples.sort(key=lambda x: x[1]['plume_size'], reverse=True)
    

    return [s[0] for s in valid_samples[:num_samples]]


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='查找 case2 和 case6 对应的样本')
    parser.add_argument('--data_path', type=str, default='data/rows',
                       help='数据根目录')
    parser.add_argument('--data_format', type=str, choices=['dataset', 'dataset_convert'], default=None,
                       help='数据格式（自动检测）')
    parser.add_argument('--num_samples', type=int, default=6,
                       help='返回的样本数量')
    
    args = parser.parse_args()
    
    base_path = Path(args.data_path)
    

    if args.data_format is None:
        data_format = detect_data_format(base_path)
    else:
        data_format = args.data_format
    
    print(f"数据格式: {data_format}")
    print(f"数据路径: {base_path}\n")
    

    samples = find_samples_by_plume_size(base_path, data_format, args.num_samples)
    
    if len(samples) >= 6:
        print(f"\n找到的样本（按羽流大小排序）:")
        for i, sample in enumerate(samples, 1):
            quality = check_sample_quality_fast(sample, base_path, data_format)
            plume_size = quality.get('plume_size', 0)
            plume_ratio = quality.get('plume_ratio', 0) * 100
            print(f"  Case {i}: {sample}")
            print(f"    羽流大小: {plume_size:,} 像素, 占比: {plume_ratio:.3f}%")
        
        print(f"\n根据图片描述:")
        print(f"  - Case 2: 有一个小的、不规则形状的白色区域（羽流）")
        print(f"  - Case 6: 有一个更大、更突出的不规则白色区域（羽流）")
        print(f"\n建议:")
        print(f"  如果 Case 2 的羽流较小，可能是样本: {samples[1] if len(samples) > 1 else 'N/A'}")
        print(f"  如果 Case 6 的羽流较大，可能是样本: {samples[5] if len(samples) > 5 else 'N/A'}")
        
        print(f"\n分析命令（请根据实际情况调整样本名称）:")
        print(f"python scripts/analyze_label_quality.py --sample_names {samples[1]} {samples[5]} --data_path {args.data_path} --visualize --output checkpoints/emit/label_analysis.json")
    else:
        print(f"只找到 {len(samples)} 个正样本，不足6个")


if __name__ == '__main__':
    main()



