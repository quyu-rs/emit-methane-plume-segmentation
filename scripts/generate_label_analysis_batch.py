# -*- coding: utf-8 -*-
"""
批量生成所有样本的 label_analysis.json
从 data/rows 目录中读取所有样本，生成完整的分析报告
"""
import sys
from pathlib import Path
import json
from tqdm import tqdm

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.analyze_label_quality import (
    analyze_label_quality, 
    load_label_mask_tif, 
    load_label_mask_png
)

def get_all_sample_names(base_path: Path, data_format: str) -> list:
    """
    获取所有样本名称
    
    参数:
        base_path: 数据根目录
        data_format: 数据格式 ('dataset' 或 'dataset_convert')
    
    返回:
        样本名称列表
    """
    if data_format == 'dataset_convert':
        plm_dir = base_path / 'plm'
        if not plm_dir.exists():
            return []
        
        plm_files = sorted(plm_dir.glob('*_0.tif'))
        sample_names = [f.stem.rsplit('_', 1)[0] for f in plm_files]
    else:
        mask_dir = base_path / 'mask'
        if not mask_dir.exists():
            return []
        
        mask_files = sorted(mask_dir.glob('*.png'))
        sample_names = [f.stem for f in mask_files]
    
    return sample_names

def generate_label_analysis_batch(
    base_path: str = 'data/rows',
    output_path: str = 'checkpoints/emit/label_analysis.json',
    data_format: str = None,
    skip_existing: bool = True
):
    """
    批量生成所有样本的 label_analysis.json
    
    参数:
        base_path: 数据根目录
        output_path: 输出 JSON 文件路径
        data_format: 数据格式（自动检测）
        skip_existing: 如果输出文件已存在，是否跳过已有样本
    """
    base_path = Path(base_path)
    output_path = Path(output_path)
    
    if data_format is None:
        rgb_dir = base_path / 'rgb'
        if rgb_dir.exists():
            tif_files = list(rgb_dir.glob('*.tif'))
            if len(tif_files) > 0:
                data_format = 'dataset_convert'
            else:
                data_format = 'dataset'
        else:
            data_format = 'dataset_convert'
    
    print(f"数据格式: {data_format}")
    print(f"数据路径: {base_path}")
    
    print("\n正在获取所有样本名称...")
    all_sample_names = get_all_sample_names(base_path, data_format)
    print(f"找到 {len(all_sample_names)} 个样本")
    
    if len(all_sample_names) == 0:
        print("错误: 未找到任何样本")
        return
    
    existing_analyses = {}
    if skip_existing and output_path.exists():
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
                for item in existing_data:
                    sample_name = item.get('sample_name', '')
                    if sample_name:
                        existing_analyses[sample_name] = item
            print(f"已加载 {len(existing_analyses)} 个已有分析结果")
        except Exception as e:
            print(f"警告: 无法加载已有分析结果: {e}")
            existing_analyses = {}
    
    import scripts.analyze_label_quality as analyze_module
    try:
        from scipy import ndimage
        HAS_SCIPY = True
    except ImportError:
        HAS_SCIPY = False
        print("警告: scipy未安装，将无法进行连通域分析")
    
    analyze_module.HAS_SCIPY = HAS_SCIPY
    
    all_analyses = []
    skipped_count = 0
    error_count = 0
    
    print(f"\n开始分析样本...")
    for idx, sample_name in enumerate(tqdm(all_sample_names, desc="分析进度")):
        if skip_existing and sample_name in existing_analyses:
            all_analyses.append(existing_analyses[sample_name])
            skipped_count += 1
            continue
        
        if data_format == 'dataset_convert':
            label_path = base_path / 'plm' / f"{sample_name}_0.tif"
        else:
            label_path = base_path / 'mask' / f"{sample_name}.png"
        
        if not label_path.exists():
            error_count += 1
            if (idx + 1) % 100 == 0:
                print(f"  警告: 样本 {sample_name} 的标签文件不存在")
            continue
        
        try:
            if data_format == 'dataset_convert':
                label = load_label_mask_tif(label_path)
            else:
                label = load_label_mask_png(label_path)
            
            analysis = analyze_label_quality(label, sample_name)
            all_analyses.append(analysis)
            
        except Exception as e:
            error_count += 1
            if (idx + 1) % 100 == 0:
                print(f"\n错误: 处理样本 {sample_name} 时出错: {e}")
            continue
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    serializable_analyses = []
    for analysis in all_analyses:
        serializable = analysis.copy()
        if 'component_sizes' in serializable:
            component_sizes = serializable['component_sizes']
            if isinstance(component_sizes, list) and len(component_sizes) > 10:
                serializable['component_sizes'] = sorted(component_sizes, reverse=True)[:10]
        serializable_analyses.append(serializable)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_analyses, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*70}")
    print("分析完成")
    print(f"{'='*70}")
    print(f"总样本数: {len(all_sample_names)}")
    print(f"成功分析: {len(all_analyses)}")
    print(f"跳过已有: {skipped_count}")
    print(f"错误/缺失: {error_count}")
    print(f"\n分析结果已保存到: {output_path}")
    
    if all_analyses:
        plume_ratios = [a.get('plume_ratio_percent', 0.0) for a in all_analyses if a.get('has_plume', False)]
        if plume_ratios:
            print(f"\n羽流占比统计:")
            print(f"  有羽流样本数: {len(plume_ratios)}")
            print(f"  最小占比: {min(plume_ratios):.2f}%")
            print(f"  最大占比: {max(plume_ratios):.2f}%")
            print(f"  平均占比: {sum(plume_ratios)/len(plume_ratios):.2f}%")
            
            range_samples = [r for r in plume_ratios if 30.0 <= r <= 40.0]
            print(f"  30%~40%范围样本数: {len(range_samples)}")
            
            if range_samples:
                print(f"  30%~40%范围样本占比: {len(range_samples)/len(plume_ratios)*100:.2f}%")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='批量生成所有样本的 label_analysis.json')
    parser.add_argument('--data_path', type=str, default='data/rows',
                       help='数据根目录')
    parser.add_argument('--output', type=str, default='checkpoints/emit/label_analysis.json',
                       help='输出分析报告文件路径（JSON格式）')
    parser.add_argument('--data_format', type=str, choices=['dataset', 'dataset_convert'], default=None,
                       help='数据格式（自动检测）')
    parser.add_argument('--no-skip-existing', action='store_true',
                       help='不跳过已有样本，重新分析所有样本')
    
    args = parser.parse_args()
    
    generate_label_analysis_batch(
        base_path=args.data_path,
        output_path=args.output,
        data_format=args.data_format,
        skip_existing=not args.no_skip_existing
    )



