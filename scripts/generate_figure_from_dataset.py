# -*- coding: utf-8 -*-
"""
从data/dataset格式生成对比图脚本
生成类似result.png格式的5行×4列网格图，展示：
- 第1列：Methane Enhancement Data（甲烷增强数据）
- 第2列：RGB Image（RGB图像）
- 第3列：Label（标签）
- 第4列：DIT-Mamba预测结果
"""
import sys
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from PIL import Image
from tqdm import tqdm
try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False
    print("警告: rasterio未安装，将无法读取.tif文件")
try:
    from scipy import ndimage
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("警告: scipy未安装，将使用numpy实现轮廓检测")


project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.dit_mamba import DiTMamba, DiT
from models.diffusion_refiner import DiffusionRefiner
from training.config import TrainingConfig
from typing import Optional


def detect_data_format(base_path: Path):
    """
    检测数据格式：dataset格式（.jpg/.png）或dataset_convert格式（.tif）
    
    返回:
        'dataset': 使用.jpg和.png格式
        'dataset_convert': 使用.tif格式
    """
    rgb_dir = base_path / 'rgb'
    if not rgb_dir.exists():

        return 'dataset'


    tif_files = list(rgb_dir.glob('*.tif'))
    if len(tif_files) > 0:
        return 'dataset_convert'


    jpg_files = list(rgb_dir.glob('*.jpg'))
    if len(jpg_files) > 0:
        return 'dataset'


    return 'dataset'


def load_raw_enh_png(enh_path: Path) -> np.ndarray:
    """从PNG文件加载原始ENH数据（未归一化）"""
    if not enh_path.exists():
        raise FileNotFoundError(f"ENH文件不存在: {enh_path}")

    img = Image.open(enh_path)
    if img.mode != 'L':
        img = img.convert('L')

    enh = np.array(img, dtype=np.float32)

    return enh


def load_raw_enh_tif(enh_path: Path) -> np.ndarray:
    """从TIF文件加载原始ENH数据（未归一化）"""
    if not HAS_RASTERIO:
        raise ImportError("需要rasterio库来读取.tif文件")
    if not enh_path.exists():
        raise FileNotFoundError(f"ENH文件不存在: {enh_path}")

    with rasterio.open(enh_path) as src:
        enh = src.read(1)


    nodata_value = -9999.0
    enh[enh == nodata_value] = 0

    return enh.astype(np.float32)


def load_rgb_image_jpg(rgb_path: Path) -> np.ndarray:
    """从JPG文件加载RGB图像"""
    if not rgb_path.exists():
        raise FileNotFoundError(f"RGB文件不存在: {rgb_path}")

    img = Image.open(rgb_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')

    rgb = np.array(img, dtype=np.uint8)

    return rgb


def load_rgb_image_tif(rgb_path: Path) -> np.ndarray:
    """从TIF文件加载RGB图像"""
    if not HAS_RASTERIO:
        raise ImportError("需要rasterio库来读取.tif文件")
    if not rgb_path.exists():
        raise FileNotFoundError(f"RGB文件不存在: {rgb_path}")

    with rasterio.open(rgb_path) as src:
        rgb = src.read()
        rgb = np.transpose(rgb, (1, 2, 0))


        if rgb.shape[2] > 3:
            rgb = rgb[:, :, :3]
        elif rgb.shape[2] == 1:
            rgb = np.repeat(rgb, 3, axis=2)

        rgb = rgb.astype(np.uint8)

    return rgb


def load_label_mask_png(mask_path: Path) -> np.ndarray:
    """从PNG文件加载标签掩码"""
    if not mask_path.exists():
        raise FileNotFoundError(f"掩码文件不存在: {mask_path}")

    img = Image.open(mask_path)
    if img.mode != 'L':
        img = img.convert('L')

    mask = np.array(img, dtype=np.float32)


    mask_binary = (mask > 0).astype(np.uint8)

    return mask_binary


def load_label_mask_tif(plm_path: Path) -> np.ndarray:
    """从TIF文件加载标签掩码（PLM）"""
    if not HAS_RASTERIO:
        raise ImportError("需要rasterio库来读取.tif文件")
    if not plm_path.exists():
        raise FileNotFoundError(f"掩码文件不存在: {plm_path}")

    with rasterio.open(plm_path) as src:
        mask = src.read(1)


    mask_binary = (mask > 0).astype(np.uint8)

    return mask_binary


def check_sample_quality_fast(sample_name: str, base_path: Path, data_format: str) -> dict:
    """
    快速检查样本质量（只检查标签，不加载ENH文件，大幅提升性能）
    
    参数:
        sample_name: 样本名称
        base_path: 数据根目录
        data_format: 数据格式 ('dataset' 或 'dataset_convert')
    
    返回:
        {
            'is_valid': bool,
            'has_plume': bool,
            'plume_size': int,
            'data_completeness': float,
            'plume_ratio': float
        }
    """
    try:
        if data_format == 'dataset_convert':
            enh_path = base_path / 'enh' / f"{sample_name}_0.tif"
            rgb_path = base_path / 'rgb' / f"{sample_name}_0.tif"
            mask_path = base_path / 'plm' / f"{sample_name}_0.tif"
        else:
            enh_path = base_path / 'enh' / f"{sample_name}.png"
            rgb_path = base_path / 'rgb' / f"{sample_name}.jpg"
            mask_path = base_path / 'mask' / f"{sample_name}.png"

        if not all([enh_path.exists(), rgb_path.exists(), mask_path.exists()]):
            return {'is_valid': False}


        if data_format == 'dataset_convert':
            label = load_label_mask_tif(mask_path)
        else:
            label = load_label_mask_png(mask_path)


        has_plume = np.any(label > 0)
        plume_size = int((label > 0).sum())
        total_pixels = label.size
        plume_ratio = plume_size / total_pixels if total_pixels > 0 else 0.0


        data_completeness = 1.0


        is_valid = True

        return {
            'is_valid': is_valid,
            'has_plume': has_plume,
            'plume_size': plume_size,
            'data_completeness': data_completeness,
            'plume_ratio': plume_ratio
        }
    except Exception as e:
        return {'is_valid': False}


def check_sample_quality(sample_name: str, base_path: Path, data_format: str) -> dict:
    """
    完整检查样本质量（包含数据完整性检查，较慢）
    
    参数:
        sample_name: 样本名称
        base_path: 数据根目录
        data_format: 数据格式 ('dataset' 或 'dataset_convert')
    
    返回:
        {
            'is_valid': bool,
            'has_plume': bool,
            'plume_size': int,
            'data_completeness': float,
            'plume_ratio': float
        }
    """
    try:
        if data_format == 'dataset_convert':
            enh_path = base_path / 'enh' / f"{sample_name}_0.tif"
            rgb_path = base_path / 'rgb' / f"{sample_name}_0.tif"
            mask_path = base_path / 'plm' / f"{sample_name}_0.tif"
        else:
            enh_path = base_path / 'enh' / f"{sample_name}.png"
            rgb_path = base_path / 'rgb' / f"{sample_name}.jpg"
            mask_path = base_path / 'mask' / f"{sample_name}.png"

        if not all([enh_path.exists(), rgb_path.exists(), mask_path.exists()]):
            return {'is_valid': False}


        if data_format == 'dataset_convert':
            label = load_label_mask_tif(mask_path)
        else:
            label = load_label_mask_png(mask_path)


        if data_format == 'dataset_convert':
            enh_raw = load_raw_enh_tif(enh_path)


            valid_mask = (enh_raw > 0) & (enh_raw != -9999.0)
            data_completeness = float(valid_mask.sum() / valid_mask.size) if valid_mask.size > 0 else 0.0
        else:

            enh_raw = load_raw_enh_png(enh_path)
            valid_mask = (enh_raw > 0)
            data_completeness = float(valid_mask.sum() / valid_mask.size) if valid_mask.size > 0 else 0.0


        has_plume = np.any(label > 0)
        plume_size = int((label > 0).sum())
        total_pixels = label.size
        plume_ratio = plume_size / total_pixels if total_pixels > 0 else 0.0


        is_valid = data_completeness >= 0.8

        return {
            'is_valid': is_valid,
            'has_plume': has_plume,
            'plume_size': plume_size,
            'data_completeness': data_completeness,
            'plume_ratio': plume_ratio
        }
    except Exception as e:
        return {'is_valid': False}


def filter_samples_by_plume_ratio(valid_samples: list, label_analysis_path: str, 
                                  min_ratio: float, max_ratio: float) -> list:
    """
    根据 label_analysis.json 中的羽流占比筛选样本
    
    参数:
        valid_samples: 有效样本名称列表
        label_analysis_path: label_analysis.json 文件路径
        min_ratio: 最小羽流占比（百分比）
        max_ratio: 最大羽流占比（百分比）
    
    返回:
        筛选后的样本名称列表
    """
    import json
    
    label_analysis_file = Path(label_analysis_path)
    if not label_analysis_file.exists():
        print(f"警告: label_analysis.json 文件不存在: {label_analysis_path}")
        return valid_samples
    
    with open(label_analysis_file, 'r', encoding='utf-8') as f:
        analysis_data = json.load(f)
    

    sample_ratio_map = {}
    for item in analysis_data:
        sample_name = item.get('sample_name', '')
        plume_ratio_percent = item.get('plume_ratio_percent', 0.0)
        sample_ratio_map[sample_name] = plume_ratio_percent
    

    filtered_samples = []
    for sample_name in valid_samples:


        base_name = sample_name.rsplit('_', 1)[0] if '_' in sample_name else sample_name
        
        if base_name in sample_ratio_map:
            ratio = sample_ratio_map[base_name]
            if min_ratio <= ratio <= max_ratio:
                filtered_samples.append(sample_name)
        elif sample_name in sample_ratio_map:
            ratio = sample_ratio_map[sample_name]
            if min_ratio <= ratio <= max_ratio:
                filtered_samples.append(sample_name)
    
    print(f"\n根据羽流占比筛选样本 ({min_ratio}%~{max_ratio}%):")
    print(f"  筛选前样本数: {len(valid_samples)}")
    print(f"  筛选后样本数: {len(filtered_samples)}")
    
    if len(filtered_samples) == 0:
        print(f"  警告: 没有找到羽流占比在 {min_ratio}%~{max_ratio}% 之间的样本")
        print(f"  将使用所有有效样本")
        return valid_samples
    
    return filtered_samples


def load_sample_info_from_json(label_analysis_path: str) -> dict:
    """
    从 label_analysis.json 加载样本信息
    
    参数:
        label_analysis_path: label_analysis.json 文件路径
    
    返回:
        sample_info_dict: 字典，格式为 {sample_name: quality_dict}
        quality_dict 包含: has_plume, plume_size, plume_ratio, plume_ratio_percent 等
    """
    import json
    
    label_analysis_file = Path(label_analysis_path)
    if not label_analysis_file.exists():
        return {}
    
    with open(label_analysis_file, 'r', encoding='utf-8') as f:
        analysis_data = json.load(f)
    
    sample_info_dict = {}
    for item in analysis_data:
        sample_name = item.get('sample_name', '')
        if not sample_name:
            continue
        

        sample_info_dict[sample_name] = {
            'is_valid': True,
            'has_plume': item.get('has_plume', False),
            'plume_size': item.get('plume_pixels', 0),
            'plume_ratio': item.get('plume_ratio', 0.0),
            'plume_ratio_percent': item.get('plume_ratio_percent', 0.0),
            'data_completeness': 1.0,
            'total_pixels': item.get('total_pixels', 0),
            'num_components': item.get('num_components', 0)
        }
    
    return sample_info_dict


def select_representative_samples(valid_samples: list, base_path: Path,
                                  data_format: str, num_samples: int = 6,
                                  min_data_completeness: float = 0.8,
                                  pos_ratio: float = 0.8,
                                  group_id: int = 1,
                                  confidence_scores: dict = None,
                                  min_confidence: float = 0.0,
                                  label_analysis_path: str = None) -> list:
    """
    选择有代表性的样本（优化版本：支持从JSON文件加载样本信息）
    
    策略：
    1. 优先选择有标签的正样本（至少占pos_ratio比例）
    2. 可以包含少量负样本（无标签）用于展示模型在负样本上的表现
    3. 正样本按羽流大小分类：小、中、大
    4. 使用快速检查（只检查标签，不检查数据完整性）或从JSON加载
    5. 根据group_id偏移选择位置，确保每组选择不同的样本
    6. 如果提供了置信度分数，优先选择高置信度样本
    7. 最终排序：Case 1-5按羽流占比和羽流像素数升序，Case 6选择最大的
    
    参数:
        valid_samples: 有效样本名称列表
        base_path: 数据根目录
        data_format: 数据格式
        num_samples: 需要选择的样本数量
        min_data_completeness: 最小数据完整性阈值（快速模式下不使用）
        pos_ratio: 正样本占比（0-1之间）
        group_id: 组号（从1开始），用于偏移选择位置，确保每组选择不同的样本
        confidence_scores: 可选的置信度分数字典，格式为 {sample_name: confidence_score}
        min_confidence: 最小置信度阈值，如果提供了置信度分数，只选择置信度>=此值的样本
        label_analysis_path: 可选的 label_analysis.json 文件路径，如果提供则从JSON加载样本信息
    
    返回:
        选择的样本名称列表（已排序：Case 1-5升序，Case 6最大）
    """


    sample_quality = {}
    if label_analysis_path:
        print(f"  从JSON文件加载样本信息: {label_analysis_path}")
        json_sample_info = load_sample_info_from_json(label_analysis_path)
        

        for sample_name in valid_samples:

            base_name = sample_name.rsplit('_', 1)[0] if '_' in sample_name and sample_name.rsplit('_', 1)[-1].isdigit() else sample_name
            
            if base_name in json_sample_info:
                sample_quality[sample_name] = json_sample_info[base_name]
            elif sample_name in json_sample_info:
                sample_quality[sample_name] = json_sample_info[sample_name]
        
        print(f"  从JSON加载了 {len(sample_quality)} 个样本的信息")
    

    total = len(valid_samples)
    missing_samples = [s for s in valid_samples if s not in sample_quality]
    
    if missing_samples:
        print(f"  对 {len(missing_samples)} 个样本进行快速检查（JSON中未找到）...")
        for idx, sample_name in enumerate(missing_samples):
            quality = check_sample_quality_fast(sample_name, base_path, data_format)
            if quality.get('is_valid', False):
                sample_quality[sample_name] = quality
            
            if (idx + 1) % 1000 == 0:
                print(f"    已处理 {idx + 1}/{len(missing_samples)} 个样本... (有效: {len(sample_quality)})")
    elif not label_analysis_path:

        print(f"  正在评估 {total} 个样本的质量（快速模式）...")
        for idx, sample_name in enumerate(valid_samples):
            quality = check_sample_quality_fast(sample_name, base_path, data_format)
            if quality.get('is_valid', False):
                sample_quality[sample_name] = quality

            if (idx + 1) % 1000 == 0:
                print(f"    已处理 {idx + 1}/{total} 个样本... (有效: {len(sample_quality)})")

    print(f"  有效样本数: {len(sample_quality)}")

    if len(sample_quality) == 0:
        return []


    pos_samples = []
    neg_samples = []

    for sample_name, quality in sample_quality.items():

        if confidence_scores is not None:
            conf_score = confidence_scores.get(sample_name, 0.0)
            if conf_score < min_confidence:
                continue

        if quality['has_plume']:
            pos_samples.append((sample_name, quality))
        else:
            neg_samples.append((sample_name, quality))

    print(f"  正样本数: {len(pos_samples)}, 负样本数: {len(neg_samples)}")


    if confidence_scores is not None:

        pos_samples.sort(key=lambda x: confidence_scores.get(x[0], 0.0), reverse=True)
        print(f"  使用置信度排序（阈值: {min_confidence:.4f}）")
    else:

        pos_samples.sort(key=lambda x: x[1]['plume_size'], reverse=True)


    selected = []
    num_pos_needed = max(1, int(num_samples * pos_ratio))



    offset = (group_id - 1) * num_samples


    if len(pos_samples) >= num_pos_needed:
        if confidence_scores is not None:


            start_idx = min(offset, len(pos_samples) - num_pos_needed) if offset < len(pos_samples) else 0
            end_idx = start_idx + num_pos_needed
            if end_idx > len(pos_samples):

                start_idx = max(0, len(pos_samples) - num_pos_needed)
                end_idx = len(pos_samples)
            selected = [pos_samples[i][0] for i in range(start_idx, end_idx)]
        else:


            if len(pos_samples) >= 3:
                large_samples = pos_samples[:len(pos_samples)//3]
                medium_samples = pos_samples[len(pos_samples)//3:2*len(pos_samples)//3]
                small_samples = pos_samples[2*len(pos_samples)//3:]


                if large_samples:

                    idx = min(offset, len(large_samples) - 1) if offset < len(large_samples) else 0
                    selected.append(large_samples[idx][0])
                if medium_samples and len(selected) < num_pos_needed:

                    idx = min(offset, len(medium_samples) - 1) if offset < len(medium_samples) else 0
                    selected.append(medium_samples[idx][0])
                if small_samples and len(selected) < num_pos_needed:

                    idx = min(offset, len(small_samples) - 1) if offset < len(small_samples) else 0
                    selected.append(small_samples[idx][0])


            remaining_pos = [s[0] for s in pos_samples if s[0] not in selected]
            if remaining_pos and len(selected) < num_pos_needed:
                needed = num_pos_needed - len(selected)

                start_idx = min(offset, len(remaining_pos) - needed) if offset < len(remaining_pos) else 0
                for i in range(min(needed, len(remaining_pos))):
                    idx = start_idx + i
                    if idx >= len(remaining_pos):
                        break
                    if remaining_pos[idx] not in selected:
                        selected.append(remaining_pos[idx])
                        if len(selected) >= num_pos_needed:
                            break
    elif len(pos_samples) > 0:


        start_idx = min(offset, len(pos_samples) - 1) if offset < len(pos_samples) else 0
        for i in range(min(len(pos_samples), num_samples - 1)):
            idx = start_idx + i
            if idx >= len(pos_samples):
                break
            selected.append(pos_samples[idx][0])



    num_needed = num_samples - len(selected)
    if num_needed > 0:

        remaining_pos = [s[0] for s in pos_samples if s[0] not in selected]
        if remaining_pos:

            start_idx = min(offset, len(remaining_pos) - num_needed) if offset < len(remaining_pos) else 0
            for i in range(min(num_needed, len(remaining_pos))):
                idx = start_idx + i
                if idx >= len(remaining_pos):
                    break
                if remaining_pos[idx] not in selected:
                    selected.append(remaining_pos[idx])
                    if len(selected) >= num_samples:
                        break
        else:

            print(f"  警告: 只有{len(selected)}个正样本可用，少于需要的{num_samples}个")
            print(f"  将只使用这{len(selected)}个正样本（不添加负样本）")



    if len(selected) < num_samples:
        print(f"  警告: 只能选择{len(selected)}个正样本，少于需要的{num_samples}个")
        print(f"  将只使用这{len(selected)}个正样本（完全排除负样本）")

    selected = selected[:num_samples]
    



    if len(selected) == 6:

        selected_with_quality = []
        for sample_name in selected:
            quality = sample_quality.get(sample_name)
            if quality and quality.get('has_plume', False):
                selected_with_quality.append({
                    'sample_name': sample_name,
                    'plume_size': quality.get('plume_size', 0),
                    'plume_ratio': quality.get('plume_ratio', 0.0)
                })
            else:

                selected_with_quality.append({
                    'sample_name': sample_name,
                    'plume_size': 0,
                    'plume_ratio': 0.0
                })
        

        selected_with_quality.sort(key=lambda x: (x['plume_ratio'], x['plume_size']))
        


        case1_5_with_quality = selected_with_quality[:5]
        case1_5 = [s['sample_name'] for s in case1_5_with_quality]
        

        print(f"  Case 1-5 样本范围:")
        for idx, s in enumerate(case1_5_with_quality, 1):
            print(f"    Case {idx}: {s['sample_name']} - 羽流占比: {s['plume_ratio']*100:.3f}%, 羽流像素数: {s['plume_size']}")
        

        max_ratio_in_case1_5 = max([s['plume_ratio'] for s in case1_5_with_quality])
        max_size_in_case1_5 = max([s['plume_size'] for s in case1_5_with_quality])
        min_ratio_in_case1_5 = min([s['plume_ratio'] for s in case1_5_with_quality])
        

        ratio_range_case1_5 = max_ratio_in_case1_5 - min_ratio_in_case1_5
        




        min_ratio_for_case6 = max(max_ratio_in_case1_5 * 2.5, 0.02)
        min_size_for_case6 = max_size_in_case1_5 * 2.0
        
        print(f"  Case 6 选择标准:")
        print(f"    最小羽流占比: {min_ratio_for_case6*100:.2f}% (Case 1-5最大值的 {min_ratio_for_case6/max_ratio_in_case1_5:.2f} 倍)")
        print(f"    最小羽流像素数: {min_size_for_case6:.0f} (Case 1-5最大值的 {min_size_for_case6/max_size_in_case1_5:.2f} 倍)")
        
        case6_candidates_strong = []
        case6_candidates_weak = []
        
        for sample_name, quality in pos_samples:
            if sample_name in case1_5:
                continue
            plume_ratio = quality.get('plume_ratio', 0.0)
            plume_size = quality.get('plume_size', 0)
            

            if plume_ratio > max_ratio_in_case1_5 and plume_size > max_size_in_case1_5:

                ratio_gap = plume_ratio - max_ratio_in_case1_5
                ratio_multiple = plume_ratio / max_ratio_in_case1_5 if max_ratio_in_case1_5 > 0 else float('inf')
                
                candidate_info = (sample_name, quality, plume_ratio, plume_size, ratio_gap, ratio_multiple)
                

                if plume_ratio >= min_ratio_for_case6 and plume_size >= min_size_for_case6:
                    case6_candidates_strong.append(candidate_info)
                else:

                    case6_candidates_weak.append(candidate_info)
        

        if case6_candidates_strong:

            case6_candidates_strong.sort(key=lambda x: (x[5], x[4]), reverse=True)
            case6_sample = case6_candidates_strong[0][0]
            case6_quality = case6_candidates_strong[0][1]
            case6_ratio = case6_candidates_strong[0][2]
            case6_size = case6_candidates_strong[0][3]
            ratio_multiple = case6_candidates_strong[0][5]
            print(f"  已选择Case 6样本（强对比）: {case6_sample}")
            print(f"    羽流占比: {case6_ratio*100:.4f}% (是Case 1-5最大值的 {ratio_multiple:.2f} 倍)")
            print(f"    羽流像素数: {case6_size} (是Case 1-5最大值的 {case6_size/max_size_in_case1_5:.2f} 倍)")
        elif case6_candidates_weak:

            case6_candidates_weak.sort(key=lambda x: (x[5], x[4]), reverse=True)
            case6_sample = case6_candidates_weak[0][0]
            case6_quality = case6_candidates_weak[0][1]
            case6_ratio = case6_candidates_weak[0][2]
            case6_size = case6_candidates_weak[0][3]
            ratio_multiple = case6_candidates_weak[0][5]
            print(f"  已选择Case 6样本（弱对比）: {case6_sample}")
            print(f"    羽流占比: {case6_ratio*100:.4f}% (是Case 1-5最大值的 {ratio_multiple:.2f} 倍)")
            print(f"    羽流像素数: {case6_size} (是Case 1-5最大值的 {case6_size/max_size_in_case1_5:.2f} 倍)")
        else:

            all_remaining = [(s[0], s[1]) for s in pos_samples if s[0] not in case1_5]
            if all_remaining:
                all_remaining.sort(key=lambda x: (x[1].get('plume_ratio', 0.0), x[1].get('plume_size', 0)), reverse=True)
                case6_sample = all_remaining[0][0]
                case6_quality = all_remaining[0][1]
                case6_ratio = case6_quality.get('plume_ratio', 0.0)
                case6_size = case6_quality.get('plume_size', 0)
                ratio_multiple = case6_ratio / max_ratio_in_case1_5 if max_ratio_in_case1_5 > 0 else 0
                print(f"  警告: 未找到比前5个都大的样本，选择剩余样本中最大的: {case6_sample}")
                print(f"    羽流占比: {case6_ratio*100:.4f}% (是Case 1-5最大值的 {ratio_multiple:.2f} 倍)")
                print(f"    羽流像素数: {case6_size}")
            else:

                case6_sample = selected_with_quality[-1]['sample_name']
                case6_quality = sample_quality.get(case6_sample, {})
                case6_ratio = case6_quality.get('plume_ratio', 0.0)
                case6_size = case6_quality.get('plume_size', 0)
                ratio_multiple = case6_ratio / max_ratio_in_case1_5 if max_ratio_in_case1_5 > 0 else 0
                print(f"  警告: 所有样本都在前5个中，使用已选样本中最大的作为Case 6: {case6_sample}")
                print(f"    羽流占比: {case6_ratio*100:.4f}% (是Case 1-5最大值的 {ratio_multiple:.2f} 倍)")
                print(f"    羽流像素数: {case6_size}")
        

        selected = case1_5 + [case6_sample]
        
        print(f"  最终排序结果:")
        for idx, sample_name in enumerate(selected, 1):
            quality = sample_quality.get(sample_name, {})
            plume_ratio = quality.get('plume_ratio', 0.0) * 100
            plume_size = quality.get('plume_size', 0)
            print(f"    Case {idx}: {sample_name} (羽流占比: {plume_ratio:.4f}%, 羽流像素数: {plume_size})")
    
    return selected


def calculate_iou_numpy(pred: np.ndarray, target: np.ndarray) -> float:
    """
    计算IoU（Intersection over Union）
    
    参数:
        pred: 预测掩码，形状为 (H, W)，值为0或1
        target: 目标掩码，形状为 (H, W)，值为0或1
    
    返回:
        IoU值（0-1之间）
    """
    pred_binary = (pred > 0.5).astype(np.uint8)
    target_binary = (target > 0).astype(np.uint8)

    intersection = np.logical_and(pred_binary, target_binary).sum()
    union = np.logical_or(pred_binary, target_binary).sum()

    if union == 0:
        return 1.0 if intersection == 0 else 0.0

    return float(intersection) / float(union)


def load_model(checkpoint_path: str, device: torch.device, config_path: str = None, model_type: str = 'ditmamba'):
    """加载模型
    
    参数:
        checkpoint_path: 检查点文件路径
        device: 设备
        config_path: 可选的配置文件路径（当checkpoint没有config时使用）
        model_type: 模型类型，'dit' 或 'ditmamba'（默认: 'ditmamba'）
            - 'dit': 纯DiT模型（不包含Mamba模块）
            - 'ditmamba': DiT-Mamba模型（包含Mamba模块）
    
    返回:
        加载的模型实例（DiT 或 DiTMamba）
    
    异常:
        ValueError: 检查点文件格式错误
        RuntimeError: 模型类型不匹配或加载失败
    """
    print(f"加载模型: {checkpoint_path} (类型: {model_type})")

    checkpoint = torch.load(checkpoint_path, map_location=device)


    if 'model_state_dict' not in checkpoint:
        raise ValueError(f"检查点文件格式错误: 缺少 'model_state_dict' 键。文件: {checkpoint_path}")


    has_mamba_keys = False
    if 'model_state_dict' in checkpoint:
        checkpoint_keys = set(checkpoint['model_state_dict'].keys())
        has_mamba_keys = any('mamba' in key.lower() for key in checkpoint_keys)

        if model_type == 'dit' and has_mamba_keys:
            print(f"警告: 检查点包含 Mamba 模块的键，但指定了 DiT 模型类型")
            print(f"  建议: 此检查点可能是 DiT-Mamba 模型，请使用 model_type='ditmamba'")
        elif model_type == 'ditmamba' and not has_mamba_keys:
            print(f"警告: 检查点不包含 Mamba 模块的键，但指定了 DiT-Mamba 模型类型")
            print(f"  建议: 此检查点可能是 DiT 模型，请使用 model_type='dit'")


    if 'config' in checkpoint and checkpoint['config'] is not None:
        model_config = checkpoint['config'].get('model', {})
        base_channels = model_config.get('base_channels', 64)
        num_encoder_layers = model_config.get('num_encoder_layers', 4)
        mamba_dim = model_config.get('mamba_dim', 256)
        dropout = checkpoint['config'].get('regularization', {}).get('dropout', 0.1)
        if model_type == 'ditmamba':
            print(f"从checkpoint读取模型配置: base_channels={base_channels}, layers={num_encoder_layers}, mamba_dim={mamba_dim}")
        else:
            print(f"从checkpoint读取模型配置: base_channels={base_channels}, layers={num_encoder_layers}")
    elif config_path and Path(config_path).exists():

        import yaml
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        model_config = config.get('model', {})
        base_channels = model_config.get('base_channels', 64)
        num_encoder_layers = model_config.get('num_encoder_layers', 4)
        mamba_dim = model_config.get('mamba_dim', 256)
        dropout = config.get('regularization', {}).get('dropout', 0.1)
        if model_type == 'ditmamba':
            print(f"从配置文件读取模型配置: base_channels={base_channels}, layers={num_encoder_layers}, mamba_dim={mamba_dim}")
        else:
            print(f"从配置文件读取模型配置: base_channels={base_channels}, layers={num_encoder_layers}")
    else:

        base_channels = 64
        num_encoder_layers = 4
        mamba_dim = 256
        dropout = 0.1
        print("警告: 使用默认模型参数，可能与checkpoint不匹配")
        print("建议: 使用config_path参数指定配置文件，或确保checkpoint包含config字段")


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


    try:
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    except RuntimeError as e:
        if "Missing key(s)" in str(e) or "Unexpected key(s)" in str(e):
            print(f"错误: 模型类型不匹配！")
            print(f"  指定的模型类型: {model_type}")

            if not has_mamba_keys and 'model_state_dict' in checkpoint:
                checkpoint_keys = set(checkpoint['model_state_dict'].keys())
                has_mamba_keys = any('mamba' in key.lower() for key in checkpoint_keys)
            if has_mamba_keys:
                print(f"  检查点实际类型: DiT-Mamba (包含 Mamba 模块)")
            else:
                print(f"  检查点实际类型: DiT (不包含 Mamba 模块)")
            print(f"  详细错误: {str(e)}")
            print(f"  建议: 请确保模型类型与检查点匹配")
            raise RuntimeError(f"模型类型不匹配: 无法加载 {model_type} 模型。检查点可能是 {'DiT-Mamba' if has_mamba_keys else 'DiT'} 模型") from e
        else:
            raise

    model.to(device)
    model.eval()

    print(f"模型加载成功 (epoch {checkpoint.get('epoch', 'unknown')})")
    print(f"最佳Val MIoU: {checkpoint.get('best_val_miou', 0.0):.4f}")

    return model


def predict_with_model(model: nn.Module, rgb: np.ndarray, enh_normalized: np.ndarray,
                       enh_raw: np.ndarray, device: torch.device,
                       threshold: float = 0.5, diffusion_model: Optional[nn.Module] = None,
                       diffusion_config: Optional[dict] = None):
    """
    使用模型进行预测
    
    参数:
        model: 训练好的模型
        rgb: RGB图像 (H, W, 3), uint8
        enh_normalized: 归一化后的ENH数据 (H, W), uint8（已归一化，可直接使用）
        enh_raw: 原始ENH数据 (H, W), float32（用于可视化）
        device: 计算设备
        threshold: 预测阈值，默认0.5
        diffusion_model: Diffusion模型（可选）
        diffusion_config: Diffusion配置（可选）
    
    返回:
        pred_enh: 预测区域的ENH值 (H, W), float32
        pred_mask: 预测掩码 (H, W), float32，0或1
        pred_prob: 预测概率 (H, W), float32，值域为[0, 1]
        prob_stats: 预测概率统计信息字典
    """

    rgb_tensor = torch.from_numpy(rgb).permute(2, 0, 1).float()
    enh_tensor = torch.from_numpy(enh_normalized).unsqueeze(0).float()
    input_tensor = torch.cat([rgb_tensor, enh_tensor], dim=0)
    input_tensor = input_tensor.unsqueeze(0).to(device)


    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        

        if diffusion_model is not None:
            diffusion_model.eval()
            inference_config = diffusion_config.get('diffusion_refiner', {}).get('inference', {}) if diffusion_config else {}
            num_steps = inference_config.get('num_steps', 20)
            

            condition = torch.cat([input_tensor, output], dim=1)
            

            refined_residuals = diffusion_model.sample(condition, num_steps=num_steps)
            

            output = output + refined_residuals
            output = torch.clamp(output, 0.0, 1.0)


        pred_prob = output
        pred_binary = (pred_prob > threshold).float()


    pred_mask = pred_binary[0, 0].cpu().numpy().astype(np.float32)
    pred_prob_np = pred_prob[0, 0].cpu().numpy().astype(np.float32)


    prob_max = float(pred_prob_np.max())
    prob_min = float(pred_prob_np.min())
    prob_mean = float(pred_prob_np.mean())
    prob_std = float(pred_prob_np.std())
    prob_median = float(np.median(pred_prob_np))
    pixels_above_threshold = int((pred_prob_np > threshold).sum())
    pixels_above_03 = int((pred_prob_np > 0.3).sum())
    pixels_above_02 = int((pred_prob_np > 0.2).sum())
    pixels_above_01 = int((pred_prob_np > 0.1).sum())
    total_pixels = int(pred_prob_np.size)

    prob_stats = {
        'max': prob_max,
        'min': prob_min,
        'mean': prob_mean,
        'std': prob_std,
        'median': prob_median,
        'pixels_above_threshold': pixels_above_threshold,
        'pixels_above_03': pixels_above_03,
        'pixels_above_02': pixels_above_02,
        'pixels_above_01': pixels_above_01,
        'total_pixels': total_pixels,
        'threshold': threshold
    }




    if enh_raw.shape != pred_mask.shape:

        enh_raw_resized = np.array(Image.fromarray(enh_raw.astype(np.float32)).resize(
            (pred_mask.shape[1], pred_mask.shape[0]), Image.LANCZOS))
        pred_enh = enh_raw_resized * pred_mask
    else:
        pred_enh = enh_raw * pred_mask

    return pred_enh.astype(np.float32), pred_mask, pred_prob_np, prob_stats


def evaluate_sample_confidence(
    model: nn.Module,
    sample_name: str,
    base_path: Path,
    device: torch.device,
    data_format: str,
    enh_x0: float = 81.15,
    enh_x1: float = 62.18
) -> dict:
    """
    评估样本的预测置信度（快速推理，只获取预测概率）
    
    参数:
        model: 训练好的模型
        sample_name: 样本名称
        base_path: 数据根目录
        device: 计算设备
        data_format: 数据格式 ('dataset' 或 'dataset_convert')
        enh_x0: ENH归一化参数x0
        enh_x1: ENH归一化参数x1
    
    返回:
        confidence_info: 包含置信度信息的字典，如果失败返回None
    """
    try:
        if data_format == 'dataset_convert':
            enh_path = base_path / 'enh' / f"{sample_name}_0.tif"
            rgb_path = base_path / 'rgb' / f"{sample_name}_0.tif"
            mask_path = base_path / 'plm' / f"{sample_name}_0.tif"
        else:
            enh_path = base_path / 'enh' / f"{sample_name}.png"
            rgb_path = base_path / 'rgb' / f"{sample_name}.jpg"
            mask_path = base_path / 'mask' / f"{sample_name}.png"

        if not all([enh_path.exists(), rgb_path.exists(), mask_path.exists()]):
            return None


        if data_format == 'dataset_convert':
            enh_raw = load_raw_enh_tif(enh_path)
            rgb = load_rgb_image_tif(rgb_path)
            label = load_label_mask_tif(mask_path)
        else:
            enh_raw = load_raw_enh_png(enh_path)
            rgb = load_rgb_image_jpg(rgb_path)
            label = load_label_mask_png(mask_path)


        if data_format == 'dataset_convert':
            enh_normalized = ((enh_raw - enh_x0) / enh_x1 * 255).clip(0, 255).astype(np.uint8)
        else:
            enh_raw_scaled = (enh_raw / 255.0) * 1000.0
            enh_normalized = ((enh_raw_scaled - enh_x0) / enh_x1 * 255).clip(0, 255).astype(np.uint8)


        rgb_tensor = torch.from_numpy(rgb).permute(2, 0, 1).float()
        enh_tensor = torch.from_numpy(enh_normalized).unsqueeze(0).float()
        input_tensor = torch.cat([rgb_tensor, enh_tensor], dim=0).unsqueeze(0).to(device)

        model.eval()
        with torch.no_grad():
            output = model(input_tensor)

            pred_prob = output
            pred_prob_np = pred_prob[0, 0].cpu().numpy().astype(np.float32)


        prob_max = float(pred_prob_np.max())
        prob_mean = float(pred_prob_np.mean())


        label_pixels = int((label > 0).sum())
        if label_pixels > 0:
            label_region_probs = pred_prob_np[label > 0]
            max_prob_in_label = float(label_region_probs.max()) if len(label_region_probs) > 0 else 0.0
            mean_prob_in_label = float(label_region_probs.mean()) if len(label_region_probs) > 0 else 0.0
        else:
            max_prob_in_label = 0.0
            mean_prob_in_label = 0.0


        if label_pixels > 0:
            confidence_score = max_prob_in_label
        else:
            confidence_score = prob_max

        return {
            'sample_name': sample_name,
            'prob_max': prob_max,
            'prob_mean': prob_mean,
            'max_prob_in_label': max_prob_in_label,
            'mean_prob_in_label': mean_prob_in_label,
            'has_plume': label_pixels > 0,
            'label_pixels': label_pixels,
            'confidence_score': confidence_score
        }
    except Exception as e:
        return None


def evaluate_sample_iou(
    dit_model: nn.Module,
    ditmamba_model: nn.Module,
    sample_name: str,
    base_path: Path,
    device: torch.device,
    data_format: str,
    threshold: float = 0.5,
    enh_x0: float = 81.15,
    enh_x1: float = 62.18
) -> dict:
    """
    评估样本的实际IoU（两个模型都评估）
    
    参数:
        dit_model: DiT模型
        ditmamba_model: DiT-Mamba模型
        sample_name: 样本名称
        base_path: 数据根目录
        device: 计算设备
        data_format: 数据格式 ('dataset' 或 'dataset_convert')
        threshold: 预测阈值（用于二值化）
        enh_x0: ENH归一化参数x0
        enh_x1: ENH归一化参数x1
    
    返回:
        iou_info: 包含IoU信息的字典，如果失败返回None
    """
    try:
        if data_format == 'dataset_convert':
            enh_path = base_path / 'enh' / f"{sample_name}_0.tif"
            rgb_path = base_path / 'rgb' / f"{sample_name}_0.tif"
            mask_path = base_path / 'plm' / f"{sample_name}_0.tif"
        else:
            enh_path = base_path / 'enh' / f"{sample_name}.png"
            rgb_path = base_path / 'rgb' / f"{sample_name}.jpg"
            mask_path = base_path / 'mask' / f"{sample_name}.png"

        if not all([enh_path.exists(), rgb_path.exists(), mask_path.exists()]):
            return None


        if data_format == 'dataset_convert':
            enh_raw = load_raw_enh_tif(enh_path)
            rgb = load_rgb_image_tif(rgb_path)
            label = load_label_mask_tif(mask_path)
        else:
            enh_raw = load_raw_enh_png(enh_path)
            rgb = load_rgb_image_jpg(rgb_path)
            label = load_label_mask_png(mask_path)


        if data_format == 'dataset_convert':
            enh_normalized = ((enh_raw - enh_x0) / enh_x1 * 255).clip(0, 255).astype(np.uint8)
        else:
            enh_raw_scaled = (enh_raw / 255.0) * 1000.0
            enh_normalized = ((enh_raw_scaled - enh_x0) / enh_x1 * 255).clip(0, 255).astype(np.uint8)


        rgb_tensor = torch.from_numpy(rgb).permute(2, 0, 1).float()
        enh_tensor = torch.from_numpy(enh_normalized).unsqueeze(0).float()
        input_tensor = torch.cat([rgb_tensor, enh_tensor], dim=0).unsqueeze(0).to(device)


        dit_model.eval()
        with torch.no_grad():
            output_dit = dit_model(input_tensor)
            pred_prob_dit = output_dit[0, 0].cpu().numpy().astype(np.float32)
            pred_mask_dit = (pred_prob_dit > threshold).astype(np.uint8)


        ditmamba_model.eval()
        with torch.no_grad():
            output_ditmamba = ditmamba_model(input_tensor)
            pred_prob_ditmamba = output_ditmamba[0, 0].cpu().numpy().astype(np.float32)
            pred_mask_ditmamba = (pred_prob_ditmamba > threshold).astype(np.uint8)


        if pred_mask_dit.shape != label.shape:
            pred_mask_dit = np.array(Image.fromarray(pred_mask_dit).resize(
                (label.shape[1], label.shape[0]), Image.NEAREST))
        if pred_mask_ditmamba.shape != label.shape:
            pred_mask_ditmamba = np.array(Image.fromarray(pred_mask_ditmamba).resize(
                (label.shape[1], label.shape[0]), Image.NEAREST))


        iou_dit = calculate_iou_numpy(pred_mask_dit, label)
        iou_ditmamba = calculate_iou_numpy(pred_mask_ditmamba, label)


        label_pixels = int((label > 0).sum())

        return {
            'sample_name': sample_name,
            'iou_dit': iou_dit,
            'iou_ditmamba': iou_ditmamba,
            'has_plume': label_pixels > 0,
            'label_pixels': label_pixels
        }
    except Exception as e:
        return None


def create_methane_colormap():
    """创建甲烷增强数据的颜色映射（0-1000 ppm-m）"""
    colors = ['#2E0854', '#000080', '#0000FF', '#00FFFF', '#00FF00', '#FFFF00', '#FFD700']
    n_bins = 256
    cmap = mcolors.LinearSegmentedColormap.from_list('methane', colors, N=n_bins)
    return cmap


def visualize_enh_data(ax, enh_data: np.ndarray, cmap, vmin=0, vmax=1000):
    """可视化ENH数据（使用统一的颜色映射范围）"""

    im = ax.imshow(enh_data, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.axis('off')
    return im


def visualize_rgb(ax, rgb: np.ndarray):
    """可视化RGB图像"""
    ax.imshow(rgb)
    ax.axis('off')


def visualize_label(ax, label: np.ndarray):
    """可视化标签（白色区域）"""
    label_rgb = np.zeros((*label.shape, 3), dtype=np.uint8)
    label_rgb[label > 0] = 255
    ax.imshow(label_rgb)
    ax.axis('off')


def visualize_prediction(ax, pred_enh: np.ndarray, pred_mask: np.ndarray, cmap,
                         enh_raw: np.ndarray = None, vmin: float = 0, vmax: float = 1000):
    """可视化预测结果（带颜色映射和白色轮廓）"""
    background = np.zeros_like(pred_enh)
    display_data = np.where(pred_mask > 0.5, pred_enh, background)

    im = ax.imshow(display_data, cmap=cmap, vmin=vmin, vmax=vmax)


    if HAS_SCIPY:
        eroded = ndimage.binary_erosion(pred_mask.astype(bool))
        edges = pred_mask.astype(bool) & ~eroded
    else:
        pred_mask_bool = pred_mask.astype(bool)
        h, w = pred_mask_bool.shape
        edges = np.zeros_like(pred_mask_bool)

        if h > 2 and w > 2:
            center = pred_mask_bool[1:h-1, 1:w-1]
            up = pred_mask_bool[0:h-2, 1:w-1]
            down = pred_mask_bool[2:h, 1:w-1]
            left = pred_mask_bool[1:h-1, 0:w-2]
            right = pred_mask_bool[1:h-1, 2:w]

            edges[1:h-1, 1:w-1] = center & (~up | ~down | ~left | ~right)

            if h > 1:
                edges[0, :] = pred_mask_bool[0, :]
                edges[h-1, :] = pred_mask_bool[h-1, :]
            if w > 1:
                edges[:, 0] = pred_mask_bool[:, 0]
                edges[:, w-1] = pred_mask_bool[:, w-1]

    if np.any(edges):
        contour = np.zeros((*pred_mask.shape, 4), dtype=np.float32)
        contour[edges, :] = [1.0, 1.0, 1.0, 1.0]
        ax.imshow(contour)

    ax.axis('off')
    return im


def generate_comparison_figure(
    dit_model: nn.Module,
    ditmamba_model: nn.Module,
    sample_names: list,
    base_path: str = 'data/dataset',
    device: torch.device = None,
    output_path: str = 'docs/result/result_comparison.png',
    enh_x0: float = 81.15,
    enh_x1: float = 62.18,
    threshold: float = 0.5,
    data_format: str = None
):
    """
    生成对比图
    
    参数:
        dit_model: 训练好的DiT模型
        ditmamba_model: 训练好的DiT-Mamba模型
        sample_names: 样本名称列表（5个）
            - dataset格式: 数字编号字符串（如"00001"）
            - dataset_convert格式: 样本名称（如"sample_00001"）
        base_path: 数据根目录
        device: 计算设备
        output_path: 输出路径
        enh_x0: ENH归一化参数x0
        enh_x1: ENH归一化参数x1
        threshold: 预测阈值，默认0.5
        data_format: 数据格式 ('dataset' 或 'dataset_convert')，如果为None则自动检测
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    if dit_model is None or ditmamba_model is None:
        raise ValueError("两个模型都必须提供，不能为 None")


    if not isinstance(dit_model, DiT):
        raise TypeError(f"dit_model 必须是 DiT 类型，实际类型: {type(dit_model)}")
    if not isinstance(ditmamba_model, DiTMamba):
        raise TypeError(f"ditmamba_model 必须是 DiTMamba 类型，实际类型: {type(ditmamba_model)}")

    base_path = Path(base_path)


    if data_format is None:
        data_format = detect_data_format(base_path)
    print(f"检测到数据格式: {data_format}")

    num_cases = len(sample_names)


    if num_cases < 6:
        print(f"注意: 只有{num_cases}个正样本，将使用{num_cases}列布局（完全排除负样本）")

        num_cols = num_cases
        total_cols = 1 + num_cols + 1
        colorbar_col = total_cols - 1

        fig_width = 4 + num_cols * 4
        fig = plt.figure(figsize=(fig_width, 20))

        width_ratios = [0.3] + [1] * num_cols + [0.25]
        gs = fig.add_gridspec(5, total_cols, width_ratios=width_ratios,
                              height_ratios=[1]*5, hspace=0.3, wspace=0.15)
        col_titles = [f'Case {i+1}' for i in range(num_cases)]
    elif num_cases > 6:
        print(f"警告: 得到{num_cases}个样本，将使用前6个")
        sample_names = sample_names[:6]
        num_cases = 6
        total_cols = 8
        colorbar_col = 7
        fig = plt.figure(figsize=(28, 20))
        gs = fig.add_gridspec(5, 8, width_ratios=[0.3, 1, 1, 1, 1, 1, 1, 0.25],
                              height_ratios=[1]*5, hspace=0.3, wspace=0.15)
        col_titles = [f'Case {i+1}' for i in range(6)]
    else:

        num_cases = 6
        total_cols = 8
        colorbar_col = 7
        fig = plt.figure(figsize=(28, 20))
        gs = fig.add_gridspec(5, 8, width_ratios=[0.3, 1, 1, 1, 1, 1, 1, 0.25],
                              height_ratios=[1]*5, hspace=0.3, wspace=0.15)
        col_titles = [f'Case {i+1}' for i in range(6)]


    cmap = create_methane_colormap()


    row_titles = [
        'DATA',
        'RGB',
        'Label',
        'DIT',
        'DIT-Mamba'
    ]


    all_data = []
    all_enh_values = []
    print("\n加载数据和生成预测...")
    for case_idx, sample_name in enumerate(sample_names):
        try:

            if data_format == 'dataset_convert':

                enh_path = base_path / 'enh' / f"{sample_name}_0.tif"
                rgb_path = base_path / 'rgb' / f"{sample_name}_0.tif"
                mask_path = base_path / 'plm' / f"{sample_name}_0.tif"
            else:

                enh_path = base_path / 'enh' / f"{sample_name}.png"
                rgb_path = base_path / 'rgb' / f"{sample_name}.jpg"
                mask_path = base_path / 'mask' / f"{sample_name}.png"


            missing_files = []
            if not enh_path.exists():
                missing_files.append(f"ENH: {enh_path}")
            if not rgb_path.exists():
                missing_files.append(f"RGB: {rgb_path}")
            if not mask_path.exists():
                missing_files.append(f"Mask: {mask_path}")

            if missing_files:
                error_msg = f"样本 '{sample_name}' 的文件不存在:\n"
                for f in missing_files:
                    error_msg += f"  - {f}\n"
                raise FileNotFoundError(error_msg)

            print(f"  处理样本 {case_idx+1}/6: {sample_name}")


            if data_format == 'dataset_convert':
                enh_raw = load_raw_enh_tif(enh_path)
                rgb = load_rgb_image_tif(rgb_path)
                label = load_label_mask_tif(mask_path)
            else:
                enh_raw = load_raw_enh_png(enh_path)
                rgb = load_rgb_image_jpg(rgb_path)
                label = load_label_mask_png(mask_path)


            if rgb.shape[:2] != enh_raw.shape:
                print(f"    警告: RGB尺寸 {rgb.shape[:2]} 与ENH尺寸 {enh_raw.shape} 不一致，调整ENH尺寸")
                enh_raw = np.array(Image.fromarray(enh_raw).resize(rgb.shape[:2], Image.LANCZOS))

            if rgb.shape[:2] != label.shape:
                print(f"    警告: RGB尺寸 {rgb.shape[:2]} 与Label尺寸 {label.shape} 不一致，调整Label尺寸")
                label = np.array(Image.fromarray(label).resize(rgb.shape[:2], Image.NEAREST))


            if data_format == 'dataset_convert':

                enh_raw_scaled = enh_raw.copy()
                enh_normalized = ((enh_raw_scaled - enh_x0) / enh_x1 * 255).clip(0, 255).astype(np.uint8)
            else:

                enh_raw_scaled = (enh_raw / 255.0) * 1000.0
                enh_normalized = ((enh_raw_scaled - enh_x0) / enh_x1 * 255).clip(0, 255).astype(np.uint8)


            input_h, input_w = rgb.shape[:2]


            print(f"    使用DiT模型生成预测...")
            pred_enh_dit, pred_mask_dit, pred_prob_dit, prob_stats_dit = predict_with_model(
                dit_model, rgb, enh_normalized, enh_raw_scaled, device, threshold,
                diffusion_model=None,
                diffusion_config=None
            )


            print(f"    使用DiT-Mamba模型生成预测...")
            pred_enh_ditmamba, pred_mask_ditmamba, pred_prob_ditmamba, prob_stats_ditmamba = predict_with_model(
                ditmamba_model, rgb, enh_normalized, enh_raw_scaled, device, threshold,
                diffusion_model=None,
                diffusion_config=None
            )



            if pred_mask_ditmamba.shape != (input_h, input_w):
                print(f"    注意: 预测输出尺寸 {pred_mask_ditmamba.shape} 与输入尺寸 {(input_h, input_w)} 不同，调整所有数据")


                if pred_mask_dit.shape != pred_mask_ditmamba.shape:
                    print(f"    注意: DiT输出尺寸 {pred_mask_dit.shape} 与DiT-Mamba输出尺寸 {pred_mask_ditmamba.shape} 不同，调整DiT输出")
                    pred_mask_dit = np.array(Image.fromarray(pred_mask_dit).resize(
                        (pred_mask_ditmamba.shape[1], pred_mask_ditmamba.shape[0]), Image.NEAREST))
                    pred_prob_dit = np.array(Image.fromarray(pred_prob_dit).resize(
                        (pred_mask_ditmamba.shape[1], pred_mask_ditmamba.shape[0]), Image.LANCZOS))


                enh_raw_scaled = np.array(Image.fromarray(enh_raw_scaled.astype(np.float32)).resize(
                    (pred_mask_ditmamba.shape[1], pred_mask_ditmamba.shape[0]), Image.LANCZOS))


                label = np.array(Image.fromarray(label.astype(np.uint8)).resize(
                    (pred_mask_ditmamba.shape[1], pred_mask_ditmamba.shape[0]), Image.NEAREST))


                rgb = np.array(Image.fromarray(rgb).resize(
                    (pred_mask_ditmamba.shape[1], pred_mask_ditmamba.shape[0]), Image.LANCZOS))


                pred_enh_dit = enh_raw_scaled * pred_mask_dit
                pred_enh_ditmamba = enh_raw_scaled * pred_mask_ditmamba


                if not (pred_mask_dit.shape == pred_mask_ditmamba.shape == label.shape == enh_raw_scaled.shape):
                    print(f"    警告: 尺寸调整后仍不匹配")
                    print(f"      pred_mask_dit: {pred_mask_dit.shape}")
                    print(f"      pred_mask_ditmamba: {pred_mask_ditmamba.shape}")
                    print(f"      label: {label.shape}")
                    print(f"      enh_raw_scaled: {enh_raw_scaled.shape}")

                    target_shape = pred_mask_ditmamba.shape
                    if pred_mask_dit.shape != target_shape:
                        pred_mask_dit = np.array(Image.fromarray(pred_mask_dit).resize(
                            (target_shape[1], target_shape[0]), Image.NEAREST))
                        pred_prob_dit = np.array(Image.fromarray(pred_prob_dit).resize(
                            (target_shape[1], target_shape[0]), Image.LANCZOS))
                    if label.shape != target_shape:
                        label = np.array(Image.fromarray(label).resize(
                            (target_shape[1], target_shape[0]), Image.NEAREST))
                    if enh_raw_scaled.shape != target_shape:
                        enh_raw_scaled = np.array(Image.fromarray(enh_raw_scaled.astype(np.float32)).resize(
                            (target_shape[1], target_shape[0]), Image.LANCZOS))

                    pred_enh_dit = enh_raw_scaled * pred_mask_dit
                    pred_enh_ditmamba = enh_raw_scaled * pred_mask_ditmamba


            iou_dit = calculate_iou_numpy(pred_mask_dit, label)
            iou_ditmamba = calculate_iou_numpy(pred_mask_ditmamba, label)


            print(f"    DiT模型预测统计:")
            print(f"      IoU: {iou_dit:.4f}")
            print(f"      大于阈值{threshold}的像素: {prob_stats_dit['pixels_above_threshold']}/{prob_stats_dit['total_pixels']} ({100*prob_stats_dit['pixels_above_threshold']/prob_stats_dit['total_pixels']:.2f}%)")
            print(f"    DiT-Mamba模型预测统计:")
            print(f"      IoU: {iou_ditmamba:.4f}")
            print(f"      大于阈值{threshold}的像素: {prob_stats_ditmamba['pixels_above_threshold']}/{prob_stats_ditmamba['total_pixels']} ({100*prob_stats_ditmamba['pixels_above_threshold']/prob_stats_ditmamba['total_pixels']:.2f}%)")

            all_data.append({
                'enh_raw': enh_raw_scaled,
                'rgb': rgb,
                'label': label,
                'pred_enh_dit': pred_enh_dit,
                'pred_mask_dit': pred_mask_dit,
                'pred_prob_dit': pred_prob_dit,
                'prob_stats_dit': prob_stats_dit,
                'iou_dit': iou_dit,
                'pred_enh_ditmamba': pred_enh_ditmamba,
                'pred_mask_ditmamba': pred_mask_ditmamba,
                'pred_prob_ditmamba': pred_prob_ditmamba,
                'prob_stats_ditmamba': prob_stats_ditmamba,
                'iou_ditmamba': iou_ditmamba,
                'sample_name': sample_name
            })



            has_plume = np.any(label > 0)
            if has_plume:

                label_mask = label > 0
                plume_enh = enh_raw_scaled[label_mask]

                valid_enh = plume_enh[plume_enh > 0]
                if len(valid_enh) > 0:
                    all_enh_values.extend(valid_enh.tolist())
                    print(f"    样本有羽流，收集了 {len(valid_enh)} 个ENH值用于颜色映射")
            else:
                print(f"    样本无羽流，跳过ENH值收集")

        except Exception as e:
            print(f"\n错误: 处理样本 {sample_name} 时发生异常")
            print(f"异常信息: {str(e)}")
            import traceback
            traceback.print_exc()
            raise


    if len(all_enh_values) > 0:
        all_enh_array = np.array(all_enh_values)


        non_zero_values = all_enh_array[all_enh_array > 0]

        if len(non_zero_values) > 0:

            global_vmin = max(0, np.percentile(non_zero_values, 10))
            global_vmax = min(1000, np.percentile(non_zero_values, 90))


            if global_vmax - global_vmin < 10:

                global_vmin = max(0, np.min(non_zero_values))
                global_vmax = min(1000, np.max(non_zero_values))
                if global_vmax - global_vmin < 10:
                    print(f"警告: 数据范围过小 ({global_vmax - global_vmin:.2f} ppm-m)，使用默认范围 [0, 1000]")
                    global_vmin, global_vmax = 0, 1000
        else:

            print("警告: 所有ENH值都是0，使用默认颜色映射范围 [0, 1000]")
            global_vmin, global_vmax = 0, 1000
    else:
        print("警告: 未找到有效的ENH数据，使用默认颜色映射范围 [0, 1000]")
        global_vmin, global_vmax = 0, 1000


    print(f"\n统一颜色映射范围: [{global_vmin:.2f}, {global_vmax:.2f}] ppm-m")
    print(f"  （注意：颜色映射范围仅基于有羽流样本的label区域内的ENH值）")
    if len(all_enh_values) > 0:
        all_enh_array = np.array(all_enh_values)
        non_zero = all_enh_array[all_enh_array > 0]
        if len(non_zero) > 0:
            print(f"  数据统计（仅羽流区域）: 最小值={np.min(non_zero):.2f}, 最大值={np.max(non_zero):.2f}, "
                  f"中位数={np.median(non_zero):.2f}, 平均值={np.mean(non_zero):.2f}")
            print(f"  10%分位数={np.percentile(non_zero, 10):.2f}, 90%分位数={np.percentile(non_zero, 90):.2f}")



    print("\n生成可视化...")


    for row_type_idx, row_type_name in enumerate(row_titles):

        ax_row_title = fig.add_subplot(gs[row_type_idx, 0])
        ax_row_title.text(0.5, 0.5, row_type_name,
                         fontsize=16, fontweight='bold', rotation=90,
                         ha='center', va='center', transform=ax_row_title.transAxes,
                         color='black')
        ax_row_title.set_facecolor('white')
        ax_row_title.axis('off')


        for case_idx in range(num_cases):
            data = all_data[case_idx]
            col_idx = case_idx + 1


            if row_type_name == 'DATA':
                ax = fig.add_subplot(gs[row_type_idx, col_idx])
                visualize_enh_data(ax, data['enh_raw'], cmap, vmin=global_vmin, vmax=global_vmax)

                if row_type_idx == 0:
                    ax.set_title(col_titles[case_idx], fontsize=16, fontweight='bold', pad=15, color='black')
            elif row_type_name == 'RGB':
                ax = fig.add_subplot(gs[row_type_idx, col_idx])
                visualize_rgb(ax, data['rgb'])

                if row_type_idx == 0:
                    ax.set_title(col_titles[case_idx], fontsize=16, fontweight='bold', pad=15, color='black')
            elif row_type_name == 'Label':
                ax = fig.add_subplot(gs[row_type_idx, col_idx])
                visualize_label(ax, data['label'])

                if row_type_idx == 0:
                    ax.set_title(col_titles[case_idx], fontsize=16, fontweight='bold', pad=15, color='black')
            elif row_type_name == 'DIT':
                ax = fig.add_subplot(gs[row_type_idx, col_idx])
                visualize_prediction(ax, data['pred_enh_dit'], data['pred_mask_dit'], cmap,
                                   enh_raw=data['enh_raw'], vmin=global_vmin, vmax=global_vmax)

                if row_type_idx == 0:
                    ax.set_title(col_titles[case_idx], fontsize=16, fontweight='bold', pad=15, color='black')





            elif row_type_name == 'DIT-Mamba':
                ax = fig.add_subplot(gs[row_type_idx, col_idx])
                visualize_prediction(ax, data['pred_enh_ditmamba'], data['pred_mask_ditmamba'], cmap,
                                   enh_raw=data['enh_raw'], vmin=global_vmin, vmax=global_vmax)

                if row_type_idx == 0:
                    ax.set_title(col_titles[case_idx], fontsize=16, fontweight='bold', pad=15, color='black')







    ax_colorbar = fig.add_subplot(gs[:, colorbar_col])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=global_vmin, vmax=global_vmax))
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=ax_colorbar, orientation='vertical')
    cbar.set_label('Methane column enhancement (ppm-m)', fontsize=14, fontweight='bold', labelpad=20, color='black')
    cbar.ax.tick_params(labelsize=12, colors='black')
    cbar.ax.yaxis.label.set_color('black')


    num_ticks = 6
    ticks = np.linspace(global_vmin, global_vmax, num_ticks)
    cbar.set_ticks(ticks)

    cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.0f}'))


    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.2, facecolor='white')
    print(f"\n对比图已保存到: {output_path}")
    plt.close()


def generate_comparison_figure_with_diffusion(
    dit_model: nn.Module,
    ditmamba_model: nn.Module,
    diffusion_model: nn.Module,
    sample_names: list,
    base_path: str = 'data/dataset',
    device: torch.device = None,
    output_path: str = 'docs/result/result_comparison_with_diffusion.png',
    enh_x0: float = 81.15,
    enh_x1: float = 62.18,
    threshold: float = 0.5,
    data_format: str = None,
    diffusion_config: dict = None,
    only_diffusion: bool = False
):
    """
    生成包含Diffusion Refiner的对比图
    
    参数:
        dit_model: 训练好的DiT模型
        ditmamba_model: 训练好的DiT-Mamba模型
        diffusion_model: 训练好的Diffusion Refiner模型
        sample_names: 样本名称列表（5个或6个）
            - dataset格式: 数字编号字符串（如"00001"）
            - dataset_convert格式: 样本名称（如"sample_00001"）
        base_path: 数据根目录
        device: 计算设备
        output_path: 输出路径
        enh_x0: ENH归一化参数x0
        enh_x1: ENH归一化参数x1
        threshold: 预测阈值，默认0.5
        data_format: 数据格式 ('dataset' 或 'dataset_convert')，如果为None则自动检测
        diffusion_config: Diffusion配置字典（可选）
        only_diffusion: 如果为True，只显示4行（DATA, RGB, Label, DIT-Mamba + Diffusion），不显示DIT和DIT-Mamba行
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    if dit_model is None or ditmamba_model is None:
        raise ValueError("DiT和DiT-Mamba模型都必须提供，不能为 None")
    if diffusion_model is None:
        raise ValueError("Diffusion模型必须提供，不能为 None")

    if not isinstance(dit_model, DiT):
        raise TypeError(f"dit_model 必须是 DiT 类型，实际类型: {type(dit_model)}")
    if not isinstance(ditmamba_model, DiTMamba):
        raise TypeError(f"ditmamba_model 必须是 DiTMamba 类型，实际类型: {type(ditmamba_model)}")

    base_path = Path(base_path)


    if data_format is None:
        data_format = detect_data_format(base_path)
    print(f"检测到数据格式: {data_format}")

    num_cases = len(sample_names)


    if only_diffusion:
        num_rows = 4
        row_titles = [
            'DATA',
            'RGB',
            'Label',
            'DIT-Mamba + Diffusion'
        ]
    else:
        num_rows = 6
        row_titles = [
            'DATA',
            'RGB',
            'Label',
            'DIT',
            'DIT-Mamba',
            'DIT-Mamba + Diffusion'
        ]


    if num_cases < 6:
        print(f"注意: 只有{num_cases}个正样本，将使用{num_cases}列布局")
        num_cols = num_cases
        total_cols = 1 + num_cols + 1
        colorbar_col = total_cols - 1
        fig_width = 4 + num_cols * 4
        fig_height = num_rows * 4
        fig = plt.figure(figsize=(fig_width, fig_height))
        width_ratios = [0.3] + [1] * num_cols + [0.25]
        gs = fig.add_gridspec(num_rows, total_cols, width_ratios=width_ratios,
                              height_ratios=[1]*num_rows, hspace=0.3, wspace=0.15)
        col_titles = [f'Case {i+1}' for i in range(num_cases)]
    elif num_cases > 6:
        print(f"警告: 得到{num_cases}个样本，将使用前6个")
        sample_names = sample_names[:6]
        num_cases = 6
        total_cols = 8
        colorbar_col = 7
        fig_height = num_rows * 4
        fig = plt.figure(figsize=(28, fig_height))
        gs = fig.add_gridspec(num_rows, 8, width_ratios=[0.3, 1, 1, 1, 1, 1, 1, 0.25],
                              height_ratios=[1]*num_rows, hspace=0.3, wspace=0.15)
        col_titles = [f'Case {i+1}' for i in range(6)]
    else:
        num_cases = 6
        total_cols = 8
        colorbar_col = 7
        fig_height = num_rows * 4
        fig = plt.figure(figsize=(28, fig_height))
        gs = fig.add_gridspec(num_rows, 8, width_ratios=[0.3, 1, 1, 1, 1, 1, 1, 0.25],
                              height_ratios=[1]*num_rows, hspace=0.3, wspace=0.15)
        col_titles = [f'Case {i+1}' for i in range(6)]

    cmap = create_methane_colormap()

    all_data = []
    all_enh_values = []
    print("\n加载数据和生成预测...")
    
    for case_idx, sample_name in enumerate(sample_names):
        try:

            if data_format == 'dataset_convert':
                enh_path = base_path / 'enh' / f"{sample_name}_0.tif"
                rgb_path = base_path / 'rgb' / f"{sample_name}_0.tif"
                mask_path = base_path / 'plm' / f"{sample_name}_0.tif"
            else:
                enh_path = base_path / 'enh' / f"{sample_name}.png"
                rgb_path = base_path / 'rgb' / f"{sample_name}.jpg"
                mask_path = base_path / 'mask' / f"{sample_name}.png"

            missing_files = []
            if not enh_path.exists():
                missing_files.append(f"ENH: {enh_path}")
            if not rgb_path.exists():
                missing_files.append(f"RGB: {rgb_path}")
            if not mask_path.exists():
                missing_files.append(f"Mask: {mask_path}")

            if missing_files:
                error_msg = f"样本 '{sample_name}' 的文件不存在:\n"
                for f in missing_files:
                    error_msg += f"  - {f}\n"
                raise FileNotFoundError(error_msg)

            print(f"  处理样本 {case_idx+1}/{num_cases}: {sample_name}")


            if data_format == 'dataset_convert':
                enh_raw = load_raw_enh_tif(enh_path)
                rgb = load_rgb_image_tif(rgb_path)
                label = load_label_mask_tif(mask_path)
            else:
                enh_raw = load_raw_enh_png(enh_path)
                rgb = load_rgb_image_jpg(rgb_path)
                label = load_label_mask_png(mask_path)


            if rgb.shape[:2] != enh_raw.shape:
                print(f"    警告: RGB尺寸 {rgb.shape[:2]} 与ENH尺寸 {enh_raw.shape} 不一致，调整ENH尺寸")
                enh_raw = np.array(Image.fromarray(enh_raw).resize(rgb.shape[:2], Image.LANCZOS))

            if rgb.shape[:2] != label.shape:
                print(f"    警告: RGB尺寸 {rgb.shape[:2]} 与Label尺寸 {label.shape} 不一致，调整Label尺寸")
                label = np.array(Image.fromarray(label).resize(rgb.shape[:2], Image.NEAREST))


            if data_format == 'dataset_convert':
                enh_raw_scaled = enh_raw.copy()
                enh_normalized = ((enh_raw_scaled - enh_x0) / enh_x1 * 255).clip(0, 255).astype(np.uint8)
            else:
                enh_raw_scaled = (enh_raw / 255.0) * 1000.0
                enh_normalized = ((enh_raw_scaled - enh_x0) / enh_x1 * 255).clip(0, 255).astype(np.uint8)

            input_h, input_w = rgb.shape[:2]


            if not only_diffusion:

                print(f"    使用DiT模型生成预测...")
                pred_enh_dit, pred_mask_dit, pred_prob_dit, prob_stats_dit = predict_with_model(
                    dit_model, rgb, enh_normalized, enh_raw_scaled, device, threshold,
                    diffusion_model=None,
                    diffusion_config=None
                )


                print(f"    使用DiT-Mamba模型生成预测（不带Diffusion）...")
                pred_enh_ditmamba, pred_mask_ditmamba, pred_prob_ditmamba, prob_stats_ditmamba = predict_with_model(
                    ditmamba_model, rgb, enh_normalized, enh_raw_scaled, device, threshold,
                    diffusion_model=None,
                    diffusion_config=None
                )
            else:

                pred_enh_dit = None
                pred_mask_dit = None
                pred_prob_dit = None
                prob_stats_dit = None
                pred_enh_ditmamba = None
                pred_mask_ditmamba = None
                pred_prob_ditmamba = None
                prob_stats_ditmamba = None


            print(f"    使用DiT-Mamba模型生成预测（带Diffusion细化）...")
            pred_enh_ditmamba_diffusion, pred_mask_ditmamba_diffusion, pred_prob_ditmamba_diffusion, prob_stats_ditmamba_diffusion = predict_with_model(
                ditmamba_model, rgb, enh_normalized, enh_raw_scaled, device, threshold,
                diffusion_model=diffusion_model,
                diffusion_config=diffusion_config
            )


            if pred_mask_ditmamba_diffusion.shape != (input_h, input_w):
                print(f"    注意: 预测输出尺寸 {pred_mask_ditmamba_diffusion.shape} 与输入尺寸 {(input_h, input_w)} 不同，调整所有数据")
                
                if not only_diffusion:
                    if pred_mask_dit is not None and pred_mask_dit.shape != pred_mask_ditmamba_diffusion.shape:
                        pred_mask_dit = np.array(Image.fromarray(pred_mask_dit).resize(
                            (pred_mask_ditmamba_diffusion.shape[1], pred_mask_ditmamba_diffusion.shape[0]), Image.NEAREST))
                        pred_prob_dit = np.array(Image.fromarray(pred_prob_dit).resize(
                            (pred_mask_ditmamba_diffusion.shape[1], pred_mask_ditmamba_diffusion.shape[0]), Image.LANCZOS))
                    
                    if pred_mask_ditmamba is not None and pred_mask_ditmamba.shape != pred_mask_ditmamba_diffusion.shape:
                        pred_mask_ditmamba = np.array(Image.fromarray(pred_mask_ditmamba).resize(
                            (pred_mask_ditmamba_diffusion.shape[1], pred_mask_ditmamba_diffusion.shape[0]), Image.NEAREST))
                        pred_prob_ditmamba = np.array(Image.fromarray(pred_prob_ditmamba).resize(
                            (pred_mask_ditmamba_diffusion.shape[1], pred_mask_ditmamba_diffusion.shape[0]), Image.LANCZOS))

                enh_raw_scaled = np.array(Image.fromarray(enh_raw_scaled.astype(np.float32)).resize(
                    (pred_mask_ditmamba_diffusion.shape[1], pred_mask_ditmamba_diffusion.shape[0]), Image.LANCZOS))
                label = np.array(Image.fromarray(label.astype(np.uint8)).resize(
                    (pred_mask_ditmamba_diffusion.shape[1], pred_mask_ditmamba_diffusion.shape[0]), Image.NEAREST))
                rgb = np.array(Image.fromarray(rgb).resize(
                    (pred_mask_ditmamba_diffusion.shape[1], pred_mask_ditmamba_diffusion.shape[0]), Image.LANCZOS))

                if not only_diffusion:
                    if pred_mask_dit is not None:
                        pred_enh_dit = enh_raw_scaled * pred_mask_dit
                    if pred_mask_ditmamba is not None:
                        pred_enh_ditmamba = enh_raw_scaled * pred_mask_ditmamba
                pred_enh_ditmamba_diffusion = enh_raw_scaled * pred_mask_ditmamba_diffusion


            if not only_diffusion:
                iou_dit = calculate_iou_numpy(pred_mask_dit, label) if pred_mask_dit is not None else 0.0
                iou_ditmamba = calculate_iou_numpy(pred_mask_ditmamba, label) if pred_mask_ditmamba is not None else 0.0
                print(f"    DiT模型 IoU: {iou_dit:.4f}")
                print(f"    DiT-Mamba模型 IoU: {iou_ditmamba:.4f}")
            else:
                iou_dit = 0.0
                iou_ditmamba = 0.0
            iou_ditmamba_diffusion = calculate_iou_numpy(pred_mask_ditmamba_diffusion, label)
            print(f"    DiT-Mamba + Diffusion模型 IoU: {iou_ditmamba_diffusion:.4f}")

            data_dict = {
                'enh_raw': enh_raw_scaled,
                'rgb': rgb,
                'label': label,
                'pred_enh_ditmamba_diffusion': pred_enh_ditmamba_diffusion,
                'pred_mask_ditmamba_diffusion': pred_mask_ditmamba_diffusion,
                'pred_prob_ditmamba_diffusion': pred_prob_ditmamba_diffusion,
                'prob_stats_ditmamba_diffusion': prob_stats_ditmamba_diffusion,
                'iou_ditmamba_diffusion': iou_ditmamba_diffusion,
                'sample_name': sample_name
            }
            
            if not only_diffusion:
                data_dict.update({
                    'pred_enh_dit': pred_enh_dit,
                    'pred_mask_dit': pred_mask_dit,
                    'pred_prob_dit': pred_prob_dit,
                    'prob_stats_dit': prob_stats_dit,
                    'iou_dit': iou_dit,
                    'pred_enh_ditmamba': pred_enh_ditmamba,
                    'pred_mask_ditmamba': pred_mask_ditmamba,
                    'pred_prob_ditmamba': pred_prob_ditmamba,
                    'prob_stats_ditmamba': prob_stats_ditmamba,
                    'iou_ditmamba': iou_ditmamba
                })
            
            all_data.append(data_dict)


            has_plume = np.any(label > 0)
            if has_plume:
                label_mask = label > 0
                plume_enh = enh_raw_scaled[label_mask]
                valid_enh = plume_enh[plume_enh > 0]
                if len(valid_enh) > 0:
                    all_enh_values.extend(valid_enh.tolist())
                    print(f"    样本有羽流，收集了 {len(valid_enh)} 个ENH值用于颜色映射")
            else:
                print(f"    样本无羽流，跳过ENH值收集")

        except Exception as e:
            print(f"\n错误: 处理样本 {sample_name} 时发生异常")
            print(f"异常信息: {str(e)}")
            import traceback
            traceback.print_exc()
            raise


    if len(all_enh_values) > 0:
        all_enh_array = np.array(all_enh_values)
        non_zero_values = all_enh_array[all_enh_array > 0]
        if len(non_zero_values) > 0:
            global_vmin = max(0, np.percentile(non_zero_values, 10))
            global_vmax = min(1000, np.percentile(non_zero_values, 90))
            if global_vmax - global_vmin < 10:
                global_vmin = max(0, np.min(non_zero_values))
                global_vmax = min(1000, np.max(non_zero_values))
                if global_vmax - global_vmin < 10:
                    print(f"警告: 数据范围过小，使用默认范围 [0, 1000]")
                    global_vmin, global_vmax = 0, 1000
        else:
            print("警告: 所有ENH值都是0，使用默认颜色映射范围 [0, 1000]")
            global_vmin, global_vmax = 0, 1000
    else:
        print("警告: 未找到有效的ENH数据，使用默认颜色映射范围 [0, 1000]")
        global_vmin, global_vmax = 0, 1000

    print(f"\n统一颜色映射范围: [{global_vmin:.2f}, {global_vmax:.2f}] ppm-m")


    print("\n生成可视化...")
    for row_type_idx, row_type_name in enumerate(row_titles):

        ax_row_title = fig.add_subplot(gs[row_type_idx, 0])
        ax_row_title.text(0.5, 0.5, row_type_name,
                         fontsize=16, fontweight='bold', rotation=90,
                         ha='center', va='center', transform=ax_row_title.transAxes,
                         color='black')
        ax_row_title.set_facecolor('white')
        ax_row_title.axis('off')


        for case_idx in range(num_cases):
            data = all_data[case_idx]
            col_idx = case_idx + 1

            if row_type_name == 'DATA':
                ax = fig.add_subplot(gs[row_type_idx, col_idx])
                visualize_enh_data(ax, data['enh_raw'], cmap, vmin=global_vmin, vmax=global_vmax)
                if row_type_idx == 0:
                    ax.set_title(col_titles[case_idx], fontsize=16, fontweight='bold', pad=15, color='black')
            elif row_type_name == 'RGB':
                ax = fig.add_subplot(gs[row_type_idx, col_idx])
                visualize_rgb(ax, data['rgb'])
                if row_type_idx == 0:
                    ax.set_title(col_titles[case_idx], fontsize=16, fontweight='bold', pad=15, color='black')
            elif row_type_name == 'Label':
                ax = fig.add_subplot(gs[row_type_idx, col_idx])
                visualize_label(ax, data['label'])
                if row_type_idx == 0:
                    ax.set_title(col_titles[case_idx], fontsize=16, fontweight='bold', pad=15, color='black')
            elif row_type_name == 'DIT':
                if not only_diffusion:
                    ax = fig.add_subplot(gs[row_type_idx, col_idx])
                    visualize_prediction(ax, data['pred_enh_dit'], data['pred_mask_dit'], cmap,
                                       enh_raw=data['enh_raw'], vmin=global_vmin, vmax=global_vmax)
                    if row_type_idx == 0:
                        ax.set_title(col_titles[case_idx], fontsize=16, fontweight='bold', pad=15, color='black')
            elif row_type_name == 'DIT-Mamba':
                if not only_diffusion:
                    ax = fig.add_subplot(gs[row_type_idx, col_idx])
                    visualize_prediction(ax, data['pred_enh_ditmamba'], data['pred_mask_ditmamba'], cmap,
                                       enh_raw=data['enh_raw'], vmin=global_vmin, vmax=global_vmax)
                    if row_type_idx == 0:
                        ax.set_title(col_titles[case_idx], fontsize=16, fontweight='bold', pad=15, color='black')
            elif row_type_name == 'DIT-Mamba + Diffusion':
                ax = fig.add_subplot(gs[row_type_idx, col_idx])
                visualize_prediction(ax, data['pred_enh_ditmamba_diffusion'], data['pred_mask_ditmamba_diffusion'], cmap,
                                   enh_raw=data['enh_raw'], vmin=global_vmin, vmax=global_vmax)
                if row_type_idx == 0:
                    ax.set_title(col_titles[case_idx], fontsize=16, fontweight='bold', pad=15, color='black')


    ax_colorbar = fig.add_subplot(gs[:, colorbar_col])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=global_vmin, vmax=global_vmax))
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=ax_colorbar, orientation='vertical')
    cbar.set_label('Methane column enhancement (ppm-m)', fontsize=14, fontweight='bold', labelpad=20, color='black')
    cbar.ax.tick_params(labelsize=12, colors='black')
    cbar.ax.yaxis.label.set_color('black')


    num_ticks = 6
    ticks = np.linspace(global_vmin, global_vmax, num_ticks)
    cbar.set_ticks(ticks)
    cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.0f}'))


    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.2, facecolor='white')
    print(f"\n对比图已保存到: {output_path}")
    plt.close()


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(
        description='从data/dataset格式生成DiT和DiT-Mamba对比图',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:

  python scripts/generate_figure_from_dataset.py \\
      --dit_checkpoint checkpoints/best_model_dit.pth \\
      --ditmamba_checkpoint checkpoints/best_model_ditmamba.pth


  python scripts/generate_figure_from_dataset.py \\
      --dit_checkpoint checkpoints/best_model_dit.pth \\
      --ditmamba_checkpoint checkpoints/best_model_ditmamba.pth \\
      --use_random


  python scripts/generate_figure_from_dataset.py \\
      --dit_checkpoint checkpoints/best_model_dit.pth \\
      --ditmamba_checkpoint checkpoints/best_model_ditmamba.pth \\
      --samples 00001 00002 00003 00004 00005


  python scripts/generate_figure_from_dataset.py \\
      --dit_checkpoint checkpoints/best_model_dit.pth \\
      --ditmamba_checkpoint checkpoints/best_model_ditmamba.pth \\
      --output docs/result/result_comparison.png \\
      --min_data_completeness 0.9 \\
      --pos_ratio 0.8


  python scripts/generate_figure_from_dataset.py \\
      --dit_checkpoint checkpoints/best_model_dit.pth \\
      --ditmamba_checkpoint checkpoints/best_model_ditmamba.pth \\
      --num_groups 3 --output docs/result/result_comparison.png


  python scripts/generate_figure_from_dataset.py \\
      --dit_checkpoint checkpoints/best_model_dit.pth \\
      --ditmamba_checkpoint checkpoints/best_model_ditmamba.pth \\
      --select_by_confidence \\
      --min_confidence 0.2 \\
      --confidence_model ditmamba \\
      --output docs/result/result_comparison.png


  python scripts/generate_figure_from_dataset.py \\
      --dit_checkpoint checkpoints/best_model_dit.pth \\
      --ditmamba_checkpoint checkpoints/best_model_ditmamba.pth \\
      --select_by_confidence \\
      --min_confidence 0.2 \\
      --confidence_model both \\
      --output docs/result/result_comparison.png


  python scripts/generate_figure_from_dataset.py \\
      --dit_checkpoint checkpoints/best_model_dit.pth \\
      --ditmamba_checkpoint checkpoints/best_model_ditmamba.pth \\
      --select_by_confidence \\
      --min_confidence 0.2 \\
      --confidence_model average \\
      --output docs/result/result_comparison.png


  python scripts/generate_figure_from_dataset.py \\
      --dit_checkpoint checkpoints/best_model_dit.pth \\
      --ditmamba_checkpoint checkpoints/best_model_ditmamba.pth \\
      --diffusion_checkpoint checkpoints/emit/diffusion_refiner_best-ditmamba.pth \\
      --use_diffusion_figure \\
      --output docs/result/result_comparison_with_diffusion.png

注意:
  - 使用 --num_groups 时，输出文件名会自动添加组号（如 result_comparison_1.png）
  - 如果指定了 --samples，则 --num_groups 参数无效
  - 默认使用智能选择策略，优先选择有标签、数据完整、羽流大小多样的样本
  - 使用 --select_by_confidence 可以根据模型预测置信度选择样本，确保可视化效果更好
  - 使用 --use_random 可以恢复原有的随机选择行为
  - 使用 --use_diffusion_figure 可以生成包含Diffusion Refiner的对比图（6行布局，包含DIT-Mamba + Diffusion行）
  - 使用 --use_diffusion_figure 时必须提供 --diffusion_checkpoint 参数或在config.yaml中配置diffusion_refiner.model_path
  - 使用 --only_diffusion 可以只显示Diffusion结果（4行布局：DATA, RGB, Label, DIT-Mamba + Diffusion），需要与--use_diffusion_figure一起使用
  - 使用 --generate_all_types 可以为每组生成所有类型的图例（standard、with_diffusion、only_diffusion），需要提供--diffusion_checkpoint。使用此参数时，--use_diffusion_figure和--only_diffusion将被忽略
  - 使用 --label_analysis_json 可以从JSON文件加载样本信息，大幅提升样本选择性能，并确保Case 1-5按羽流占比升序，Case 6选择最大的样本
        """
    )
    parser.add_argument('--dit_checkpoint', type=str, default=None,
                       help='DiT模型检查点路径（必需）')
    parser.add_argument('--ditmamba_checkpoint', type=str,
                       default='checkpoints/best_model.pth',
                       help='DiT-Mamba模型检查点路径（默认: checkpoints/best_model.pth）')
    parser.add_argument('--config', type=str, default=None,
                       help='配置文件路径（当checkpoint没有config时使用）')
    parser.add_argument('--output', type=str, default='docs/result/result_from_dataset.png',
                       help='输出图像路径')
    parser.add_argument('--samples', type=str, nargs='+', default=None,
                       help='指定样本名称列表（5个），如果不指定则从数据集随机选择。样本名称格式: 数字编号（如"00001"）')
    parser.add_argument('--data_path', type=str, default='data/dataset',
                       help='数据根目录')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='预测阈值（0-1之间），默认0.5')
    parser.add_argument('--num_groups', type=int, default=1,
                       help='生成多少组对比图（每组选择不同的样本），默认1。如果指定了--samples，此参数无效')
    parser.add_argument('--min_data_completeness', type=float, default=0.8,
                       help='最小数据完整性阈值（0-1之间），默认0.8。低于此值的样本将被过滤')
    parser.add_argument('--pos_ratio', type=float, default=0.8,
                       help='正样本占比（0-1之间），默认0.8。例如0.8表示5个样本中至少4个有标签')
    parser.add_argument('--use_random', action='store_true',
                       help='使用随机选择而非智能选择（默认使用智能选择策略）')
    parser.add_argument('--select_by_confidence', action='store_true',
                       help='根据模型预测置信度选择样本（推荐，可确保选择模型预测效果好的样本）')
    parser.add_argument('--min_confidence', type=float, default=0.1,
                       help='最小置信度阈值（0-1之间），默认0.1。使用--select_by_confidence时，只选择置信度>=此值的样本')
    parser.add_argument('--confidence_model', type=str, default='ditmamba',
                       choices=['dit', 'ditmamba', 'both', 'average'],
                       help='用于评估置信度的模型（默认: ditmamba）。'
                            'both: 使用两个模型，要求两个模型置信度都>=阈值，然后按平均置信度排序（确保两个模型都表现好，推荐用于对比）；'
                            'average: 使用两个模型，取平均置信度')
    parser.add_argument('--use_diffusion_figure', action='store_true',
                       help='使用包含Diffusion Refiner的对比图生成函数（6行布局，包含DIT-Mamba + Diffusion行）')
    parser.add_argument('--only_diffusion', action='store_true',
                       help='只显示Diffusion结果（4行布局：DATA, RGB, Label, DIT-Mamba + Diffusion），不显示DIT和DIT-Mamba行。需要与--use_diffusion_figure一起使用')
    parser.add_argument('--generate_all_types', action='store_true',
                       help='为每组生成所有类型的图例（standard、with_diffusion、only_diffusion）。需要提供--diffusion_checkpoint。如果使用此参数，--use_diffusion_figure和--only_diffusion将被忽略')
    parser.add_argument('--diffusion_checkpoint', type=str, default=None,
                       help='Diffusion模型检查点路径（使用--use_diffusion_figure或--generate_all_types时必需）')
    parser.add_argument('--label_analysis_json', type=str, default=None,
                       help='label_analysis.json文件路径，用于智能选择样本（从JSON加载样本信息，大幅提升性能）')
    parser.add_argument('--plume_ratio_min', type=float, default=None,
                       help='最小羽流占比（百分比），例如30.0表示30%%')
    parser.add_argument('--plume_ratio_max', type=float, default=None,
                       help='最大羽流占比（百分比），例如40.0表示40%%')

    args = parser.parse_args()

    print("=" * 70)
    print("DiT-Mamba 对比图生成 (支持data/dataset和data/dataset_convert格式)")
    print("=" * 70)

    try:

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {device}")


        if args.dit_checkpoint is None:
            print("错误: 必须指定 --dit_checkpoint 参数")
            print("  使用示例: --dit_checkpoint checkpoints/best_model_dit.pth")
            print("  完整命令示例:")
            print("    python scripts/generate_figure_from_dataset.py \\")
            print("        --dit_checkpoint checkpoints/best_model_dit.pth \\")
            print("        --ditmamba_checkpoint checkpoints/best_model_ditmamba.pth")
            return

        if not Path(args.dit_checkpoint).exists():
            print(f"错误: DiT检查点文件不存在: {args.dit_checkpoint}")
            print(f"  请检查文件路径是否正确")
            return

        if not Path(args.ditmamba_checkpoint).exists():
            print(f"错误: DiT-Mamba检查点文件不存在: {args.ditmamba_checkpoint}")
            print(f"  请检查文件路径是否正确")
            return

        if args.generate_all_types:

            if args.diffusion_checkpoint is None:

                if config_path and Path(config_path).exists():
                    import yaml
                    with open(config_path, 'r', encoding='utf-8') as f:
                        config = yaml.safe_load(f)
                    if config and 'diffusion_refiner' in config and config['diffusion_refiner'].get('model_path'):
                        args.diffusion_checkpoint = config['diffusion_refiner']['model_path']
                        print(f"从配置文件读取Diffusion模型路径: {args.diffusion_checkpoint}")
            
            if args.diffusion_checkpoint is None or not Path(args.diffusion_checkpoint).exists():
                print("错误: 使用 --generate_all_types 时必须提供有效的Diffusion模型路径")
                print("  请使用 --diffusion_checkpoint 参数指定模型路径")
                print("  或在config.yaml中配置 diffusion_refiner.model_path")
                return
            

            print("注意: 使用 --generate_all_types，将忽略 --use_diffusion_figure 和 --only_diffusion 参数")
        elif args.only_diffusion and not args.use_diffusion_figure:
            print("错误: 使用 --only_diffusion 时必须同时使用 --use_diffusion_figure")
            print("  正确用法:")
            print("    python scripts/generate_figure_from_dataset.py \\")
            print("        --dit_checkpoint checkpoints/best_model_dit.pth \\")
            print("        --ditmamba_checkpoint checkpoints/best_model_ditmamba.pth \\")
            print("        --use_diffusion_figure \\")
            print("        --only_diffusion \\")
            print("        --diffusion_checkpoint checkpoints/diffusion_refiner_best.pth")
            return


        config_path = args.config
        if config_path is None:
            project_root = Path(__file__).parent.parent
            default_config = project_root / 'config.yaml'
            if default_config.exists():
                config_path = str(default_config)
                print(f"使用默认配置文件: {config_path}")



        config_dict = {}
        if config_path and Path(config_path).exists():
            config_obj = TrainingConfig(config_path)
            config_dict = config_obj.config


        diffusion_model = None
        diffusion_config = None
        if args.use_diffusion_figure or args.generate_all_types:

            diffusion_checkpoint = args.diffusion_checkpoint
            if diffusion_checkpoint is None:

                if config_dict.get('diffusion_refiner', {}).get('enable', False):
                    diffusion_checkpoint = config_dict.get('diffusion_refiner', {}).get('model_path')
            
            if diffusion_checkpoint and Path(diffusion_checkpoint).exists():
                from scripts.evaluate import load_diffusion_model
                diffusion_model = load_diffusion_model(diffusion_checkpoint, device, config_dict)
                diffusion_config = config_dict
                print(f"已加载Diffusion模型: {diffusion_checkpoint}")
            else:
                if args.use_diffusion_figure:
                    print(f"错误: 使用 --use_diffusion_figure 时必须提供有效的Diffusion模型路径")
                else:
                    print(f"错误: 使用 --generate_all_types 时必须提供有效的Diffusion模型路径")
                print(f"  请使用 --diffusion_checkpoint 参数指定模型路径")
                print(f"  或在config.yaml中配置 diffusion_refiner.model_path")
                return

        print("\n加载DiT模型...")
        dit_model = load_model(args.dit_checkpoint, device, config_path=config_path, model_type='dit')

        print("\n加载DiT-Mamba模型...")
        ditmamba_model = load_model(args.ditmamba_checkpoint, device, config_path=config_path, model_type='ditmamba')


        if args.samples is not None:

            num_groups = 1
        else:
            num_groups = max(1, args.num_groups)


        successful_groups = 0

        global_selected_samples = set()
        for group_id in range(1, num_groups + 1):
            print(f"\n{'='*70}")
            print(f"生成第 {group_id}/{num_groups} 组对比图")
            print(f"{'='*70}")


            if args.generate_all_types:
                figure_types = ['standard', 'with_diffusion', 'only_diffusion']
            elif args.use_diffusion_figure:
                if args.only_diffusion:
                    figure_types = ['only_diffusion']
                else:
                    figure_types = ['with_diffusion']
            else:
                figure_types = ['standard']
            

            output_path = Path(args.output)
            stem = output_path.stem
            suffix = output_path.suffix
            parent = output_path.parent

            base_path = Path(args.data_path)


            data_format = detect_data_format(base_path)
            print(f"检测到数据格式: {data_format}")

            if args.samples is not None:
                sample_names = args.samples
                if len(sample_names) != 5:
                    print(f"警告: 需要5个样本，得到{len(sample_names)}个")
                    if len(sample_names) > 5:
                        sample_names = sample_names[:5]
                    else:
                        print(f"错误: 样本数量不足5个")
                        if num_groups > 1:
                            print(f"      跳过第 {group_id} 组，继续下一组...")
                            continue
                        else:
                            return


                print("\n验证样本是否存在...")
                invalid_samples = []
                for sample_name in sample_names:
                    if data_format == 'dataset_convert':
                        enh_path = base_path / 'enh' / f"{sample_name}_0.tif"
                        rgb_path = base_path / 'rgb' / f"{sample_name}_0.tif"
                        mask_path = base_path / 'plm' / f"{sample_name}_0.tif"
                    else:
                        enh_path = base_path / 'enh' / f"{sample_name}.png"
                        rgb_path = base_path / 'rgb' / f"{sample_name}.jpg"
                        mask_path = base_path / 'mask' / f"{sample_name}.png"

                    if not (enh_path.exists() and rgb_path.exists() and mask_path.exists()):
                        invalid_samples.append(sample_name)

                if invalid_samples:
                    print(f"\n错误: 以下样本的文件不存在:")
                    for s in invalid_samples:
                        print(f"  - {s}")
                    if num_groups > 1:
                        print(f"      跳过第 {group_id} 组，继续下一组...")
                        continue
                    else:
                        return
                else:
                    print(f"  所有样本验证通过")
            else:

                print("\n从数据集获取样本...")
                rgb_dir = base_path / 'rgb'

                if data_format == 'dataset_convert':
                    all_rgb_files = sorted(rgb_dir.glob('*.tif'))
                else:
                    all_rgb_files = sorted(rgb_dir.glob('*.jpg'))

                if len(all_rgb_files) == 0:
                    ext = '.tif' if data_format == 'dataset_convert' else '.jpg'
                    print(f"错误: 在 {rgb_dir} 中未找到任何{ext}文件")
                    if num_groups > 1:
                        print(f"      跳过第 {group_id} 组，继续下一组...")
                        continue
                    else:
                        return

                print(f"  数据集共有 {len(all_rgb_files)} 个样本")


                valid_samples = []
                for rgb_file in all_rgb_files:
                    if data_format == 'dataset_convert':

                        file_stem = rgb_file.stem.rsplit('_', 1)[0]
                        enh_path = base_path / 'enh' / f"{file_stem}_0.tif"
                        mask_path = base_path / 'plm' / f"{file_stem}_0.tif"
                    else:
                        file_stem = rgb_file.stem
                        enh_path = base_path / 'enh' / f"{file_stem}.png"
                        mask_path = base_path / 'mask' / f"{file_stem}.png"

                    if enh_path.exists() and mask_path.exists():
                        valid_samples.append(file_stem)

                if len(valid_samples) == 0:
                    print(f"错误: 未找到任何完整的样本（需要RGB、ENH和Mask文件都存在）")
                    if num_groups > 1:
                        print(f"      跳过第 {group_id} 组，继续下一组...")
                        continue
                    else:
                        return

                print(f"  完整样本数: {len(valid_samples)}")


                if args.label_analysis_json and args.plume_ratio_min is not None and args.plume_ratio_max is not None:
                    valid_samples = filter_samples_by_plume_ratio(
                        valid_samples, 
                        args.label_analysis_json,
                        args.plume_ratio_min,
                        args.plume_ratio_max
                    )
                    if len(valid_samples) == 0:
                        print(f"错误: 筛选后没有可用样本")
                        if num_groups > 1:
                            print(f"      跳过第 {group_id} 组，继续下一组...")
                            continue
                        else:
                            return

                if args.use_random:

                    import random
                    random.seed(42 + group_id - 1)
                    if num_groups > 1:
                        print(f"使用随机种子: {42 + group_id - 1}")


                    available_samples = [s for s in valid_samples if s not in global_selected_samples]

                    if len(available_samples) == 0:
                        print(f"警告: 所有样本都已被选择，将使用所有样本")
                        available_samples = valid_samples
                    elif len(available_samples) < 6:
                        print(f"警告: 可用样本数少于6个（已排除{len(global_selected_samples)}个已选样本），将使用所有{len(available_samples)}个可用样本")
                        sample_names = available_samples
                    else:
                        sample_names = random.sample(available_samples, 6)


                    global_selected_samples.update(sample_names)

                    print(f"\n选择的样本（随机，已排除{len(global_selected_samples) - len(sample_names)}个已选样本）: {sample_names}")
                else:

                    confidence_scores = None


                    if args.select_by_confidence:
                        print("\n评估所有正样本的置信度...")


                        print("  快速检查样本标签...")
                        pos_sample_names = []
                        for sample_name in valid_samples:
                            quality = check_sample_quality_fast(sample_name, base_path, data_format)
                            if quality.get('has_plume', False):
                                pos_sample_names.append(sample_name)

                        print(f"  找到 {len(pos_sample_names)} 个正样本，开始评估置信度...")


                        if args.confidence_model == 'both' or args.confidence_model == 'average':
                            if args.confidence_model == 'both':

                                print(f"  使用DiT和DiT-Mamba两个模型评估IoU（both模式）")
                                print(f"  这将确保选择的样本在两个模型上都表现良好")

                                all_iou_scores = {}


                                available_pos_samples = [s for s in pos_sample_names if s not in global_selected_samples]

                                if len(available_pos_samples) == 0:
                                    print(f"  警告: 所有正样本都已被选择，将使用所有正样本")
                                    available_pos_samples = pos_sample_names

                                print(f"  评估 {len(available_pos_samples)} 个正样本的IoU（排除已选样本: {len(global_selected_samples)}）...")

                                for idx, sample_name in enumerate(available_pos_samples):
                                    iou_info = evaluate_sample_iou(
                                        dit_model, ditmamba_model, sample_name, base_path, device,
                                        data_format, threshold=args.threshold, enh_x0=81.15, enh_x1=62.18
                                    )
                                    if iou_info and iou_info['has_plume']:
                                        iou_dit = iou_info['iou_dit']
                                        iou_ditmamba = iou_info['iou_ditmamba']


                                        if iou_dit >= args.min_confidence and iou_ditmamba >= args.min_confidence:



                                            if iou_dit > 0 and iou_dit < 1.0 and iou_ditmamba > 0 and iou_ditmamba < 1.0:

                                                combined_score = (iou_dit + iou_ditmamba) / 2.0
                                                all_iou_scores[sample_name] = {
                                                    'iou_dit': iou_dit,
                                                    'iou_ditmamba': iou_ditmamba,
                                                    'combined_score': combined_score
                                                }

                                    if (idx + 1) % 10 == 0:
                                        print(f"    已评估 {idx + 1}/{len(available_pos_samples)} 个样本...")


                                all_confidences = {
                                    k: v['combined_score'] for k, v in all_iou_scores.items()
                                }

                                print(f"\nIoU评估完成（both模式）:")
                                print(f"  评估样本数: {len(available_pos_samples)}")
                                print(f"  两个模型IoU都>={args.min_confidence:.4f}的样本数: {len(all_confidences)}")
                                print(f"  （both模式：要求两个模型IoU都达到阈值，按平均IoU排序）")
                            else:

                                print(f"  使用DiT和DiT-Mamba两个模型评估置信度（模式: average）")

                                all_confidences_dit = {}
                                all_confidences_ditmamba = {}


                                available_pos_samples = [s for s in pos_sample_names if s not in global_selected_samples]

                                if len(available_pos_samples) == 0:
                                    print(f"  警告: 所有正样本都已被选择，将使用所有正样本")
                                    available_pos_samples = pos_sample_names


                                print("  评估DiT模型置信度...")
                                for idx, sample_name in enumerate(available_pos_samples):
                                    conf_info = evaluate_sample_confidence(
                                        dit_model, sample_name, base_path, device,
                                        data_format, enh_x0=81.15, enh_x1=62.18
                                    )
                                    if conf_info:
                                        all_confidences_dit[sample_name] = conf_info['confidence_score']

                                    if (idx + 1) % 10 == 0:
                                        print(f"    已评估 {idx + 1}/{len(available_pos_samples)} 个样本（DiT）...")


                                print("  评估DiT-Mamba模型置信度...")
                                for idx, sample_name in enumerate(available_pos_samples):
                                    conf_info = evaluate_sample_confidence(
                                        ditmamba_model, sample_name, base_path, device,
                                        data_format, enh_x0=81.15, enh_x1=62.18
                                    )
                                    if conf_info:
                                        all_confidences_ditmamba[sample_name] = conf_info['confidence_score']

                                    if (idx + 1) % 10 == 0:
                                        print(f"    已评估 {idx + 1}/{len(available_pos_samples)} 个样本（DiT-Mamba）...")


                                all_confidences = {}
                                common_samples = set(all_confidences_dit.keys()) & set(all_confidences_ditmamba.keys())

                                for sample_name in common_samples:
                                    conf_dit = all_confidences_dit[sample_name]
                                    conf_ditmamba = all_confidences_ditmamba[sample_name]

                                    combined_score = (conf_dit + conf_ditmamba) / 2.0
                                    all_confidences[sample_name] = combined_score

                                print(f"\n置信度评估完成（两个模型，模式: average）:")
                                print(f"  DiT模型评估样本数: {len(all_confidences_dit)}")
                                print(f"  DiT-Mamba模型评估样本数: {len(all_confidences_ditmamba)}")
                                print(f"  共同评估样本数: {len(all_confidences)}")
                                print(f"  （average模式：使用平均置信度）")

                        else:

                            if args.confidence_model == 'dit':
                                conf_model = dit_model
                                print(f"  使用DiT模型评估置信度")
                            else:
                                conf_model = ditmamba_model
                                print(f"  使用DiT-Mamba模型评估置信度")


                            available_pos_samples = [s for s in pos_sample_names if s not in global_selected_samples]

                            if len(available_pos_samples) == 0:
                                print(f"  警告: 所有正样本都已被选择，将使用所有正样本")
                                available_pos_samples = pos_sample_names

                            all_confidences = {}
                            for idx, sample_name in enumerate(available_pos_samples):
                                conf_info = evaluate_sample_confidence(
                                    conf_model, sample_name, base_path, device,
                                    data_format, enh_x0=81.15, enh_x1=62.18
                                )
                                if conf_info:
                                    all_confidences[sample_name] = conf_info['confidence_score']

                                if (idx + 1) % 10 == 0:
                                    print(f"    已评估 {idx + 1}/{len(available_pos_samples)} 个样本...")


                        if len(all_confidences) == 0:
                            print("  警告: 无法评估任何样本的置信度，将使用默认选择策略")
                            confidence_scores = None
                        else:

                            if args.confidence_model == 'both':

                                confidence_scores = all_confidences
                                print(f"\n置信度评估完成（both模式，已要求两个模型都>={args.min_confidence:.4f}）:")
                                print(f"  符合条件的样本数: {len(confidence_scores)}")
                            else:

                                filtered_confidences = {
                                    k: v for k, v in all_confidences.items()
                                    if v >= args.min_confidence
                                }
                                print(f"\n置信度评估完成:")
                                print(f"  评估样本数: {len(all_confidences)}")
                                print(f"  通过阈值过滤（>={args.min_confidence:.4f}）: {len(filtered_confidences)}")

                                if len(filtered_confidences) < 5:
                                    print(f"  警告: 过滤后只有{len(filtered_confidences)}个样本，少于需要的5个")
                                    if len(filtered_confidences) > 0:
                                        print(f"  将使用所有{len(filtered_confidences)}个高置信度样本")
                                        confidence_scores = filtered_confidences
                                    else:
                                        print(f"  将使用所有评估的样本（不应用阈值过滤）")
                                        confidence_scores = all_confidences
                                else:
                                    confidence_scores = filtered_confidences
                                    print(f"  过滤成功，从{len(confidence_scores)}个高置信度样本中选择")


                            if len(confidence_scores) < 5:
                                print(f"  警告: 最终只有{len(confidence_scores)}个样本，少于需要的5个")
                                if len(confidence_scores) == 0:
                                    print("  将使用默认选择策略（不使用置信度）")
                                    confidence_scores = None
                    else:
                        confidence_scores = None

                    print("\n评估样本质量并选择代表性样本...")

                    available_samples = [s for s in valid_samples if s not in global_selected_samples]
                    if len(available_samples) == 0:
                        print(f"警告: 所有样本都已被选择，将使用所有样本")
                        available_samples = valid_samples
                    elif len(available_samples) < len(valid_samples):
                        print(f"  已排除 {len(global_selected_samples)} 个已选样本，剩余 {len(available_samples)} 个可用样本")



                    min_conf_for_selection = 0.0 if (args.select_by_confidence and args.confidence_model == 'both') else (args.min_confidence if args.select_by_confidence else 0.0)
                    sample_names = select_representative_samples(
                        available_samples, base_path, data_format, num_samples=6,
                        min_data_completeness=args.min_data_completeness,
                        pos_ratio=args.pos_ratio,
                        group_id=group_id,
                        confidence_scores=confidence_scores,
                        min_confidence=min_conf_for_selection,
                        label_analysis_path=args.label_analysis_json
                    )

                    if len(sample_names) < 5:
                        print(f"警告: 只能选择{len(sample_names)}个有效样本")
                        if len(sample_names) == 0:
                            print("错误: 没有找到任何有效样本")
                            if num_groups > 1:
                                print(f"      跳过第 {group_id} 组，继续下一组...")
                                continue
                            else:
                                return


                    global_selected_samples.update(sample_names)


                    selection_method = "智能选择（按置信度）" if args.select_by_confidence else "智能选择"
                    if args.select_by_confidence and args.confidence_model == 'both':
                        selection_method = "智能选择（按IoU，both模式）"
                    print(f"\n选择的样本（{selection_method}）:")
                    for sample_name in sample_names:
                        quality = check_sample_quality_fast(sample_name, base_path, data_format)
                        plume_info = "有标签" if quality.get('has_plume') else "无标签"
                        size_info = f", 羽流大小: {quality.get('plume_size', 0)}" if quality.get('has_plume') else ""
                        conf_info = ""
                        if confidence_scores and sample_name in confidence_scores:
                            if args.select_by_confidence and args.confidence_model == 'both':
                                conf_info = f", 平均IoU: {confidence_scores[sample_name]:.4f}"
                            else:
                                conf_info = f", 置信度: {confidence_scores[sample_name]:.4f}"
                        print(f"  - {sample_name}: {plume_info}{size_info}{conf_info}")


            print(f"\n开始生成对比图...")
            print(f"数据路径: {args.data_path}")
            if len(figure_types) > 1:
                print(f"将生成 {len(figure_types)} 种类型的图例: {', '.join(figure_types)}")
            

            group_successful = False


            for fig_type in figure_types:

                if args.generate_all_types:
                    if fig_type == 'standard':
                        type_suffix = 'standard'
                    elif fig_type == 'with_diffusion':
                        type_suffix = 'with_diffusion'
                    else:
                        type_suffix = 'diffusion_only'
                    
                    if num_groups > 1:
                        group_output = str(parent / f"{stem}_{type_suffix}_{group_id}{suffix}")
                    else:
                        group_output = str(parent / f"{stem}_{type_suffix}{suffix}")
                else:

                    if num_groups > 1:
                        group_output = str(parent / f"{stem}_{group_id}{suffix}")
                    else:
                        group_output = args.output
                
                print(f"\n生成 {fig_type} 类型图例...")
                print(f"输出路径: {group_output}")

                try:
                    if fig_type == 'standard':

                        generate_comparison_figure(
                            dit_model=dit_model,
                            ditmamba_model=ditmamba_model,
                            sample_names=sample_names,
                            base_path=args.data_path,
                            device=device,
                            output_path=group_output,
                            threshold=args.threshold,
                            data_format=data_format
                        )
                    elif fig_type in ['with_diffusion', 'only_diffusion']:

                        if diffusion_model is None:
                            print(f"错误: 生成 {fig_type} 类型图例需要Diffusion模型，但模型未加载")
                            if num_groups > 1:
                                print(f"      跳过此类型，继续其他类型...")
                                continue
                            else:
                                return
                        generate_comparison_figure_with_diffusion(
                            dit_model=dit_model,
                            ditmamba_model=ditmamba_model,
                            diffusion_model=diffusion_model,
                            sample_names=sample_names,
                            base_path=args.data_path,
                            device=device,
                            output_path=group_output,
                            threshold=args.threshold,
                            data_format=data_format,
                            diffusion_config=diffusion_config,
                            only_diffusion=(fig_type == 'only_diffusion')
                        )
                    group_successful = True
                except Exception as e:
                    print(f"\n错误: 生成第 {group_id} 组 {fig_type} 类型图例时发生异常")
                    print(f"异常信息: {str(e)}")
                    if num_groups > 1:
                        print(f"      跳过此类型，继续其他类型...")
                        continue
                    else:
                        raise
            
            if group_successful:
                successful_groups += 1
                if num_groups > 1:
                    print(f"\n第 {group_id} 组所有类型图例生成完成")


            if num_groups == 1:
                break




        if num_groups > 1:
            print("\n" + "=" * 70)
            print(f"所有组生成完成! 共成功生成 {successful_groups}/{num_groups} 组对比图")
            print("=" * 70)

    except Exception as e:
        import traceback
        print(f"\n错误: 生成对比图时发生异常")
        print(f"异常类型: {type(e).__name__}")
        print(f"异常信息: {str(e)}")
        print("\n详细错误信息:")
        traceback.print_exc()
        return


if __name__ == '__main__':
    main()



