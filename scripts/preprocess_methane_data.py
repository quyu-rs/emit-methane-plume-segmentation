# -*- coding: utf-8 -*-
"""
Methane_DATA 数据预处理脚本
将 Methane_DATA 转换为可用的格式（统一尺寸、处理NoData、转换目录结构）
"""

import argparse
from pathlib import Path
from collections import defaultdict
import numpy as np
from PIL import Image
from tqdm import tqdm
try:
    import rasterio
    from rasterio.transform import from_bounds
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False
    print("错误: 需要安装 rasterio 库来读取 .tif 文件")
    print("安装命令: pip install rasterio")
    exit(1)


def extract_base_name(filename: str) -> str:
    """
    从文件名中提取基础名称
    
    例如：
    - ang20200708t200101-1_l3a-ch4-mf-v1_concentrations.tif -> ang20200708t200101-1
    """
    name = filename.replace('.tif', '')
    parts = name.split('_')
    if len(parts) >= 1:
        base = parts[0]
        return base
    return name


def resize_array(array: np.ndarray, target_shape: tuple, method: str = 'bilinear') -> np.ndarray:
    """
    调整数组尺寸
    
    参数:
        array: 输入数组
        target_shape: 目标尺寸 (height, width)
        method: 插值方法 ('bilinear', 'nearest')
    """
    if array.shape[:2] == target_shape:
        return array
    
    if len(array.shape) == 2:
        # 单通道
        img = Image.fromarray(array)
        if method == 'nearest':
            resized = img.resize((target_shape[1], target_shape[0]), Image.NEAREST)
        else:
            resized = img.resize((target_shape[1], target_shape[0]), Image.LANCZOS)
        return np.array(resized, dtype=array.dtype)
    elif len(array.shape) == 3:
        # 多通道
        channels = []
        for i in range(array.shape[2]):
            channel = array[:, :, i]
            img = Image.fromarray(channel)
            if method == 'nearest':
                resized = img.resize((target_shape[1], target_shape[0]), Image.NEAREST)
            else:
                resized = img.resize((target_shape[1], target_shape[0]), Image.LANCZOS)
            channels.append(np.array(resized, dtype=array.dtype))
        return np.stack(channels, axis=2)
    else:
        raise ValueError(f"不支持的数组维度: {array.shape}")


def process_concentrations(conc_path: Path, target_shape: tuple) -> np.ndarray:
    """
    处理 concentrations 数据：读取、处理NoData、调整尺寸
    
    参数:
        conc_path: concentrations 文件路径
        target_shape: 目标尺寸 (height, width)
    
    返回:
        处理后的数组（float32，已处理NoData）
    """
    with rasterio.open(conc_path) as src:
        conc_data = src.read(1).astype(np.float32)
        
        # 处理 NoData 值
        nodata_value = src.nodata if src.nodata is not None else -9999.0
        valid_mask = (conc_data != nodata_value) & (conc_data > 0)
        
        # 将 NoData 区域设为 0
        conc_data[~valid_mask] = 0.0
        
        # 调整尺寸
        if conc_data.shape != target_shape:
            conc_data = resize_array(conc_data, target_shape, method='bilinear')
        
        return conc_data


def process_rgb(rgb_path: Path, target_shape: tuple = None) -> tuple:
    """
    处理 RGB 数据：读取、调整尺寸
    
    参数:
        rgb_path: RGB 文件路径
        target_shape: 目标尺寸 (height, width)，如果为None则保持原尺寸
    
    返回:
        (rgb_array, actual_shape) - RGB数组和实际尺寸
    """
    with rasterio.open(rgb_path) as src:
        rgb_data = src.read()
        rgb_array = np.transpose(rgb_data, (1, 2, 0))
        
        # 确保是3通道
        if rgb_array.shape[2] > 3:
            rgb_array = rgb_array[:, :, :3]
        elif rgb_array.shape[2] == 1:
            rgb_array = np.repeat(rgb_array, 3, axis=2)
        
        actual_shape = rgb_array.shape[:2]
        
        # 调整尺寸（如果需要）
        if target_shape is not None and actual_shape != target_shape:
            rgb_array = resize_array(rgb_array, target_shape, method='bilinear')
            actual_shape = target_shape
        
        # 确保值在 0-255 范围内
        rgb_array = np.clip(rgb_array, 0, 255).astype(np.uint8)
        
        return rgb_array, actual_shape


def process_plume(plume_path: Path, target_shape: tuple) -> np.ndarray:
    """
    处理 plume 数据：读取、转换为二值掩码、调整尺寸
    
    参数:
        plume_path: plume 文件路径
        target_shape: 目标尺寸 (height, width)
    
    返回:
        处理后的二值掩码数组（uint8，0或1）
    """
    with rasterio.open(plume_path) as src:
        plume_data = src.read(1)
        
        # 转换为二值掩码
        plume_binary = (plume_data > 0).astype(np.uint8)
        
        # 调整尺寸
        if plume_binary.shape != target_shape:
            plume_binary = resize_array(plume_binary, target_shape, method='nearest')
        
        return plume_binary


def save_tif_with_metadata(array: np.ndarray, output_path: Path, 
                           nodata: float = None, dtype: str = None):
    """
    保存数组为 TIF 文件，保留元数据
    
    参数:
        array: 要保存的数组
        output_path: 输出路径
        nodata: NoData 值
        dtype: 数据类型（如果为None则自动推断）
    """
    if dtype is None:
        dtype = array.dtype
    
    array = array.astype(dtype)
    
    if len(array.shape) == 2:
        # 单通道
        height, width = array.shape
        count = 1
        array_reshaped = array[np.newaxis, :, :]
    elif len(array.shape) == 3:
        # 多通道
        height, width = array.shape[:2]
        count = array.shape[2]
        array_reshaped = np.transpose(array, (2, 0, 1))
    else:
        raise ValueError(f"不支持的数组维度: {array.shape}")
    
    # 创建简单的变换矩阵
    transform = from_bounds(0, 0, width, height, width, height)
    
    with rasterio.open(
        output_path,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=count,
        dtype=dtype,
        crs=rasterio.crs.CRS.from_epsg(4326),
        transform=transform,
        nodata=nodata,
        compress='lzw'
    ) as dst:
        dst.write(array_reshaped)


def preprocess_methane_data(
    source_path: str = 'data/Methane_DATA',
    output_path: str = 'data/methane_processed',
    target_size: tuple = None,
    use_rgb_size: bool = True,
    min_data_completeness: float = 0.2,
    check_all_years: bool = True
) -> dict:
    """
    预处理 Methane_DATA 数据集
    
    参数:
        source_path: 源数据目录
        output_path: 输出目录
        target_size: 目标尺寸 (height, width)，如果为None则使用RGB尺寸
        use_rgb_size: 是否使用RGB尺寸作为目标尺寸
        min_data_completeness: 最小数据完整性阈值（低于此值的样本将被跳过）
        check_all_years: 是否处理所有年份
    
    返回:
        处理统计信息
    """
    print("=" * 70)
    print("Methane_DATA 数据预处理")
    print("=" * 70)
    
    source = Path(source_path)
    output = Path(output_path)
    
    if not source.exists():
        print(f"错误: 源目录不存在: {source_path}")
        return {}
    
    # 创建输出目录结构
    output.mkdir(parents=True, exist_ok=True)
    enh_dir = output / 'enh'
    rgb_dir = output / 'rgb'
    plm_dir = output / 'plm'
    enh_dir.mkdir(exist_ok=True)
    rgb_dir.mkdir(exist_ok=True)
    plm_dir.mkdir(exist_ok=True)
    
    # 收集所有年份
    years = []
    if check_all_years:
        for year_dir in sorted(source.iterdir()):
            if year_dir.is_dir() and year_dir.name.isdigit():
                years.append(year_dir.name)
    else:
        for year_dir in sorted(source.iterdir()):
            if year_dir.is_dir() and year_dir.name.isdigit():
                years.append(year_dir.name)
                break
    
    print(f"\n处理年份: {', '.join(years)}")
    
    # 收集所有样本
    all_samples = {}
    
    for year in years:
        year_path = source / year
        for source_dir in sorted(year_path.iterdir()):
            if not source_dir.is_dir():
                continue
            
            source_name = source_dir.name
            for split_dir in ['train', 'val', 'test']:
                split_path = source_dir / split_dir
                if not split_path.exists():
                    continue
                
                concentrations_dir = split_path / 'concentrations'
                rgb_dir_source = split_path / 'rgb'
                plume_dir_source = split_path / 'plume'
                
                # 收集文件
                if concentrations_dir.exists() and rgb_dir_source.exists() and plume_dir_source.exists():
                    for conc_file in concentrations_dir.glob('*.tif'):
                        base_name = extract_base_name(conc_file.name)
                        
                        # 查找对应的 RGB 和 Plume 文件
                        rgb_file = None
                        plume_file = None
                        
                        for rgb_f in rgb_dir_source.glob('*.tif'):
                            if extract_base_name(rgb_f.name) == base_name:
                                rgb_file = rgb_f
                                break
                        
                        for plume_f in plume_dir_source.glob('*.tif'):
                            if extract_base_name(plume_f.name) == base_name:
                                plume_file = plume_f
                                break
                        
                        if rgb_file and plume_file:
                            key = f"{year}/{source_name}/{split_dir}/{base_name}"
                            all_samples[key] = {
                                'year': year,
                                'source': source_name,
                                'split': split_dir,
                                'base_name': base_name,
                                'concentrations': conc_file,
                                'rgb': rgb_file,
                                'plume': plume_file
                            }
    
    print(f"\n找到 {len(all_samples)} 个完整样本")
    
    # 统计信息
    stats = {
        'total_samples': len(all_samples),
        'processed_samples': 0,
        'skipped_samples': 0,
        'skipped_reasons': defaultdict(int),
        'size_statistics': defaultdict(int)
    }
    
    # 处理每个样本
    print(f"\n开始处理样本...")
    for key, sample in tqdm(all_samples.items(), desc="处理进度"):
        try:
            # 1. 读取 RGB 确定目标尺寸
            rgb_array, rgb_shape = process_rgb(sample['rgb'], target_shape=None)
            
            if target_size is not None:
                target_shape = target_size
            elif use_rgb_size:
                target_shape = rgb_shape
            else:
                # 使用 concentrations 尺寸
                with rasterio.open(sample['concentrations']) as src:
                    conc_shape = src.read(1).shape
                target_shape = conc_shape
            
            # 2. 检查数据完整性
            with rasterio.open(sample['concentrations']) as src:
                conc_data = src.read(1)
                nodata_value = src.nodata if src.nodata is not None else -9999.0
                valid_mask = (conc_data != nodata_value) & (conc_data > 0)
                data_completeness = float(valid_mask.sum() / valid_mask.size) if valid_mask.size > 0 else 0.0
            
            if data_completeness < min_data_completeness:
                stats['skipped_samples'] += 1
                stats['skipped_reasons']['数据完整性不足'] += 1
                continue
            
            # 3. 处理 concentrations
            conc_processed = process_concentrations(sample['concentrations'], target_shape)
            
            # 4. 处理 RGB（确保尺寸一致）
            rgb_processed, _ = process_rgb(sample['rgb'], target_shape=target_shape)
            
            # 5. 处理 Plume
            plume_processed = process_plume(sample['plume'], target_shape)
            
            # 6. 保存文件
            # 使用 base_name_0.tif 格式（符合 dataset_convert 格式）
            output_base_name = f"{sample['base_name']}_0"
            
            enh_output = enh_dir / f"{output_base_name}.tif"
            rgb_output = rgb_dir / f"{output_base_name}.tif"
            plm_output = plm_dir / f"{output_base_name}.tif"
            
            # 保存 concentrations（作为 enh）
            save_tif_with_metadata(conc_processed, enh_output, nodata=-9999.0, dtype='float32')
            
            # 保存 RGB
            save_tif_with_metadata(rgb_processed, rgb_output, dtype='uint8')
            
            # 保存 Plume（作为 plm）
            save_tif_with_metadata(plume_processed, plm_output, dtype='uint8')
            
            stats['processed_samples'] += 1
            stats['size_statistics'][str(target_shape)] += 1
            
        except Exception as e:
            stats['skipped_samples'] += 1
            stats['skipped_reasons'][f'处理错误: {str(e)[:50]}'] += 1
            print(f"\n警告: 处理样本 {key} 时出错: {e}")
            continue
    
    # 打印统计信息
    print(f"\n" + "=" * 70)
    print("处理完成统计")
    print("=" * 70)
    print(f"总样本数: {stats['total_samples']}")
    print(f"成功处理: {stats['processed_samples']}")
    print(f"跳过样本: {stats['skipped_samples']}")
    
    if stats['skipped_reasons']:
        print(f"\n跳过原因:")
        for reason, count in stats['skipped_reasons'].items():
            print(f"  {reason}: {count}")
    
    if stats['size_statistics']:
        print(f"\n尺寸分布:")
        for size, count in sorted(stats['size_statistics'].items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {size}: {count} 个样本")
    
    print(f"\n输出目录: {output_path}")
    print(f"  - enh/: {stats['processed_samples']} 个文件")
    print(f"  - rgb/: {stats['processed_samples']} 个文件")
    print(f"  - plm/: {stats['processed_samples']} 个文件")
    
    return stats


def main() -> None:
    """命令行入口"""
    parser = argparse.ArgumentParser(description='Methane_DATA 数据预处理')
    parser.add_argument(
        '--source_path',
        type=str,
        default='data/Methane_DATA',
        help='源数据目录路径',
    )
    parser.add_argument(
        '--output_path',
        type=str,
        default='data/methane_processed',
        help='输出目录路径',
    )
    parser.add_argument(
        '--target_size',
        type=str,
        default=None,
        help='目标尺寸，格式: height,width (例如: 512,512)。如果未指定，将使用RGB尺寸',
    )
    parser.add_argument(
        '--use_rgb_size',
        action='store_true',
        default=True,
        help='使用RGB尺寸作为目标尺寸（默认启用）',
    )
    parser.add_argument(
        '--min_data_completeness',
        type=float,
        default=0.2,
        help='最小数据完整性阈值（0-1），低于此值的样本将被跳过（默认: 0.2）',
    )
    parser.add_argument(
        '--check_all_years',
        action='store_true',
        default=True,
        help='处理所有年份（默认启用）',
    )
    
    args = parser.parse_args()
    
    # 解析目标尺寸
    target_size = None
    if args.target_size:
        try:
            parts = args.target_size.split(',')
            target_size = (int(parts[0]), int(parts[1]))
        except:
            print(f"警告: 无法解析目标尺寸 {args.target_size}，将使用RGB尺寸")
            target_size = None
    
    stats = preprocess_methane_data(
        source_path=args.source_path,
        output_path=args.output_path,
        target_size=target_size,
        use_rgb_size=args.use_rgb_size,
        min_data_completeness=args.min_data_completeness,
        check_all_years=args.check_all_years
    )
    
    print("=" * 70)


if __name__ == '__main__':
    main()

