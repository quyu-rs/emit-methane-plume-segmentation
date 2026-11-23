# -*- coding: utf-8 -*-
"""
数据集划分脚本
将数据集划分为训练集、验证集和测试集，保持类别平衡
"""
import json
import numpy as np
from pathlib import Path
from collections import defaultdict
import rasterio

def check_sample_label(plm_path: Path) -> bool:
    """
    检查样本是否有正标签（羽流）
    
    参数:
        plm_path: PLM文件路径
    
    返回:
        True表示有正标签，False表示负样本
    """
    try:
        with rasterio.open(plm_path) as src:
            mask = src.read(1)
            return np.any(mask > 0)
    except Exception as e:
        print(f"警告: 无法读取 {plm_path}: {e}")
        return False

def get_all_samples(base_path: str = 'data/rows') -> dict:
    """
    获取所有样本及其标签
    
    参数:
        base_path: 数据集根目录
    
    返回:
        字典，键为样本名称，值为是否为正样本
    """
    base = Path(base_path)
    plm_dir = base / 'plm'


    plm_files = sorted(plm_dir.glob('*_0.tif'))

    samples = {}
    for plm_file in plm_files:

        sample_name = plm_file.stem.rsplit('_', 1)[0]
        is_positive = check_sample_label(plm_file)
        samples[sample_name] = is_positive

    return samples

def split_dataset(
    base_path: str = 'data/rows',
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    output_dir: str = 'datasets',
    seed: int = 42
):
    """
    划分数据集，保持类别平衡
    
    参数:
        base_path: 数据集根目录
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        output_dir: 输出目录
        seed: 随机种子
    """

    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "比例之和必须为1.0"


    np.random.seed(seed)


    print("正在加载样本...")
    all_samples = get_all_samples(base_path)


    pos_samples = [name for name, is_pos in all_samples.items() if is_pos]
    neg_samples = [name for name, is_pos in all_samples.items() if not is_pos]

    print(f"总样本数: {len(all_samples)}")
    print(f"正样本: {len(pos_samples)} ({len(pos_samples)/len(all_samples)*100:.1f}%)")
    print(f"负样本: {len(neg_samples)} ({len(neg_samples)/len(all_samples)*100:.1f}%)")


    np.random.shuffle(pos_samples)
    np.random.shuffle(neg_samples)


    total = len(all_samples)
    n_train = int(total * train_ratio)
    n_val = int(total * val_ratio)
    n_test = total - n_train - n_val


    n_train_pos = int(n_train * len(pos_samples) / total)
    n_train_neg = n_train - n_train_pos

    n_val_pos = int(n_val * len(pos_samples) / total)
    n_val_neg = n_val - n_val_pos

    n_test_pos = len(pos_samples) - n_train_pos - n_val_pos
    n_test_neg = len(neg_samples) - n_train_neg - n_val_neg


    train_samples = pos_samples[:n_train_pos] + neg_samples[:n_train_neg]
    val_samples = pos_samples[n_train_pos:n_train_pos+n_val_pos] + neg_samples[n_train_neg:n_train_neg+n_val_neg]
    test_samples = pos_samples[n_train_pos+n_val_pos:] + neg_samples[n_train_neg+n_val_neg:]


    np.random.shuffle(train_samples)
    np.random.shuffle(val_samples)
    np.random.shuffle(test_samples)


    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    splits = {
        'train': train_samples,
        'val': val_samples,
        'test': test_samples
    }

    for split_name, samples in splits.items():
        filename = output_path / f'{split_name}_split.json'
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(samples, f, indent=2, ensure_ascii=False)


        pos_count = sum(1 for name in samples if all_samples[name])
        neg_count = len(samples) - pos_count

        print(f"\n{split_name}集:")
        print(f"  样本数: {len(samples)}")
        print(f"  正样本: {pos_count} ({pos_count/len(samples)*100:.1f}%)")
        print(f"  负样本: {neg_count} ({neg_count/len(samples)*100:.1f}%)")
        print(f"  保存到: {filename}")

    print("\n数据集划分完成！")

if __name__ == '__main__':
    split_dataset()



