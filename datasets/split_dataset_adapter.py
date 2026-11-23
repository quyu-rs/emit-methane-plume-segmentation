# -*- coding: utf-8 -*-
"""
数据集划分脚本（适配DatasetAdapter格式）
将data/dataset格式的数据集划分为训练集、验证集和测试集，保持类别平衡
"""
import json
import numpy as np
from pathlib import Path
from PIL import Image

def check_sample_label(mask_path: Path) -> bool:
    """
    检查样本是否有正标签（羽流）
    
    参数:
        mask_path: Mask文件路径（.png格式）
    
    返回:
        True表示有正标签，False表示负样本
    """
    try:
        img = Image.open(mask_path)

        if img.mode != 'L':
            img = img.convert('L')
        mask = np.array(img)
        return np.any(mask > 0)
    except Exception as e:
        print(f"警告: 无法读取 {mask_path}: {e}")
        return False

def get_all_samples(base_path: str = 'data/dataset') -> dict:
    """
    获取所有样本及其标签
    
    参数:
        base_path: 数据集根目录
    
    返回:
        字典，键为样本名称（数字编号），值为是否为正样本
    """
    base = Path(base_path)
    mask_dir = base / 'mask'

    if not mask_dir.exists():
        raise FileNotFoundError(f"Mask目录不存在: {mask_dir}")


    mask_files = sorted(mask_dir.glob('*.png'))

    if len(mask_files) == 0:
        raise ValueError(f"在 {mask_dir} 中未找到任何 .png 文件")

    samples = {}
    print(f"正在检查 {len(mask_files)} 个样本的标签...")

    for idx, mask_file in enumerate(mask_files):
        if (idx + 1) % 500 == 0:
            print(f"  处理进度: {idx + 1}/{len(mask_files)}")


        sample_name = mask_file.stem
        is_positive = check_sample_label(mask_file)
        samples[sample_name] = is_positive

    return samples

def split_dataset_adapter(
    base_path: str = 'data/dataset',
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    output_dir: str = 'datasets',
    seed: int = 42,
    output_prefix: str = 'dataset_adapter'
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
        output_prefix: 输出文件名前缀
    """

    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "比例之和必须为1.0"


    np.random.seed(seed)


    print("=" * 70)
    print("数据集划分（DatasetAdapter格式）")
    print("=" * 70)
    print("正在加载样本...")
    all_samples = get_all_samples(base_path)


    pos_samples = [name for name, is_pos in all_samples.items() if is_pos]
    neg_samples = [name for name, is_pos in all_samples.items() if not is_pos]

    print(f"\n总样本数: {len(all_samples)}")
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
        filename = output_path / f'{output_prefix}_{split_name}_split.json'
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(samples, f, indent=2, ensure_ascii=False)


        pos_count = sum(1 for name in samples if all_samples[name])
        neg_count = len(samples) - pos_count

        print(f"\n{split_name}集:")
        print(f"  样本数: {len(samples)}")
        print(f"  正样本: {pos_count} ({pos_count/len(samples)*100:.1f}%)")
        print(f"  负样本: {neg_count} ({neg_count/len(samples)*100:.1f}%)")
        print(f"  保存到: {filename}")

    print("\n" + "=" * 70)
    print("数据集划分完成！")
    print("=" * 70)

    return splits

if __name__ == '__main__':
    split_dataset_adapter()



