# -*- coding: utf-8 -*-
"""
测试数据加载器
验证多通道数据加载和数据增强
"""
import sys
from pathlib import Path


project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from datasets.rows_dataset import RowsDataset
from datasets.transforms import Rotate180, Compose
from torch.utils.data import DataLoader
import torch

def test_dataloader():
    """测试数据加载器"""
    print("=" * 70)
    print("测试数据加载器")
    print("=" * 70)


    print("\n1. 测试单通道加载 (channel=0)")
    dataset_single = RowsDataset(
        base_path='data/rows',
        channels=0,
        mask_mode='enhanced'
    )
    print(f"   样本数: {len(dataset_single)}")


    print("\n2. 测试多通道加载 (channels=[0,1,2])")
    dataset_multi = RowsDataset(
        base_path='data/rows',
        channels=[0, 1, 2],
        mask_mode='enhanced'
    )
    print(f"   样本数: {len(dataset_multi)} (预期: {len(dataset_single) * 3})")


    print("\n3. 测试数据加载")
    sample = dataset_multi[0]
    print(f"   输入形状: {sample['input'].shape} (预期: (4, 512, 512))")
    print(f"   掩码形状: {sample['mask'].shape} (预期: (1, 512, 512))")
    print(f"   样本名称: {sample['name']}")


    input_data = sample['input']
    print(f"   输入数据范围: [{input_data.min():.1f}, {input_data.max():.1f}] (预期: [0, 255])")


    has_nan = torch.isnan(input_data).any()
    has_inf = torch.isinf(input_data).any()
    print(f"   包含NaN: {has_nan.item()}")
    print(f"   包含Inf: {has_inf.item()}")


    print("\n4. 测试数据增强（180度旋转）")
    transform = Rotate180()
    dataset_aug = RowsDataset(
        base_path='data/rows',
        channels=[0, 1, 2],
        mask_mode='enhanced',
        transform=transform
    )
    sample_orig = dataset_multi[0]
    sample_aug = dataset_aug[0]


    print(f"   原始形状: {sample_orig['input'].shape}")
    print(f"   增强后形状: {sample_aug['input'].shape}")


    print("\n5. 测试使用数据集划分")
    import json
    with open('datasets/train_split.json', 'r', encoding='utf-8') as f:
        train_samples = json.load(f)

    dataset_split = RowsDataset(
        base_path='data/rows',
        channels=[0, 1, 2],
        mask_mode='enhanced',
        sample_list=train_samples
    )
    print(f"   训练集样本数: {len(dataset_split)} (预期: {len(train_samples) * 3})")


    print("\n6. 测试DataLoader")
    dataloader = DataLoader(
        dataset_multi,
        batch_size=2,
        shuffle=False,
        num_workers=0
    )

    batch = next(iter(dataloader))
    print(f"   批次输入形状: {batch['input'].shape} (预期: (2, 4, 512, 512))")
    print(f"   批次掩码形状: {batch['mask'].shape} (预期: (2, 1, 512, 512))")


    print("\n7. 计算扩展后样本数")

    base_samples = len(dataset_multi)

    augmented_samples = base_samples * 2
    print(f"   基础样本数（多通道）: {base_samples}")
    print(f"   扩展后样本数（+旋转）: {augmented_samples}")
    print(f"   预期样本数: 480")

    print("\n" + "=" * 70)
    print("数据加载器测试完成！")
    print("=" * 70)

if __name__ == '__main__':
    test_dataloader()



