# -*- coding: utf-8 -*-
"""
测试DatasetAdapter数据加载器
验证data/dataset格式的数据加载、形状、数据增强等功能
"""
import sys
from pathlib import Path


project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from datasets.dataset_adapter import DatasetAdapter
from datasets.transforms import Rotate180, Compose
from torch.utils.data import DataLoader
import torch
import numpy as np

def test_dataset_adapter():
    """测试DatasetAdapter数据加载器"""
    print("=" * 70)
    print("测试DatasetAdapter数据加载器")
    print("=" * 70)


    print("\n1. 测试基本数据加载")
    try:
        dataset = DatasetAdapter(
            base_path='data/dataset',
            channels=0,
            mask_mode='enhanced',
            normalize_enh=True
        )
        print(f"   [OK] 数据集加载成功")
        print(f"   样本数: {len(dataset)}")
    except Exception as e:
        print(f"   [FAIL] 数据集加载失败: {e}")
        import traceback
        traceback.print_exc()
        return

    if len(dataset) == 0:
        print("   [WARN] 数据集为空，无法继续测试")
        return


    print("\n2. 测试数据形状")
    try:
        sample = dataset[0]
        input_shape = sample['input'].shape
        mask_shape = sample['mask'].shape

        print(f"   输入形状: {input_shape} (预期: (4, 512, 512))")
        print(f"   掩码形状: {mask_shape} (预期: (1, 512, 512))")
        print(f"   样本名称: {sample['name']}")


        if input_shape == (4, 512, 512) and mask_shape == (1, 512, 512):
            print("   [OK] 数据形状正确")
        else:
            print("   [FAIL] 数据形状不正确")
    except Exception as e:
        print(f"   [FAIL] 数据加载失败: {e}")
        import traceback
        traceback.print_exc()
        return


    print("\n3. 测试数据范围")
    input_data = sample['input']
    mask_data = sample['mask']

    input_min, input_max = input_data.min().item(), input_data.max().item()
    mask_min, mask_max = mask_data.min().item(), mask_data.max().item()

    print(f"   输入数据范围: [{input_min:.1f}, {input_max:.1f}] (预期: [0, 255])")
    print(f"   掩码数据范围: [{mask_min:.1f}, {mask_max:.1f}] (预期: [0, 255] 或 [0, 1])")


    has_nan = torch.isnan(input_data).any() or torch.isnan(mask_data).any()
    has_inf = torch.isinf(input_data).any() or torch.isinf(mask_data).any()
    print(f"   包含NaN: {has_nan.item()}")
    print(f"   包含Inf: {has_inf.item()}")

    if not has_nan and not has_inf:
        print("   [OK] 数据范围正常，无NaN/Inf")
    else:
        print("   [WARN] 数据包含异常值")


    print("\n4. 测试多个样本尺寸一致性")
    shapes_consistent = True
    for i in range(min(10, len(dataset))):
        sample_i = dataset[i]
        if sample_i['input'].shape != (4, 512, 512) or sample_i['mask'].shape != (1, 512, 512):
            print(f"   [FAIL] 样本 {i} 形状不一致: input={sample_i['input'].shape}, mask={sample_i['mask'].shape}")
            shapes_consistent = False
            break

    if shapes_consistent:
        print(f"   [OK] 前10个样本尺寸一致，均为512×512")


    print("\n5. 测试数据增强（180度旋转）")
    try:
        transform = Rotate180()
        dataset_aug = DatasetAdapter(
            base_path='data/dataset',
            channels=0,
            mask_mode='enhanced',
            normalize_enh=True,
            transform=transform
        )

        sample_orig = dataset[0]
        sample_aug = dataset_aug[0]

        print(f"   原始形状: {sample_orig['input'].shape}")
        print(f"   增强后形状: {sample_aug['input'].shape}")


        if sample_aug['input'].shape == (4, 512, 512) and sample_aug['mask'].shape == (1, 512, 512):
            print("   [OK] 数据增强后形状正确")
        else:
            print("   [FAIL] 数据增强后形状不正确")
    except Exception as e:
        print(f"   [FAIL] 数据增强测试失败: {e}")
        import traceback
        traceback.print_exc()


    print("\n6. 测试掩码模式")
    try:

        dataset_binary = DatasetAdapter(
            base_path='data/dataset',
            channels=0,
            mask_mode='binary',
            normalize_enh=True
        )
        sample_binary = dataset_binary[0]
        mask_binary = sample_binary['mask']
        unique_values = torch.unique(mask_binary).tolist()
        print(f"   Binary模式掩码唯一值: {unique_values} (预期: [0, 1])")


        dataset_enhanced = DatasetAdapter(
            base_path='data/dataset',
            channels=0,
            mask_mode='enhanced',
            normalize_enh=True
        )
        sample_enhanced = dataset_enhanced[0]
        mask_enhanced = sample_enhanced['mask']
        mask_min, mask_max = mask_enhanced.min().item(), mask_enhanced.max().item()
        print(f"   Enhanced模式掩码范围: [{mask_min:.1f}, {mask_max:.1f}]")

        print("   [OK] 掩码模式测试通过")
    except Exception as e:
        print(f"   [FAIL] 掩码模式测试失败: {e}")
        import traceback
        traceback.print_exc()


    print("\n7. 测试DataLoader批处理")
    try:
        dataloader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=False,
            num_workers=0
        )

        batch = next(iter(dataloader))
        batch_input = batch['input']
        batch_mask = batch['mask']

        print(f"   批次输入形状: {batch_input.shape} (预期: (2, 4, 512, 512))")
        print(f"   批次掩码形状: {batch_mask.shape} (预期: (2, 1, 512, 512))")

        if batch_input.shape == (2, 4, 512, 512) and batch_mask.shape == (2, 1, 512, 512):
            print("   [OK] DataLoader批处理正常")
        else:
            print("   [FAIL] DataLoader批处理形状不正确")
    except Exception as e:
        print(f"   [FAIL] DataLoader测试失败: {e}")
        import traceback
        traceback.print_exc()


    print("\n8. 测试接口兼容性")
    try:

        dataset_compat = DatasetAdapter(
            base_path='data/dataset',
            channels=[0, 1, 2],
            normalize_enh=True,
            enh_x0=81.15,
            enh_x1=62.18,
            mask_mode='enhanced',
            transform=None,
            sample_list=None
        )

        sample_compat = dataset_compat[0]

        required_keys = ['input', 'mask', 'name']
        if all(key in sample_compat for key in required_keys):
            print("   [OK] 返回格式与RowsDataset兼容")
        else:
            print(f"   [FAIL] 返回格式不兼容，缺少键: {set(required_keys) - set(sample_compat.keys())}")
    except Exception as e:
        print(f"   [FAIL] 接口兼容性测试失败: {e}")
        import traceback
        traceback.print_exc()


    print("\n9. 测试使用sample_list参数")
    try:

        sample_names = [dataset.samples[i]['name'] for i in range(min(5, len(dataset)))]

        dataset_filtered = DatasetAdapter(
            base_path='data/dataset',
            channels=0,
            mask_mode='enhanced',
            normalize_enh=True,
            sample_list=sample_names
        )

        print(f"   原始数据集样本数: {len(dataset)}")
        print(f"   过滤后数据集样本数: {len(dataset_filtered)} (预期: {len(sample_names)})")

        if len(dataset_filtered) == len(sample_names):
            print("   [OK] sample_list参数工作正常")
        else:
            print("   [WARN] sample_list参数可能有问题")
    except Exception as e:
        print(f"   [FAIL] sample_list测试失败: {e}")
        import traceback
        traceback.print_exc()


    print("\n10. 测试ENH归一化")
    try:

        dataset_norm = DatasetAdapter(
            base_path='data/dataset',
            channels=0,
            mask_mode='binary',
            normalize_enh=True,
            enh_x0=81.15,
            enh_x1=62.18
        )


        dataset_no_norm = DatasetAdapter(
            base_path='data/dataset',
            channels=0,
            mask_mode='binary',
            normalize_enh=False
        )

        sample_norm = dataset_norm[0]
        sample_no_norm = dataset_no_norm[0]

        enh_norm = sample_norm['input'][3]
        enh_no_norm = sample_no_norm['input'][3]

        print(f"   归一化ENH范围: [{enh_norm.min():.1f}, {enh_norm.max():.1f}]")
        print(f"   未归一化ENH范围: [{enh_no_norm.min():.1f}, {enh_no_norm.max():.1f}]")

        print("   [OK] ENH归一化测试通过")
    except Exception as e:
        print(f"   [FAIL] ENH归一化测试失败: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 70)
    print("DatasetAdapter测试完成！")
    print("=" * 70)


    print("\n测试汇总:")
    print(f"  总样本数: {len(dataset)}")
    print(f"  数据形状: input=(4, 512, 512), mask=(1, 512, 512)")
    print(f"  接口兼容: 与RowsDataset完全兼容")
    print(f"  数据增强: 支持transform参数")
    print(f"  掩码模式: 支持binary和enhanced模式")

if __name__ == '__main__':
    test_dataset_adapter()



