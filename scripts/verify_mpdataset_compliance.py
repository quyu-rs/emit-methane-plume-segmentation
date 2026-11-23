# -*- coding: utf-8 -*-
"""
验证Rows数据集是否符合MPDataset要求
检查ENH归一化、掩码格式、通道合并等关键要求
"""

import sys
from pathlib import Path


project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import torch
from datasets.rows_dataset import RowsDataset
import json

def verify_mpdataset_compliance():
    """验证数据集实现是否符合MPDataset要求"""

    print("="*70)
    print("MPDataset符合性验证")
    print("="*70)


    print("\n1. 创建数据集（enhanced掩码模式）...")
    dataset_enhanced = RowsDataset(
        base_path='data/rows',
        channels=0,
        normalize_enh=True,
        enh_x0=81.15,
        enh_x1=62.18,
        mask_mode='enhanced'
    )

    if len(dataset_enhanced) == 0:
        print("[ERROR] 错误: 数据集为空，无法验证")
        return

    print(f"[OK] 数据集加载成功: {len(dataset_enhanced)} 个样本")


    num_samples_to_check = min(5, len(dataset_enhanced))
    print(f"\n2. 验证前 {num_samples_to_check} 个样本...")

    results = {
        'image_size': [],
        'rgb_format': [],
        'enh_format': [],
        'mask_format': [],
        'channel_merge': [],
        'overall': True
    }

    for idx in range(num_samples_to_check):
        print(f"\n--- 样本 {idx+1}: {dataset_enhanced.samples[idx]['name']} ---")
        sample = dataset_enhanced[idx]

        input_data = sample['input']
        mask = sample['mask']


        h, w = input_data.shape[1], input_data.shape[2]
        if h == 512 and w == 512:
            print(f"[OK] 图像尺寸: {h}x{w} (符合要求)")
            results['image_size'].append(True)
        else:
            print(f"[FAIL] 图像尺寸: {h}x{w} (应为512x512)")
            results['image_size'].append(False)
            results['overall'] = False


        rgb_channels = input_data[:3].numpy()
        rgb_valid = (
            rgb_channels.dtype == np.uint8 or
            (rgb_channels.min() >= 0 and rgb_channels.max() <= 255)
        )
        if rgb_valid:
            print(f"[OK] RGB格式: 3通道, 范围[{rgb_channels.min():.0f}, {rgb_channels.max():.0f}]")
            results['rgb_format'].append(True)
        else:
            print(f"[FAIL] RGB格式: 范围[{rgb_channels.min():.2f}, {rgb_channels.max():.2f}] (应为[0, 255])")
            results['rgb_format'].append(False)
            results['overall'] = False


        enh_channel = input_data[3].numpy()
        enh_min, enh_max = enh_channel.min(), enh_channel.max()
        enh_unique = len(np.unique(enh_channel))

        enh_valid = (
            enh_channel.dtype == np.uint8 or
            (enh_min >= 0 and enh_max <= 255)
        )

        if enh_valid:
            print(f"[OK] ENH格式: 范围[{enh_min:.0f}, {enh_max:.0f}], 唯一值数量: {enh_unique}")
            results['enh_format'].append(True)
        else:
            print(f"[FAIL] ENH格式: 范围[{enh_min:.2f}, {enh_max:.2f}] (应为[0, 255])")
            results['enh_format'].append(False)
            results['overall'] = False


        mask_data = mask[0].numpy()
        mask_min, mask_max = mask_data.min(), mask_data.max()
        mask_unique = len(np.unique(mask_data))
        mask_nonzero = np.count_nonzero(mask_data)


        mask_valid = mask_min >= 0 and mask_max <= 255

        if mask_valid:
            print(f"[OK] 掩码格式: 范围[{mask_min:.0f}, {mask_max:.0f}], 唯一值: {mask_unique}, 非零像素: {mask_nonzero}")
            results['mask_format'].append(True)
        else:
            print(f"[FAIL] 掩码格式: 范围[{mask_min:.2f}, {mask_max:.2f}] (应为[0, 255])")
            results['mask_format'].append(False)
            results['overall'] = False


        if input_data.shape[0] == 4:
            print(f"[OK] 通道合并: 4通道 (RGB 3 + ENH 1)")
            results['channel_merge'].append(True)
        else:
            print(f"[FAIL] 通道合并: {input_data.shape[0]} 通道 (应为4)")
            results['channel_merge'].append(False)
            results['overall'] = False


    print("\n" + "="*70)
    print("验证结果汇总")
    print("="*70)

    checks = {
        '图像尺寸 (512×512)': results['image_size'],
        'RGB格式 (3通道, [0,255])': results['rgb_format'],
        'ENH格式 (归一化到[0,255])': results['enh_format'],
        '掩码格式 ([0,255])': results['mask_format'],
        '通道合并 (4通道)': results['channel_merge']
    }

    for check_name, check_results in checks.items():
        if all(check_results):
            status = "[PASS] 通过"
        else:
            status = "[FAIL] 失败"
            results['overall'] = False
        passed = sum(check_results)
        total = len(check_results)
        print(f"{check_name:30s}: {status} ({passed}/{total})")

    print("\n" + "="*70)
    if results['overall']:
        print("[PASS] 总体评估: 符合MPDataset要求")
    else:
        print("[FAIL] 总体评估: 部分项目不符合要求，请检查上述错误")
    print("="*70)


    print("\n" + "="*70)
    print("掩码模式对比 (binary vs enhanced)")
    print("="*70)

    dataset_binary = RowsDataset(
        base_path='data/rows',
        channels=0,
        normalize_enh=True,
        mask_mode='binary'
    )

    if len(dataset_binary) > 0:
        sample_binary = dataset_binary[0]
        sample_enhanced = dataset_enhanced[0]

        mask_binary = sample_binary['mask'][0].numpy()
        mask_enhanced = sample_enhanced['mask'][0].numpy()

        print(f"\nBinary掩码模式:")
        print(f"  唯一值: {np.unique(mask_binary)}")
        print(f"  非零像素: {np.count_nonzero(mask_binary)}")

        print(f"\nEnhanced掩码模式 (MPDataset标准):")
        unique_vals = np.unique(mask_enhanced)
        print(f"  唯一值数量: {len(unique_vals)}")
        print(f"  唯一值范围: [{unique_vals.min()}, {unique_vals.max()}]")
        print(f"  非零像素: {np.count_nonzero(mask_enhanced)}")

        if len(unique_vals) > 2:
            print(f"  [OK] Enhanced模式保留了增强值（符合MPDataset要求）")
        else:
            print(f"  [WARN] Enhanced模式可能未正确保留增强值")

    return results

if __name__ == '__main__':
    results = verify_mpdataset_compliance()


    if results:
        output_file = project_root / 'mpdataset_compliance_results.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'overall_compliant': results['overall'],
                'checks': {
                    'image_size': all(results['image_size']),
                    'rgb_format': all(results['rgb_format']),
                    'enh_format': all(results['enh_format']),
                    'mask_format': all(results['mask_format']),
                    'channel_merge': all(results['channel_merge'])
                },
                'sample_count': len(results['image_size'])
            }, f, indent=2, ensure_ascii=False)
        print(f"\n验证结果已保存到: {output_file}")



