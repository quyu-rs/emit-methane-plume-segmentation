# -*- coding: utf-8 -*-
import os
import copy
import numpy as np
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset
import torch
from typing import Tuple, Optional, Dict, List, Union

class DatasetAdapter(Dataset):
    """
    数据集适配器
    支持 data/dataset 格式的数据加载（.jpg/.png文件，mask/目录，数字编号命名）
    保持与 RowsDataset 完全相同的接口
    """

    def __init__(
        self,
        base_path: str = 'data/dataset',
        channels: Union[int, List[int]] = 0,
        normalize_enh: bool = True,
        enh_x0: float = 81.15,
        enh_x1: float = 62.18,
        mask_mode: str = 'binary',
        transform=None,
        sample_list: Optional[List[str]] = None,
        target_size: Tuple[int, int] = (512, 512),
        cache_enabled: bool = False
    ):
        """
        参数:
            base_path: 数据集根目录
            channels: 使用的RGB通道编号（为接口兼容性保留，data/dataset格式不使用）
            normalize_enh: 是否归一化ENH数据
            enh_x0: ENH归一化参数x0 (MPDataset默认: 81.15)
            enh_x1: ENH归一化参数x1 (MPDataset默认: 62.18)
            mask_mode: 掩码模式
                - 'binary': 二值掩码 (0/1)
                - 'enhanced': 保留增强值的掩码
            transform: 数据增强变换
            sample_list: 可选的样本列表，用于数据集划分
            target_size: 目标图像尺寸，默认(512, 512)
            cache_enabled: 是否启用数据缓存（需要大量内存约17GB，默认关闭）
                注意：
                - 如果启用transform（数据增强），缓存将不会生效
                - 当使用多进程DataLoader（num_workers > 0）时，每个worker进程会创建独立的缓存，
                  导致内存浪费（每个worker缓存相同数据）。建议启用缓存时使用num_workers=0
        """
        self.base_path = Path(base_path)


        if isinstance(channels, int):
            self.channels = [channels]
        elif isinstance(channels, list):
            self.channels = channels
        else:
            raise ValueError(f"channels必须是int或list，得到: {type(channels)}")

        self.normalize_enh = normalize_enh
        self.enh_x0 = enh_x0
        self.enh_x1 = enh_x1
        self.mask_mode = mask_mode
        self.transform = transform
        self.target_size = target_size
        self.cache_enabled = cache_enabled


        self.cache = {} if cache_enabled else None


        self.samples = self._load_sample_list(sample_list)

        print(f"加载数据集: {len(self.samples)} 个样本")
        print(f"使用通道: {self.channels} (注意: data/dataset格式不使用通道概念)")
        print(f"ENH归一化: {normalize_enh}")
        print(f"掩码模式: {mask_mode}")
        print(f"目标尺寸: {target_size}")

    def _load_sample_list(self, sample_list: Optional[List[str]] = None):
        """加载样本列表，支持数字编号命名格式"""
        rgb_dir = self.base_path / 'rgb'
        mask_dir = self.base_path / 'mask'
        enh_dir = self.base_path / 'enh'

        samples = []


        if sample_list is not None:

            for sample_name in sample_list:


                if '.' in sample_name:
                    file_stem = Path(sample_name).stem
                else:
                    file_stem = sample_name

                rgb_file = rgb_dir / f"{file_stem}.jpg"
                enh_file = enh_dir / f"{file_stem}.png"
                mask_file = mask_dir / f"{file_stem}.png"

                if rgb_file.exists() and enh_file.exists() and mask_file.exists():
                    samples.append({
                        'name': file_stem,
                        'base_name': file_stem,
                        'channel_id': 0,
                        'rgb': rgb_file,
                        'enh': enh_file,
                        'mask': mask_file
                    })
        else:


            all_rgb_files = sorted(rgb_dir.glob('*.jpg'))

            for rgb_file in all_rgb_files:
                file_stem = rgb_file.stem
                enh_file = enh_dir / f"{file_stem}.png"
                mask_file = mask_dir / f"{file_stem}.png"

                if enh_file.exists() and mask_file.exists():
                    samples.append({
                        'name': file_stem,
                        'base_name': file_stem,
                        'channel_id': 0,
                        'rgb': rgb_file,
                        'enh': enh_file,
                        'mask': mask_file
                    })
                else:
                    print(f"警告: 样本 {file_stem} 缺少对应文件 (ENH: {enh_file.exists()}, Mask: {mask_file.exists()})")

        return samples

    def _load_rgb(self, rgb_path: Path) -> np.ndarray:
        """加载RGB图像，使用PIL读取并resize到目标尺寸"""
        if not rgb_path.exists():
            raise FileNotFoundError(f"RGB文件不存在: {rgb_path}")

        try:

            img = Image.open(rgb_path)


            if img.mode != 'RGB':
                img = img.convert('RGB')


            img = img.resize(self.target_size, Image.LANCZOS)


            rgb = np.array(img, dtype=np.uint8)

            return rgb
        except Exception as e:
            raise RuntimeError(f"无法读取RGB文件（可能已损坏）: {rgb_path} - {e}")

    def _load_enh(self, enh_path: Path) -> np.ndarray:
        """加载并处理ENH数据，使用PIL读取并resize到目标尺寸"""
        if not enh_path.exists():
            raise FileNotFoundError(f"ENH文件不存在: {enh_path}")

        try:

            img = Image.open(enh_path)


            if img.mode != 'L':
                img = img.convert('L')


            img = img.resize(self.target_size, Image.LANCZOS)


            enh = np.array(img, dtype=np.float32)


            if self.normalize_enh:

                enh_normalized = (enh - self.enh_x0) / self.enh_x1 * 255
                enh_normalized = np.clip(enh_normalized, 0, 255)
                enh_processed = enh_normalized.astype(np.uint8)
            else:

                enh_processed = np.clip(enh, 0, 255).astype(np.uint8)

            return enh_processed
        except Exception as e:
            raise RuntimeError(f"无法读取ENH文件（可能已损坏）: {enh_path} - {e}")

    def _load_mask(self, mask_path: Path, enh_data: Optional[np.ndarray] = None) -> np.ndarray:
        """加载并处理掩码，使用PIL读取并resize到目标尺寸"""
        if not mask_path.exists():
            raise FileNotFoundError(f"掩码文件不存在: {mask_path}")

        try:

            img = Image.open(mask_path)


            if img.mode != 'L':
                img = img.convert('L')


            img = img.resize(self.target_size, Image.LANCZOS)


            mask = np.array(img, dtype=np.float32)


            if self.mask_mode == 'binary':

                mask_processed = (mask > 0).astype(np.uint8)
            elif self.mask_mode == 'enhanced':

                if enh_data is not None:
                    mask_processed = np.where(mask > 0, enh_data, 0).astype(np.uint8)
                else:

                    mask_processed = (mask > 0).astype(np.uint8)
            else:
                raise ValueError(f"未知的掩码模式: {self.mask_mode}")

            return mask_processed
        except Exception as e:
            raise RuntimeError(f"无法读取掩码文件（可能已损坏）: {mask_path} - {e}")

    def _merge_channels(self, rgb: np.ndarray, enh: np.ndarray) -> np.ndarray:
        """合并RGB和ENH为4通道输入"""

        enh_expanded = np.expand_dims(enh, axis=2)


        merged = np.concatenate([rgb, enh_expanded], axis=2)

        return merged

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        sample_name = sample['name']


        if self.cache_enabled and self.cache is not None and self.transform is None:
            if sample_name in self.cache:
                return self.cache[sample_name]


        try:
            rgb = self._load_rgb(sample['rgb'])
        except FileNotFoundError as e:
            raise FileNotFoundError(f"RGB文件不存在: {sample['rgb']} - {e}")
        except Exception as e:
            raise RuntimeError(f"加载RGB文件失败: {sample['rgb']} - {e}")

        try:
            enh = self._load_enh(sample['enh'])
        except FileNotFoundError as e:
            raise FileNotFoundError(f"ENH文件不存在: {sample['enh']} - {e}")
        except Exception as e:
            raise RuntimeError(f"加载ENH文件失败: {sample['enh']} - {e}")

        try:
            mask = self._load_mask(sample['mask'], enh_data=enh if self.mask_mode == 'enhanced' else None)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"掩码文件不存在: {sample['mask']} - {e}")
        except Exception as e:
            raise RuntimeError(f"加载掩码文件失败: {sample['mask']} - {e}")


        input_data = self._merge_channels(rgb, enh)


        input_data = np.transpose(input_data, (2, 0, 1))
        mask = np.expand_dims(mask, axis=0)



        input_tensor = torch.from_numpy(input_data).float()



        mask_normalized = (mask.astype(np.float32) / 255.0).clip(0.0, 1.0)
        mask_tensor = torch.from_numpy(mask_normalized).float()


        if self.transform:

            input_tensor, mask_tensor = self.transform(input_tensor, mask_tensor)

        result = {
            'input': input_tensor,
            'mask': mask_tensor,
            'name': sample_name
        }



        if self.cache_enabled and self.cache is not None and self.transform is None:
            self.cache[sample_name] = copy.deepcopy(result)

        return result



