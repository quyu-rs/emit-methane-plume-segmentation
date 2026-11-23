# -*- coding: utf-8 -*-
import os
import numpy as np
import rasterio
from pathlib import Path
from torch.utils.data import Dataset
import torch
from typing import Tuple, Optional, Dict, List, Union

class RowsDataset(Dataset):
    """
    Rows数据集加载器
    支持多通道加载、ENH归一化、NoData处理、通道合并和掩码格式处理
    """

    def __init__(
        self,
        base_path: str = 'data/rows',
        channels: Union[int, List[int]] = 0,
        normalize_enh: bool = True,
        enh_x0: float = 81.15,
        enh_x1: float = 62.18,
        mask_mode: str = 'binary',
        transform=None,
        sample_list: Optional[List[str]] = None
    ):
        """
        参数:
            base_path: 数据集根目录
            channels: 使用的RGB通道编号，可以是单个int (0, 1, 2) 或列表 [0, 1, 2]
            normalize_enh: 是否归一化ENH数据
            enh_x0: ENH归一化参数x0 (MPDataset默认: 81.15)
            enh_x1: ENH归一化参数x1 (MPDataset默认: 62.18)
            mask_mode: 掩码模式
                - 'binary': 二值掩码 (0/1)
                - 'enhanced': 保留增强值的掩码
            transform: 数据增强变换
            sample_list: 可选的样本列表，用于数据集划分
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


        self.samples = self._load_sample_list(sample_list)

        print(f"加载数据集: {len(self.samples)} 个样本")
        print(f"使用通道: {self.channels}")
        print(f"ENH归一化: {normalize_enh}")
        print(f"掩码模式: {mask_mode}")

    def _load_sample_list(self, sample_list: Optional[List[str]] = None):
        """加载样本列表，支持多通道"""
        rgb_dir = self.base_path / 'rgb'
        samples = []


        if sample_list is not None:

            for sample_name in sample_list:
                for channel_id in self.channels:
                    rgb_file = self.base_path / 'rgb' / f"{sample_name}_{channel_id}.tif"
                    enh_file = self.base_path / 'enh' / f"{sample_name}_{channel_id}.tif"
                    plm_file = self.base_path / 'plm' / f"{sample_name}_{channel_id}.tif"

                    if rgb_file.exists() and enh_file.exists() and plm_file.exists():
                        samples.append({
                            'name': f"{sample_name}_{channel_id}",
                            'base_name': sample_name,
                            'channel_id': channel_id,
                            'rgb': rgb_file,
                            'enh': enh_file,
                            'plm': plm_file
                        })
        else:


            all_rgb_files = sorted(rgb_dir.glob('*.tif'))
            unique_samples = set()

            for rgb_file in all_rgb_files:
                stem = rgb_file.stem


                sample_name = stem.rsplit('_', 1)[0]
                unique_samples.add(sample_name)


            for sample_name in sorted(unique_samples):
                for channel_id in self.channels:
                    rgb_file = self.base_path / 'rgb' / f"{sample_name}_{channel_id}.tif"
                    enh_file = self.base_path / 'enh' / f"{sample_name}_{channel_id}.tif"
                    plm_file = self.base_path / 'plm' / f"{sample_name}_{channel_id}.tif"

                    if rgb_file.exists() and enh_file.exists() and plm_file.exists():
                        samples.append({
                            'name': f"{sample_name}_{channel_id}",
                            'base_name': sample_name,
                            'channel_id': channel_id,
                            'rgb': rgb_file,
                            'enh': enh_file,
                            'plm': plm_file
                        })
                    else:
                        print(f"警告: 样本 {sample_name} 通道 {channel_id} 缺少文件")

        return samples

    def _load_rgb(self, rgb_path: Path) -> np.ndarray:
        """加载RGB图像"""
        if not rgb_path.exists():
            raise FileNotFoundError(f"RGB文件不存在: {rgb_path}")

        try:
            with rasterio.open(rgb_path) as src:
                rgb = src.read()
                rgb = np.transpose(rgb, (1, 2, 0))
                rgb = rgb.astype(np.uint8)
            return rgb
        except rasterio.errors.RasterioIOError as e:
            raise RuntimeError(f"无法读取RGB文件（可能已损坏）: {rgb_path} - {e}")
        except PermissionError as e:
            raise PermissionError(f"无权限读取RGB文件: {rgb_path} - {e}")

    def _load_enh(self, enh_path: Path) -> np.ndarray:
        """加载并处理ENH数据"""
        if not enh_path.exists():
            raise FileNotFoundError(f"ENH文件不存在: {enh_path}")

        try:
            with rasterio.open(enh_path) as src:
                enh = src.read(1)
        except rasterio.errors.RasterioIOError as e:
            raise RuntimeError(f"无法读取ENH文件（可能已损坏）: {enh_path} - {e}")
        except PermissionError as e:
            raise PermissionError(f"无权限读取ENH文件: {enh_path} - {e}")


        nodata_value = -9999.0
        valid_mask = enh != nodata_value

        if not np.any(valid_mask):

            enh_processed = np.zeros_like(enh, dtype=np.uint8)
        else:

            if self.normalize_enh:


                enh_normalized = (enh - self.enh_x0) / self.enh_x1 * 255
                enh_normalized = np.clip(enh_normalized, 0, 255)


                enh_normalized[~valid_mask] = 0
                enh_processed = enh_normalized.astype(np.uint8)
            else:

                valid_data = enh[valid_mask]
                if len(valid_data) > 0:
                    enh_min, enh_max = valid_data.min(), valid_data.max()
                    if enh_max > enh_min:
                        enh_normalized = (enh - enh_min) / (enh_max - enh_min) * 255
                    else:
                        enh_normalized = np.zeros_like(enh)
                    enh_normalized[~valid_mask] = 0
                    enh_processed = enh_normalized.astype(np.uint8)
                else:
                    enh_processed = np.zeros_like(enh, dtype=np.uint8)

        return enh_processed

    def _load_mask(self, plm_path: Path, enh_data: Optional[np.ndarray] = None) -> np.ndarray:
        """加载并处理掩码"""
        if not plm_path.exists():
            raise FileNotFoundError(f"掩码文件不存在: {plm_path}")

        try:
            with rasterio.open(plm_path) as src:
                mask = src.read(1)
        except rasterio.errors.RasterioIOError as e:
            raise RuntimeError(f"无法读取掩码文件（可能已损坏）: {plm_path} - {e}")
        except PermissionError as e:
            raise PermissionError(f"无权限读取掩码文件: {plm_path} - {e}")

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

    def _merge_channels(self, rgb: np.ndarray, enh: np.ndarray) -> np.ndarray:
        """合并RGB和ENH为4通道输入"""

        enh_expanded = np.expand_dims(enh, axis=2)


        merged = np.concatenate([rgb, enh_expanded], axis=2)

        return merged

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]


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
            mask = self._load_mask(sample['plm'], enh_data=enh if self.mask_mode == 'enhanced' else None)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"掩码文件不存在: {sample['plm']} - {e}")
        except Exception as e:
            raise RuntimeError(f"加载掩码文件失败: {sample['plm']} - {e}")


        input_data = self._merge_channels(rgb, enh)


        input_data = np.transpose(input_data, (2, 0, 1))
        mask = np.expand_dims(mask, axis=0)



        input_tensor = torch.from_numpy(input_data).float()



        mask_normalized = (mask.astype(np.float32) / 255.0).clip(0.0, 1.0)
        mask_tensor = torch.from_numpy(mask_normalized).float()


        if self.transform:

            input_tensor, mask_tensor = self.transform(input_tensor, mask_tensor)

        return {
            'input': input_tensor,
            'mask': mask_tensor,
            'name': sample['name']
        }



