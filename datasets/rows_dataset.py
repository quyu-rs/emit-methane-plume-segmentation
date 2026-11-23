# -*- coding: utf-8 -*-
import numpy as np
import rasterio
from pathlib import Path
from torch.utils.data import Dataset
import torch
from typing import Optional, List, Union


class RowsDataset(Dataset):
    """
    Rows 数据集加载器
    支持模态组合:
      1) RGB only
      2) SWIR only
      3) SWIR + RGB
      4) SWIR + ENH
      5) RGB + ENH
      6) SWIR + RGB + ENH
      7) ENH only
      

    说明:
    - channels 参数保留原有风格，但它表示文件后缀编号，如 0 / [0,1,2]，
      不是 SWIR 的 41 个光谱通道
    - sample_list 应传入 split json 中的 base_name 列表
    """

    def __init__(
        self,
        base_path: str = 'data/rows',
        channels: Union[int, List[int]] = 0,   # 文件后缀 id，如 0 / [0,1,2]
        use_rgb: bool = True,
        use_swir: bool = True,
        use_enh: bool = True,
        normalize_enh: bool = True,
        enh_x0: float = 81.15,
        enh_x1: float = 62.18,
        normalize_swir: bool = True,
        swir_percentile_low: float = 2.0,
        swir_percentile_high: float = 98.0,
        mask_mode: str = 'binary',
        transform=None,
        sample_list: Optional[List[str]] = None
    ):
        self.base_path = Path(base_path)

        if isinstance(channels, int):
            self.channels = [channels]
        elif isinstance(channels, list):
            self.channels = channels
        else:
            raise ValueError(f"channels 必须是 int 或 list，得到: {type(channels)}")

        self.use_rgb = use_rgb
        self.use_swir = use_swir
        self.use_enh = use_enh

        if not (self.use_rgb or self.use_swir or self.use_enh):
            raise ValueError("至少需要启用一个模态: use_rgb/use_swir/use_enh")

        if mask_mode == 'enhanced' and not self.use_enh:
            raise ValueError("mask_mode='enhanced' 时必须 use_enh=True")

        self.normalize_enh = normalize_enh
        self.enh_x0 = enh_x0
        self.enh_x1 = enh_x1

        self.normalize_swir = normalize_swir
        self.swir_percentile_low = swir_percentile_low
        self.swir_percentile_high = swir_percentile_high

        self.mask_mode = mask_mode
        self.transform = transform

        self.samples = self._load_sample_list(sample_list)

        print("=" * 60)
        print(f"加载数据集: {len(self.samples)} 个样本")
        print(f"文件后缀 channels: {self.channels}")
        print(f"模态设置: RGB={self.use_rgb}, SWIR={self.use_swir}, ENH={self.use_enh}")
        print(f"ENH归一化: {self.normalize_enh}")
        print(f"SWIR归一化: {self.normalize_swir}")
        print(f"掩码模式: {self.mask_mode}")
        print("=" * 60)

    def _resolve_tif(self, folder: str, sample_name: str, channel_id: int) -> Optional[Path]:
        """兼容 .tif / .TIF"""
        p = self.base_path / folder / f"{sample_name}_{channel_id}.tif"
        if p.exists():
            return p
        p2 = self.base_path / folder / f"{sample_name}_{channel_id}.TIF"
        if p2.exists():
            return p2
        return None

    def _load_sample_list(self, sample_list: Optional[List[str]] = None):
        """加载样本列表，sample_list 是 base_name 列表"""
        samples = []

        if sample_list is None:
            plm_dir = self.base_path / 'plm'
            tif_files = list(plm_dir.glob('*.tif')) + list(plm_dir.glob('*.TIF'))
            unique_samples = sorted({p.stem.rsplit('_', 1)[0] for p in tif_files})
        else:
            unique_samples = sample_list

        for sample_name in unique_samples:
            for channel_id in self.channels:
                rgb_file = self._resolve_tif('rgb', sample_name, channel_id)
                swir_file = self._resolve_tif('swir', sample_name, channel_id)
                enh_file = self._resolve_tif('enh', sample_name, channel_id)
                plm_file = self._resolve_tif('plm', sample_name, channel_id)

                # label 必须存在
                if plm_file is None:
                    continue

                # 仅检查当前启用的模态
                if self.use_rgb and rgb_file is None:
                    continue
                if self.use_swir and swir_file is None:
                    continue
                if self.use_enh and enh_file is None:
                    continue

                samples.append({
                    'name': f"{sample_name}_{channel_id}",
                    'base_name': sample_name,
                    'channel_id': channel_id,
                    'rgb': rgb_file,
                    'swir': swir_file,
                    'enh': enh_file,
                    'plm': plm_file
                })

        return samples

    def _load_rgb(self, rgb_path: Path) -> np.ndarray:
        """加载 RGB，输出 [H, W, 3]，归一化到 [0,1]"""
        if rgb_path is None or not rgb_path.exists():
            raise FileNotFoundError(f"RGB 文件不存在: {rgb_path}")

        try:
            with rasterio.open(rgb_path) as src:
                rgb = src.read().astype(np.float32)  # [3, H, W]
            rgb = np.transpose(rgb, (1, 2, 0))      # [H, W, 3]

            # 若已是 0-255，直接 /255；否则裁剪到 0-255 再 /255
            rgb = np.clip(rgb, 0, 255) / 255.0
            return rgb.astype(np.float32)
        except rasterio.errors.RasterioIOError as e:
            raise RuntimeError(f"无法读取 RGB 文件: {rgb_path} - {e}")

    def _load_swir(self, swir_path: Path) -> np.ndarray:
        """加载 SWIR 多波段影像，输出 [H, W, C]，按 band 做 2%-98% 分位归一化到 [0,1]"""
        if swir_path is None or not swir_path.exists():
            raise FileNotFoundError(f"SWIR 文件不存在: {swir_path}")

        try:
            with rasterio.open(swir_path) as src:
                swir = src.read().astype(np.float32)   # [C, H, W]
                nodata = src.nodata
        except rasterio.errors.RasterioIOError as e:
            raise RuntimeError(f"无法读取 SWIR 文件: {swir_path} - {e}")

        swir = np.transpose(swir, (1, 2, 0))  # [H, W, C]

        if not self.normalize_swir:
            return swir.astype(np.float32)

        swir_norm = np.zeros_like(swir, dtype=np.float32)

        for b in range(swir.shape[2]):
            band = swir[:, :, b]

            if nodata is not None:
                valid_mask = np.isfinite(band) & (band != nodata)
            else:
                valid_mask = np.isfinite(band)

            if not np.any(valid_mask):
                continue

            valid_pixels = band[valid_mask]
            p_low = np.percentile(valid_pixels, self.swir_percentile_low)
            p_high = np.percentile(valid_pixels, self.swir_percentile_high)

            if p_high > p_low:
                band_norm = (band - p_low) / (p_high - p_low)
                band_norm = np.clip(band_norm, 0.0, 1.0)
            else:
                band_norm = np.zeros_like(band, dtype=np.float32)

            band_norm[~valid_mask] = 0.0
            swir_norm[:, :, b] = band_norm

        return swir_norm

    def _load_enh(self, enh_path: Path) -> np.ndarray:
        """加载 ENH，输出 [H, W]，归一化到 [0,1]"""
        if enh_path is None or not enh_path.exists():
            raise FileNotFoundError(f"ENH 文件不存在: {enh_path}")

        try:
            with rasterio.open(enh_path) as src:
                enh = src.read(1).astype(np.float32)
        except rasterio.errors.RasterioIOError as e:
            raise RuntimeError(f"无法读取 ENH 文件: {enh_path} - {e}")

        nodata_value = -9999.0
        valid_mask = np.isfinite(enh) & (enh != nodata_value)

        if not np.any(valid_mask):
            return np.zeros_like(enh, dtype=np.float32)

        if self.normalize_enh:
            # 与你原代码保持一致，只是直接归一化到 [0,1]
            enh_norm = (enh - self.enh_x0) / self.enh_x1
            enh_norm = np.clip(enh_norm, 0.0, 1.0)
        else:
            valid_data = enh[valid_mask]
            enh_min, enh_max = valid_data.min(), valid_data.max()
            if enh_max > enh_min:
                enh_norm = (enh - enh_min) / (enh_max - enh_min)
            else:
                enh_norm = np.zeros_like(enh, dtype=np.float32)

        enh_norm[~valid_mask] = 0.0
        return enh_norm.astype(np.float32)

    def _load_mask(self, plm_path: Path, enh_data: Optional[np.ndarray] = None) -> np.ndarray:
        """加载掩码，binary 输出 0/1；enhanced 输出 [0,1]"""
        if plm_path is None or not plm_path.exists():
            raise FileNotFoundError(f"掩码文件不存在: {plm_path}")

        try:
            with rasterio.open(plm_path) as src:
                mask = src.read(1)
        except rasterio.errors.RasterioIOError as e:
            raise RuntimeError(f"无法读取掩码文件: {plm_path} - {e}")

        if self.mask_mode == 'binary':
            return (mask > 0).astype(np.float32)

        if self.mask_mode == 'enhanced':
            if enh_data is None:
                raise ValueError("mask_mode='enhanced' 时 enh_data 不能为空")
            return np.where(mask > 0, enh_data, 0.0).astype(np.float32)

        raise ValueError(f"未知的 mask_mode: {self.mask_mode}")

    def _merge_modalities(
        self,
        rgb: Optional[np.ndarray],
        swir: Optional[np.ndarray],
        enh: Optional[np.ndarray]
    ) -> np.ndarray:
        """按启用模态拼接输入"""
        parts = []

        if self.use_swir:
            if swir is None:
                raise ValueError("use_swir=True 但 swir 数据为空")
            parts.append(swir)  # [H, W, 41]

        if self.use_rgb:
            if rgb is None:
                raise ValueError("use_rgb=True 但 rgb 数据为空")
            parts.append(rgb)   # [H, W, 3]

        if self.use_enh:
            if enh is None:
                raise ValueError("use_enh=True 但 enh 数据为空")
            parts.append(np.expand_dims(enh, axis=2))  # [H, W, 1]

        merged = np.concatenate(parts, axis=2).astype(np.float32)
        return merged

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        rgb = self._load_rgb(sample['rgb']) if self.use_rgb else None
        swir = self._load_swir(sample['swir']) if self.use_swir else None
        enh = self._load_enh(sample['enh']) if self.use_enh else None

        mask = self._load_mask(
            sample['plm'],
            enh_data=enh if (self.use_enh and self.mask_mode == 'enhanced') else None
        )

        input_data = self._merge_modalities(rgb=rgb, swir=swir, enh=enh)  # [H, W, C]
        input_data = np.transpose(input_data, (2, 0, 1))                   # [C, H, W]
        mask = np.expand_dims(mask, axis=0)                                # [1, H, W]

        input_tensor = torch.from_numpy(input_data).float()
        mask_tensor = torch.from_numpy(mask).float()

        if self.transform:
            input_tensor, mask_tensor = self.transform(input_tensor, mask_tensor)

        return {
            'input': input_tensor,
            'mask': mask_tensor,
            'name': sample['name']
        }
