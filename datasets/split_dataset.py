# -*- coding: utf-8 -*-
"""
数据集划分脚本（输出 base_name 列表）

特点：
- 文件名格式: {base_name}_{channel}.tif 或 .TIF
- 默认使用统一样本池：要求 rgb / swir / enh / plm 四件套齐全
- 不强制每个 base_name 必须有 0/1/2 全部后缀；
  只要 candidate_channels 中存在任一完整后缀即可入库
- 默认比例改为 7:1:2，与论文一致
"""

import json
import numpy as np
from pathlib import Path
import rasterio
from typing import Dict, List, Sequence, Tuple, Optional


def _exists_any_case(p: Path) -> Optional[Path]:
    """兼容 .tif / .TIF；存在则返回真实路径，否则 None"""
    if p.exists():
        return p
    if p.suffix.lower() == ".tif":
        p2 = p.with_suffix(".TIF")
        if p2.exists():
            return p2
    return None


def file_path(base: Path, folder: str, name: str, ch: int) -> Optional[Path]:
    return _exists_any_case(base / folder / f"{name}_{ch}.tif")


def find_first_complete_channel(
    base: Path,
    name: str,
    candidate_channels: Sequence[int],
    require_rgb: bool = True,
    require_swir: bool = True,
    require_enh: bool = True,
) -> Optional[int]:
    """
    返回第一个“plm + 当前要求模态”都存在的 channel id
    """
    for ch in candidate_channels:
        plm_ok = file_path(base, "plm", name, ch) is not None
        rgb_ok = (not require_rgb) or (file_path(base, "rgb", name, ch) is not None)
        swir_ok = (not require_swir) or (file_path(base, "swir", name, ch) is not None)
        enh_ok = (not require_enh) or (file_path(base, "enh", name, ch) is not None)

        if plm_ok and rgb_ok and swir_ok and enh_ok:
            return ch
    return None


def check_positive(plm_path: Path) -> bool:
    """PLM 是否有正像元"""
    try:
        with rasterio.open(plm_path) as src:
            m = src.read(1)
        return bool(np.any(m > 0))
    except Exception as e:
        print(f"警告: 无法读取 {plm_path}: {e}")
        return False


def collect_base_names_from_plm(plm_dir: Path) -> List[str]:
    """从 plm 目录收集所有 base_name"""
    tif_files = list(plm_dir.glob("*.tif")) + list(plm_dir.glob("*.TIF"))
    base_names = sorted({p.stem.rsplit("_", 1)[0] for p in tif_files})
    return base_names


def get_all_samples(
    base_path: str,
    candidate_channels: Sequence[int] = (0, 1, 2),
    require_rgb: bool = True,
    require_swir: bool = True,
    require_enh: bool = True,
    debug_show_missing: int = 20,
) -> Dict[str, bool]:
    """
    返回 {base_name: is_positive}

    规则：
    - 对每个 base_name：在 candidate_channels 中找到一个完整 channel
    - 标签：用该 channel 的 plm 判定正负
    """
    base = Path(base_path)
    plm_dir = base / "plm"

    if not plm_dir.exists():
        print(f"ERROR: plm 目录不存在: {plm_dir.resolve()}")
        return {}

    base_names = collect_base_names_from_plm(plm_dir)
    print(f"DEBUG: 收集到 base_name 数量: {len(base_names)}")

    samples: Dict[str, bool] = {}
    missing = []

    for name in base_names:
        ch = find_first_complete_channel(
            base=base,
            name=name,
            candidate_channels=candidate_channels,
            require_rgb=require_rgb,
            require_swir=require_swir,
            require_enh=require_enh,
        )

        if ch is None:
            if len(missing) < debug_show_missing:
                missing.append(name)
            continue

        plm = file_path(base, "plm", name, ch)
        if plm is None:
            continue

        samples[name] = check_positive(plm)

    if missing:
        print("\nDEBUG: 以下样本被过滤（找不到任何完整通道组合）:")
        for n in missing:
            print("  ", n)
        print("提示：最常见原因是 swir / rgb / enh 缺文件或命名不一致。\n")

    return samples


def stratified_split(
    pos: List[str],
    neg: List[str],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> Tuple[List[str], List[str], List[str]]:
    n_pos = len(pos)
    n_neg = len(neg)

    n_pos_train = int(n_pos * train_ratio)
    n_pos_val = int(n_pos * val_ratio)

    n_neg_train = int(n_neg * train_ratio)
    n_neg_val = int(n_neg * val_ratio)

    train = pos[:n_pos_train] + neg[:n_neg_train]
    val = pos[n_pos_train:n_pos_train + n_pos_val] + neg[n_neg_train:n_neg_train + n_neg_val]
    test = pos[n_pos_train + n_pos_val:] + neg[n_neg_train + n_neg_val:]

    return train, val, test


def split_dataset(
    base_path: str = r"data/rows",
    candidate_channels: Sequence[int] = (0, 1, 2),
    train_ratio: float = 0.70,
    val_ratio: float = 0.10,
    test_ratio: float = 0.20,
    output_dir: str = "datasets",
    seed: int = 42,
    require_rgb: bool = True,
    require_swir: bool = True,
    require_enh: bool = True,
):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "比例之和必须为1.0"
    np.random.seed(seed)

    print("正在扫描数据并构建可用样本列表（base_name）...")
    all_samples = get_all_samples(
        base_path=base_path,
        candidate_channels=candidate_channels,
        require_rgb=require_rgb,
        require_swir=require_swir,
        require_enh=require_enh,
        debug_show_missing=20
    )

    if len(all_samples) == 0:
        raise RuntimeError("没有找到可用样本：请检查 base_path 是否正确，或模态文件是否齐全。")

    pos = [n for n, is_pos in all_samples.items() if is_pos]
    neg = [n for n, is_pos in all_samples.items() if not is_pos]

    print(f"\n可用总样本数: {len(all_samples)}")
    print(f"正样本: {len(pos)} ({len(pos)/len(all_samples)*100:.1f}%)")
    print(f"负样本: {len(neg)} ({len(neg)/len(all_samples)*100:.1f}%)")

    np.random.shuffle(pos)
    np.random.shuffle(neg)

    train, val, test = stratified_split(pos, neg, train_ratio, val_ratio, test_ratio)
    np.random.shuffle(train)
    np.random.shuffle(val)
    np.random.shuffle(test)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    split_dict = {"train": train, "val": val, "test": test}

    for split_name, samples in split_dict.items():
        fn = out / f"{split_name}_split.json"
        with open(fn, "w", encoding="utf-8") as f:
            json.dump(samples, f, indent=2, ensure_ascii=False)

        pos_count = sum(1 for n in samples if all_samples[n])
        neg_count = len(samples) - pos_count
        print(f"{split_name}: {len(samples)}  正:{pos_count}  负:{neg_count}  -> {fn}")

    # 额外保存划分元信息，便于复现
    meta = {
        "base_path": str(Path(base_path).resolve()),
        "candidate_channels": list(candidate_channels),
        "seed": seed,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "require_rgb": require_rgb,
        "require_swir": require_swir,
        "require_enh": require_enh,
        "total_samples": len(all_samples),
        "positive_samples": len(pos),
        "negative_samples": len(neg),
        "train_samples": len(train),
        "val_samples": len(val),
        "test_samples": len(test),
    }

    meta_fn = out / "split_meta.json"
    with open(meta_fn, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"\n元信息保存到: {meta_fn}")
    print("数据集划分完成！")


if __name__ == "__main__":
    split_dataset(
        base_path=r"data/rows",          # 建议改成绝对路径更稳
        candidate_channels=(0, 1, 2),
        train_ratio=0.70,
        val_ratio=0.10,
        test_ratio=0.20,
        output_dir="datasets",
        seed=42,
        require_rgb=True,
        require_swir=True,
        require_enh=True,
    )

# SWIR + ENH
# train_dataset = RowsDataset(
#     base_path=r"/你的绝对路径/data/rows",
#     channels=[0, 1, 2],
#     use_rgb=False,
#     use_swir=True,
#     use_enh=True,
#     sample_list=train_split
# )

# RGB + ENH
# train_dataset = RowsDataset(
#     base_path=r"/你的绝对路径/data/rows",
#     channels=[0, 1, 2],
#     use_rgb=True,
#     use_swir=False,
#     use_enh=True,
#     sample_list=train_split
# )

# Full: SWIR + RGB + ENH
# train_dataset = RowsDataset(
#     base_path=r"/你的绝对路径/data/rows",
#     channels=[0, 1, 2],
#     use_rgb=True,
#     use_swir=True,
#     use_enh=True,
#     sample_list=train_split
# )
