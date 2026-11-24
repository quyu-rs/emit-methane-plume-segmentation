# -*- coding: utf-8 -*-
import argparse
import sys
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data.dataloader import default_collate

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from train import create_data_loaders, load_class_weights_from_calculation
from models.dit_mamba import DiTMamba
from losses.dice_focal_loss import DiceFocalLoss
from training.config import TrainingConfig
from training.trainer import Trainer


def collate_enh_only(samples):
    # RowsDataset 返回 [R, G, B, ENH]。保留 ENH 的单通道维度。
    rows = []
    for sample in samples:
        image = sample["input"]
        if image.ndim != 3 or image.shape[0] != 4:
            raise ValueError(f"Expected RGB + ENH, got {tuple(image.shape)}")
        rows.append({**sample, "input": image[3:4]})
    return default_collate(rows)


def make_optimizer(model, cfg):
    # 沿用原 train.py 中五组 Adam 参数各自的学习率。
    keys = ("mamba.A_log", "mamba.A_proj", "mamba.x_proj", "mamba.dt_proj")
    groups = [[] for _ in range(5)]
    for name, parameter in model.named_parameters():
        index = next((i for i, key in enumerate(keys) if key in name), 4)
        groups[index].append(parameter)

    lr = cfg.get("lr", 1e-4)
    scales = (0.01, 0.1, 0.1, 0.1, 1.0)
    return torch.optim.Adam(
        [{"params": params, "lr": lr * scale} for params, scale in zip(groups, scales)],
        lr=lr,
        weight_decay=cfg.get("weight_decay", 1e-4),
        betas=(cfg.get("beta1", 0.9), cfg.get("beta2", 0.999)),
        eps=cfg.get("eps", 1e-8),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config-first.yaml")
    parser.add_argument("--model_type", choices=["ditmamba"], default="ditmamba")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    config_path = (ROOT / args.config).resolve(strict=True)
    cfg = TrainingConfig(str(config_path)).config
    if cfg["data"].get("use_dataset_adapter", False):
        raise ValueError("Set data.use_dataset_adapter to false for EMIT RowsDataset.")

    cfg["data"]["input_modality"] = "enh_only"
    model_cfg = cfg["model"]
    model_cfg["in_channels"] = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device, flush=True)

    model = DiTMamba(
        in_channels=1,
        base_channels=model_cfg.get("base_channels", 64),
        num_encoder_layers=model_cfg.get("num_encoder_layers", 4),
        mamba_dim=model_cfg.get("mamba_dim", 256),
        num_classes=1,
        dropout=cfg.get("regularization", {}).get("dropout", 0.1),
    ).to(device)

    # 使用原工程的 train_split.json 和 val_split.json。
    train_loader, val_loader = create_data_loaders(cfg, device)
    train_loader.collate_fn = collate_enh_only
    val_loader.collate_fn = collate_enh_only
    if not len(train_loader.dataset) or not len(val_loader.dataset):
        raise ValueError("Empty training or validation set. Check data/rows and split files.")

    batch = next(iter(train_loader))
    shape = tuple(batch["input"].shape)
    if len(shape) != 4 or shape[1] != 1 or tuple(batch["mask"].shape) != shape:
        raise ValueError(f"Input {shape}, mask {tuple(batch['mask'].shape)}")
    print("ENH-only training input:", shape, flush=True)
    del batch

    if args.check_only:
        sample = collate_enh_only([val_loader.dataset[0]])
        x = sample["input"].to(device)
        model.eval()
        with torch.no_grad():
            pred = model(x)
        if pred.shape != x.shape or not torch.isfinite(pred).all().item():
            raise ValueError("Invalid validation prediction")
        print("Forward pass OK:", tuple(pred.shape), flush=True)
        return

    loss_cfg = cfg["loss"]
    weights, alpha = load_class_weights_from_calculation()
    if weights and alpha is not None:
        loss_cfg.update(class_weights=weights, focal_alpha=alpha)

    criterion = DiceFocalLoss(
        dice_weight=loss_cfg.get("dice_weight", 1.0),
        focal_weight=loss_cfg.get("focal_weight", 1.0),
        focal_alpha=loss_cfg.get("focal_alpha", 0.25),
        focal_gamma=loss_cfg.get("focal_gamma", 2.0),
        class_weights=torch.tensor(
            loss_cfg.get("class_weights", [0.5, 1.5]),
            dtype=torch.float32,
            device=device,
        ),
    ).to(device)

    run_dir = ROOT / "checkpoints" / "enh_only" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir.mkdir(parents=True, exist_ok=False)
    print("Run directory:", run_dir, flush=True)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=make_optimizer(model, cfg["optimizer"]),
        criterion=criterion,
        device=device,
        config=cfg,
        checkpoint_dir=str(run_dir),
        log_dir=str(run_dir / "logs"),
        model_type="ditmamba",
    )
    trainer.train()


if __name__ == "__main__":
    main()