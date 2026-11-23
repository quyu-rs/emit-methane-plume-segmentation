# -*- coding: utf-8 -*-
"""
训练 CNN Refiner
复用 diffusion_preprocessed_data.pt
"""

import sys
from pathlib import Path
import argparse
from typing import Dict, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.cnn_refiner import CNNRefiner
from training.config import TrainingConfig


class RefinerDataset(Dataset):
    """
    第二阶段通用数据集
    从 diffusion_preprocessed_data.pt 加载
    """
    def __init__(self, preprocessed_data: list):
        self.data = preprocessed_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        def to_tensor(x):
            if isinstance(x, torch.Tensor):
                return x.float()
            elif isinstance(x, np.ndarray):
                return torch.from_numpy(x).float()
            else:
                return torch.tensor(x, dtype=torch.float32)

        return {
            "input_image": to_tensor(item["input_image"]),   # [4,H,W]
            "pred_coarse": to_tensor(item["pred_coarse"]),   # [1,H,W]
            "residual": to_tensor(item["residual"]),         # [1,H,W]
            "target_mask": to_tensor(item["target_mask"]),   # [1,H,W]
            "name": item["name"],
        }


def create_data_loaders(preprocessed_data: Dict[str, Any], config: dict):
    train_data = preprocessed_data.get("train", [])
    val_data = preprocessed_data.get("val", [])

    train_dataset = RefinerDataset(train_data)
    val_dataset = RefinerDataset(val_data)

    # 复用 diffusion_refiner.training 配置，便于公平对比
    train_config = config.get("diffusion_refiner", {}).get("training", {})
    data_config = config.get("data", {})

    batch_size = train_config.get("batch_size", data_config.get("batch_size", 32))
    num_workers = data_config.get("num_workers", 1)
    pin_memory = data_config.get("pin_memory", True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    return train_loader, val_loader


def dice_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    pred = pred.contiguous()
    target = target.contiguous()

    intersection = (pred * target).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = (2.0 * intersection + eps) / (union + eps)
    return 1.0 - dice.mean()


class RefinerLoss(nn.Module):
    """
    第二阶段 deterministic residual refinement loss
    total = seg_loss + lambda * residual_loss
    """
    def __init__(self, lambda_residual: float = 0.2):
        super().__init__()
        self.lambda_residual = lambda_residual
        self.bce = nn.BCELoss()
        self.smooth_l1 = nn.SmoothL1Loss()

    def forward(
        self,
        pred_refined: torch.Tensor,
        target_mask: torch.Tensor,
        pred_residual: torch.Tensor,
        target_residual: torch.Tensor
    ):
        loss_bce = self.bce(pred_refined, target_mask)
        loss_dice = dice_loss(pred_refined, target_mask)
        loss_res = self.smooth_l1(pred_residual, target_residual)

        total = loss_bce + loss_dice + self.lambda_residual * loss_res
        return total, {
            "loss_bce": loss_bce.item(),
            "loss_dice": loss_dice.item(),
            "loss_res": loss_res.item(),
        }


@torch.no_grad()
def compute_metrics(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5):
    pred_bin = (pred >= threshold).float()
    target_bin = (target >= threshold).float()

    tp = (pred_bin * target_bin).sum(dim=(1, 2, 3))
    fp = (pred_bin * (1 - target_bin)).sum(dim=(1, 2, 3))
    fn = ((1 - pred_bin) * target_bin).sum(dim=(1, 2, 3))

    iou = tp / (tp + fp + fn + 1e-6)
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)

    return {
        "iou": iou.mean().item(),
        "precision": precision.mean().item(),
        "recall": recall.mean().item(),
    }


def run_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
    train: bool = True
):
    model.train() if train else model.eval()

    total_loss = 0.0
    total_iou = 0.0
    total_precision = 0.0
    total_recall = 0.0
    num_batches = 0

    pbar = tqdm(loader, desc="Train" if train else "Val", leave=False)

    for batch in pbar:
        input_image = batch["input_image"].to(device)      # [B,4,H,W]
        pred_coarse = batch["pred_coarse"].to(device)      # [B,1,H,W]
        residual = batch["residual"].to(device)            # [B,1,H,W]
        target_mask = batch["target_mask"].to(device)      # [B,1,H,W]

        condition = torch.cat([input_image, pred_coarse], dim=1)  # [B,5,H,W]

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            pred_residual = model(condition)                              # [B,1,H,W]
            pred_refined = torch.clamp(pred_coarse + pred_residual, 0, 1)

            loss, loss_dict = criterion(
                pred_refined=pred_refined,
                target_mask=target_mask,
                pred_residual=pred_residual,
                target_residual=residual
            )

            if train:
                loss.backward()
                optimizer.step()

        metrics = compute_metrics(pred_refined.detach(), target_mask)

        total_loss += loss.item()
        total_iou += metrics["iou"]
        total_precision += metrics["precision"]
        total_recall += metrics["recall"]
        num_batches += 1

        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "iou": f"{metrics['iou']:.4f}"
        })

    return {
        "loss": total_loss / max(num_batches, 1),
        "iou": total_iou / max(num_batches, 1),
        "precision": total_precision / max(num_batches, 1),
        "recall": total_recall / max(num_batches, 1),
    }


def save_checkpoint(
    path: Path,
    model,
    optimizer,
    epoch: int,
    best_val_iou: float,
    config: dict
):
    ckpt = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_iou": best_val_iou,
        "config": config,
    }
    torch.save(ckpt, path)


def main():
    parser = argparse.ArgumentParser(description="训练 CNN Refiner")
    parser.add_argument("--preprocessed_data", type=str, required=True, help="预处理数据文件路径")
    parser.add_argument("--config", type=str, default="config.yaml", help="配置文件路径")
    parser.add_argument("--resume", type=str, default=None, help="恢复训练的检查点路径")
    args = parser.parse_args()

    print("=" * 70)
    print("CNN Refiner训练")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

    config_path = project_root / args.config
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    config = TrainingConfig(str(config_path))

    print(f"\n加载预处理数据: {args.preprocessed_data}")
    preprocessed_data = torch.load(args.preprocessed_data, map_location="cpu")
    print(f"训练集: {len(preprocessed_data.get('train', []))} 个样本")
    print(f"验证集: {len(preprocessed_data.get('val', []))} 个样本")

    train_loader, val_loader = create_data_loaders(preprocessed_data, config.config)

    print("\n创建 CNN Refiner 模型...")
    model = CNNRefiner(
        in_channels=5,
        base_channels=64,
        num_blocks=4,
        dropout=config.get("regularization", {}).get("dropout", 0.1),
        residual_range=0.2
    ).to(device)

    print(f"模型参数量: {model.count_parameters() / 1e6:.2f}M")

    diffusion_cfg = config.get("diffusion_refiner", {}).get("training", {})
    optimizer_cfg = config.get("optimizer", {})

    learning_rate = diffusion_cfg.get("learning_rate", optimizer_cfg.get("lr", 1e-4))
    num_epochs = diffusion_cfg.get("num_epochs", 40)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=optimizer_cfg.get("weight_decay", 1e-4),
        betas=(
            optimizer_cfg.get("beta1", 0.9),
            optimizer_cfg.get("beta2", 0.999),
        ),
        eps=optimizer_cfg.get("eps", 1e-8)
    )

    criterion = RefinerLoss(lambda_residual=0.2)

    checkpoint_dir = project_root / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    latest_ckpt = checkpoint_dir / "cnn_refiner_latest.pth"
    best_ckpt = checkpoint_dir / "cnn_refiner_best.pth"

    start_epoch = 0
    best_val_iou = 0.0

    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.exists():
            print(f"\n恢复训练: {resume_path}")
            ckpt = torch.load(resume_path, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            start_epoch = ckpt.get("epoch", 0) + 1
            best_val_iou = ckpt.get("best_val_iou", 0.0)
            print(f"从 epoch {start_epoch} 继续训练, best_val_iou={best_val_iou:.4f}")
        else:
            print(f"警告: resume文件不存在，忽略: {resume_path}")

    print("\n开始训练...")
    for epoch in range(start_epoch, num_epochs):
        print(f"\n[Epoch {epoch + 1}/{num_epochs}]")

        train_stats = run_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            train=True
        )

        val_stats = run_one_epoch(
            model=model,
            loader=val_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            train=False
        )

        print(
            f"Train Loss: {train_stats['loss']:.4f} | "
            f"Train IoU: {train_stats['iou']:.4f} | "
            f"Val Loss: {val_stats['loss']:.4f} | "
            f"Val IoU: {val_stats['iou']:.4f} | "
            f"Val Precision: {val_stats['precision']:.4f} | "
            f"Val Recall: {val_stats['recall']:.4f}"
        )

        save_checkpoint(
            path=latest_ckpt,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            best_val_iou=best_val_iou,
            config=config.config
        )

        if val_stats["iou"] > best_val_iou:
            best_val_iou = val_stats["iou"]
            save_checkpoint(
                path=best_ckpt,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_val_iou=best_val_iou,
                config=config.config
            )
            print(f"保存最佳模型: {best_ckpt} (Val IoU={best_val_iou:.4f})")

    print("\n训练完成!")
    print(f"最佳验证IoU: {best_val_iou:.4f}")


if __name__ == "__main__":
    main()