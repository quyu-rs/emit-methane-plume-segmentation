# -*- coding: utf-8 -*-
"""
Dice Loss和Focal Loss实现
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class DiceLoss(nn.Module):
    """
    Dice Loss实现
    添加平滑项防止数值不稳定
    """
    def __init__(self, smooth: float = 1e-6, class_weights: Optional[torch.Tensor] = None):
        """
        参数:
            smooth: 平滑项，防止分母为0
            class_weights: 类别权重，形状为 (num_classes,)
        """
        super().__init__()
        self.smooth = smooth
        self.class_weights = class_weights

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        计算Dice Loss
        
        参数:
            pred: 预测值，形状为 (B, C, H, W) 或 (B, 1, H, W)，值在[0,1]
            target: 目标值，形状为 (B, C, H, W) 或 (B, 1, H, W)，值在[0,1]
        
        返回:
            Dice Loss标量
        """

        if pred.shape != target.shape:
            if pred.shape[1] == 1 and target.shape[1] == 1:
                pass
            else:

                pred = pred.argmax(dim=1, keepdim=True).float()


        pred_flat = pred.view(pred.size(0), -1)
        target_flat = target.view(target.size(0), -1)


        intersection = (pred_flat * target_flat).sum(dim=1)
        union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)


        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)


        dice_loss = 1.0 - dice


        if self.class_weights is not None:





            background_mask = (target_flat < 0.5).float()
            foreground_mask = (target_flat >= 0.5).float()


            bg_pred = pred_flat * background_mask
            bg_target = target_flat * background_mask
            bg_intersection = (bg_pred * bg_target).sum(dim=1)
            bg_union = bg_pred.sum(dim=1) + bg_target.sum(dim=1)


            bg_valid = bg_union > self.smooth
            bg_dice = torch.where(
                bg_valid,
                (2.0 * bg_intersection + self.smooth) / (bg_union + self.smooth),
                torch.ones_like(bg_union)
            )
            bg_dice_loss = 1.0 - bg_dice


            fg_pred = pred_flat * foreground_mask
            fg_target = target_flat * foreground_mask
            fg_intersection = (fg_pred * fg_target).sum(dim=1)
            fg_union = fg_pred.sum(dim=1) + fg_target.sum(dim=1)


            fg_valid = fg_union > self.smooth
            fg_dice = torch.where(
                fg_valid,
                (2.0 * fg_intersection + self.smooth) / (fg_union + self.smooth),
                torch.ones_like(fg_union)
            )
            fg_dice_loss = 1.0 - fg_dice


            weights = self.class_weights.to(dice_loss.device)
            weighted_loss = weights[0] * bg_dice_loss + weights[1] * fg_dice_loss
            dice_loss = weighted_loss.mean()
        else:
            dice_loss = dice_loss.mean()

        return dice_loss


class FocalLoss(nn.Module):
    """
    Focal Loss实现
    数值稳定版本
    """
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        class_weights: Optional[torch.Tensor] = None
    ):
        """
        参数:
            alpha: 类别权重平衡因子
            gamma: 难样本聚焦参数
            class_weights: 类别权重，形状为 (num_classes,)
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.class_weights = class_weights

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        计算Focal Loss
        
        参数:
            pred: 预测值，形状为 (B, C, H, W) 或 (B, 1, H, W)，值在[0,1]
            target: 目标值，形状为 (B, C, H, W) 或 (B, 1, H, W)，值在[0,1]
        
        返回:
            Focal Loss标量
        """

        if pred.shape != target.shape:
            if pred.shape[1] == 1 and target.shape[1] == 1:
                pass
            else:

                target_onehot = F.one_hot(target.long().squeeze(1), num_classes=pred.shape[1]).permute(0, 3, 1, 2).float()
                target = target_onehot


        eps = 1e-6
        pred = torch.clamp(pred, min=eps, max=1.0 - eps)


        target = torch.clamp(target, min=0.0, max=1.0)



        log_pred = torch.log(pred + eps)
        log_1_minus_pred = torch.log(1 - pred + eps)
        ce_loss = -target * log_pred - (1 - target) * log_1_minus_pred


        ce_loss = torch.where(torch.isnan(ce_loss) | torch.isinf(ce_loss),
                              torch.zeros_like(ce_loss), ce_loss)


        ce_loss = torch.clamp(ce_loss, min=0.0, max=50.0)


        p_t = pred * target + (1 - pred) * (1 - target)

        p_t = torch.clamp(p_t, min=eps, max=1.0 - eps)


        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        alpha_t = torch.clamp(alpha_t, min=eps, max=1.0 - eps)




        log_1_minus_p_t = torch.log(1 - p_t + eps)

        log_1_minus_p_t = torch.clamp(log_1_minus_p_t, min=-10.0, max=0.0)


        gamma_log_term = self.gamma * log_1_minus_p_t
        gamma_log_term = torch.clamp(gamma_log_term, min=-10.0, max=0.0)
        focal_weight = torch.exp(gamma_log_term)


        focal_weight = torch.clamp(focal_weight, min=eps, max=1.0)


        focal_weight = torch.where(torch.isnan(focal_weight) | torch.isinf(focal_weight),
                                   torch.ones_like(focal_weight) * eps, focal_weight)

        focal_loss = alpha_t * focal_weight * ce_loss


        focal_loss = torch.where(torch.isnan(focal_loss) | torch.isinf(focal_loss),
                                 torch.zeros_like(focal_loss), focal_loss)


        focal_loss = torch.clamp(focal_loss, min=0.0, max=50.0)


        if self.class_weights is not None:






            weights = self.class_weights.to(focal_loss.device)



            background_mask = (target < 0.5).float()
            foreground_mask = (target >= 0.5).float()


            weighted_focal_loss = weights[0] * focal_loss * background_mask + \
                                weights[1] * focal_loss * foreground_mask


            focal_loss = weighted_focal_loss.mean()
        else:
            focal_loss = focal_loss.mean()


        if torch.isnan(focal_loss) or torch.isinf(focal_loss):
            focal_loss = torch.tensor(0.05, device=focal_loss.device, dtype=focal_loss.dtype)

        return focal_loss


class DiceFocalLoss(nn.Module):
    """
    Dice Loss + Focal Loss联合损失函数
    """
    def __init__(
        self,
        dice_weight: float = 1.0,
        focal_weight: float = 1.0,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        class_weights: Optional[torch.Tensor] = None
    ):
        """
        参数:
            dice_weight: Dice Loss权重
            focal_weight: Focal Loss权重
            focal_alpha: Focal Loss alpha参数
            focal_gamma: Focal Loss gamma参数
            class_weights: 类别权重，形状为 (num_classes,)
                例如: torch.tensor([0.5, 1.5]) 表示背景=0.5, 羽流=1.5
        """
        super().__init__()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight

        self.dice_loss = DiceLoss(smooth=1e-6, class_weights=class_weights)
        self.focal_loss = FocalLoss(
            alpha=focal_alpha,
            gamma=focal_gamma,
            class_weights=class_weights
        )

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        计算联合损失
        
        参数:
            pred: 预测值，形状为 (B, 1, H, W)，值在[0,1]
            target: 目标值，形状为 (B, 1, H, W)，值在[0,1]（数据加载时已归一化）
        
        返回:
            联合损失标量
        """

        if torch.isnan(pred).any() or torch.isinf(pred).any():

            pred = torch.where(
                torch.isnan(pred) | torch.isinf(pred),
                torch.tensor(0.5, device=pred.device, dtype=pred.dtype),
                pred
            )


        pred = torch.clamp(pred, min=0.0, max=1.0)



        if target.dtype != torch.float32:
            target = target.float()


        if torch.isnan(target).any() or torch.isinf(target).any():
            target = torch.where(
                torch.isnan(target) | torch.isinf(target),
                torch.zeros_like(target),
                target
            )




        target = torch.clamp(target, min=0.0, max=1.0)


        import logging
        logger = logging.getLogger('DiT-Mamba-Training')

        try:
            dice = self.dice_loss(pred, target)

            if torch.isnan(dice) or torch.isinf(dice):
                logger.warning(f"Dice loss包含NaN/Inf: dice={dice.item() if hasattr(dice, 'item') else dice}, 使用回退值")

                dice = torch.tensor(0.1, device=pred.device, requires_grad=True)
        except Exception as e:
            logger.error(f"Dice loss计算异常: {str(e)}, 使用回退值")

            dice = torch.tensor(0.1, device=pred.device, requires_grad=True)

        try:
            focal = self.focal_loss(pred, target)

            if torch.isnan(focal) or torch.isinf(focal):
                logger.warning(f"Focal loss包含NaN/Inf: focal={focal.item() if hasattr(focal, 'item') else focal}, 使用回退值")

                focal = torch.tensor(0.05, device=pred.device, requires_grad=True)
        except Exception as e:
            logger.error(f"Focal loss计算异常: {str(e)}, 使用回退值")

            focal = torch.tensor(0.05, device=pred.device, requires_grad=True)


        total_loss = self.dice_weight * dice + self.focal_weight * focal


        if torch.isnan(total_loss) or torch.isinf(total_loss):
            logger.error(f"总损失包含NaN/Inf: total_loss={total_loss.item() if hasattr(total_loss, 'item') else total_loss}, dice={dice.item() if hasattr(dice, 'item') else dice}, focal={focal.item() if hasattr(focal, 'item') else focal}")

            total_loss = torch.tensor(0.1, device=pred.device, requires_grad=True)


        total_loss = torch.clamp(total_loss, min=1e-4, max=10.0)

        return total_loss



