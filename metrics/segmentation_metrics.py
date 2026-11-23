# -*- coding: utf-8 -*-
"""
分割评估指标实现
包括MIoU, MPrecision, Recall, MPA, F1 Score
"""
import torch
import torch.nn.functional as F
import numpy as np

def normalize_target_to_binary(target: torch.Tensor) -> torch.Tensor:
    """
    将目标掩码归一化为二值掩码（类别索引）
    
    参数:
        target: 目标掩码，可能是[0,1]或[0,255]范围
    
    返回:
        二值掩码，值为0或1（long类型）
    """
    if target.dtype == torch.long:

        if target.max() > 1:
            return (target > 0).long()
        else:
            return target
    else:

        if target.max() > 1.0:

            return (target > 0).long()
        else:

            return (target > 0.5).long()

def calculate_iou(pred: torch.Tensor, target: torch.Tensor, num_classes: int = 2) -> torch.Tensor:
    """
    计算每个类别的IoU
    
    参数:
        pred: 预测值，形状为 (B, H, W)，值为类别索引
        target: 目标值，形状为 (B, H, W)，值为类别索引
        num_classes: 类别数
    
    返回:
        IoU，形状为 (num_classes,)
    """
    ious = []

    for cls in range(num_classes):
        pred_cls = (pred == cls)
        target_cls = (target == cls)

        intersection = (pred_cls & target_cls).sum().float()
        union = (pred_cls | target_cls).sum().float()

        if union == 0:
            iou = torch.tensor(1.0, device=pred.device)
        else:
            iou = intersection / union

        ious.append(iou)

    return torch.stack(ious)

def calculate_miou(pred: torch.Tensor, target: torch.Tensor, num_classes: int = 2) -> float:
    """
    计算平均IoU (MIoU)
    
    参数:
        pred: 预测值，形状为 (B, H, W) 或 (B, 1, H, W)，值为类别索引
        target: 目标值，形状为 (B, H, W) 或 (B, 1, H, W)，值为类别索引
        num_classes: 类别数
    
    返回:
        MIoU标量
    """

    if len(pred.shape) == 4:
        pred = pred.squeeze(1)
    if len(target.shape) == 4:
        target = target.squeeze(1)


    if pred.dtype != torch.long:
        pred = (pred > 0.5).long()
    target = normalize_target_to_binary(target)


    ious = calculate_iou(pred, target, num_classes)


    miou = ious.mean().item()

    return miou

def calculate_precision(pred: torch.Tensor, target: torch.Tensor, num_classes: int = 2) -> torch.Tensor:
    """
    计算每个类别的精确率
    
    参数:
        pred: 预测值，形状为 (B, H, W)，值为类别索引
        target: 目标值，形状为 (B, H, W)，值为类别索引
        num_classes: 类别数
    
    返回:
        精确率，形状为 (num_classes,)
    """
    precisions = []

    for cls in range(num_classes):
        pred_cls = (pred == cls)
        target_cls = (target == cls)

        true_positives = (pred_cls & target_cls).sum().float()
        predicted_positives = pred_cls.sum().float()

        if predicted_positives == 0:
            precision = torch.tensor(1.0, device=pred.device)
        else:
            precision = true_positives / predicted_positives

        precisions.append(precision)

    return torch.stack(precisions)

def calculate_mprecision(pred: torch.Tensor, target: torch.Tensor, num_classes: int = 2) -> float:
    """
    计算平均精确率 (MPrecision)
    
    参数:
        pred: 预测值，形状为 (B, H, W) 或 (B, 1, H, W)，值为类别索引
        target: 目标值，形状为 (B, H, W) 或 (B, 1, H, W)，值为类别索引
        num_classes: 类别数
    
    返回:
        MPrecision标量
    """

    if len(pred.shape) == 4:
        pred = pred.squeeze(1)
    if len(target.shape) == 4:
        target = target.squeeze(1)


    if pred.dtype != torch.long:
        pred = (pred > 0.5).long()
    if target.dtype != torch.long:
        target = (target > 0.5).long()


    precisions = calculate_precision(pred, target, num_classes)


    mprecision = precisions.mean().item()

    return mprecision

def calculate_recall(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    计算召回率
    
    参数:
        pred: 预测值，形状为 (B, H, W) 或 (B, 1, H, W)，值为类别索引
        target: 目标值，形状为 (B, H, W) 或 (B, 1, H, W)，值为类别索引
    
    返回:
        Recall标量
    """

    if len(pred.shape) == 4:
        pred = pred.squeeze(1)
    if len(target.shape) == 4:
        target = target.squeeze(1)


    if pred.dtype != torch.long:
        pred = (pred > 0.5).long()
    if target.dtype != torch.long:
        target = (target > 0.5).long()


    true_positives = ((pred == 1) & (target == 1)).sum().float()
    false_negatives = ((pred == 0) & (target == 1)).sum().float()


    if (true_positives + false_negatives) == 0:
        recall = 1.0
    else:
        recall = (true_positives / (true_positives + false_negatives)).item()

    return recall

def calculate_pixel_accuracy(pred: torch.Tensor, target: torch.Tensor, num_classes: int = 2) -> torch.Tensor:
    """
    计算每个类别的像素精度
    
    参数:
        pred: 预测值，形状为 (B, H, W)，值为类别索引
        target: 目标值，形状为 (B, H, W)，值为类别索引
        num_classes: 类别数
    
    返回:
        像素精度，形状为 (num_classes,)
    """
    accuracies = []

    for cls in range(num_classes):
        target_cls = (target == cls)

        if target_cls.sum() == 0:
            accuracy = torch.tensor(1.0, device=pred.device)
        else:
            correct = ((pred == cls) & target_cls).sum().float()
            total = target_cls.sum().float()
            accuracy = correct / total

        accuracies.append(accuracy)

    return torch.stack(accuracies)

def calculate_mpa(pred: torch.Tensor, target: torch.Tensor, num_classes: int = 2) -> float:
    """
    计算平均像素精度 (MPA)
    
    参数:
        pred: 预测值，形状为 (B, H, W) 或 (B, 1, H, W)，值为类别索引
        target: 目标值，形状为 (B, H, W) 或 (B, 1, H, W)，值为类别索引
        num_classes: 类别数
    
    返回:
        MPA标量
    """

    if len(pred.shape) == 4:
        pred = pred.squeeze(1)
    if len(target.shape) == 4:
        target = target.squeeze(1)


    if pred.dtype != torch.long:
        pred = (pred > 0.5).long()
    if target.dtype != torch.long:
        target = (target > 0.5).long()


    accuracies = calculate_pixel_accuracy(pred, target, num_classes)


    mpa = accuracies.mean().item()

    return mpa

def calculate_f1(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    计算F1 Score（正类的F1，即前景类的F1）
    
    参数:
        pred: 预测值，形状为 (B, H, W) 或 (B, 1, H, W)，值为类别索引
        target: 目标值，形状为 (B, H, W) 或 (B, 1, H, W)，值为类别索引
    
    返回:
        F1 Score标量
    """
    if len(pred.shape) == 4:
        pred = pred.squeeze(1)
    if len(target.shape) == 4:
        target = target.squeeze(1)
    
    if pred.dtype != torch.long:
        pred = (pred > 0.5).long()
    if target.dtype != torch.long:
        target = (target > 0.5).long()
    
    true_positives = ((pred == 1) & (target == 1)).sum().float()
    false_positives = ((pred == 1) & (target == 0)).sum().float()
    false_negatives = ((pred == 0) & (target == 1)).sum().float()
    
    if (true_positives + false_positives) == 0:
        precision = 1.0
    else:
        precision = (true_positives / (true_positives + false_positives)).item()
    
    if (true_positives + false_negatives) == 0:
        recall = 1.0
    else:
        recall = (true_positives / (true_positives + false_negatives)).item()
    
    if (precision + recall) == 0:
        f1 = 0.0
    else:
        f1 = (2.0 * precision * recall / (precision + recall))
    
    return f1

def calculate_mf1(pred: torch.Tensor, target: torch.Tensor, num_classes: int = 2) -> float:
    """
    计算平均F1 Score（所有类别的平均F1）
    
    参数:
        pred: 预测值，形状为 (B, H, W) 或 (B, 1, H, W)，值为类别索引
        target: 目标值，形状为 (B, H, W) 或 (B, 1, H, W)，值为类别索引
        num_classes: 类别数
    
    返回:
        平均F1 Score标量
    """
    if len(pred.shape) == 4:
        pred = pred.squeeze(1)
    if len(target.shape) == 4:
        target = target.squeeze(1)
    
    if pred.dtype != torch.long:
        pred = (pred > 0.5).long()
    target = normalize_target_to_binary(target)
    
    f1_scores = []
    for cls in range(num_classes):
        pred_cls = (pred == cls)
        target_cls = (target == cls)
        
        true_positives = (pred_cls & target_cls).sum().float()
        false_positives = (pred_cls & ~target_cls).sum().float()
        false_negatives = (~pred_cls & target_cls).sum().float()
        
        if (true_positives + false_positives) == 0:
            precision = 1.0
        else:
            precision = (true_positives / (true_positives + false_positives)).item()
        
        if (true_positives + false_negatives) == 0:
            recall = 1.0
        else:
            recall = (true_positives / (true_positives + false_negatives)).item()
        
        if (precision + recall) == 0:
            f1 = 0.0
        else:
            f1 = (2.0 * precision * recall / (precision + recall))
        
        f1_scores.append(f1)
    
    mf1 = sum(f1_scores) / len(f1_scores)
    return mf1

def calculate_all_metrics(pred: torch.Tensor, target: torch.Tensor, num_classes: int = 2) -> dict:
    """
    计算所有评估指标
    
    参数:
        pred: 预测值，形状为 (B, H, W) 或 (B, 1, H, W)
        target: 目标值，形状为 (B, H, W) 或 (B, 1, H, W)
        num_classes: 类别数
    
    返回:
        包含所有指标的字典
    """
    metrics = {
        'miou': calculate_miou(pred, target, num_classes),
        'mprecision': calculate_mprecision(pred, target, num_classes),
        'recall': calculate_recall(pred, target),
        'mpa': calculate_mpa(pred, target, num_classes),
        'f1': calculate_f1(pred, target),
        'mf1': calculate_mf1(pred, target, num_classes)
    }

    return metrics



