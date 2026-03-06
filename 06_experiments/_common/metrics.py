"""
Common evaluation metrics for image segmentation.

Provides IoU (Intersection over Union), Dice coefficient, and pixel accuracy
metrics implemented in PyTorch with support for both binary and multi-class
segmentation tasks.

Usage:
    from _common.metrics import iou_score, dice_score, pixel_accuracy

    # Binary segmentation
    iou = iou_score(pred_mask, true_mask)
    dice = dice_score(pred_mask, true_mask)
    acc = pixel_accuracy(pred_mask, true_mask)

    # Multi-class segmentation
    iou = iou_score(pred_mask, true_mask, num_classes=5)
"""

import torch
import torch.nn.functional as F
from typing import Optional


def iou_score(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int = 2,
    smooth: float = 1e-6,
    per_class: bool = False,
    ignore_index: Optional[int] = None,
) -> torch.Tensor:
    """Compute Intersection over Union (Jaccard Index).

    Args:
        pred: Predicted logits of shape (B, C, H, W) or binary mask (B, 1, H, W).
        target: Ground truth labels of shape (B, H, W) with integer class indices,
                or binary mask of shape (B, 1, H, W).
        num_classes: Number of segmentation classes.
        smooth: Smoothing factor to avoid division by zero.
        per_class: If True, return per-class IoU. Otherwise, return mean IoU.
        ignore_index: Class index to ignore in the computation.

    Returns:
        IoU score as a scalar tensor (mean IoU) or tensor of shape (C,) if per_class=True.
    """
    if pred.dim() == 4 and pred.shape[1] > 1:
        # Multi-class: convert logits to class predictions
        pred_classes = pred.argmax(dim=1)  # (B, H, W)
    elif pred.dim() == 4 and pred.shape[1] == 1:
        # Binary: threshold at 0.5 (assumes sigmoid already applied or logits > 0)
        pred_classes = (pred.squeeze(1) > 0).long()
    else:
        pred_classes = pred.long()

    if target.dim() == 4:
        target = target.squeeze(1)
    target = target.long()

    ious = []
    for cls in range(num_classes):
        if ignore_index is not None and cls == ignore_index:
            continue

        pred_mask = (pred_classes == cls)
        target_mask = (target == cls)

        intersection = (pred_mask & target_mask).float().sum()
        union = (pred_mask | target_mask).float().sum()

        iou = (intersection + smooth) / (union + smooth)
        ious.append(iou)

    ious = torch.stack(ious)

    if per_class:
        return ious
    return ious.mean()


def dice_score(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int = 2,
    smooth: float = 1e-6,
    per_class: bool = False,
    ignore_index: Optional[int] = None,
) -> torch.Tensor:
    """Compute Dice coefficient (F1 score for segmentation).

    Args:
        pred: Predicted logits of shape (B, C, H, W) or binary mask (B, 1, H, W).
        target: Ground truth labels of shape (B, H, W) with integer class indices,
                or binary mask of shape (B, 1, H, W).
        num_classes: Number of segmentation classes.
        smooth: Smoothing factor to avoid division by zero.
        per_class: If True, return per-class Dice. Otherwise, return mean Dice.
        ignore_index: Class index to ignore in the computation.

    Returns:
        Dice score as a scalar tensor (mean Dice) or tensor of shape (C,) if per_class=True.
    """
    if pred.dim() == 4 and pred.shape[1] > 1:
        pred_classes = pred.argmax(dim=1)
    elif pred.dim() == 4 and pred.shape[1] == 1:
        pred_classes = (pred.squeeze(1) > 0).long()
    else:
        pred_classes = pred.long()

    if target.dim() == 4:
        target = target.squeeze(1)
    target = target.long()

    dices = []
    for cls in range(num_classes):
        if ignore_index is not None and cls == ignore_index:
            continue

        pred_mask = (pred_classes == cls).float()
        target_mask = (target == cls).float()

        intersection = (pred_mask * target_mask).sum()
        dice = (2.0 * intersection + smooth) / (pred_mask.sum() + target_mask.sum() + smooth)
        dices.append(dice)

    dices = torch.stack(dices)

    if per_class:
        return dices
    return dices.mean()


def pixel_accuracy(
    pred: torch.Tensor,
    target: torch.Tensor,
    ignore_index: Optional[int] = None,
) -> torch.Tensor:
    """Compute pixel-wise accuracy.

    Args:
        pred: Predicted logits of shape (B, C, H, W) or binary mask (B, 1, H, W).
        target: Ground truth labels of shape (B, H, W) with integer class indices.
        ignore_index: Class index to ignore in the computation.

    Returns:
        Pixel accuracy as a scalar tensor.
    """
    if pred.dim() == 4 and pred.shape[1] > 1:
        pred_classes = pred.argmax(dim=1)
    elif pred.dim() == 4 and pred.shape[1] == 1:
        pred_classes = (pred.squeeze(1) > 0).long()
    else:
        pred_classes = pred.long()

    if target.dim() == 4:
        target = target.squeeze(1)
    target = target.long()

    if ignore_index is not None:
        valid_mask = target != ignore_index
        correct = ((pred_classes == target) & valid_mask).float().sum()
        total = valid_mask.float().sum()
    else:
        correct = (pred_classes == target).float().sum()
        total = torch.tensor(target.numel(), dtype=torch.float32, device=target.device)

    return correct / total.clamp(min=1)


def confusion_matrix(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    """Compute confusion matrix for segmentation predictions.

    Args:
        pred: Predicted logits of shape (B, C, H, W) or class indices (B, H, W).
        target: Ground truth labels of shape (B, H, W).
        num_classes: Number of classes.

    Returns:
        Confusion matrix of shape (num_classes, num_classes).
        Entry (i, j) = number of pixels with true class i predicted as class j.
    """
    if pred.dim() == 4 and pred.shape[1] > 1:
        pred_classes = pred.argmax(dim=1)
    elif pred.dim() == 4 and pred.shape[1] == 1:
        pred_classes = (pred.squeeze(1) > 0).long()
    else:
        pred_classes = pred.long()

    if target.dim() == 4:
        target = target.squeeze(1)
    target = target.long()

    mask = (target >= 0) & (target < num_classes)
    idx = num_classes * target[mask] + pred_classes[mask]
    cm = torch.bincount(idx, minlength=num_classes ** 2).reshape(num_classes, num_classes)

    return cm
