"""Numpy evaluation metrics for the Segmentation Lab.

Classical algorithms output *unlabeled* segments (segment 3 has no idea it
is "the cat"), so before computing supervised metrics the predicted
segments must be mapped onto ground-truth classes. Two standard mappings
are provided:

- ``majority`` (many-to-one): every predicted segment is assigned the GT
  class it overlaps most. This measures the *achievable* accuracy of a
  segmentation and flatters over-segmentation (many tiny segments can
  each pick their best class).
- ``hungarian`` (one-to-one): optimal bipartite matching between segments
  and classes; leftover segments stay unmatched and count as errors.
  Harsher, but honest about over- and under-segmentation.

The gap between the two mappings is itself a diagnostic: a large gap
means the algorithm over-segments.

All functions take integer label maps of shape (H, W).
"""

from __future__ import annotations

import numpy as np


def contingency_table(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Return matrix C where C[i, j] = pixels with pred segment i and GT class j."""
    n_pred = int(pred.max()) + 1
    n_gt = int(gt.max()) + 1
    idx = pred.ravel().astype(np.int64) * n_gt + gt.ravel().astype(np.int64)
    return np.bincount(idx, minlength=n_pred * n_gt).reshape(n_pred, n_gt)


def relabel_by_overlap(
    pred: np.ndarray, gt: np.ndarray, method: str = "majority"
) -> tuple[np.ndarray, int]:
    """Map raw predicted segments onto GT class ids.

    Returns (mapped label map with values in {-1} ∪ [0, n_classes), and
    the number of raw predicted segments). Unmatched segments become -1.
    """
    pred = np.ascontiguousarray(pred)
    # Compact segment ids to 0..n-1
    uniq, pred_c = np.unique(pred, return_inverse=True)
    pred_c = pred_c.reshape(pred.shape)
    table = contingency_table(pred_c, gt)
    n_segments = table.shape[0]

    if method == "hungarian":
        from scipy.optimize import linear_sum_assignment

        rows, cols = linear_sum_assignment(-table)
        mapping = np.full(n_segments, -1, dtype=np.int64)
        for r, c in zip(rows, cols):
            if table[r, c] > 0:
                mapping[r] = c
    else:  # majority
        mapping = table.argmax(axis=1)

    return mapping[pred_c], n_segments


def pixel_accuracy(mapped: np.ndarray, gt: np.ndarray) -> float:
    """Fraction of pixels whose mapped prediction equals the GT class."""
    return float((mapped == gt).mean())


def iou_dice_per_class(
    mapped: np.ndarray, gt: np.ndarray, n_classes: int
) -> tuple[np.ndarray, np.ndarray]:
    """Per-class IoU and Dice. Classes absent from the GT are NaN."""
    ious = np.full(n_classes, np.nan)
    dices = np.full(n_classes, np.nan)
    for c in range(n_classes):
        gt_c = gt == c
        pr_c = mapped == c
        gt_sum = gt_c.sum()
        if gt_sum == 0:
            continue
        inter = np.logical_and(gt_c, pr_c).sum()
        union = np.logical_or(gt_c, pr_c).sum()
        ious[c] = inter / union if union else np.nan
        denom = gt_sum + pr_c.sum()
        dices[c] = 2.0 * inter / denom if denom else np.nan
    return ious, dices


def _boundary(mask: np.ndarray) -> np.ndarray:
    """One-pixel-wide inner boundary of a binary mask."""
    from scipy.ndimage import binary_erosion

    return mask & ~binary_erosion(mask, border_value=1)


def boundary_f1_per_class(
    mapped: np.ndarray, gt: np.ndarray, n_classes: int, tolerance: int = 2
) -> np.ndarray:
    """Boundary F1 (BF score): precision/recall of boundary pixels within a tolerance."""
    from scipy.ndimage import distance_transform_edt

    scores = np.full(n_classes, np.nan)
    for c in range(n_classes):
        gt_c = gt == c
        if gt_c.sum() == 0:
            continue
        gt_b = _boundary(gt_c)
        pr_b = _boundary(mapped == c)
        n_gt, n_pr = gt_b.sum(), pr_b.sum()
        if n_pr == 0 or n_gt == 0:
            scores[c] = 0.0
            continue
        # Distance from every pixel to nearest boundary pixel of the other set
        d_to_gt = distance_transform_edt(~gt_b)
        d_to_pr = distance_transform_edt(~pr_b)
        precision = (d_to_gt[pr_b] <= tolerance).mean()
        recall = (d_to_pr[gt_b] <= tolerance).mean()
        if precision + recall > 0:
            scores[c] = 2 * precision * recall / (precision + recall)
        else:
            scores[c] = 0.0
    return scores


def hausdorff95_per_class(
    mapped: np.ndarray, gt: np.ndarray, n_classes: int
) -> np.ndarray:
    """Symmetric 95th-percentile Hausdorff distance per class (pixels).

    NaN when the class is absent from the GT; the image diagonal when the
    prediction missed the class entirely (worst case, keeps the mean finite).
    """
    from scipy.ndimage import distance_transform_edt

    diag = float(np.hypot(*gt.shape))
    dists = np.full(n_classes, np.nan)
    for c in range(n_classes):
        gt_c = gt == c
        if gt_c.sum() == 0:
            continue
        pr_c = mapped == c
        if pr_c.sum() == 0:
            dists[c] = diag
            continue
        gt_b, pr_b = _boundary(gt_c), _boundary(pr_c)
        if gt_b.sum() == 0 or pr_b.sum() == 0:
            dists[c] = 0.0 if (gt_c == pr_c).all() else diag
            continue
        d_to_gt = distance_transform_edt(~gt_b)
        d_to_pr = distance_transform_edt(~pr_b)
        d1 = d_to_gt[pr_b]  # pred boundary -> nearest GT boundary
        d2 = d_to_pr[gt_b]  # GT boundary -> nearest pred boundary
        dists[c] = max(np.percentile(d1, 95), np.percentile(d2, 95))
    return dists


def adjusted_rand_index(pred: np.ndarray, gt: np.ndarray) -> float:
    """Adjusted Rand Index between two labelings (no class mapping needed).

    Measures pairwise agreement: do two pixels that share a GT class also
    share a predicted segment? Chance-corrected: 1 = identical partition
    (up to renaming), 0 ≈ random, can be slightly negative.
    """
    table = contingency_table(pred, gt).astype(np.float64)
    n = table.sum()
    sum_comb = (table * (table - 1) / 2).sum()
    a = table.sum(axis=1)
    b = table.sum(axis=0)
    comb_a = (a * (a - 1) / 2).sum()
    comb_b = (b * (b - 1) / 2).sum()
    total = n * (n - 1) / 2
    expected = comb_a * comb_b / total if total else 0.0
    max_index = (comb_a + comb_b) / 2
    if max_index == expected:
        return 1.0
    return float((sum_comb - expected) / (max_index - expected))


def compute_all(
    pred_raw: np.ndarray,
    gt: np.ndarray,
    n_classes: int,
    mapping: str = "majority",
    boundary_tol: int = 2,
) -> dict:
    """Compute the full metric suite for one prediction against the GT.

    Returns a dict with scalar metrics, per-class arrays, and the mapped
    label map (for display).
    """
    # Compact raw segments before mapping so counts are meaningful
    mapped, n_segments = relabel_by_overlap(pred_raw, gt, method=mapping)
    ious, dices = iou_dice_per_class(mapped, gt, n_classes)
    bf1 = boundary_f1_per_class(mapped, gt, n_classes, tolerance=boundary_tol)
    hd95 = hausdorff95_per_class(mapped, gt, n_classes)

    return {
        "n_segments": n_segments,
        "mapped": mapped,
        "pixel_accuracy": pixel_accuracy(mapped, gt),
        "mean_iou": float(np.nanmean(ious)),
        "mean_dice": float(np.nanmean(dices)),
        "boundary_f1": float(np.nanmean(bf1)),
        "hd95": float(np.nanmean(hd95)),
        "ari": adjusted_rand_index(pred_raw, gt),
        "per_class": {"iou": ious, "dice": dices, "boundary_f1": bf1, "hd95": hd95},
    }
