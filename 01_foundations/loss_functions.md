---
title: "Loss Function Catalog for Image Segmentation"
date: 2025-03-06
status: in-progress
tags: [loss-functions, cross-entropy, dice-loss, focal-loss, tversky, lovasz, boundary-loss, training]
difficulty: intermediate
---

# Loss Function Catalog for Image Segmentation

The choice of loss function has a profound effect on segmentation model performance, especially under class imbalance, boundary ambiguity, or small object prevalence. This document catalogs the most commonly used loss functions with mathematical formulations, PyTorch implementations, and guidance on when to use each.

---

## Notation

- $p_i \in [0, 1]$: predicted probability for pixel $i$ belonging to the foreground (binary) or a specific class (multi-class)
- $g_i \in \{0, 1\}$: ground-truth label for pixel $i$
- $N$: total number of pixels
- $K$: number of classes
- $p_{i,c}$: predicted probability of pixel $i$ for class $c$
- $g_{i,c}$: ground-truth one-hot encoding of pixel $i$ for class $c$

---

## 1. Cross-Entropy Loss

### Binary Cross-Entropy (BCE)

$$\mathcal{L}_{\text{BCE}} = -\frac{1}{N} \sum_{i=1}^{N} \Big[ g_i \log(p_i) + (1 - g_i) \log(1 - p_i) \Big]$$

### Multi-Class Cross-Entropy

$$\mathcal{L}_{\text{CE}} = -\frac{1}{N} \sum_{i=1}^{N} \sum_{c=1}^{K} g_{i,c} \log(p_{i,c})$$

### Weighted Cross-Entropy

Apply class weights $w_c$ to address class imbalance:

$$\mathcal{L}_{\text{WCE}} = -\frac{1}{N} \sum_{i=1}^{N} \sum_{c=1}^{K} w_c \, g_{i,c} \log(p_{i,c})$$

Common weight strategies:
- **Inverse frequency:** $w_c = \frac{N}{K \cdot N_c}$ where $N_c$ is the number of pixels of class $c$.
- **Median frequency balancing:** $w_c = \frac{\text{median}(\{N_1, \dots, N_K\})}{N_c}$.

### Pros
- Well-understood, convex (in logit space), stable gradients.
- Works well as a general-purpose baseline loss.
- Directly optimizes per-pixel classification accuracy.

### Cons
- Treats each pixel independently -- ignores spatial structure.
- Sensitive to class imbalance: the loss is dominated by the majority class unless weights are applied.
- Does not directly optimize overlap metrics (IoU, Dice).

### PyTorch Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropySegmentationLoss(nn.Module):
    """Standard cross-entropy loss for multi-class segmentation.

    Args:
        weight: Optional class weights tensor of shape (K,).
        ignore_index: Label index to ignore (e.g., 255 for void pixels).
    """
    def __init__(self, weight=None, ignore_index=255):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)

    def forward(self, logits, targets):
        """
        Args:
            logits: (B, K, H, W) raw model output (before softmax).
            targets: (B, H, W) integer class labels.
        """
        return self.ce(logits, targets)
```

---

## 2. Dice Loss

### Motivation

Dice loss directly optimizes the Dice coefficient (F1 score), making it robust to class imbalance without requiring explicit class weights. Particularly popular in medical image segmentation.

### Formula (Binary)

$$\mathcal{L}_{\text{Dice}} = 1 - \frac{2 \sum_{i=1}^{N} p_i \, g_i + \epsilon}{\sum_{i=1}^{N} p_i + \sum_{i=1}^{N} g_i + \epsilon}$$

where $\epsilon$ is a small smoothing constant (e.g., $10^{-5}$) to prevent division by zero.

### Formula (Multi-Class)

Compute per-class Dice and average:

$$\mathcal{L}_{\text{Dice}} = 1 - \frac{1}{K} \sum_{c=1}^{K} \frac{2 \sum_{i} p_{i,c} \, g_{i,c} + \epsilon}{\sum_{i} p_{i,c} + \sum_{i} g_{i,c} + \epsilon}$$

### Pros
- Naturally handles class imbalance: small objects contribute proportionally to the loss.
- Directly related to the Dice evaluation metric.
- Smooth and differentiable (the soft Dice formulation using probabilities).

### Cons
- **Gradient instability** when the foreground region is very small: the denominator approaches $\epsilon$, leading to large, noisy gradients.
- Non-convex -- more difficult optimization landscape than cross-entropy.
- Can produce poorly calibrated probabilities (since it optimizes overlap, not per-pixel likelihood).
- Training can be unstable if used alone; often combined with cross-entropy.

### PyTorch Implementation

```python
class DiceLoss(nn.Module):
    """Soft Dice loss for multi-class segmentation.

    Args:
        smooth: Smoothing constant to avoid division by zero.
        per_image: If True, compute Dice per image then average.
                   If False, compute over the entire batch (recommended).
    """
    def __init__(self, smooth=1e-5, per_image=False):
        super().__init__()
        self.smooth = smooth
        self.per_image = per_image

    def forward(self, logits, targets):
        """
        Args:
            logits: (B, K, H, W) raw model output.
            targets: (B, H, W) integer class labels.
        """
        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)  # (B, K, H, W)

        # One-hot encode targets: (B, H, W) -> (B, K, H, W)
        targets_one_hot = F.one_hot(targets, num_classes)  # (B, H, W, K)
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()  # (B, K, H, W)

        dims = (0, 2, 3) if not self.per_image else (2, 3)

        intersection = (probs * targets_one_hot).sum(dim=dims)
        cardinality = probs.sum(dim=dims) + targets_one_hot.sum(dim=dims)

        dice_score = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - dice_score.mean()
```

---

## 3. Focal Loss

### Motivation

Introduced by Lin et al. (2017) for object detection (RetinaNet). Addresses the extreme foreground-background imbalance by down-weighting easy, well-classified examples and focusing training on hard examples.

### Formula (Binary)

$$\mathcal{L}_{\text{Focal}} = -\frac{1}{N} \sum_{i=1}^{N} \Big[ \alpha \, g_i (1 - p_i)^\gamma \log(p_i) + (1 - \alpha)(1 - g_i) \, p_i^\gamma \log(1 - p_i) \Big]$$

where:
- $\gamma \geq 0$ is the **focusing parameter**. When $\gamma = 0$, focal loss reduces to standard weighted cross-entropy. Typical values: $\gamma \in \{1, 2, 3\}$; $\gamma = 2$ is the most common default.
- $\alpha \in [0, 1]$ balances the importance of positive vs. negative classes.

### Intuition

Consider a well-classified foreground pixel with $p_i = 0.95$. The modulating factor $(1 - 0.95)^2 = 0.0025$ reduces its loss contribution by 400x compared to standard CE. Conversely, a hard example with $p_i = 0.1$ has a factor of $(0.9)^2 = 0.81$, losing only ~20% of its original contribution.

### Multi-Class Extension

$$\mathcal{L}_{\text{Focal}} = -\frac{1}{N} \sum_{i=1}^{N} \sum_{c=1}^{K} \alpha_c \, g_{i,c} (1 - p_{i,c})^\gamma \log(p_{i,c})$$

### Pros
- Elegantly handles class imbalance through the modulating factor, without discarding data or complex sampling.
- Easy to implement -- a minor modification of cross-entropy.
- Particularly effective when there is a large imbalance between easy background pixels and hard foreground pixels.

### Cons
- Introduces two hyperparameters ($\alpha$, $\gamma$) that require tuning.
- May underweight the learning signal from majority-class pixels too aggressively, hurting performance on well-represented classes.
- Does not optimize overlap metrics directly.

### PyTorch Implementation

```python
class FocalLoss(nn.Module):
    """Focal loss for multi-class segmentation.

    Args:
        alpha: Class balancing weight. Can be a scalar or tensor of shape (K,).
        gamma: Focusing parameter. Higher values focus more on hard examples.
        ignore_index: Label index to ignore.
    """
    def __init__(self, alpha=0.25, gamma=2.0, ignore_index=255):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        """
        Args:
            logits: (B, K, H, W) raw model output.
            targets: (B, H, W) integer class labels.
        """
        num_classes = logits.shape[1]

        # Create mask for valid pixels
        valid_mask = (targets != self.ignore_index)
        targets_safe = targets.clone()
        targets_safe[~valid_mask] = 0

        # Compute softmax probabilities
        probs = F.softmax(logits, dim=1)  # (B, K, H, W)

        # Gather probabilities for the true class
        targets_one_hot = F.one_hot(targets_safe, num_classes)  # (B, H, W, K)
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()  # (B, K, H, W)

        pt = (probs * targets_one_hot).sum(dim=1)  # (B, H, W)

        # Compute focal weight
        focal_weight = (1.0 - pt) ** self.gamma

        # Compute CE
        ce_loss = F.cross_entropy(logits, targets_safe, reduction='none')  # (B, H, W)

        # Apply focal weight and alpha
        loss = self.alpha * focal_weight * ce_loss

        # Mask invalid pixels
        loss = loss * valid_mask.float()

        return loss.sum() / valid_mask.float().sum().clamp(min=1.0)
```

---

## 4. Tversky Loss

### Motivation

Generalizes Dice loss by allowing asymmetric weighting of false positives and false negatives. Useful when recall (sensitivity) is more important than precision, or vice versa.

### Formula

$$\mathcal{L}_{\text{Tversky}} = 1 - \frac{\sum_{i} p_i \, g_i + \epsilon}{\sum_{i} p_i \, g_i + \alpha \sum_{i} p_i (1 - g_i) + \beta \sum_{i} (1 - p_i) g_i + \epsilon}$$

where:
- $\alpha$ controls the penalty for false positives ($p_i = 1, g_i = 0$).
- $\beta$ controls the penalty for false negatives ($p_i = 0, g_i = 1$).
- When $\alpha = \beta = 0.5$, Tversky loss reduces to Dice loss.
- When $\alpha = \beta = 1.0$, it reduces to the Jaccard (IoU) loss.

### Focal Tversky Loss

Apply a focal-like power term to further emphasize hard examples:

$$\mathcal{L}_{\text{FTL}} = (1 - TI)^{1/\gamma}$$

where $TI$ is the Tversky Index (the expression inside the subtraction above) and $\gamma$ controls focusing. Proposed by Abraham & Khan (2019).

### Pros
- Flexibility to tune the precision-recall trade-off via $\alpha$ and $\beta$.
- Setting $\beta > \alpha$ emphasizes recall, which is often desired in medical segmentation (better to over-segment than miss a lesion).
- Subsumes Dice and IoU losses as special cases.

### Cons
- Two additional hyperparameters to tune.
- Shares the gradient instability issues of Dice loss for very small regions.
- Less commonly used than Dice or CE, so there is less empirical guidance on hyperparameter settings.

### PyTorch Implementation

```python
class TverskyLoss(nn.Module):
    """Tversky loss for binary or multi-class segmentation.

    Args:
        alpha: Weight for false positives.
        beta: Weight for false negatives.
        smooth: Smoothing constant.
    """
    def __init__(self, alpha=0.3, beta=0.7, smooth=1e-5):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, logits, targets):
        """
        Args:
            logits: (B, K, H, W) raw model output.
            targets: (B, H, W) integer class labels.
        """
        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)

        targets_one_hot = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()

        # Compute Tversky index per class
        tp = (probs * targets_one_hot).sum(dim=(0, 2, 3))
        fp = (probs * (1 - targets_one_hot)).sum(dim=(0, 2, 3))
        fn = ((1 - probs) * targets_one_hot).sum(dim=(0, 2, 3))

        tversky_index = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)

        return 1.0 - tversky_index.mean()
```

---

## 5. Lovasz-Softmax Loss

### Motivation

Directly optimizes the Jaccard index (IoU) by exploiting the Lovasz extension of submodular functions. Unlike Dice loss, which uses a soft approximation of the Dice coefficient, Lovasz-Softmax provides a convex surrogate that is a tighter approximation of the actual IoU loss.

### Mathematical Background

The Jaccard loss for class $c$ is:

$$\Delta_{J_c}(\hat{y}, y) = 1 - \frac{|y_c \cap \hat{y}_c|}{|y_c \cup \hat{y}_c|}$$

This is a submodular set function of the mis-predictions. The Lovasz extension $\bar{\Delta}_{J_c}$ provides a tight convex relaxation. For a vector of pixel errors $\mathbf{m}(c)$ sorted in decreasing order:

$$\bar{\Delta}_{J_c}(\mathbf{m}(c)) = \sum_{i=1}^{N} m_{\pi_i}(c) \cdot g_{\pi_i}$$

where $\pi$ is the sorted permutation and $g_{\pi_i}$ are the incremental changes in the Jaccard loss.

### Pros
- **Theoretically grounded:** Directly optimizes a convex surrogate of the IoU, unlike Dice loss which optimizes a related but different quantity.
- Consistently outperforms or matches Dice loss on semantic segmentation benchmarks.
- No per-class weighting needed -- the loss naturally focuses on classes with poor IoU.

### Cons
- More complex to implement than Dice or CE.
- Requires sorting per class per batch, adding computational overhead.
- Less intuitive than other losses.
- The per-image variant can be noisy; batch-level computation is preferred.

### PyTorch Implementation

```python
class LovaszSoftmaxLoss(nn.Module):
    """Lovász-Softmax loss for multi-class segmentation.

    Based on: Berman, Triki, & Blaschko (2018). "The Lovász-Softmax loss:
    A tractable surrogate for the optimization of the intersection-over-union
    measure in neural networks." CVPR.
    """
    def __init__(self, per_image=False, ignore_index=255):
        super().__init__()
        self.per_image = per_image
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        """
        Args:
            logits: (B, K, H, W) raw model output.
            targets: (B, H, W) integer class labels.
        """
        probas = F.softmax(logits, dim=1)
        if self.per_image:
            losses = []
            for prob, target in zip(probas, targets):
                losses.append(self._lovasz_softmax_flat(
                    prob.unsqueeze(0), target.unsqueeze(0)))
            return torch.stack(losses).mean()
        else:
            return self._lovasz_softmax_flat(probas, targets)

    def _lovasz_softmax_flat(self, probas, targets):
        B, C, H, W = probas.shape
        probas = probas.permute(0, 2, 3, 1).reshape(-1, C)  # (N, C)
        targets = targets.reshape(-1)  # (N,)

        # Filter out ignore index
        valid = targets != self.ignore_index
        probas = probas[valid]
        targets = targets[valid]

        if probas.numel() == 0:
            return probas.sum() * 0.0

        losses = []
        for c in range(C):
            fg = (targets == c).float()
            if fg.sum() == 0 and (1 - fg).sum() == 0:
                continue
            errors = (1.0 - probas[:, c]) * fg + probas[:, c] * (1.0 - fg)
            errors_sorted, perm = torch.sort(errors, descending=True)
            fg_sorted = fg[perm]
            grad = self._lovasz_grad(fg_sorted)
            losses.append(torch.dot(F.relu(errors_sorted), grad))

        return torch.stack(losses).mean() if losses else torch.tensor(0.0, device=probas.device)

    @staticmethod
    def _lovasz_grad(gt_sorted):
        """Compute the gradient of the Lovász extension w.r.t. sorted errors."""
        n = len(gt_sorted)
        gts = gt_sorted.sum()
        intersection = gts - gt_sorted.float().cumsum(0)
        union = gts + (1 - gt_sorted).float().cumsum(0)
        jaccard = 1.0 - intersection / union
        if n > 1:
            jaccard[1:] = jaccard[1:] - jaccard[:-1]
        return jaccard
```

---

## 6. Boundary Loss

### Motivation

Overlap-based losses (Dice, CE) treat all pixels equally regardless of their distance from the object boundary. Boundary loss explicitly penalizes boundary errors by incorporating distance information, which is especially useful for segmenting objects with elongated or thin structures.

### Formula

Introduced by Kervadec et al. (2019). The boundary loss is formulated as the integral of the distance to the boundary over the predicted region:

$$\mathcal{L}_{\text{Boundary}} = \sum_{i=1}^{N} \phi_G(i) \cdot (p_i - g_i)$$

where $\phi_G(i)$ is the signed distance map computed from the ground-truth boundary:

$$\phi_G(i) = \begin{cases} -D(\partial G, i) & \text{if } i \in G \\ D(\partial G, i) & \text{if } i \notin G \end{cases}$$

$D(\partial G, i)$ is the Euclidean distance from pixel $i$ to the nearest ground-truth boundary point $\partial G$.

### Intuition

- Pixels far inside the ground truth have large negative $\phi_G$ values. Predicting them as foreground ($p_i$ large) yields a large negative (good) loss contribution.
- Pixels far outside the ground truth have large positive $\phi_G$ values. Predicting them as foreground yields a large positive (bad) loss contribution.
- Pixels near the boundary have $\phi_G \approx 0$ and contribute little.

### Training Strategy

Boundary loss alone is not sufficient (it needs a coarse initial segmentation to provide meaningful gradients). The recommended approach is to start training with Dice loss and gradually increase the boundary loss weight:

$$\mathcal{L} = (1 - \lambda) \, \mathcal{L}_{\text{Dice}} + \lambda \, \mathcal{L}_{\text{Boundary}}$$

where $\lambda$ increases linearly from 0 to 1 during training (e.g., over the first 50% of epochs).

### Pros
- Directly penalizes distance from the true boundary, improving Hausdorff distance.
- Effective for thin, elongated structures where overlap losses struggle (e.g., blood vessels, nerve fibers).
- Differentiable and easy to combine with other losses.

### Cons
- Requires precomputing distance transform maps for each ground-truth mask (adds preprocessing overhead).
- Not meaningful on its own -- must be combined with a region-based loss.
- The distance map must be recomputed whenever the ground truth changes (e.g., with data augmentation that modifies geometry).

### PyTorch Implementation

```python
from scipy.ndimage import distance_transform_edt


class BoundaryLoss(nn.Module):
    """Boundary loss using signed distance maps.

    Note: Distance maps must be precomputed from ground-truth masks
    and passed alongside targets. This implementation computes them
    on-the-fly for simplicity, but precomputation is recommended for
    efficiency.
    """
    def __init__(self):
        super().__init__()

    def forward(self, logits, dist_maps):
        """
        Args:
            logits: (B, 1, H, W) raw model output for binary segmentation.
            dist_maps: (B, 1, H, W) precomputed signed distance maps.
        """
        probs = torch.sigmoid(logits)
        return (probs * dist_maps).mean()

    @staticmethod
    def compute_distance_map(mask):
        """Compute signed distance map from a binary mask.

        Args:
            mask: numpy array of shape (H, W), binary ground truth.

        Returns:
            Signed distance map: negative inside, positive outside.
        """
        import numpy as np
        mask = mask.astype(bool)
        if mask.any():
            pos_dist = distance_transform_edt(mask)
            neg_dist = distance_transform_edt(~mask)
            dist_map = neg_dist - pos_dist  # positive outside, negative inside
        else:
            dist_map = distance_transform_edt(~mask)
        return dist_map.astype(np.float32)
```

---

## 7. Combo Loss

### Motivation

Combine the strengths of multiple loss functions to compensate for individual weaknesses. The most common combination is cross-entropy (stable gradients, per-pixel optimization) plus Dice loss (overlap-aware, imbalance-robust).

### Common Combinations

**CE + Dice (most popular):**

$$\mathcal{L}_{\text{Combo}} = \alpha \, \mathcal{L}_{\text{CE}} + (1 - \alpha) \, \mathcal{L}_{\text{Dice}}$$

Typical: $\alpha = 0.5$ (equal weight).

**CE + Dice + Boundary:**

$$\mathcal{L} = \alpha \, \mathcal{L}_{\text{CE}} + \beta \, \mathcal{L}_{\text{Dice}} + \gamma \, \mathcal{L}_{\text{Boundary}}$$

**Focal + Dice:**

$$\mathcal{L} = \alpha \, \mathcal{L}_{\text{Focal}} + (1 - \alpha) \, \mathcal{L}_{\text{Dice}}$$

### Combo Loss (Tag et al., 2019)

The specific "Combo Loss" from the literature:

$$\mathcal{L}_{\text{Combo}} = \alpha \Big(-\frac{1}{N}\sum_i \big[\beta \, g_i \log p_i + (1-\beta)(1-g_i)\log(1-p_i)\big]\Big) + (1-\alpha)\Big(1 - \frac{2\sum_i p_i g_i + \epsilon}{\sum_i p_i + \sum_i g_i + \epsilon}\Big)$$

where $\alpha$ balances the two terms and $\beta$ controls the CE weighting.

### Pros
- Combines stable gradient flow (from CE) with overlap optimization (from Dice).
- Empirically, CE + Dice consistently outperforms either loss alone across many segmentation tasks.
- Flexible: can incorporate any combination of losses.

### Cons
- Introduces weighting hyperparameters between loss terms.
- Loss terms may have different scales -- ensure they are normalized or the weighting may be misleading.
- More complex to tune and debug.

### PyTorch Implementation

```python
class ComboLoss(nn.Module):
    """Combination of Cross-Entropy and Dice loss.

    Args:
        alpha: Weight for CE component. (1 - alpha) is used for Dice.
        ce_weight: Optional per-class weights for CE.
        ignore_index: Label to ignore.
    """
    def __init__(self, alpha=0.5, ce_weight=None, ignore_index=255):
        super().__init__()
        self.alpha = alpha
        self.ce = nn.CrossEntropyLoss(weight=ce_weight, ignore_index=ignore_index)
        self.dice = DiceLoss(smooth=1e-5)

    def forward(self, logits, targets):
        """
        Args:
            logits: (B, K, H, W) raw model output.
            targets: (B, H, W) integer class labels.
        """
        ce_loss = self.ce(logits, targets)
        dice_loss = self.dice(logits, targets)
        return self.alpha * ce_loss + (1 - self.alpha) * dice_loss
```

---

## 8. Comparison Summary

| Loss | Handles Imbalance | Optimizes Overlap | Boundary Aware | Gradient Stability | Hyperparameters |
|------|:-:|:-:|:-:|:-:|:--|
| Cross-Entropy | With weights | No | No | High | weight, ignore_index |
| Dice | Yes (inherent) | Yes (Dice/F1) | No | Medium | smooth |
| Focal | Yes (modulating factor) | No | No | High | $\alpha$, $\gamma$ |
| Tversky | Yes (tunable FP/FN) | Yes (Tversky Index) | No | Medium | $\alpha$, $\beta$ |
| Lovasz-Softmax | Yes (inherent) | Yes (IoU) | No | Medium | per_image |
| Boundary | N/A (complementary) | No | Yes | Low (needs warm-up) | $\lambda$ schedule |
| Combo (CE + Dice) | Yes | Yes | No | High | $\alpha$ |

---

## 9. Practical Recommendations

1. **Start with CE + Dice** ($\alpha = 0.5$). This is the strongest general-purpose baseline and works well across medical, natural, and remote sensing domains.

2. **For extreme imbalance** (foreground < 5% of pixels), consider Focal Loss or Tversky Loss with $\beta > \alpha$.

3. **For boundary-critical tasks** (vessels, cell membranes), add Boundary Loss with gradual warm-up.

4. **For direct IoU optimization**, use Lovasz-Softmax. It often provides a small but consistent gain over Dice loss.

5. **Deep supervision:** When using architectures with auxiliary outputs at multiple scales (e.g., U-Net++), apply the loss at each scale with decreasing weight.

6. **Online Hard Example Mining (OHEM):** An alternative to focal loss -- compute the loss for all pixels, then backpropagate only through the top-$k$ hardest pixels. Can be combined with any loss function.

---

## References

1. Lin, T.-Y., Goyal, P., Girshick, R., He, K., & Dollar, P. (2017). Focal Loss for Dense Object Detection. ICCV.
2. Milletari, F., Navab, N., & Ahmadi, S.-A. (2016). V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation. 3DV.
3. Salehi, S. S. M., Erdogmus, D., & Gholipour, A. (2017). Tversky Loss Function for Image Segmentation Using 3D Fully Convolutional Deep Networks. MLMI.
4. Berman, M., Triki, A. R., & Blaschko, M. B. (2018). The Lovasz-Softmax Loss. CVPR.
5. Kervadec, H., Bouchtiba, J., Desrosiers, C., Granger, E., de Guise, J., & Ben Ayed, I. (2019). Boundary Loss for Highly Unbalanced Segmentation. MIDL.
6. Abraham, N., & Khan, N. M. (2019). A Novel Focal Tversky Loss Function with Improved Attention U-Net for Lesion Segmentation. ISBI.
7. Taghanaki, S. A., et al. (2019). Combo Loss: Handling Input and Output Imbalance in Multi-Organ Segmentation. CMIG.
