---
title: "Evaluation Metrics for Image Segmentation"
date: 2025-03-06
status: in-progress
tags: [evaluation, metrics, IoU, Dice, mIoU, PQ, AP, Hausdorff, pixel-accuracy]
difficulty: intermediate
---

# Evaluation Metrics for Image Segmentation

Choosing the right evaluation metric is as important as choosing the right model. Different metrics emphasize different aspects of segmentation quality -- region overlap, boundary accuracy, class balance, or instance-level recognition. This document provides a comprehensive reference for the most widely used segmentation metrics with formulas, intuitions, and trade-offs.

---

## Notation

Throughout this document:

- $\hat{Y}$ = predicted segmentation (binary mask or label map)
- $Y$ = ground-truth segmentation
- $TP$ = true positives (correctly predicted foreground pixels)
- $FP$ = false positives (background pixels incorrectly predicted as foreground)
- $FN$ = false negatives (foreground pixels missed by the prediction)
- $TN$ = true negatives (correctly predicted background pixels)
- $K$ = number of classes
- $N$ = total number of pixels

---

## 1. Pixel Accuracy (PA)

### Definition

The simplest metric: the fraction of pixels that are correctly classified.

$$\text{PA} = \frac{\sum_{i=1}^{K} p_{ii}}{\sum_{i=1}^{K} \sum_{j=1}^{K} p_{ij}}$$

where $p_{ij}$ is the number of pixels of class $i$ predicted as class $j$.

For binary segmentation:

$$\text{PA} = \frac{TP + TN}{TP + TN + FP + FN}$$

### Mean Pixel Accuracy (mPA)

Per-class accuracy averaged over all classes:

$$\text{mPA} = \frac{1}{K} \sum_{i=1}^{K} \frac{p_{ii}}{\sum_{j=1}^{K} p_{ij}}$$

### Pros

- Intuitive and easy to compute.
- Gives a quick sanity check.

### Cons

- **Dominated by majority classes.** If "background" covers 90% of the image and the model predicts everything as background, PA = 90%. This makes it almost useless for imbalanced datasets.
- Does not reflect per-class performance.
- Rarely used as a primary metric in modern research; typically reported alongside mIoU.

---

## 2. Intersection over Union (IoU) / Jaccard Index

### Definition

For a single class $c$:

$$\text{IoU}_c = \frac{|Y_c \cap \hat{Y}_c|}{|Y_c \cup \hat{Y}_c|} = \frac{TP_c}{TP_c + FP_c + FN_c}$$

Equivalently, using the confusion matrix:

$$\text{IoU}_c = \frac{p_{cc}}{\sum_{j} p_{cj} + \sum_{j} p_{jc} - p_{cc}}$$

### Mean IoU (mIoU)

The standard metric for semantic segmentation. Average IoU over all $K$ classes:

$$\text{mIoU} = \frac{1}{K} \sum_{c=1}^{K} \text{IoU}_c$$

### Frequency-Weighted IoU (FWIoU)

Weight each class IoU by its pixel frequency:

$$\text{FWIoU} = \frac{1}{\sum_{i,j} p_{ij}} \sum_{c=1}^{K} \frac{\sum_j p_{cj}}{1} \cdot \text{IoU}_c$$

### Pros

- **Class-balanced** (in the mean version): each class contributes equally regardless of its pixel count.
- Penalizes both false positives and false negatives symmetrically.
- Universally adopted: the standard metric for PASCAL VOC, Cityscapes, ADE20K.
- Range $[0, 1]$ with clear interpretation: 1.0 means perfect overlap.

### Cons

- **Harsh on small objects:** A few misclassified pixels on a small object cause a large IoU drop, while the same number of errors on a large object barely affects it. This can be seen as either a pro (sensitivity to small objects) or a con (instability).
- Does not directly measure boundary quality.
- Binary (per-class) -- does not distinguish between instances.
- Sensitive to class definition: adding or merging classes changes mIoU values.

### Practical Notes

- When a class is absent from both prediction and ground truth in an image, it is typically excluded from the mean for that image (to avoid division by zero / inflating the score).
- Standard practice: compute IoU per class over the **entire dataset** (not per image), then average. This avoids instability from images where a class has very few pixels.

---

## 3. Dice Coefficient / F1 Score

### Definition

$$\text{Dice} = \frac{2 \cdot |Y \cap \hat{Y}|}{|Y| + |\hat{Y}|} = \frac{2 \cdot TP}{2 \cdot TP + FP + FN}$$

### Relationship to IoU

Dice and IoU are monotonically related:

$$\text{Dice} = \frac{2 \cdot \text{IoU}}{1 + \text{IoU}}, \qquad \text{IoU} = \frac{\text{Dice}}{2 - \text{Dice}}$$

Dice is always $\geq$ IoU for the same prediction. If IoU = 0.5, Dice = 0.667.

### Pros

- Widely used in medical image segmentation (the de facto standard for challenges like BraTS, ISIC, ACDC).
- Gives more weight to true positives relative to IoU, which can be desirable when the foreground is small.
- Identical to the F1 score from information retrieval, making it familiar across communities.

### Cons

- Same core limitations as IoU (no boundary information, no instance distinction).
- Not a proper metric in the mathematical sense (does not satisfy the triangle inequality), though this rarely matters in practice.
- The monotonic relationship with IoU means it carries the same information; the choice between them is largely a matter of convention.

---

## 4. Hausdorff Distance (HD)

### Definition

The Hausdorff distance measures the worst-case boundary error between the predicted and ground-truth contours.

Let $\partial Y$ and $\partial \hat{Y}$ be the sets of boundary points of the ground truth and prediction, respectively. The directed Hausdorff distance is:

$$\overrightarrow{HD}(\partial Y, \partial \hat{Y}) = \max_{a \in \partial Y} \min_{b \in \partial \hat{Y}} \|a - b\|_2$$

The (symmetric) Hausdorff distance is:

$$HD(\partial Y, \partial \hat{Y}) = \max\big(\overrightarrow{HD}(\partial Y, \partial \hat{Y}),\ \overrightarrow{HD}(\partial \hat{Y}, \partial Y)\big)$$

### 95th Percentile Hausdorff Distance (HD95)

Because HD is sensitive to single outlier points, the 95th percentile variant is commonly used:

$$HD_{95}(\partial Y, \partial \hat{Y}) = \text{percentile}_{95}\big(\{d(a, \partial \hat{Y})\}_{a \in \partial Y} \cup \{d(b, \partial Y)\}_{b \in \partial \hat{Y}}\big)$$

where $d(a, S) = \min_{s \in S} \|a - s\|_2$.

### Pros

- Directly measures boundary accuracy in spatial units (pixels or mm).
- Captures localized errors that overlap-based metrics may miss (e.g., a thin protrusion that barely changes IoU but is far from the true boundary).
- Essential in medical imaging where boundary precision is clinically relevant (e.g., radiation therapy planning).

### Cons

- **Extremely sensitive to outliers** (mitigated by HD95, but not eliminated).
- Computationally more expensive than overlap metrics, especially for large 3D volumes.
- Does not capture region overlap -- a prediction can have perfect HD but poor IoU if it covers only the boundary.
- Requires extracting boundary points, which adds implementation complexity.
- Not differentiable, so it cannot be directly used as a loss function (though approximations exist).

---

## 5. Panoptic Quality (PQ)

### Definition

Introduced by Kirillov et al. (2019) for evaluating panoptic segmentation. PQ decomposes evaluation into recognition (did you find the segment?) and segmentation (how well did you segment it?).

First, predicted and ground-truth segments are matched using IoU > 0.5:

- $TP$: matched pairs (IoU > 0.5 between predicted and ground-truth segment)
- $FP$: predicted segments with no match
- $FN$: ground-truth segments with no match

Then:

$$PQ = \underbrace{\frac{|TP|}{|TP| + \frac{1}{2}|FP| + \frac{1}{2}|FN|}}_{\text{Recognition Quality (RQ)}} \times \underbrace{\frac{\sum_{(p,g) \in TP} \text{IoU}(p, g)}{|TP|}}_{\text{Segmentation Quality (SQ)}}$$

$$PQ = RQ \times SQ$$

- **RQ** is the F1 score for segment detection.
- **SQ** is the average IoU of matched segments (always $> 0.5$ by construction).

PQ is computed per class and then averaged:

$$PQ = \frac{1}{K} \sum_{c=1}^{K} PQ_c$$

Often reported separately for "things" ($PQ^{Th}$) and "stuff" ($PQ^{St}$) classes.

### Pros

- **Unified metric** for both stuff and thing classes in a single number.
- Clean decomposition into RQ and SQ helps diagnose whether errors are due to missed detections or poor mask quality.
- The IoU > 0.5 matching threshold ensures a unique 1-to-1 assignment between predictions and ground truth.

### Cons

- The IoU > 0.5 threshold is arbitrary and can be harsh: a prediction with IoU = 0.49 counts as a false positive and a false negative.
- SQ is always in $[0.5, 1.0]$, compressing the segmentation quality range and reducing discriminability.
- Does not account for pixel-level errors within matched segments beyond their aggregate IoU.
- Not suitable for comparing semantic-only or instance-only models.

---

## 6. Average Precision (AP)

### Definition

AP is the standard metric for instance segmentation (and object detection). It is computed from the precision-recall curve.

For a given IoU threshold $\tau$:

1. Rank all predicted instances by confidence score (descending).
2. For each prediction, determine if it matches a ground-truth instance (IoU $\geq \tau$ and not already matched). If yes, it is a TP; otherwise, a FP.
3. Compute precision and recall at each rank position.
4. AP is the area under the precision-recall curve (using all-point interpolation).

$$AP = \int_0^1 p(r) \, dr$$

### COCO-Style AP

COCO averages AP across 10 IoU thresholds from 0.50 to 0.95 in steps of 0.05:

$$AP = \frac{1}{10} \sum_{\tau \in \{0.50, 0.55, \dots, 0.95\}} AP_\tau$$

Additional breakdowns:
- $AP_{50}$: AP at IoU = 0.50 (lenient, PASCAL-style)
- $AP_{75}$: AP at IoU = 0.75 (strict)
- $AP_S$, $AP_M$, $AP_L$: AP for small, medium, large objects

### Mask AP vs. Box AP

In instance segmentation, mask AP computes IoU between predicted and ground-truth **masks** (not bounding boxes). This is strictly harder than box AP.

### Pros

- **Comprehensive:** Evaluates both detection (did you find it?) and segmentation (how accurate is the mask?) quality.
- **Threshold-averaged** (COCO-style): rewards models that produce tight, accurate masks, not just rough localizations.
- Well-established evaluation protocol with mature tooling (pycocotools).

### Cons

- **Complex to compute** and understand compared to mIoU.
- Depends on confidence score ordering -- two models with identical masks but different score distributions can get different AP.
- Does not evaluate stuff classes.
- AP can be unstable for classes with very few instances.
- COCO's averaged AP penalizes methods that are good at detection but produce slightly noisy masks.

---

## 7. Boundary-Aware Metrics

### Boundary IoU

Proposed by Cheng et al. (2021) to evaluate mask quality specifically at object boundaries.

1. Compute the boundary region $B_d(M)$ of a mask $M$ as the set of pixels within distance $d$ of the mask boundary.
2. Compute IoU only within the boundary region:

$$\text{Boundary IoU} = \frac{|B_d(Y) \cap B_d(\hat{Y}) \cap Y \cap \hat{Y}|}{|B_d(Y) \cap B_d(\hat{Y}) \cap (Y \cup \hat{Y})|}$$

### Boundary F1

Compute precision and recall of boundary pixels (within a tolerance distance), then take the F1 score.

**Usage:** These metrics are reported alongside standard metrics when boundary quality is a specific concern (e.g., Cityscapes benchmark reports boundary AP).

---

## 8. Metric Comparison Summary

| Metric | Task | Measures | Handles Class Imbalance | Boundary Sensitive | Computational Cost |
|--------|------|----------|------------------------|--------------------|--------------------|
| Pixel Accuracy | Semantic | Overall correctness | No | No | Very low |
| mIoU | Semantic | Per-class region overlap | Yes (mean over classes) | Indirectly | Low |
| Dice / F1 | Semantic / Binary | Region overlap | Somewhat | No | Low |
| HD / HD95 | Any (boundary) | Boundary distance | N/A | Yes (primary purpose) | Medium--High |
| PQ | Panoptic | Detection + segmentation | Yes (mean over classes) | Indirectly | Medium |
| AP (mask) | Instance | Detection + segmentation | Evaluated per class | Via IoU threshold | Medium |
| Boundary IoU | Instance / Semantic | Boundary overlap | Depends on usage | Yes | Medium |

---

## 9. Practical Recommendations

1. **Semantic segmentation:** Report mIoU as the primary metric. Include pixel accuracy for completeness. For medical imaging, also report Dice and HD95.

2. **Instance segmentation:** Report COCO-style mask AP (averaged over IoU thresholds). Include AP50 and AP75 breakdowns. Report AP by object size if the dataset supports it.

3. **Panoptic segmentation:** Report PQ, with SQ and RQ decomposition. Report PQ-Things and PQ-Stuff separately.

4. **When boundary quality matters:** Supplement overlap metrics with Boundary IoU or HD95.

5. **Statistical significance:** Report standard deviation or confidence intervals when comparing methods. Small differences in mIoU (< 0.5%) may not be statistically significant.

6. **Implementation:** Use official evaluation toolkits when available (pycocotools for COCO, cityscapesscripts for Cityscapes, panopticapi for panoptic) to ensure comparability.

---

## References

1. Everingham, M., et al. (2010). The Pascal Visual Object Classes (VOC) Challenge. IJCV.
2. Cordts, M., et al. (2016). The Cityscapes Dataset for Semantic Urban Scene Understanding. CVPR.
3. Lin, T.-Y., et al. (2014). Microsoft COCO: Common Objects in Context. ECCV.
4. Kirillov, A., He, K., Girshick, R., & Dollar, P. (2019). Panoptic Segmentation. CVPR.
5. Cheng, B., et al. (2021). Boundary IoU: Improving Object-Centric Image Segmentation Evaluation. CVPR.
6. Taha, A. A., & Hanbury, A. (2015). Metrics for Evaluating 3D Medical Image Segmentation. BMC Medical Imaging.
