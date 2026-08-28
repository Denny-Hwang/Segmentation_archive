"""Knowledge base for segmentation evaluation metrics.

Used by the Segmentation Lab (inline explanations next to results) and by
the Metrics Guide page (full study material). Each entry documents what
the metric measures, its formula, strengths, weaknesses, and known
limitations - the "why", not just the number.
"""

from __future__ import annotations

METRIC_INFO: dict[str, dict] = {
    "pixel_accuracy": {
        "name": "Pixel Accuracy",
        "aka": "Overall accuracy",
        "range": "0 – 1 (higher is better)",
        "higher_better": True,
        "formula": r"\text{Acc} = \frac{\sum_i \mathbb{1}[\hat{y}_i = y_i]}{N}",
        "intuition": (
            "The fraction of all pixels labeled correctly. The most obvious "
            "metric - and the most misleading one for segmentation."
        ),
        "pros": [
            "Trivially simple to compute and explain",
            "Fine as a coarse progress signal during early training",
        ],
        "cons": [
            "Dominated by large classes: 95% background ⇒ 95% accuracy for "
            "predicting 'all background'",
            "Says nothing about *which* classes are wrong",
            "Insensitive to boundary quality",
        ],
        "limitations": (
            "Class imbalance makes pixel accuracy nearly meaningless as a "
            "standalone score. In the Road example, missing every lane "
            "marking costs under 2% accuracy. Never report it alone; pair "
            "it with mIoU or Dice."
        ),
        "use_when": "Quick sanity checks; roughly balanced classes only.",
    },
    "mean_iou": {
        "name": "Mean IoU",
        "aka": "Jaccard index, mIoU",
        "range": "0 – 1 (higher is better)",
        "higher_better": True,
        "formula": r"\text{IoU}_c = \frac{|P_c \cap G_c|}{|P_c \cup G_c|},\qquad"
        r" \text{mIoU} = \frac{1}{C}\sum_c \text{IoU}_c",
        "intuition": (
            "Overlap divided by union, per class, then averaged with every "
            "class weighted equally. The de-facto standard for semantic "
            "segmentation benchmarks (Cityscapes, ADE20K, PASCAL VOC)."
        ),
        "pros": [
            "Robust to class imbalance: a tiny class counts as much as a huge one",
            "Penalizes both false positives and false negatives",
            "Universal benchmark currency - comparable across papers",
        ],
        "cons": [
            "Harsh on small objects: a 2-pixel shift on a thin structure "
            "destroys its IoU",
            "Averaging hides per-class failures - always inspect the per-class table",
            "Insensitive to *where* boundary errors occur (blob vs. sliver)",
        ],
        "limitations": (
            "IoU is systematically lower than Dice for the same prediction "
            "(IoU = D/(2-D)), so never compare an IoU number against a Dice "
            "number. Undefined for classes absent from both GT and "
            "prediction - implementations differ in how they skip or score "
            "these, which silently shifts reported means."
        ),
        "use_when": "Semantic segmentation benchmarking; imbalanced classes.",
    },
    "mean_dice": {
        "name": "Mean Dice",
        "aka": "F1 score, Sørensen–Dice coefficient, DSC",
        "range": "0 – 1 (higher is better)",
        "higher_better": True,
        "formula": r"\text{Dice}_c = \frac{2|P_c \cap G_c|}{|P_c| + |G_c|}"
        r" = \frac{2\,\text{IoU}_c}{1+\text{IoU}_c}",
        "intuition": (
            "Twice the overlap divided by total mass - the harmonic mean of "
            "precision and recall. The standard metric of medical image "
            "segmentation (BraTS, Medical Decathlon, nnU-Net)."
        ),
        "pros": [
            "Directly interpretable as an F1 score (precision/recall balance)",
            "More forgiving of small errors on small structures than IoU",
            "Differentiable relaxations (soft Dice) double as training losses",
        ],
        "cons": [
            "Monotonically related to IoU - reports no *new* information, "
            "just a different scale",
            "Still a pure overlap metric: blind to boundary shape quality",
            "Inflated on large structures; compare only within similar sizes",
        ],
        "limitations": (
            "Two predictions with equal Dice can differ wildly in clinical "
            "usefulness (smooth contour vs. ragged one) - which is exactly "
            "why medical challenges pair Dice with HD95. Beware the "
            "empty-class convention: Dice on an absent class is defined as "
            "1, 0, or skipped depending on the toolkit."
        ),
        "use_when": "Medical imaging; any F1-style precision/recall summary.",
    },
    "boundary_f1": {
        "name": "Boundary F1",
        "aka": "BF score, boundary IoU (variant)",
        "range": "0 – 1 (higher is better)",
        "higher_better": True,
        "formula": r"P = \frac{|\{p \in \partial P : d(p, \partial G) < \theta\}|}"
        r"{|\partial P|},\quad"
        r"R = \frac{|\{g \in \partial G : d(g, \partial P) < \theta\}|}"
        r"{|\partial G|},\quad F = \frac{2PR}{P+R}",
        "intuition": (
            "Precision and recall computed on *boundary pixels only*, with a "
            "distance tolerance θ. Measures how well the predicted contour "
            "traces the true contour - the thing region metrics ignore."
        ),
        "pros": [
            "Directly targets contour quality, invisible to IoU/Dice",
            "Tolerance θ maps to an acceptable annotation error in pixels",
            "Sensitive to the ragged edges produced by noisy predictions",
        ],
        "cons": [
            "Ignores region interiors: a hole in the middle of a mask goes unnoticed",
            "Score depends strongly on the chosen tolerance θ",
            "Boundary length differences between classes skew averages",
        ],
        "limitations": (
            "θ must scale with image resolution - the same prediction "
            "scores differently at 512px vs 2048px unless θ is scaled too. "
            "Complementary by design: report Boundary F1 *alongside* an "
            "overlap metric, never instead of one."
        ),
        "use_when": "Boundary-critical tasks: matting, lesion contours, thin structures.",
    },
    "hd95": {
        "name": "Hausdorff Distance (95%)",
        "aka": "HD95",
        "range": "0 – image diagonal, in pixels (lower is better)",
        "higher_better": False,
        "formula": r"\text{HD95} = \max\bigl(\text{P}_{95}\,d(\partial P, \partial G),"
        r"\ \text{P}_{95}\,d(\partial G, \partial P)\bigr)",
        "intuition": (
            "The worst boundary disagreement, using the 95th percentile "
            "instead of the true maximum so a single outlier pixel does not "
            "dominate. Answers: 'how far can the contour be off, at worst?'"
        ),
        "pros": [
            "Catches localized catastrophic errors that overlap metrics average away",
            "In physical units (pixels/mm) - clinically interpretable",
            "Standard companion to Dice in medical challenges",
        ],
        "cons": [
            "Says nothing about typical/average quality - one number for the tail",
            "Undefined when a class is empty in GT or prediction "
            "(conventions vary; this Lab scores a fully-missed class as the "
            "image diagonal)",
            "Sensitive to small spurious islands far from the object",
        ],
        "limitations": (
            "A prediction with perfect Dice on 99% of the boundary and one "
            "distant false-positive blob gets a terrible HD95 - sometimes "
            "that is exactly what you want to detect, sometimes it is "
            "noise. Percentile choice (95 vs 100) changes rankings; always "
            "state it."
        ),
        "use_when": "Medical/safety-critical tasks where worst-case boundary error matters.",
    },
    "ari": {
        "name": "Adjusted Rand Index",
        "aka": "ARI",
        "range": "≈ -0.5 – 1 (higher is better; 0 ≈ chance)",
        "higher_better": True,
        "formula": r"\text{ARI} = \frac{\text{RI} - \mathbb{E}[\text{RI}]}"
        r"{\max(\text{RI}) - \mathbb{E}[\text{RI}]}",
        "intuition": (
            "Do two pixels that belong together in the GT also end up "
            "together in the prediction? Compares *partitions*, so segment "
            "labels never need to be matched to classes - ideal for "
            "unsupervised methods."
        ),
        "pros": [
            "No label mapping required - evaluates clustering structure directly",
            "Chance-corrected: random segmentations score ≈ 0",
            "Penalizes both over- and under-segmentation symmetrically",
        ],
        "cons": [
            "No spatial awareness - a pixel pair 500px apart counts like neighbors",
            "Less intuitive to interpret than overlap percentages",
            "Dominated by large regions (pair counts grow quadratically)",
        ],
        "limitations": (
            "ARI treats segmentation as pure clustering: it cannot tell you "
            "*which* class failed or where. Use it to compare unsupervised "
            "methods fairly (before any best-overlap mapping inflates "
            "scores), then switch to per-class metrics for diagnosis."
        ),
        "use_when": "Unsupervised segmentation; comparing partitions without labels.",
    },
}

# Display order and formatting for results tables
METRIC_COLUMNS = [
    ("pixel_accuracy", "Pixel Acc", "{:.3f}"),
    ("mean_iou", "mIoU", "{:.3f}"),
    ("mean_dice", "mDice", "{:.3f}"),
    ("boundary_f1", "Boundary F1", "{:.3f}"),
    ("hd95", "HD95 (px)", "{:.1f}"),
    ("ari", "ARI", "{:.3f}"),
]
