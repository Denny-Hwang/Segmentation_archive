---
title: "Segmentation Result Visualization Guide"
date: 2025-01-15
status: planned
tags: [segmentation-results, comparison, qualitative-evaluation]
---

# Segmentation Result Visualization

## Purpose

This directory stores qualitative segmentation results -- side-by-side comparisons of model
predictions on the same images. These visualizations complement quantitative metrics by
revealing where and how models differ. A model with 0.85 Dice and a model with 0.87 Dice
may look almost identical on most images but fail in completely different ways. Only
qualitative visualization reveals these patterns.

Good segmentation visualizations serve three goals:

1. **Debugging.** Identify systematic failure modes: does the model miss small objects,
   produce jagged boundaries, or confuse similar-looking classes?
2. **Communication.** Convey model performance to collaborators, reviewers, and stakeholders
   more intuitively than numeric tables.
3. **Model selection.** When two models have similar quantitative scores, visualizations help
   you pick the one whose errors are more acceptable for your application.

---

## Visualization Format

### Standard Comparison Layout

For each test image, produce a horizontal row of panels showing the input, ground truth, and
each model's prediction. This layout makes differences immediately apparent.

```
| Input Image | Ground Truth | Model A | Model B | Model C |
```

```python
import matplotlib.pyplot as plt
import numpy as np

def plot_comparison_row(image, ground_truth, predictions, model_names,
                        figsize=None, save_path=None):
    """
    Create a side-by-side comparison of segmentation predictions.

    Args:
        image: Input image as numpy array [H, W, 3], values in [0, 255].
        ground_truth: Ground truth mask [H, W], integer class labels.
        predictions: List of prediction masks, each [H, W].
        model_names: List of model name strings.
        figsize: Optional figure size tuple.
        save_path: Optional path to save the figure.

    Returns:
        matplotlib Figure object.
    """
    n_panels = 2 + len(predictions)  # input + GT + predictions
    if figsize is None:
        figsize = (4 * n_panels, 4)

    fig, axes = plt.subplots(1, n_panels, figsize=figsize)

    # Input image
    axes[0].imshow(image)
    axes[0].set_title("Input", fontsize=12, fontweight="bold")
    axes[0].axis("off")

    # Ground truth
    axes[1].imshow(colorize_mask(ground_truth))
    axes[1].set_title("Ground Truth", fontsize=12, fontweight="bold")
    axes[1].axis("off")

    # Predictions
    for i, (pred, name) in enumerate(zip(predictions, model_names)):
        axes[2 + i].imshow(colorize_mask(pred))
        axes[2 + i].set_title(name, fontsize=12)
        axes[2 + i].axis("off")

    plt.tight_layout(pad=0.5)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
```

---

### Color Coding

Use a consistent color palette across all visualizations. Consistency is critical: viewers
should be able to glance at any image in the project and immediately know which color
corresponds to which class.

#### Binary Segmentation Palette

| Region | Color | Hex | RGB |
|--------|-------|-----|-----|
| Background | Black | #000000 | (0, 0, 0) |
| Foreground | Cyan | #00FFFF | (0, 255, 255) |

#### Multi-Class Palette (up to 20 classes)

We use a colorblind-friendly palette derived from the Tableau 20 color scheme:

```python
SEGMENTATION_PALETTE = {
    0:  (0, 0, 0),        # Background - Black
    1:  (255, 0, 0),      # Class 1 - Red
    2:  (0, 255, 0),      # Class 2 - Green
    3:  (0, 0, 255),      # Class 3 - Blue
    4:  (255, 255, 0),    # Class 4 - Yellow
    5:  (255, 0, 255),    # Class 5 - Magenta
    6:  (0, 255, 255),    # Class 6 - Cyan
    7:  (255, 128, 0),    # Class 7 - Orange
    8:  (128, 0, 255),    # Class 8 - Purple
    9:  (0, 128, 255),    # Class 9 - Sky Blue
    10: (255, 128, 128),  # Class 10 - Salmon
    11: (128, 255, 128),  # Class 11 - Light Green
    12: (128, 128, 255),  # Class 12 - Light Blue
    13: (255, 255, 128),  # Class 13 - Light Yellow
    14: (255, 128, 255),  # Class 14 - Pink
    15: (128, 255, 255),  # Class 15 - Light Cyan
    16: (192, 64, 0),     # Class 16 - Brown
    17: (64, 192, 0),     # Class 17 - Lime
    18: (0, 64, 192),     # Class 18 - Navy
    19: (192, 0, 64),     # Class 19 - Crimson
}

def colorize_mask(mask, palette=None):
    """
    Convert an integer-labeled mask to an RGB image using a color palette.

    Args:
        mask: Numpy array of shape [H, W] with integer class labels.
        palette: Dict mapping class_id -> (R, G, B). Defaults to SEGMENTATION_PALETTE.

    Returns:
        RGB numpy array of shape [H, W, 3], dtype uint8.
    """
    if palette is None:
        palette = SEGMENTATION_PALETTE

    H, W = mask.shape
    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    for class_id, color in palette.items():
        rgb[mask == class_id] = color

    return rgb
```

---

### Overlay Techniques

#### Semi-Transparent Mask Overlay

The most common technique: blend the colored segmentation mask with the original image using
alpha compositing. This preserves the anatomical/scene context while showing the segmentation.

```python
def overlay_mask(image, mask, alpha=0.5, palette=None):
    """
    Overlay a colored segmentation mask on the original image.

    Args:
        image: Input image [H, W, 3], values in [0, 255], dtype uint8.
        mask: Integer mask [H, W].
        alpha: Blending factor (0 = image only, 1 = mask only).
        palette: Color palette dict.

    Returns:
        Blended image [H, W, 3], dtype uint8.
    """
    colored_mask = colorize_mask(mask, palette)
    # Only blend where mask is non-background
    blended = image.copy().astype(np.float32)
    foreground = mask > 0
    blended[foreground] = (
        (1 - alpha) * image[foreground].astype(np.float32) +
        alpha * colored_mask[foreground].astype(np.float32)
    )
    return blended.astype(np.uint8)
```

#### Contour-Only Overlay

Draw only the boundaries of each segmented region on the original image. This is ideal for
assessing boundary quality without obscuring the image content.

```python
import cv2

def contour_overlay(image, mask, thickness=2, palette=None):
    """
    Draw segmentation contours on the original image.

    Args:
        image: Input image [H, W, 3], values in [0, 255].
        mask: Integer mask [H, W].
        thickness: Contour line thickness in pixels.
        palette: Color palette dict.

    Returns:
        Image with contours drawn, [H, W, 3], dtype uint8.
    """
    if palette is None:
        palette = SEGMENTATION_PALETTE

    output = image.copy()
    unique_classes = np.unique(mask)

    for class_id in unique_classes:
        if class_id == 0:  # Skip background
            continue
        binary = (mask == class_id).astype(np.uint8)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        color = palette.get(class_id, (255, 255, 255))
        cv2.drawContours(output, contours, -1, color, thickness)

    return output
```

#### Error Map Visualization

Highlight where the model got it right and where it failed. This is arguably the most
informative visualization for model debugging.

```python
def error_map(ground_truth, prediction, image=None, alpha=0.6):
    """
    Create an error map showing true positives, false positives, and false negatives.

    Color coding:
        - Green: True positives (correct foreground prediction)
        - Red: False positives (predicted foreground, actually background)
        - Blue: False negatives (predicted background, actually foreground)
        - No overlay: True negatives (correct background)

    Args:
        ground_truth: Binary mask [H, W] (0 or 1).
        prediction: Binary mask [H, W] (0 or 1).
        image: Optional original image [H, W, 3] for overlay.
        alpha: Blending factor for overlay.

    Returns:
        Error map image [H, W, 3], dtype uint8.
    """
    H, W = ground_truth.shape
    error_rgb = np.zeros((H, W, 3), dtype=np.uint8)

    tp = (ground_truth == 1) & (prediction == 1)
    fp = (ground_truth == 0) & (prediction == 1)
    fn = (ground_truth == 1) & (prediction == 0)

    error_rgb[tp] = (0, 200, 0)     # Green - true positive
    error_rgb[fp] = (220, 0, 0)     # Red - false positive
    error_rgb[fn] = (0, 0, 220)     # Blue - false negative

    if image is not None:
        has_error_or_tp = tp | fp | fn
        blended = image.copy().astype(np.float32)
        blended[has_error_or_tp] = (
            (1 - alpha) * image[has_error_or_tp].astype(np.float32) +
            alpha * error_rgb[has_error_or_tp].astype(np.float32)
        )
        return blended.astype(np.uint8)

    return error_rgb
```

---

### Comprehensive Comparison Grid

For publications and presentations, create a grid that shows the input, ground truth,
predictions from multiple models, and error maps all in one figure.

```python
def plot_full_comparison(image, ground_truth, predictions, model_names,
                         figsize=(20, 10), save_path=None):
    """
    Create a comprehensive comparison grid with overlays and error maps.

    Row 1: Input | GT | Model A prediction | Model B prediction | ...
    Row 2: (empty) | (empty) | Model A error map | Model B error map | ...
    """
    n_models = len(predictions)
    n_cols = 2 + n_models

    fig, axes = plt.subplots(2, n_cols, figsize=figsize)

    # Row 1: Input, GT, and predictions with overlay
    axes[0, 0].imshow(image)
    axes[0, 0].set_title("Input", fontsize=11, fontweight="bold")

    gt_overlay = overlay_mask(image, ground_truth, alpha=0.4)
    axes[0, 1].imshow(gt_overlay)
    axes[0, 1].set_title("Ground Truth", fontsize=11, fontweight="bold")

    for i, (pred, name) in enumerate(zip(predictions, model_names)):
        pred_overlay = overlay_mask(image, pred, alpha=0.4)
        axes[0, 2 + i].imshow(pred_overlay)
        axes[0, 2 + i].set_title(name, fontsize=11)

    # Row 2: Error maps
    axes[1, 0].axis("off")  # Empty cell under input
    axes[1, 1].axis("off")  # Empty cell under GT

    for i, (pred, name) in enumerate(zip(predictions, model_names)):
        err = error_map(ground_truth, pred, image=image, alpha=0.5)
        axes[1, 2 + i].imshow(err)
        # Compute quick stats for the title
        dice = 2 * ((pred & ground_truth).sum()) / (pred.sum() + ground_truth.sum() + 1e-8)
        axes[1, 2 + i].set_title(f"Error Map (Dice={dice:.3f})", fontsize=10)

    for ax in axes.flat:
        ax.axis("off")

    plt.tight_layout(pad=0.5)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
```

---

### Interactive Visualization with Plotly

For interactive exploration in notebooks, Plotly allows zooming, panning, and toggling layers:

```python
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from PIL import Image as PILImage
import io

def interactive_comparison(image, ground_truth, prediction, class_names=None):
    """
    Create an interactive Plotly figure with toggleable segmentation overlays.

    Args:
        image: Input image [H, W, 3], uint8.
        ground_truth: Integer mask [H, W].
        prediction: Integer mask [H, W].
        class_names: Optional dict mapping class_id -> name string.
    """
    fig = make_subplots(rows=1, cols=3,
                        subplot_titles=["Input", "Ground Truth", "Prediction"])

    # Convert images to PIL for Plotly
    for col, (data, title) in enumerate([
        (image, "Input"),
        (overlay_mask(image, ground_truth, alpha=0.5), "Ground Truth"),
        (overlay_mask(image, prediction, alpha=0.5), "Prediction"),
    ], start=1):
        img_pil = PILImage.fromarray(data)
        fig.add_layout_image(
            dict(source=img_pil, x=0, y=1, xref=f"x{col}", yref=f"y{col}",
                 sizex=data.shape[1], sizey=data.shape[0],
                 xanchor="left", yanchor="top", layer="below"),
            row=1, col=col
        )
        fig.update_xaxes(range=[0, data.shape[1]], row=1, col=col)
        fig.update_yaxes(range=[data.shape[0], 0], row=1, col=col)

    fig.update_layout(height=500, width=1400, title_text="Segmentation Comparison")
    return fig
```

---

## Recommended Comparisons

### 1. U-Net Variants

- **Input:** Same test images from the U-Net variants comparison experiment.
- **Models:** U-Net, U-Net++, Attention U-Net, U-Net + ResNet34.
- **Focus:** Boundary precision differences. Attention U-Net should show cleaner boundaries
  on small structures. U-Net++ should show smoother segmentations overall.

### 2. CNN vs Transformer

- **Input:** Images containing both small and large objects (e.g., from ADE20K).
- **Models:** DeepLabV3+, SegFormer-B2, TransUNet.
- **Focus:** Large-region coherence (transformers should excel) vs. boundary sharpness (CNNs
  should excel). Include per-class error maps.

### 3. Zero-Shot SAM 2

- **Input:** Domain-specific images (medical, satellite, industrial).
- **Models:** SAM 2 (point prompt), SAM 2 (box prompt), SAM 2 (fine-tuned), trained
  specialist U-Net.
- **Focus:** How much does fine-tuning improve SAM 2's segmentation on out-of-distribution
  images? Show the same image with different prompt types.

---

## Generating Visualizations

Use the `show_prediction_comparison()` function from `06_experiments/_common/visualization.py`:

```python
from _common.visualization import show_prediction_comparison

# Basic comparison
fig = show_prediction_comparison(image, ground_truth, prediction)
fig.savefig("comparison.png", dpi=150)

# Multi-model comparison
fig = plot_full_comparison(
    image=image,
    ground_truth=gt_mask,
    predictions=[unet_pred, unetpp_pred, attn_unet_pred],
    model_names=["U-Net", "U-Net++", "Attention U-Net"],
    save_path="unet_variants_comparison_sample01.png"
)
```

---

## Batch Visualization Script

For generating visualizations across an entire test set:

```python
def generate_all_comparisons(test_loader, models, model_names, output_dir, num_samples=20):
    """
    Generate comparison visualizations for a batch of test images.

    Args:
        test_loader: DataLoader yielding (images, masks).
        models: List of trained model objects (in eval mode).
        model_names: List of model name strings.
        output_dir: Directory to save output images.
        num_samples: Number of samples to visualize.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    for i, (images, masks) in enumerate(test_loader):
        if i >= num_samples:
            break

        image_np = denormalize(images[0]).permute(1, 2, 0).numpy().astype(np.uint8)
        gt_np = masks[0].numpy()

        predictions = []
        for model in models:
            with torch.no_grad():
                pred = model(images.to("cuda"))
                pred = pred.argmax(dim=1)[0].cpu().numpy()
            predictions.append(pred)

        fig = plot_full_comparison(
            image_np, gt_np, predictions, model_names,
            save_path=os.path.join(output_dir, f"comparison_{i:04d}.png")
        )
        plt.close(fig)

    print(f"Saved {min(num_samples, len(test_loader))} comparisons to {output_dir}")
```

---

## File Naming Convention

```
<dataset>_<image_id>_<model_name>.png          # Single model prediction
<dataset>_<image_id>_comparison.png             # Multi-model comparison grid
<dataset>_<image_id>_error_<model_name>.png     # Error map for one model
<dataset>_<image_id>_overlay_<model_name>.png   # Overlay visualization
<dataset>_summary_grid.png                      # Summary grid of best/worst cases
```

Examples:
```
isic2018_0042_unet.png
isic2018_0042_comparison.png
isic2018_0042_error_segformer_b2.png
ade20k_val_0001_overlay_deeplabv3plus.png
```

---

## Tips for Effective Visualizations

1. **Always include the input image and ground truth.** Without these references, predictions
   are impossible to evaluate visually.
2. **Use the same color palette everywhere.** Switching palettes between figures confuses
   readers.
3. **Show failure cases, not just successes.** The most informative visualizations are the
   ones where models disagree or fail.
4. **Add quantitative annotations.** Include per-image Dice or IoU scores in the subplot
   titles so viewers can correlate visual quality with numeric performance.
5. **Control figure resolution.** Use at least 150 DPI for presentations and 300 DPI for
   publications. Set `bbox_inches="tight"` to avoid wasted whitespace.
6. **Sort by difficulty.** Present images sorted from easy (high Dice) to hard (low Dice) to
   show how model performance degrades.

---

## Planned Visualizations

Generate comparison images after running experiments in `06_experiments/`:

1. U-Net variants comparison on ISIC 2018 (10 easy + 10 hard samples).
2. CNN vs Transformer comparison on ADE20K (focus on scale-dependent performance).
3. SAM 2 zero-shot vs. fine-tuned on Kvasir-SEG (show prompt sensitivity).
4. Summary grid: best and worst predictions from each model on each dataset.
