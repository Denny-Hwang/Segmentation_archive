---
title: "Segmentation Result Visualization Guide"
date: 2025-01-15
status: planned
tags: [segmentation-results, comparison, qualitative-evaluation]
---

# Segmentation Result Visualization

## Purpose

This directory stores qualitative segmentation results -- side-by-side comparisons of model predictions on the same images. These visualizations complement quantitative metrics by revealing where and how models differ.

## Visualization Format

### Standard Comparison Layout

For each test image, produce a row of panels:

```
| Input Image | Ground Truth | Model A | Model B | Model C |
```

### Color Coding

Use a consistent color palette across all visualizations:

| Class | Color | Hex |
|-------|-------|-----|
| Background | Black | #000000 |
| Class 1 | Red | #FF0000 |
| Class 2 | Green | #00FF00 |
| Class 3 | Blue | #0000FF |
| (additional classes follow the palette in `_common/visualization.py`) |

### Overlay Style

- Semi-transparent mask overlay on the original image (alpha = 0.5)
- Contour-only overlay for boundary quality assessment
- Error map: highlight false positives (red) and false negatives (blue)

## Recommended Comparisons

### 1. U-Net Variants
- Input: Same test images from the U-Net baseline experiment
- Models: U-Net, U-Net++, Attention U-Net

### 2. CNN vs Transformer
- Input: Images with both small and large objects
- Models: DeepLabV3+, SegFormer, TransUNet

### 3. Zero-Shot SAM 2
- Input: Domain-specific images (medical, satellite, etc.)
- Models: SAM 2 (point prompt), SAM 2 (box prompt), trained specialist model

## Generating Visualizations

Use the `show_prediction_comparison()` function from `06_experiments/_common/visualization.py`:

```python
from _common.visualization import show_prediction_comparison

fig = show_prediction_comparison(image, ground_truth, prediction)
fig.savefig("comparison.png", dpi=150)
```

## File Naming Convention

```
<dataset>_<image_id>_<model_name>.png
<dataset>_<image_id>_comparison.png
```

## Planned Visualizations

TODO: Generate comparison images after running experiments in `06_experiments/`
