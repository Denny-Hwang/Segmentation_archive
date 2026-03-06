---
title: "Benchmark Comparison Table: Transformer Segmentation Models"
date: 2025-03-06
status: planned
tags: [benchmark, comparison, performance, metrics]
difficulty: beginner
---

# Benchmark Comparison Table

## Overview

<!-- Comprehensive performance comparison of transformer-based segmentation models across standard benchmarks -->

## Medical Image Segmentation

### Synapse Multi-Organ (CT)

| Model | Year | DSC (%) | HD95 (mm) | Params (M) | Notes |
|-------|------|---------|-----------|------------|-------|
| TransUNet | 2021 | | | | |
| Swin-Unet | 2021 | | | | |
| DS-TransUNet | 2022 | | | | |

### ACDC (Cardiac MRI)

| Model | Year | DSC (%) | Params (M) | Notes |
|-------|------|---------|------------|-------|
| TransUNet | 2021 | | | |
| Swin-Unet | 2021 | | | |

### BraTS (Brain Tumor MRI)

| Model | Year | WT Dice | TC Dice | ET Dice | Params (M) | Notes |
|-------|------|---------|---------|---------|------------|-------|
| UNETR | 2022 | | | | | |
| Swin UNETR | 2022 | | | | | |

## Natural Image Segmentation

### ADE20K

| Model | Year | mIoU (%) | Params (M) | FLOPs (G) | Notes |
|-------|------|----------|------------|-----------|-------|
| Mask2Former | 2022 | | | | |
| OneFormer | 2023 | | | | |

### Cityscapes

| Model | Year | mIoU (%) | Params (M) | Notes |
|-------|------|----------|------------|-------|
| Mask2Former | 2022 | | | |
| OneFormer | 2023 | | | |

### COCO Panoptic

| Model | Year | PQ (%) | Params (M) | Notes |
|-------|------|--------|------------|-------|
| Mask2Former | 2022 | | | |
| OneFormer | 2023 | | | |

## Notes on Fair Comparison

<!-- Caveats about comparing across different papers, training setups, backbones, etc. -->

## References

<!-- Sources for all reported numbers -->
