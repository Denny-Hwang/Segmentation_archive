---
title: "Benchmark Comparison Table: Transformer Segmentation Models"
date: 2025-03-06
status: complete
tags: [benchmark, comparison, performance, metrics]
difficulty: beginner
---

# Benchmark Comparison Table

## Overview

This document provides comprehensive performance comparisons of transformer-based segmentation models across standard benchmarks. Numbers are collected from original papers and may vary based on training protocols, backbones, and data augmentation. All results use the best reported configuration from each paper.

## Medical Image Segmentation

### Synapse Multi-Organ (CT)

| Model | Year | DSC (%) | HD95 (mm) | Params (M) | Notes |
|-------|------|---------|-----------|------------|-------|
| U-Net | 2015 | 76.85 | 39.70 | 31.0 | CNN baseline |
| Attention U-Net | 2018 | 77.77 | 36.02 | 34.9 | Attention gating |
| TransUNet | 2021 | 77.48 | 31.69 | 105.3 | R50 + ViT-B/16 |
| Swin-Unet | 2021 | 79.13 | 21.55 | 27.2 | Swin-T backbone |
| UNETR | 2022 | 78.36 | 26.47 | 92.8 | ViT-B encoder (3D) |
| Swin UNETR | 2022 | 82.25 | 18.34 | 62.2 | Swin-T encoder (3D) |
| DS-TransUNet | 2022 | 82.58 | 17.09 | ~55 | Dual Swin encoders |
| nnU-Net | 2021 | 82.50 | 15.20 | 31.2 | Self-configuring |

### ACDC (Cardiac MRI)

| Model | Year | DSC (%) | RV | Myo | LV | Notes |
|-------|------|---------|-----|-----|-----|-------|
| U-Net | 2015 | 87.55 | 87.10 | 80.63 | 94.92 | Baseline |
| TransUNet | 2021 | 89.71 | 88.86 | 84.53 | 95.73 | R50-ViT |
| Swin-Unet | 2021 | 90.00 | 88.55 | 85.62 | 95.83 | Swin-T |
| DS-TransUNet | 2022 | 91.51 | 90.18 | 87.23 | 97.13 | Dual-scale |
| nnU-Net | 2021 | 91.61 | 90.24 | 88.40 | 96.19 | Self-configuring |

### BraTS (Brain Tumor MRI)

| Model | Year | WT Dice | TC Dice | ET Dice | Params (M) | Notes |
|-------|------|---------|---------|---------|------------|-------|
| 3D U-Net | 2016 | 88.7 | 83.2 | 78.1 | 19.1 | 3D baseline |
| UNETR | 2022 | 90.4 | 85.7 | 82.4 | 92.8 | ViT-B, 3D |
| Swin UNETR | 2022 | 92.1 | 88.3 | 84.6 | 62.2 | Swin-T, 3D |
| nnU-Net | 2021 | 91.5 | 87.5 | 83.8 | 31.2 | Self-configuring |

## Natural Image Segmentation

### ADE20K

| Model | Year | mIoU (%) | Params (M) | FLOPs (G) | Notes |
|-------|------|----------|------------|-----------|-------|
| PSPNet (R101) | 2017 | 44.4 | 68.1 | 256 | Pyramid pooling |
| DeepLab v3+ (R101) | 2018 | 45.5 | 62.7 | 255 | ASPP + decoder |
| SegFormer-B5 | 2021 | 51.8 | 84.7 | 183 | Mix Transformer |
| Mask2Former (Swin-L) | 2022 | 57.8 | 216 | 411 | Masked attention |
| OneFormer (Swin-L) | 2023 | 58.0 | 220 | 415 | Task-conditioned |

### Cityscapes

| Model | Year | mIoU (%) | Params (M) | Notes |
|-------|------|----------|------------|-------|
| PSPNet (R101) | 2017 | 79.7 | 68.1 | Pyramid pooling |
| DeepLab v3+ (R101) | 2018 | 80.9 | 62.7 | ASPP |
| SegFormer-B5 | 2021 | 84.0 | 84.7 | Mix Transformer |
| Mask2Former (Swin-L) | 2022 | 83.3 | 216 | Masked attention |
| OneFormer (Swin-L) | 2023 | 84.4 | 220 | Task-conditioned |

### COCO Panoptic

| Model | Year | PQ (%) | PQ_th | PQ_st | Params (M) | Notes |
|-------|------|--------|-------|-------|------------|-------|
| Panoptic FPN (R101) | 2019 | 40.9 | 48.3 | 29.7 | 56.0 | Baseline |
| MaskFormer (Swin-L) | 2021 | 52.7 | 58.5 | 44.0 | 212 | Mask classification |
| Mask2Former (Swin-L) | 2022 | 57.8 | 64.2 | 48.1 | 216 | Masked attention |
| OneFormer (Swin-L) | 2023 | 58.0 | 64.4 | 48.0 | 220 | Joint training |

## Notes on Fair Comparison

These numbers should be interpreted with several caveats: (1) different papers use different training schedules, data augmentation, and preprocessing — direct comparison is only fair within the same experimental setup; (2) backbone capacity varies widely (R50 ~25M vs Swin-L ~200M params); (3) 3D models process volumetric data while 2D models process slices; (4) some results use test-time augmentation while others don't; (5) pre-training data differs (ImageNet-1K vs ImageNet-22K vs self-supervised). For the most reliable comparisons, refer to papers that re-implement baselines under identical conditions (e.g., nnU-Net's systematic benchmarks for medical imaging).

## References

- Isensee et al., "nnU-Net," Nature Methods, 2021 (most comprehensive medical benchmarks)
- Cheng et al., "Mask2Former," CVPR 2022 (natural image benchmarks)
- Jain et al., "OneFormer," CVPR 2023
