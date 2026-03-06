---
title: "Transformer-Based Segmentation"
date: 2025-03-06
status: planned
tags: [transformer, segmentation, attention, vision-transformer]
difficulty: intermediate
---

# 03 - Transformer-Based Segmentation

## Overview

This section covers the evolution of transformer architectures applied to image segmentation. Starting from hybrid CNN-Transformer designs, we trace the path toward fully transformer-based segmentation models, including universal architectures that unify semantic, instance, and panoptic segmentation under a single framework.

## Table of Contents

| Model | Authors | Year | Venue | Key Contribution |
|-------|---------|------|-------|------------------|
| [TransUNet](transunet/review.md) | Chen et al. | 2021 | arXiv | CNN-Transformer hybrid encoder for medical image segmentation |
| [Swin-Unet](swin_unet/review.md) | Cao et al. | 2021 | arXiv | Pure Swin Transformer U-shaped architecture |
| [UNETR](unetr/review.md) | Hatamizadeh et al. | 2022 | WACV | ViT encoder for 3D medical segmentation |
| [Swin UNETR](swin_unetr/review.md) | Hatamizadeh et al. | 2022 | CVPR | Swin Transformer encoder for 3D segmentation |
| [DS-TransUNet](ds_transunet/review.md) | Lin et al. | 2022 | arXiv | Dual-scale transformer encoding with TIF module |
| [Mask2Former](mask2former/review.md) | Cheng et al. | 2022 | CVPR | Universal segmentation via masked attention |
| [OneFormer](oneformer/review.md) | Jain et al. | 2023 | CVPR | Task-conditioned joint training for universal segmentation |

## Comparative Analyses

- [CNN vs Transformer](_comparative/cnn_vs_transformer.md) - Systematic comparison of CNN and transformer approaches
- [Attention Mechanisms Survey](_comparative/attention_mechanisms_survey.md) - Survey of attention mechanisms in segmentation
- [Benchmark Comparison Table](_comparative/benchmark_comparison_table.md) - Performance comparison across standard benchmarks

## Key Themes

1. **Hybrid Architectures**: Combining CNN local feature extraction with transformer global attention (TransUNet, DS-TransUNet)
2. **Pure Transformer Designs**: Replacing CNN components entirely with transformer blocks (Swin-Unet, UNETR)
3. **Efficient Attention**: Window-based and shifted-window attention for computational tractability (Swin-Unet, Swin UNETR)
4. **Universal Segmentation**: Unifying semantic, instance, and panoptic tasks (Mask2Former, OneFormer)
5. **3D Medical Segmentation**: Extending transformer architectures to volumetric data (UNETR, Swin UNETR)
