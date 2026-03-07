---
title: "Self-Configuring Pipeline in nnU-Net"
date: 2025-03-06
status: complete
tags: [nnunet, self-configuring, pipeline, 2d, 3d, cascade]
difficulty: advanced
---

# Self-Configuring Pipeline

## Overview

nnU-Net's self-configuring pipeline automatically selects the optimal segmentation configuration from three options: 2D U-Net, 3D full-resolution U-Net, and 3D cascade U-Net. The selection is based on dataset properties and cross-validation performance.

## Three Configurations

### 2D U-Net
- Processes individual 2D slices independently
- Best for: highly anisotropic data (thick slices), datasets with large in-plane resolution, limited GPU memory
- Advantages: fast training, large batch sizes, handles any image size
- Limitation: no inter-slice context

### 3D Full-Resolution U-Net
- Processes 3D patches from the full-resolution volume
- Best for: isotropic or near-isotropic data, small-to-medium image sizes
- Advantages: full 3D context, best for isotropic data
- Limitation: limited patch size due to GPU memory, may miss very large structures

### 3D Cascade U-Net
- Two-stage: (1) 3D U-Net on downsampled volume → coarse segmentation; (2) 3D U-Net on full-resolution patches, conditioned on coarse prediction
- Best for: large images that don't fit in GPU memory at full resolution
- Advantages: combines global context (stage 1) with fine details (stage 2)
- Limitation: longer training (two networks), potential error propagation

## Automatic Selection Rules

1. If median image size fits in GPU memory → 3D full-res is primary candidate
2. If spacing anisotropy > 3:1 → 2D is competitive, include in evaluation
3. If image size too large for 3D full-res → 3D cascade is primary candidate
4. Always train all viable configurations → select best based on CV performance

## Ensembling

After 5-fold CV for each configuration, nnU-Net evaluates:
1. Each single configuration's mean Dice
2. All pairwise ensembles (average softmax predictions)
3. Selects the best single model OR ensemble based on validation Dice

Common outcomes: 3D full-res wins on isotropic data; 2D+3D ensemble wins on anisotropic data; cascade wins on very large images.

## Training Protocol

All configurations share:
- **Optimizer**: SGD with momentum 0.99, weight decay 3e-5
- **Learning rate**: Polynomial decay from initial lr=0.01
- **Loss**: Cross-entropy + Dice loss (equal weight)
- **Augmentation**: rotation, scaling, mirroring, gamma, elastic deformation
- **Epochs**: 1000 (with early stopping patience)
- **Batch size**: Automatically determined by GPU memory and patch size
