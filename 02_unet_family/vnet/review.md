---
title: "V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation"
date: 2025-03-06
status: complete
tags: [v-net, 3d-segmentation, dice-loss, volumetric]
difficulty: intermediate
---

# V-Net

## Meta Information

| Field | Value |
|-------|-------|
| **Paper Title** | V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation |
| **Authors** | Milletari, F., Navab, N., Ahmadi, S. |
| **Year** | 2016 |
| **Venue** | 3DV |

## One-Line Summary

V-Net introduces a 3D volumetric CNN for medical image segmentation with residual connections and the first use of Dice loss as a training objective.

## Key Contributions

1. **First 3D volumetric segmentation CNN** with an encoder-decoder architecture processing entire volumes
2. **Dice loss function** — first paper to propose training with the Dice coefficient directly as the loss, addressing the severe class imbalance problem in medical segmentation
3. **Residual connections** within each encoder/decoder stage (input added to output), improving gradient flow in 3D networks

## Architecture

V-Net processes 3D volumes (128×128×64 voxels) through an encoder with 5 stages using 5×5×5 convolutions, followed by a decoder with upsampling via transposed convolutions. Each stage contains 1-3 convolutional layers with residual connections (stage input added to stage output). Downsampling uses 2×2×2 convolutions with stride 2. The network outputs a voxel-wise probability map.

## Dice Loss

The Dice loss was introduced to handle extreme class imbalance (prostate <5% of volume):

`L_Dice = 1 - (2 Σ p_i g_i) / (Σ p_i² + Σ g_i²)`

This formulation is differentiable and inherently handles class imbalance since it measures overlap between prediction and ground truth regardless of the background size. It became the standard loss for medical segmentation, often combined with cross-entropy.

## Results

Evaluated on the PROMISE12 prostate MRI segmentation challenge. Achieved Dice score of 86.9% on the test set, competitive with the challenge leaders. The Dice loss was shown to significantly outperform weighted cross-entropy on this imbalanced dataset.

## Impact

V-Net established key paradigms: (1) 3D volumetric processing for medical images; (2) Dice loss as the standard training objective; (3) residual connections for deep 3D networks. These contributions influenced virtually all subsequent medical segmentation architectures.
