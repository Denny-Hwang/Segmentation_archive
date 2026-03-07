---
title: "3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation"
date: 2025-03-06
status: complete
tags: [3d-unet, volumetric, sparse-annotation, semi-supervised]
difficulty: intermediate
---

# 3D U-Net

## Meta Information

| Field | Value |
|-------|-------|
| **Paper Title** | 3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation |
| **Authors** | Çiçek, Ö., Abdulkadir, A., Lienkamp, S.S., Brox, T., Ronneberger, O. |
| **Year** | 2016 |
| **Venue** | MICCAI |

## One-Line Summary

3D U-Net extends the U-Net architecture to 3D volumetric data and demonstrates that dense volumetric segmentation can be learned from sparsely annotated data.

## Key Contributions

1. **3D extension of U-Net** with 3D convolutions, 3D max pooling, and 3D up-convolutions
2. **Sparse annotation training**: the network can learn from only a few annotated 2D slices within a 3D volume and generalize to segment the entire volume
3. **Semi-supervised learning**: combines a small amount of annotated data with unlabeled data for volumetric segmentation

## Architecture

The 3D U-Net follows the same encoder-decoder structure as 2D U-Net but with all operations in 3D. The encoder has three levels with 3×3×3 convolutions and 2×2×2 max pooling. The decoder uses 2×2×2 transposed convolutions for upsampling. Skip connections concatenate encoder and decoder features at each level. Batch normalization is applied before each ReLU activation.

## Sparse Annotation Strategy

The key innovation is training from sparse annotations: instead of annotating every slice in a volume, only a few slices are annotated. The network is trained using a weighted loss that only computes gradients for annotated voxels. Despite seeing labels for only ~3-10% of voxels, the network learns to segment the entire 3D volume by leveraging the spatial continuity of 3D convolutions.

## Results

Evaluated on Xenopus kidney volumetric data (confocal microscopy). With annotations from just 3 out of ~300 slices, the network achieved IoU scores within 5% of fully supervised training. This dramatically reduces the annotation burden for volumetric medical image segmentation.

## Impact

3D U-Net became a foundational architecture for volumetric medical imaging. It showed that 3D context is valuable even with very limited annotations. The sparse annotation approach influenced later works on semi-supervised and weakly supervised 3D segmentation. nnU-Net adopted and refined the 3D U-Net architecture as one of its three configuration options.
