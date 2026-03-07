---
title: "Dataset Fingerprint Analysis in nnU-Net"
date: 2025-03-06
status: complete
tags: [nnunet, fingerprint, dataset-analysis, auto-configuration]
difficulty: advanced
---

# Dataset Fingerprint Analysis

## Overview

The dataset fingerprint is nnU-Net's mechanism for automatically understanding a new dataset's properties. By analyzing the dataset fingerprint, nnU-Net derives all preprocessing, architecture, and training decisions without manual intervention.

## Fingerprint Components

The DatasetFingerprintExtractor analyzes:

1. **Image properties**: sizes, spacings (voxel dimensions), number of channels, modality
2. **Intensity statistics**: per-channel mean, std, percentiles (0.5, 99.5) — crucial for normalization
3. **Label properties**: number of classes, class frequencies, foreground/background ratio
4. **Spacing analysis**: anisotropy ratio (spacing along z vs x/y), determines 2D vs 3D preference
5. **Size statistics**: median image size, size range — determines patch size and network depth

## Configuration Decisions from Fingerprint

| Fingerprint Property | Configuration Decision |
|---------------------|----------------------|
| Anisotropy ratio > 3 | Prefer 2D or 3D cascade over 3D full-res |
| CT modality | Global normalization (clip to [0.5, 99.5] percentile → z-score) |
| MRI modality | Per-image z-score normalization |
| Small foreground ratio | Use class-weighted sampling |
| Large image size | Reduce patch size, increase network depth |
| Small dataset (<50 cases) | Reduce batch size, increase augmentation |

## Normalization Strategies

nnU-Net applies different normalization based on modality:

- **CT**: Global clip to [p0.5, p99.5] across entire dataset → subtract global mean → divide by global std
- **MRI (per-channel)**: Per-image z-score normalization (subtract mean, divide by std, computed within foreground mask)
- **Other (RGB, microscopy)**: Per-channel z-score or rescale to [0, 1]

The choice is automatic based on the modality field in the dataset descriptor.

## Patch Size and Network Topology

The fingerprint determines the training patch size by considering:
1. Maximum GPU memory (default: 1 GPU with batch size 2)
2. Median image size (patches should cover representative regions)
3. Anisotropy (patches may be non-cubic for anisotropic data)
4. Network depth = number of pooling operations that fit within the patch size (minimum 32 voxels per dimension)

Example: for a dataset with median size 512×512×150 and spacing 0.8×0.8×2.5mm:
- Target spacing (resampled): 0.8×0.8×2.5mm (preserve native spacing)
- Patch size: 128×128×64 (fits in GPU memory)
- Network depth: 5 levels (128→64→32→16→8)
