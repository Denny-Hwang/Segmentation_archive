---
title: "UNETR: Transformers for 3D Medical Image Segmentation"
date: 2025-03-06
status: planned
tags: [vit, 3d-segmentation, medical-segmentation, volumetric]
difficulty: intermediate
---

# UNETR

## Meta Information

| Field | Value |
|-------|-------|
| **Paper Title** | UNETR: Transformers for 3D Medical Image Segmentation |
| **Authors** | Hatamizadeh, A., Tang, Y., Nath, V., Yang, D., Myronenko, A., Landman, B., Roth, H.R., Xu, D. |
| **Year** | 2022 |
| **Venue** | WACV |
| **arXiv** | [2103.10504](https://arxiv.org/abs/2103.10504) |
| **Difficulty** | Intermediate |

## One-Line Summary

UNETR uses a pure Vision Transformer as the encoder for 3D medical image segmentation, connecting intermediate transformer representations to a CNN-based decoder via skip connections.

## Motivation and Problem Statement

<!-- Why was this work needed? What gap does it address? -->

## Architecture Overview

<!-- High-level description of the model architecture with diagram reference -->

### Key Components

- **ViT Encoder for 3D Data**: See [vit_encoder_3d.md](vit_encoder_3d.md)

## Technical Details

### 3D Input Tokenization

<!-- How 3D volumes are divided into patches and tokenized -->

### ViT Encoder

<!-- Details of the Vision Transformer encoder for volumetric data -->

### CNN Decoder with Skip Connections

<!-- How transformer features are connected to the CNN decoder -->

### Loss Function

<!-- Training objective(s) -->

## Experiments and Results

### Datasets

<!-- Benchmark datasets used (e.g., BTCV, MSD) -->

### Key Results

<!-- Main quantitative results -->

### Ablation Studies

<!-- Important ablations and findings -->

## Strengths

<!-- List key strengths of the approach -->

## Limitations

<!-- List key limitations -->

## Connections

<!-- How this work relates to other papers in the archive -->

## References

<!-- Key references cited in the paper -->
