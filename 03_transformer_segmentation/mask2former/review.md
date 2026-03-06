---
title: "Masked-attention Mask Transformer for Universal Image Segmentation"
date: 2025-03-06
status: planned
tags: [universal-segmentation, masked-attention, panoptic, instance, semantic]
difficulty: advanced
---

# Mask2Former

## Meta Information

| Field | Value |
|-------|-------|
| **Paper Title** | Masked-attention Mask Transformer for Universal Image Segmentation |
| **Authors** | Cheng, B., Misra, I., Schwing, A.G., Kirillov, A., Girdhar, R. |
| **Year** | 2022 |
| **Venue** | CVPR |
| **arXiv** | [2112.01527](https://arxiv.org/abs/2112.01527) |
| **Difficulty** | Advanced |

## One-Line Summary

Mask2Former introduces masked attention in a transformer decoder to restrict cross-attention to predicted mask regions, achieving state-of-the-art on semantic, instance, and panoptic segmentation with a single architecture.

## Motivation and Problem Statement

<!-- Why was this work needed? What gap does it address compared to MaskFormer? -->

## Architecture Overview

<!-- High-level description of the model architecture with diagram reference -->

### Key Components

- **Masked Attention**: See [masked_attention.md](masked_attention.md)
- **Universal Segmentation**: See [universal_segmentation.md](universal_segmentation.md)

## Technical Details

### Backbone and Pixel Decoder

<!-- Feature extraction and multi-scale feature maps -->

### Transformer Decoder with Masked Attention

<!-- How masked attention improves over standard cross-attention -->

### Query Design

<!-- Learnable queries and their role -->

### Multi-Scale Strategy

<!-- How multi-scale features are leveraged -->

### Loss Function

<!-- Training objectives including Hungarian matching -->

## Experiments and Results

### Datasets

<!-- Benchmark datasets used (ADE20K, Cityscapes, COCO) -->

### Key Results

<!-- Main quantitative results across all three segmentation tasks -->

### Ablation Studies

<!-- Important ablations and findings -->

## Strengths

<!-- List key strengths of the approach -->

## Limitations

<!-- List key limitations -->

## Connections

<!-- How this work relates to MaskFormer, DETR, and other papers -->

## References

<!-- Key references cited in the paper -->
