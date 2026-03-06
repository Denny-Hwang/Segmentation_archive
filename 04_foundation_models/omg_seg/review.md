---
title: "OMG-Seg: Is One Model Good Enough For All Segmentation?"
date: 2025-03-06
status: planned
tags: [universal-segmentation, clip, unified-architecture, multi-dataset]
difficulty: advanced
---

# OMG-Seg

## Meta Information

| Field | Value |
|-------|-------|
| **Paper Title** | OMG-Seg: Is One Model Good Enough For All Segmentation? |
| **Authors** | Li, X., Yuan, H., Li, W., Ding, H., Wu, S., Zhang, W., Li, Y., Chen, K., Loy, C.C. |
| **Year** | 2024 |
| **Venue** | CVPR |
| **arXiv** | [2401.10229](https://arxiv.org/abs/2401.10229) |
| **Difficulty** | Advanced |

## One-Line Summary

OMG-Seg presents a unified segmentation model that handles image-level, video-level, and interactive segmentation using a CLIP backbone and shared decoder, achieving competitive performance across all tasks.

## Motivation and Problem Statement

<!-- Why a single model for all segmentation tasks? -->

## Architecture Overview

<!-- High-level description: CLIP backbone, unified decoder -->

### Key Components

- **CLIP Backbone**: See [clip_backbone.md](clip_backbone.md)

## Technical Details

### CLIP-Based Feature Extraction

<!-- How CLIP features are used for segmentation -->

### Unified Decoder

<!-- Shared decoder design across tasks -->

### Task Routing

<!-- How different segmentation tasks are handled -->

### Multi-Dataset Training

<!-- Training across multiple datasets and task types -->

### Loss Function

<!-- Training objective(s) -->

## Experiments and Results

### Image Segmentation

<!-- Results on semantic, instance, panoptic benchmarks -->

### Video Segmentation

<!-- Results on video segmentation benchmarks -->

### Interactive Segmentation

<!-- Results on interactive/promptable segmentation -->

### Key Results

<!-- Main quantitative results -->

## Strengths

<!-- List key strengths of the approach -->

## Limitations

<!-- List key limitations -->

## Connections

<!-- How this work relates to SAM, Mask2Former, OneFormer, and other papers -->

## References

<!-- Key references cited in the paper -->
