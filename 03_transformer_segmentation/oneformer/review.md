---
title: "OneFormer: One Transformer to Rule Universal Image Segmentation"
date: 2025-03-06
status: planned
tags: [universal-segmentation, task-conditioned, contrastive-learning, multi-task]
difficulty: advanced
---

# OneFormer

## Meta Information

| Field | Value |
|-------|-------|
| **Paper Title** | OneFormer: One Transformer to Rule Universal Image Segmentation |
| **Authors** | Jain, J., Li, J., Chiu, M., Hassani, A., Orber, N., Shi, H. |
| **Year** | 2023 |
| **Venue** | CVPR |
| **arXiv** | [2211.06220](https://arxiv.org/abs/2211.06220) |
| **Difficulty** | Advanced |

## One-Line Summary

OneFormer achieves universal segmentation with a single jointly trained model by conditioning on task tokens and using query-text contrastive learning to distinguish between semantic, instance, and panoptic tasks.

## Motivation and Problem Statement

<!-- Why was this work needed? What gap does it address compared to Mask2Former? -->

## Architecture Overview

<!-- High-level description of the model architecture with diagram reference -->

### Key Components

- **Task-Conditioned Training**: See [task_conditioned_training.md](task_conditioned_training.md)
- **Query-Text Contrastive Learning**: See [query_text_contrastive.md](query_text_contrastive.md)

## Technical Details

### Task Token Conditioning

<!-- How task tokens guide the model behavior -->

### Query Initialization

<!-- How queries are initialized based on the task -->

### Contrastive Loss

<!-- Query-text contrastive learning objective -->

### Backbone and Decoder

<!-- Feature extraction and transformer decoder design -->

### Loss Function

<!-- Full training objective(s) -->

## Experiments and Results

### Datasets

<!-- Benchmark datasets used (ADE20K, Cityscapes, COCO) -->

### Key Results

<!-- Main quantitative results -->

### Ablation Studies

<!-- Important ablations and findings -->

## Strengths

<!-- List key strengths of the approach -->

## Limitations

<!-- List key limitations -->

## Connections

<!-- How this work relates to Mask2Former and other papers -->

## References

<!-- Key references cited in the paper -->
