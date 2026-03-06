---
title: "3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation"
date: 2025-03-06
status: planned
tags:
  - 3d-convolution
  - sparse-annotation
  - volumetric
  - biomedical
difficulty: beginner
---

# 3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation

## Meta Information

| Field          | Details |
|----------------|---------|
| **Paper Title**   | 3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation |
| **Authors**       | Ozgun Cicek, Ahmed Abdulkadir, Soeren S. Lienkamp, Thomas Brox, Olaf Ronneberger |
| **Year**          | 2016 |
| **Venue**         | MICCAI 2016 |
| **ArXiv ID**      | [1606.06650](https://arxiv.org/abs/1606.06650) |

## One-Line Summary

3D U-Net extends the original U-Net to volumetric segmentation by replacing all 2D operations with 3D counterparts and introduces a sparse annotation strategy that enables learning dense 3D segmentation from only a few annotated 2D slices.

---

## Motivation and Problem Statement

_TODO: Annotating every slice in a 3D volume is prohibitively expensive -- describe the sparse annotation motivation._

---

## Key Contributions

- _TODO: 3D extension of U-Net architecture_
- _TODO: Sparse annotation training strategy_
- _TODO: Batch normalization before each ReLU_
- _TODO: Weighted softmax loss with class balancing_

---

## Architecture Overview

_TODO: Describe the 3D encoder-decoder with skip connections._

---

## Method Details

### Sparse Annotation Strategy

_TODO: Reference [sparse_annotation_strategy.md](./sparse_annotation_strategy.md)._

### 3D Operations

_TODO: 3D convolutions, 3D max pooling, 3D up-convolutions._

### Loss Function

_TODO: Weighted cross-entropy with class balancing._

---

## Experimental Results

| Dataset | Metric | 3D U-Net Result | Notes |
|---------|--------|-----------------|-------|
| Xenopus kidney | IoU | _TODO_ | Semi-automated |
| Xenopus kidney | IoU | _TODO_ | Fully-automated |

---

## Strengths

- _TODO_

---

## Weaknesses and Limitations

- _TODO_

---

## Connections to Other Work

| Related Paper | Relationship |
|---------------|-------------|
| U-Net (Ronneberger et al., 2015) | Direct 2D predecessor |
| V-Net (Milletari et al., 2016) | Concurrent 3D approach with Dice loss |
| nnU-Net (Isensee et al., 2021) | Framework incorporating 3D U-Net |

---

## Open Questions

- _TODO_
