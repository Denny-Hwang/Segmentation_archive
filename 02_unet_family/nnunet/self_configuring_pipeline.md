---
title: "nnU-Net Self-Configuring Pipeline"
date: 2025-03-06
status: planned
tags:
  - self-configuring
  - pipeline
  - preprocessing
  - postprocessing
parent: nnunet/review.md
---

# nnU-Net Self-Configuring Pipeline

## Overview

_TODO: Describe how nnU-Net automatically derives all design choices from the dataset fingerprint using a set of heuristic rules._

---

## Pipeline Stages

### 1. Dataset Fingerprint Extraction

_TODO: Reference [fingerprint_analysis.md](./fingerprint_analysis.md)._

### 2. Preprocessing Configuration

_TODO: Resampling strategy, normalization scheme, cropping decisions._

| Decision | Rule |
|----------|------|
| Target spacing | _TODO: Median voxel spacing_ |
| Normalization | _TODO: Per-modality z-score or CT windowing_ |
| Cropping | _TODO: Crop to non-zero region_ |

### 3. Network Architecture Configuration

_TODO: How patch size, network depth, feature map sizes, and batch size are determined._

| Parameter | How It Is Set |
|-----------|--------------|
| Patch size | _TODO: Based on median image shape and GPU memory_ |
| Network depth | _TODO: Derived from patch size_ |
| Feature maps | _TODO: Doubling scheme capped by GPU memory_ |
| Batch size | _TODO: Remaining GPU memory after network allocation_ |

### 4. Training Configuration

_TODO: Optimizer, learning rate schedule, data augmentation, loss function._

### 5. Postprocessing Configuration

_TODO: Connected component analysis, ensembling of 2D and 3D models._

---

## Fixed vs Adaptive Decisions

| Category | Fixed (Empirical Best) | Adaptive (Data-Dependent) |
|----------|----------------------|--------------------------|
| Optimizer | _TODO_ | |
| Loss function | _TODO_ | |
| Augmentation | _TODO_ | |
| Spacing | | _TODO_ |
| Patch size | | _TODO_ |
| Network topology | | _TODO_ |

---

## Why Heuristics Over Learning?

_TODO: Discuss why nnU-Net uses hand-crafted rules rather than learned hyperparameters (NAS), and the reliability advantages._

---

## Reproducing the Pipeline

_TODO: Practical steps for running nnU-Net on a new dataset._
