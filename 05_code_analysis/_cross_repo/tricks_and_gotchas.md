---
title: "Cross-Repository Tricks and Gotchas"
date: 2025-01-15
status: planned
tags: [cross-repo, tricks, gotchas, debugging, performance]
---

# Tricks and Gotchas

## Purpose

This document collects subtle implementation details, common pitfalls, and non-obvious techniques discovered during code analysis. These are things not typically mentioned in papers or documentation.

## Numerical Stability

### Loss Function Stability
TODO: Epsilon values in Dice loss, log-softmax vs softmax + log, etc.

### Gradient Issues
TODO: Gradient clipping, exploding gradients with deep networks

## Data Pipeline Gotchas

### Mask Encoding
TODO: One-hot vs integer encoding, off-by-one errors in class indices

### Augmentation Pitfalls
TODO: Augmentations that must be applied identically to image and mask

### DataLoader Workers
TODO: Issues with `num_workers > 0` and shared memory

## Architecture Gotchas

### Spatial Dimension Mismatch
TODO: When input size is not divisible by 2^depth (padding strategies)

### Channel Count After Concatenation
TODO: Off-by-one channel bugs in skip connections

### Bilinear vs Transposed Convolution
TODO: Checkerboard artifacts, alignment issues

## Training Gotchas

### Learning Rate for Pretrained vs New Layers
TODO: Differential learning rates

### Batch Normalization with Small Batches
TODO: When to switch to InstanceNorm or GroupNorm

### Mixed Precision Pitfalls
TODO: Operations that break under FP16

## Evaluation Gotchas

### IoU vs Dice Metric Differences
TODO: Micro vs macro averaging, boundary effects

### Evaluation on Original vs Resized Images
TODO: When to evaluate on original resolution

## Repository-Specific Gotchas

### Pytorch-UNet
TODO

### SMP
TODO

### nnU-Net
TODO

### SAM 2
TODO

### MMSegmentation
TODO
