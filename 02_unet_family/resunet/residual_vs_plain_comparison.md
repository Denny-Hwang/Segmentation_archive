---
title: "Residual vs Plain U-Net Comparison"
date: 2025-03-06
status: planned
tags:
  - residual-learning
  - ablation
  - gradient-flow
parent: resunet/review.md
---

# Residual vs Plain U-Net Comparison

## Overview

_TODO: Systematically compare U-Net with plain convolution blocks vs U-Net with residual blocks._

---

## Plain Convolution Block

_TODO: Diagram and description: Conv -> BN -> ReLU -> Conv -> BN -> ReLU._

---

## Residual Convolution Block

_TODO: Diagram and description: BN -> ReLU -> Conv -> BN -> ReLU -> Conv + identity shortcut._

---

## Theoretical Advantages of Residual Blocks

### Gradient Flow

_TODO: The shortcut connection provides a direct path for gradients, mitigating vanishing gradients._

### Identity Mapping

_TODO: The network only needs to learn the residual function F(x), which is easier to optimize._

### Deeper Networks

_TODO: Residual connections enable training deeper U-Net variants without degradation._

---

## Empirical Comparison

| Metric | Plain U-Net | ResUNet | Delta |
|--------|-------------|---------|-------|
| Training convergence speed | _TODO_ | _TODO_ | _TODO_ |
| Final Dice/IoU | _TODO_ | _TODO_ | _TODO_ |
| Parameter count | _TODO_ | _TODO_ | _TODO_ |
| Gradient magnitude (early layers) | _TODO_ | _TODO_ | _TODO_ |

---

## When to Use Residual Blocks

_TODO: Guidelines for when residual connections in U-Net provide meaningful benefit vs. unnecessary overhead._

---

## Other Residual Integration Strategies

| Variant | Shortcut Type | Notes |
|---------|--------------|-------|
| Pre-activation ResUnit | Identity | Used in ResUNet |
| Bottleneck ResUnit | 1x1 proj | For wider networks |
| Dense block | Concatenation | Used in DenseNet / Tiramisu |
