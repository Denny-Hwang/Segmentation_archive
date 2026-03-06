---
title: "Recurrent Convolution Blocks Explained"
date: 2025-03-06
status: planned
tags:
  - recurrent-convolution
  - feature-accumulation
  - time-steps
parent: r2unet/review.md
---

# Recurrent Convolution Blocks Explained

## Overview

_TODO: Explain how recurrent convolutional layers unfold the same convolution over multiple time steps, accumulating features without adding new parameters._

---

## Standard Convolution vs Recurrent Convolution

### Standard Convolution

_TODO: Single forward pass through the convolutional layer._

### Recurrent Convolution

_TODO: The output of a convolution is fed back as input, iterating t times with shared weights._

---

## Mathematical Formulation

_TODO: Write out the recurrent update equations._

$$
x_l^{(t)} = f(W_f * x_{l-1} + W_r * x_l^{(t-1)} + b)
$$

Where:
- _TODO: Define each variable_

---

## Variants in R2U-Net

### RU-Net (Recurrent U-Net)

_TODO: Uses recurrent convolutional units without residual connections._

### R2U-Net (Recurrent Residual U-Net)

_TODO: Adds a residual connection around the recurrent block._

---

## Effect of Time Steps

| Time Steps (t) | Feature Quality | Computation | Parameters |
|----------------|----------------|-------------|------------|
| t = 1 | Baseline | 1x | Same |
| t = 2 | Better | ~2x | Same |
| t = 3 | Marginally better | ~3x | Same |

---

## Intuition: Why Recurrence Helps

_TODO: Discuss how recurrence allows the effective receptive field to grow without deepening the network or increasing parameters._

---

## Comparison with Other Feature Accumulation Methods

| Method | Parameters | Receptive Field Growth |
|--------|-----------|----------------------|
| Deeper network | More | Per layer |
| Dilated convolution | Same | Via dilation rate |
| Recurrent convolution | Same | Via time steps |
