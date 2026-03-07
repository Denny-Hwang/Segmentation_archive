---
title: "Recurrent Residual Convolutional Neural Network based on U-Net (R2U-Net)"
date: 2025-03-06
status: complete
tags: [r2u-net, recurrent-convolution, residual, medical-segmentation]
difficulty: intermediate
---

# R2U-Net

## Meta Information

| Field | Value |
|-------|-------|
| **Paper Title** | Recurrent Residual Convolutional Neural Network based on U-Net (R2U-Net) |
| **Authors** | Alom, M.Z., Hasan, M., Yakopcic, C., Taha, T.M., Asari, V.K. |
| **Year** | 2018 |
| **arXiv** | [1802.06955](https://arxiv.org/abs/1802.06955) |

## One-Line Summary

R2U-Net combines recurrent convolution blocks with residual connections in a U-Net framework, improving feature representation without increasing parameters.

## Key Contributions

1. **Recurrent Convolution (RC) blocks**: Replace standard convolution with recurrent units that iterate t times, progressively refining features using shared weights
2. **Residual Recurrent blocks (RRC)**: Combine recurrent convolutions with residual connections for better gradient flow
3. **No additional parameters**: The recurrent structure reuses the same weights across time steps, adding computation but not parameters

## Architecture

R2U-Net has the same U-Net encoder-decoder structure but replaces each DoubleConv block with a Recurrent Residual Convolutional Block (RRCB). Each RRCB applies the same convolution kernel t times (t=2 by default), with the output of each iteration fed back as additional input to the next iteration. A residual connection adds the input of the block to its final output.

## Recurrent Convolution Block

At time step t, the feature map is computed as:

`x^t = f(W * x^{t-1} + W_r * x^{t-1}_r + b)`

where x^{t-1} is the input, x^{t-1}_r is the recurrent input (output of previous time step), W and W_r are shared convolutional weights, and f is ReLU.

With t=2 time steps, each pixel's receptive field effectively doubles without increasing parameters.

## Results

| Task | Metric | U-Net | R2U-Net |
|------|--------|-------|---------|
| Retina (DRIVE) | Acc | 97.26% | 97.84% |
| Skin (ISIC) | Acc | 88.30% | 90.20% |
| Lung | Acc | 97.45% | 98.76% |

R2U-Net shows consistent improvements of 0.5-2% across medical imaging tasks. The improvements are most significant on tasks requiring fine detail preservation.

## Strengths

- Improved feature representation without additional parameters
- Residual connections aid training stability
- Drop-in replacement for standard U-Net blocks
- Effective on small medical datasets

## Limitations

- ~t× increase in computation per block due to recurrent iterations
- Marginal improvements may not justify the computational overhead in all settings
- Hyperparameter t (number of recurrence steps) needs tuning
