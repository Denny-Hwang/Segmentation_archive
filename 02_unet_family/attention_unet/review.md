---
title: "Attention U-Net: Learning Where to Look for the Pancreas"
date: 2025-03-06
status: complete
tags: [attention-gate, skip-connections, medical-segmentation]
difficulty: intermediate
---

# Attention U-Net

## Meta Information

| Field | Value |
|-------|-------|
| **Paper Title** | Attention U-Net: Learning Where to Look for the Pancreas |
| **Authors** | Oktay, O., Schlemper, J., Folgoc, L.L., et al. |
| **Year** | 2018 |
| **arXiv** | [1804.03999](https://arxiv.org/abs/1804.03999) |

## One-Line Summary

Attention U-Net adds attention gates to the skip connections in U-Net, allowing the model to focus on relevant spatial regions and suppress irrelevant features before concatenation with decoder features.

## Motivation

Standard U-Net skip connections pass ALL encoder features to the decoder, including irrelevant background regions. For tasks like pancreas segmentation where the target organ occupies a small fraction of the image (<1% of voxels), most skip connection features are noise. Attention gates learn to weight skip connection features spatially, amplifying relevant regions (near the target) and suppressing irrelevant ones (background).

## Architecture

Attention U-Net has the same encoder-decoder structure as U-Net, with attention gates inserted on each skip connection. Each attention gate takes two inputs: (1) the skip connection features from the encoder (g_skip), and (2) the gating signal from the decoder (g_gate). The gate produces spatial attention weights that modulate the skip features before concatenation.

## Key Results

| Dataset | Model | Dice (%) |
|---------|-------|----------|
| CT Pancreas | U-Net | 71.95 |
| CT Pancreas | Attention U-Net | 75.41 |
| CT Pancreas | Attention U-Net + DS | 77.98 |

The attention mechanism provides +3.5% Dice improvement on pancreas segmentation, with additional gains from deep supervision (DS). Improvements are most significant for small, hard-to-segment organs.

## Strengths

- Simple, lightweight module that can be added to any U-Net variant
- Provides interpretable attention maps showing where the model focuses
- Consistent improvements for small structure segmentation
- No significant computational overhead (~1% additional parameters)

## Limitations

- Improvement diminishes for large structures that occupy most of the image
- Attention maps can be noisy in early training stages
- Additive attention may not capture complex spatial relationships as effectively as self-attention
