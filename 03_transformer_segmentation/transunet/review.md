---
title: "TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation"
date: 2025-03-06
status: planned
tags: [transformer, cnn-hybrid, medical-segmentation, u-net, vit]
difficulty: intermediate
---

# TransUNet

## Meta Information

| Field | Value |
|-------|-------|
| **Paper Title** | TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation |
| **Authors** | Chen, J., Lu, Y., Yu, Q., Luo, X., Adeli, E., Wang, Y., Lu, L., Yuille, A.L., Zhou, Y. |
| **Year** | 2021 |
| **Venue** | arXiv |
| **arXiv** | [2102.04306](https://arxiv.org/abs/2102.04306) |
| **Difficulty** | Intermediate |

## One-Line Summary

TransUNet combines a CNN feature extractor with a Vision Transformer encoder to capture both local and global context for medical image segmentation, using a cascaded upsampler for dense prediction.

## Motivation and Problem Statement

<!-- Why was this work needed? What gap does it address? -->

## Architecture Overview

<!-- High-level description of the model architecture with diagram reference -->

### Key Components

- **CNN-Transformer Hybrid Encoder**: See [cnn_transformer_hybrid.md](cnn_transformer_hybrid.md)
- **Positional Encoding**: See [positional_encoding.md](positional_encoding.md)

## Technical Details

### Input Processing

<!-- How are inputs tokenized/embedded? -->

### Encoder Design

<!-- Details of the hybrid CNN + ViT encoder -->

### Decoder Design

<!-- Cascaded upsampler and skip connections -->

### Loss Function

<!-- Training objective(s) -->

## Experiments and Results

### Datasets

<!-- Benchmark datasets used -->

### Key Results

<!-- Main quantitative results -->

### Ablation Studies

<!-- Important ablations and findings -->

## Strengths

<!-- List key strengths of the approach -->

## Limitations

<!-- List key limitations -->

## Connections

<!-- How this work relates to other papers in the archive -->

## References

<!-- Key references cited in the paper -->
