---
title: "keras-unet-collection - Model Comparison"
date: 2025-01-15
status: planned
parent: "keras_unet_collection/repo_overview.md"
tags: [keras, unet-variants, comparison, tensorflow]
---

# keras-unet-collection Model Comparison

## Available Models

| Model | Function | Paper | Key Feature |
|-------|----------|-------|-------------|
| U-Net | `models.unet_2d()` | Ronneberger 2015 | Baseline encoder-decoder |
| U-Net++ | `models.unet_plus_2d()` | Zhou 2018 | Nested skip connections |
| Attention U-Net | `models.att_unet_2d()` | Oktay 2018 | Attention gates on skip connections |
| R2U-Net | `models.r2_unet_2d()` | Alom 2018 | Recurrent + residual blocks |
| TransUNet | `models.transunet_2d()` | Chen 2021 | CNN encoder + ViT + CNN decoder |
| Swin-UNET | `models.swin_unet_2d()` | Cao 2022 | Pure Swin Transformer U-Net |
| V-Net | `models.vnet_2d()` | Milletari 2016 | Residual blocks + Dice loss |

## Common Interface

TODO: Document the shared function signature across all models

## Architecture Comparison

### Encoder Design
TODO: Compare encoder approaches across models

### Skip Connection Variants
TODO: Concatenation, addition, attention-gated, nested

### Decoder Design
TODO: Compare decoder approaches

## Parameter Counts

TODO: Compare parameter counts at the same depth and filter configuration

## Keras vs PyTorch Implementation Differences

TODO: Notable differences from PyTorch implementations of the same architectures
