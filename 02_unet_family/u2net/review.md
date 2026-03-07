---
title: "U²-Net: Going Deeper with Nested U-Structure for Salient Object Detection"
date: 2025-03-06
status: complete
tags: [u2net, nested-u-structure, salient-object-detection, rsu-block]
difficulty: intermediate
---

# U²-Net

## Meta Information

| Field | Value |
|-------|-------|
| **Paper Title** | U²-Net: Going Deeper with Nested U-Structure for Salient Object Detection |
| **Authors** | Qin, X., Zhang, Z., Huang, C., Dehghan, M., Zaiane, O.R., Jagersand, M. |
| **Year** | 2020 |
| **Venue** | Pattern Recognition |
| **arXiv** | [2005.09007](https://arxiv.org/abs/2005.09007) |

## One-Line Summary

U²-Net captures multi-scale features at each encoder/decoder stage by replacing standard convolution blocks with Residual U-blocks (RSU), creating a nested U-structure for salient object detection.

## Motivation

Standard encoder-decoder networks capture multi-scale information only through the hierarchical structure (successive pooling). Individual stages operate at a single scale. U²-Net addresses this by making each stage itself a mini U-Net (RSU block), capturing multi-scale features at every level of the architecture.

## Architecture

The outer architecture follows U-Net's encoder-decoder pattern with 6 encoder stages and 5 decoder stages. However, each stage is an RSU block — a small U-Net that processes features at multiple internal resolutions. This two-level nesting (U-structure within U-structure) gives the model its name: U²-Net.

## RSU Blocks

The Residual U-block (RSU-L) has L internal levels:
- Input features are processed through L encoding stages (with pooling) and L-1 decoding stages (with upsampling)
- Skip connections within the RSU block connect encoding and decoding stages
- A residual connection adds the input to the output

The depth L varies by position: deeper RSU blocks (L=7) in the early encoder stages where resolution is high, shallower RSU blocks (L=4) in later stages where resolution is already low.

## Results

| Dataset | Metric | BASNet | CPD | U²-Net |
|---------|--------|--------|-----|--------|
| DUTS-TE | MAE | 0.048 | 0.043 | 0.023 |
| DUT-OMRON | F-measure | 0.805 | 0.825 | 0.847 |
| SOD | MAE | 0.113 | 0.110 | 0.098 |

U²-Net achieves SOTA on multiple salient object detection benchmarks. The model has 44.0M parameters (standard) or 4.7M (U²-Net-lite), making it practical for deployment.

## Impact

U²-Net became widely used for background removal (e.g., rembg library uses U²-Net). The nested U-structure concept influenced later designs. The architecture demonstrated that deeper nesting of U-structures provides rich multi-scale features without extreme computational overhead.
