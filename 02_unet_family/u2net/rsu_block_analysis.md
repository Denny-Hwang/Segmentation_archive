---
title: "RSU Block Analysis in U²-Net"
date: 2025-03-06
status: complete
tags: [rsu-block, nested-u-structure, multi-scale, u2net]
difficulty: intermediate
---

# RSU Block Analysis

## Overview

The Residual U-block (RSU) is the core building block of U²-Net. It is a mini U-Net within each encoder/decoder stage, capturing multi-scale features at each level of the architecture. This creates a two-level hierarchy of U-structures.

## RSU-L Architecture

An RSU block with depth L (RSU-L) contains:
- An input convolution transforming input features
- L-1 encoder stages with progressive downsampling (3×3 conv + BN + ReLU + maxpool)
- A bottleneck convolution
- L-1 decoder stages with upsampling and skip connections
- A residual connection from input to output

For RSU-7 (used in early encoder stages):
```
Input → Conv → En1 → En2 → En3 → En4 → En5 → En6 → Bottleneck
                ↓      ↓      ↓      ↓      ↓      ↓
               De1 ← De2 ← De3 ← De4 ← De5 ← De6 ← Bottleneck
Output = Input_conv + De1
```

## Parameterization: RSU-L-Cin-Cmid

Each RSU block is parameterized by:
- **L**: Number of internal levels (depth of the mini U-Net)
- **C_in**: Input/output channel dimension
- **C_mid**: Internal channel dimension (typically C_in // 2 or C_in // 4)

Deeper RSU blocks (higher L) are used at higher resolutions where the feature maps are large enough to support multiple pooling operations. Shallower blocks are used at lower resolutions.

| Stage | Resolution | RSU Depth | Config |
|-------|-----------|-----------|--------|
| En1 | Full | RSU-7 | Deep, captures wide context |
| En2 | 1/2 | RSU-6 | Moderate depth |
| En3 | 1/4 | RSU-5 | Moderate depth |
| En4 | 1/8 | RSU-4 | Shallower |
| En5 | 1/16 | RSU-4F | Dilated (no pooling) |
| En6 | 1/32 | RSU-4F | Dilated (no pooling) |

## RSU-4F: Dilated Variant

At the lowest resolution levels, feature maps are too small for further pooling. RSU-4F replaces the pooling-based internal U-structure with dilated convolutions at rates (1, 2, 4, 8). This maintains multi-scale receptive fields without spatial reduction.

## Why RSU Blocks Work

1. **Multi-scale within each stage**: Standard U-Net blocks operate at a single scale; RSU captures multiple scales at each level
2. **Residual learning**: The shortcut connection allows the block to learn residual multi-scale features
3. **Efficient**: Internal channels (C_mid) are smaller than external, keeping computation manageable
4. **Depth-adaptive**: L is adjusted per stage based on available spatial resolution
