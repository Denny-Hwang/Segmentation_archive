---
title: "UNet 3+: A Full-Scale Connected UNet for Medical Image Segmentation"
date: 2025-03-06
status: complete
tags: [unet3plus, full-scale-skip, classification-guided, deep-supervision]
difficulty: intermediate
---

# UNet 3+

## Meta Information

| Field | Value |
|-------|-------|
| **Paper Title** | UNet 3+: A Full-Scale Connected UNet for Medical Image Segmentation |
| **Authors** | Huang, H., Lin, L., Tong, R., et al. |
| **Year** | 2020 |
| **Venue** | ICASSP |

## One-Line Summary

UNet 3+ introduces full-scale skip connections that combine features from ALL encoder levels and ALL prior decoder levels at each decoder node, providing comprehensive multi-scale feature aggregation.

## Motivation

U-Net uses same-scale skip connections (encoder level i → decoder level i). UNet++ uses dense skip connections but still primarily connects within neighboring scales. UNet 3+ argues that EVERY decoder node should access features from ALL scales — both all encoder levels and all previously computed decoder levels — to achieve the most comprehensive multi-scale representation.

## Architecture

Each decoder node X_de^i receives inputs from:
1. **All encoder levels** (X_en^1 through X_en^5): via downsampling (for higher-resolution) or upsampling (for lower-resolution) to match the target decoder resolution
2. **Same-level encoder**: direct skip connection
3. **All prior decoder levels**: features from deeper decoder levels, upsampled to match

All inputs are first reduced to a uniform channel dimension (64) via 3×3 conv + BN + ReLU, then concatenated and fused with another 3×3 conv.

## Key Contributions

1. **Full-scale skip connections**: Each decoder node aggregates features from all 5 encoder levels and available decoder levels
2. **Classification-guided module (CGM)**: An auxiliary classification branch that predicts whether the image contains the target structure, masking the segmentation output for negative images
3. **Full-scale deep supervision**: Auxiliary losses at every decoder level

## Results

| Dataset | Metric | U-Net | UNet++ | UNet 3+ |
|---------|--------|-------|--------|---------|
| Liver | Dice (%) | 94.97 | 95.56 | 97.17 |
| Spleen | Dice (%) | 95.93 | 96.01 | 96.82 |

UNet 3+ achieves 1-2% Dice improvement over UNet++ with fewer parameters (26.97M vs 36.63M for UNet++), thanks to the unified 64-channel feature dimension at each decoder node.

## Limitations

- Complex feature routing with many resize operations
- Full-scale connections may be overkill for simple segmentation tasks
- CGM assumes binary presence/absence, not applicable to multi-class with always-present classes
