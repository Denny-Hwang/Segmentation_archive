---
title: "DS-TransUNet: Dual Swin Transformer U-Net for Medical Image Segmentation"
date: 2025-03-06
status: complete
tags: [dual-scale, swin-transformer, medical-segmentation, feature-fusion]
difficulty: advanced
---

# DS-TransUNet

## Meta Information

| Field | Value |
|-------|-------|
| **Paper Title** | DS-TransUNet: Dual Swin Transformer U-Net for Medical Image Segmentation |
| **Authors** | Lin, A., Chen, B., Xu, J., Zhang, Z., Lu, G., Zhang, D. |
| **Year** | 2022 |
| **Venue** | arXiv |
| **arXiv** | [2106.06716](https://arxiv.org/abs/2106.06716) |
| **Difficulty** | Advanced |

## One-Line Summary

DS-TransUNet employs a dual-scale Swin Transformer encoding strategy with a Transformer Interactive Fusion (TIF) module to capture both fine-grained and coarse semantic features for medical image segmentation.

## Motivation and Problem Statement

Prior transformer-based segmentation methods like TransUNet and Swin-Unet process input images at a single scale, limiting their ability to simultaneously capture fine-grained local details and coarse global semantics. Medical images often contain structures at vastly different scales — small lesions alongside large organs — making multi-scale representation critical. DS-TransUNet addresses this gap by introducing a dual-encoder architecture that processes the input at two different scales simultaneously, fusing features through a transformer-based fusion module.

## Architecture Overview

DS-TransUNet follows a U-shaped encoder-decoder design with two parallel Swin Transformer encoders operating at different input resolutions. The first encoder processes the original-resolution input for fine details, while the second encoder processes a downsampled version for broader semantic context. Features are fused at each stage using TIF modules with cross-attention, then passed to a CNN decoder with skip connections.

### Key Components

- **Dual-Scale Encoding**: See [dual_scale_encoding.md](dual_scale_encoding.md)
- **TIF Module**: See [tif_module.md](tif_module.md)

## Technical Details

### Dual Swin Transformer Encoders

Two encoders share the same Swin Transformer architecture but operate on different resolutions. The fine-scale encoder receives the original image (224×224), producing hierarchical features at H/4, H/8, H/16, and H/32. The coarse encoder receives a downsampled version (112×112) with proportionally larger effective receptive fields per window. Both use shifted window attention but maintain separate weights for specialization.

### Transformer Interactive Fusion (TIF)

At each stage, features from both encoders are fused via cross-attention. Queries from one scale attend to keys/values from the other, enabling bidirectional information exchange. A learned gating mechanism weights each scale's contribution adaptively. This produces features combining spatial details from the high-resolution path with semantic context from the low-resolution path.

### Decoder Design

A standard U-Net-style decoder with bilinear upsampling followed by 3×3 convolution blocks. Skip connections bring fused encoder features to corresponding decoder stages. Deep supervision at intermediate levels provides gradient signals throughout the network.

### Loss Function

Combination of cross-entropy and Dice loss with equal weighting. Auxiliary losses at each decoder level with decreasing weights for lower-resolution predictions.

## Experiments and Results

### Datasets

Evaluated on Synapse Multi-Organ CT (30 scans, 8 organs) and ACDC cardiac MRI (100 patients, 3 structures) using standard TransUNet protocols.

### Key Results

| Dataset | Metric | DS-TransUNet | TransUNet | Swin-Unet | nnU-Net |
|---------|--------|-------------|-----------|-----------|---------|
| Synapse | mDSC (%) | 82.58 | 77.48 | 79.13 | 82.50 |
| Synapse | mHD95 (mm) | 17.09 | 31.69 | 21.55 | 15.20 |
| ACDC | mDSC (%) | 91.51 | 89.71 | 90.00 | 91.61 |

### Ablation Studies

Removing either encoder degrades performance significantly. Replacing TIF with simple concatenation reduces mDSC by ~1.5%. Deep supervision adds ~0.5%. The dual-scale design is most impactful for organs with complex boundaries (e.g., stomach, gallbladder).

## Strengths

- Novel dual-scale encoding captures complementary multi-resolution information
- TIF cross-attention fusion is more expressive than naive concatenation
- Consistent improvements over single-encoder baselines
- Competitive with nnU-Net despite being a fixed (non-self-configuring) architecture

## Limitations

- ~2× computational cost of single-encoder models
- Optimal scale ratio requires manual tuning
- Limited to 2D slice-by-slice processing
- Memory intensive for high-resolution inputs

## Connections

Extends Swin-Unet to dual-scale processing. Shares U-shaped design with TransUNet but uses dual pure-transformer encoders. TIF generalizes FPN-style multi-scale fusion to the transformer domain. Complementary to nnU-Net's self-configuring philosophy.

## References

- Cao et al., "Swin-Unet," 2021
- Chen et al., "TransUNet," 2021
- Liu et al., "Swin Transformer," 2021
- Isensee et al., "nnU-Net," Nature Methods, 2021
