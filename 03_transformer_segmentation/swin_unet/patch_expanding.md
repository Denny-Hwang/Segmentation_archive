---
title: "Patch Expanding Layer in Swin-Unet"
date: 2025-03-06
status: planned
tags: [patch-expanding, upsampling, decoder, swin-transformer]
difficulty: intermediate
---

# Patch Expanding Layer

## Overview

The patch expanding layer is a novel upsampling operation introduced in Swin-Unet that serves as the counterpart to the patch merging layer used in the Swin Transformer encoder. While patch merging reduces spatial resolution and increases channel dimension (analogous to strided convolution or pooling in CNNs), patch expanding performs the inverse operation: it increases spatial resolution while reducing channel dimension. This operation is essential for building the decoder of a pure Transformer U-shaped architecture, enabling the progressive recovery of spatial resolution needed for dense segmentation prediction.

The patch expanding layer operates entirely through linear transformations and tensor reshaping, without relying on any convolutional operations. This maintains the pure Transformer design philosophy of Swin-Unet and provides a learnable upsampling mechanism that is jointly optimized with the rest of the network.

## Motivation

In the standard U-Net architecture, upsampling in the decoder is typically achieved through transposed convolutions (deconvolutions) or bilinear interpolation followed by convolution. However, these operations are inherently convolutional and would break the pure Transformer design of Swin-Unet. A dedicated Transformer-compatible upsampling mechanism was needed to complement the patch merging downsampling operation.

Patch merging in the encoder concatenates features from $2 \times 2$ neighboring patches and applies a linear layer to reduce the dimension from $4C$ to $2C$, effectively halving the spatial resolution while doubling the feature channels. Patch expanding reverses this process: it uses a linear layer to expand the channel dimension, followed by a reshape operation that redistributes the expanded channels into additional spatial positions. This creates a natural encoder-decoder symmetry in the Swin-Unet architecture.

## Mechanism

The patch expanding operation consists of two steps:

**Step 1: Linear expansion.** Given an input feature map of shape $\frac{H}{2^k} \times \frac{W}{2^k} \times C_k$, a linear layer first expands the channel dimension by a factor of 2, producing features of shape $\frac{H}{2^k} \times \frac{W}{2^k} \times 2C_k$.

**Step 2: Spatial rearrangement.** The expanded features are then rearranged (reshaped) to double the spatial dimensions while halving the channel count. Specifically, each token's $2C_k$-dimensional feature vector is split into 4 groups of $\frac{C_k}{2}$ dimensions, and these groups are distributed to a $2 \times 2$ spatial neighborhood:

$$\text{PatchExpand}: \mathbb{R}^{\frac{H}{2^k} \times \frac{W}{2^k} \times 2C_k} \rightarrow \mathbb{R}^{\frac{H}{2^{k-1}} \times \frac{W}{2^{k-1}} \times \frac{C_k}{2}}$$

This can be understood as the inverse of the PixelShuffle (sub-pixel convolution) operation from Shi et al. (2016), but implemented with a preceding linear layer instead of convolution. The rearrangement operation itself is parameter-free; all learnable capacity resides in the linear layer.

For the final upsampling from $\frac{H}{4} \times \frac{W}{4}$ to $H \times W$, a $4\times$ patch expanding layer is used. This layer expands channels by a factor of 16 and rearranges them into a $4 \times 4$ spatial grid per token, achieving the full resolution recovery in a single step.

## Comparison with Other Upsampling Methods

**Transposed Convolution (Deconvolution)**: Uses a learnable convolutional kernel applied in a "reverse" manner to upsample feature maps. Transposed convolutions can produce checkerboard artifacts due to uneven overlap patterns and are inherently local operations. Patch expanding avoids checkerboard artifacts since it distributes channels uniformly and can capture global patterns through its linear transformation.

**Bilinear Interpolation**: A fixed, non-learnable operation that computes output values as weighted averages of neighboring inputs. While simple and artifact-free, bilinear interpolation cannot learn task-specific upsampling patterns. It is often followed by a convolution to add learnable capacity, but this reintroduces convolutional operations.

**PixelShuffle / Sub-Pixel Convolution**: The most closely related operation. PixelShuffle rearranges channels into spatial dimensions after a convolutional expansion. Patch expanding uses the same rearrangement but with a linear (fully-connected) expansion instead of convolution, making it compatible with the Transformer framework.

**Nearest Neighbor Upsampling**: Simply replicates each feature vector to fill a larger spatial grid. Like bilinear interpolation, it is non-learnable and typically requires a subsequent convolution for refinement.

Patch expanding occupies a unique position as a learnable, non-convolutional upsampling operation that naturally pairs with the patch merging downsampling used in Swin Transformers.

## Role in the Decoder

In the Swin-Unet decoder, patch expanding layers are positioned at the beginning of each decoder stage, analogous to how patch merging layers appear at the end of each encoder stage. The decoder pipeline for each stage is:

1. **Patch Expanding**: Upsample the incoming features by $2\times$ spatially.
2. **Skip Connection Fusion**: Concatenate the upsampled features with the corresponding encoder features from the skip connection, then apply a linear layer to reduce the channel dimension.
3. **Swin Transformer Blocks**: Process the fused features through two consecutive Swin Transformer blocks (W-MSA + SW-MSA) to refine the representation.

This structure is repeated three times, producing features at progressively increasing resolutions: $\frac{H}{16} \rightarrow \frac{H}{8} \rightarrow \frac{H}{4}$. The final $4\times$ patch expanding layer then recovers the full input resolution, and a linear classifier produces the per-pixel segmentation predictions.

The use of Swin Transformer blocks after patch expanding is important because the upsampled features can be noisy or contain artifacts from the channel-to-spatial redistribution. The self-attention mechanism in the Swin Transformer blocks refines these features by capturing local and cross-window spatial relationships, producing cleaner segmentation maps.

## Implementation Notes

The patch expanding layer is straightforward to implement in PyTorch. The core operation uses `nn.Linear` for the channel expansion and `rearrange` (from the `einops` library) or manual `reshape`/`permute` operations for the spatial redistribution:

```python
# Conceptual implementation
x = self.linear_expand(x)  # (B, H/2k, W/2k, 2*C) -> expand channels
x = rearrange(x, 'b h w (p1 p2 c) -> b (h p1) (w p2) c', p1=2, p2=2)
```

Key implementation considerations include: (1) ensuring the channel dimension is divisible by the upsampling factor squared (4 for $2\times$ upsampling); (2) matching the output channel dimension with the skip connection dimension for proper concatenation; (3) applying layer normalization after the rearrangement to stabilize the distribution of the redistributed features. The final $4\times$ patch expanding uses a linear layer that expands channels by $16\times$, followed by rearrangement with $p1 = p2 = 4$.
