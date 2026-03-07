---
title: "Shifted Window Mechanism in Swin-Unet"
date: 2025-03-06
status: planned
tags: [swin-transformer, shifted-window, attention, computational-efficiency]
difficulty: intermediate
---

# Shifted Window Mechanism

## Overview

The shifted window mechanism is the core innovation of the Swin Transformer that enables efficient self-attention computation while maintaining the ability to model cross-region dependencies. In standard Vision Transformers, global self-attention computes pairwise interactions between all spatial tokens, resulting in quadratic computational complexity $O(N^2)$ where $N$ is the number of tokens. The shifted window approach partitions the feature map into non-overlapping local windows and computes self-attention within each window, reducing complexity to linear with respect to image size. By alternating between regular and shifted window partitions in consecutive Transformer layers, the mechanism enables information flow across window boundaries without the prohibitive cost of global attention.

In the context of Swin-Unet, the shifted window mechanism is fundamental to both the encoder and decoder, enabling the pure Transformer architecture to efficiently process feature maps at multiple scales while maintaining cross-region connectivity that is essential for coherent segmentation predictions.

## Window-Based Self-Attention (W-MSA)

Window-based multi-head self-attention (W-MSA) partitions the feature map into non-overlapping windows of fixed size $M \times M$ (typically $M = 7$). Self-attention is then computed independently within each window. For a feature map of size $\frac{H}{s} \times \frac{W}{s}$ (where $s$ is the downsampling factor), the feature map is divided into $\frac{H}{sM} \times \frac{W}{sM}$ windows, each containing $M^2$ tokens.

Within each window, standard multi-head self-attention is computed:

$$\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{QK^T}{\sqrt{d_k}} + B\right)V$$

where $Q, K, V \in \mathbb{R}^{M^2 \times d}$ are the query, key, and value matrices for tokens within one window, and $B \in \mathbb{R}^{M^2 \times M^2}$ is the relative position bias that encodes the spatial relationship between tokens within the window.

The key advantage is computational: instead of computing attention over all $N = \frac{H}{s} \times \frac{W}{s}$ tokens (cost $O(N^2)$), attention is computed independently in each of the $\frac{N}{M^2}$ windows (cost $O(M^2)$ per window), giving a total cost of $O(N \cdot M^2)$. Since $M$ is a small constant (7), this is linear in $N$.

However, W-MSA alone has a critical limitation: there is no information exchange between different windows. Tokens in one window cannot attend to tokens in adjacent windows, creating artificial boundaries that can fragment the representation and degrade segmentation quality, particularly at window edges.

## Shifted Window Self-Attention (SW-MSA)

Shifted window self-attention addresses the isolation problem of W-MSA by displacing the window partition by $(\lfloor M/2 \rfloor, \lfloor M/2 \rfloor)$ pixels in the second layer of each pair. For $M = 7$, this means shifting the grid by $(3, 3)$ positions. The resulting windows span across the boundaries of the previous layer's regular windows, enabling cross-window connections.

In a two-layer Swin Transformer block, the computation alternates:

$$\hat{z}^l = \text{W-MSA}(\text{LN}(z^{l-1})) + z^{l-1}$$
$$z^l = \text{MLP}(\text{LN}(\hat{z}^l)) + \hat{z}^l$$
$$\hat{z}^{l+1} = \text{SW-MSA}(\text{LN}(z^l)) + z^l$$
$$z^{l+1} = \text{MLP}(\text{LN}(\hat{z}^{l+1})) + \hat{z}^{l+1}$$

The first layer uses regular window partitioning (W-MSA), and the second layer uses shifted window partitioning (SW-MSA). This alternation ensures that every pair of spatially adjacent tokens can interact through one of the two layers. The effective receptive field grows progressively through the network as information propagates across window boundaries at each shifted-window layer.

For segmentation tasks, this alternating pattern is particularly important because segmentation boundaries can fall anywhere in the image. Without the shifted windows, features near window boundaries would lack the cross-boundary context needed for accurate boundary delineation.

## Computational Complexity

The computational complexity comparison between global self-attention and windowed self-attention is substantial:

**Global Multi-Head Self-Attention (MSA)**:
$$\Omega(\text{MSA}) = 4hwC^2 + 2(hw)^2C$$

where $h \times w$ is the spatial resolution and $C$ is the embedding dimension. The $(hw)^2$ term makes this quadratic in the number of tokens.

**Window-Based Multi-Head Self-Attention (W-MSA)**:
$$\Omega(\text{W-MSA}) = 4hwC^2 + 2M^2hwC$$

The quadratic term $(hw)^2C$ is replaced by $M^2hwC$, which is linear in the spatial dimensions since $M$ is constant. For a $56 \times 56$ feature map with $M = 7$, this represents a reduction from $56^2 \times 56^2 = 9.8M$ pairwise interactions to $56^2 \times 7^2 = 153K$ interactions per window summed over all windows, a roughly $64\times$ reduction.

This efficiency gain is critical for segmentation architectures that must process feature maps at multiple scales, including relatively high resolutions ($\frac{H}{4}$) in the early encoder and late decoder stages. Without windowed attention, applying Transformer layers at these resolutions would be computationally prohibitive.

## Cyclic Shift Implementation

A naive implementation of shifted window attention would create windows of varying sizes at the boundaries of the feature map, requiring padding and complicating batched computation. The Swin Transformer uses an efficient cyclic shift strategy to avoid this problem.

Instead of creating irregular boundary windows, the feature map is cyclically shifted by $(\lfloor M/2 \rfloor, \lfloor M/2 \rfloor)$ positions before applying the regular window partition. This cyclic shift wraps boundary regions to the opposite side of the feature map, ensuring all windows have the same $M \times M$ size. However, after shifting, some windows contain tokens that are not spatially adjacent in the original feature map. To prevent these non-adjacent tokens from attending to each other, an attention mask is applied.

The masking mechanism sets attention weights to $-\infty$ (before softmax) for pairs of tokens that were brought together by the cyclic shift but are not truly adjacent. This ensures that the shifted windows correctly model only cross-boundary interactions between genuinely neighboring regions. After attention computation, the feature map is cyclically shifted back by $(-\lfloor M/2 \rfloor, -\lfloor M/2 \rfloor)$ to restore the original spatial arrangement.

This implementation is elegant because it requires no padding, all windows are the same size enabling efficient batched computation, and the only overhead is the attention mask application, which is computationally negligible.

## Impact on Segmentation

The shifted window mechanism has several specific benefits for dense prediction tasks like segmentation:

**Boundary coherence**: By enabling cross-window information flow, the shifted window mechanism prevents the artificial fragmentation that would occur if attention were confined to isolated windows. Segmentation masks are continuous across window boundaries, producing spatially coherent predictions.

**Multi-scale context**: As shifted window attention is applied at each stage of the hierarchical encoder and decoder, the effective receptive field grows progressively. At deeper stages with larger token strides, each window covers a larger image region, capturing broader contextual information. Combined with skip connections from earlier stages, this provides both fine-grained local detail and broad spatial context.

**Relative position bias**: The learnable relative position bias $B$ in the attention computation encodes spatial relationships between tokens within a window, providing strong positional inductive biases that are particularly valuable for segmentation where spatial precision is paramount. The relative bias values are shared across all windows, ensuring consistent spatial modeling throughout the feature map.

**Efficient scaling**: The linear complexity of windowed attention allows Swin-Unet to apply Transformer blocks at high-resolution stages ($\frac{H}{4}$) where global self-attention would be impractical, enabling transformer-based feature refinement close to the output resolution.

## Implementation Notes

Key implementation details for the shifted window mechanism include:

1. **Window size**: The default window size is $M = 7$. The input feature map dimensions must be divisible by $M$ at each stage. For standard inputs of $224 \times 224$, the feature map sizes $56, 28, 14, 7$ are all divisible by 7.

2. **Relative position bias table**: A bias table of shape $(2M-1) \times (2M-1)$ stores the learned bias values for all possible relative positions within a window. The actual bias matrix $B \in \mathbb{R}^{M^2 \times M^2}$ is indexed from this table using precomputed relative position indices.

3. **Attention mask**: The mask for shifted windows is precomputed once and reused across all forward passes. It is a binary mask that is added to the attention logits (with $-100$ replacing the masked positions) before the softmax.

4. **Number of heads**: The default configuration uses $[3, 6, 12, 24]$ attention heads at the four stages respectively, with the head dimension fixed at 32. The increasing number of heads at deeper stages allows the model to capture more diverse patterns as the feature dimension grows.

5. **Drop path**: Stochastic depth (drop path) is used for regularization, with linearly increasing drop rates from 0 to a maximum value (typically 0.1--0.2) across the depth of the network.
