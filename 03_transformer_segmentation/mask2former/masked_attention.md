---
title: "Masked Attention in Mask2Former"
date: 2025-03-06
status: complete
tags: [masked-attention, cross-attention, transformer-decoder, segmentation]
difficulty: advanced
---

# Masked Attention

## Overview

Masked attention restricts cross-attention in the transformer decoder so each query only attends to feature map locations within its predicted mask region, rather than globally. This focused attention dramatically improves convergence speed, reduces computational waste, and helps distinguish nearby instances.

## Standard Cross-Attention vs. Masked Attention

Standard cross-attention: `Attn(Q,K,V) = softmax(QK^T/√d)V` — every query attends to all H×W positions, wasting computation on irrelevant regions.

Masked attention adds a mask bias: `MaskedAttn(Q,K,V) = softmax(QK^T/√d + M)V` where `M(x,y) = 0` inside the predicted mask and `M(x,y) = -∞` outside. The -∞ values become 0 after softmax, zeroing out attention outside the target region. For layer 1, no mask is applied (equivalent to standard cross-attention).

## Mask Prediction and Attention Restriction

Each decoder layer produces a mask prediction via linear projection + dot product with pixel features. This mask is binarized at threshold 0.5 to create the attention mask for the next layer. Nearest-neighbor interpolation resizes masks to match the current feature scale (due to round-robin multi-scale strategy).

## Iterative Refinement

Creates a coarse-to-fine process: Layer 1 uses global attention for initial coarse prediction → Layer 2 restricts to coarse mask → Layers 3-9 progressively refine. This is especially beneficial for complex shapes where single global attention struggles to capture the full object extent.

## Impact on Convergence

Standard cross-attention models require 300+ epochs. Masked attention achieves strong results in 50 epochs (3-4× faster) by reducing the search space for each query, making optimization smoother. The spatial prior provided by the mask is a strong initialization for the attention pattern.

## Computational Efficiency

Theoretical complexity remains O(N × H × W) per query, but effective attention is sparse. The mask bias is applied as an additive term in softmax with negligible overhead. In practice, implementations can skip masked positions for additional speedup.

## Ablation Results

| Configuration | ADE20K PQ | Convergence |
|--------------|-----------|-------------|
| Standard cross-attention | 48.1 | 200+ epochs |
| Masked attention | 52.5 | 50 epochs |
| Masked attention + multi-scale | 54.5 | 50 epochs |

The +4.4 PQ improvement from masked attention alone is significant, with additional gains from multi-scale features.

## Implementation Notes

Key details: the mask is detached from the computation graph to prevent gradient instability; a small number of padding positions are kept in fully masked regions to prevent NaN from empty softmax; the mask threshold 0.5 is fixed during training.

```python
def masked_cross_attention(query, key, value, mask_pred):
    attn = torch.bmm(query, key.transpose(1, 2)) / math.sqrt(d)
    mask = (mask_pred.detach().sigmoid() > 0.5).flatten(2)  # (B, N, HW)
    attn_bias = torch.where(mask, 0.0, float('-inf'))
    attn = F.softmax(attn + attn_bias, dim=-1)
    return torch.bmm(attn, value)
```
