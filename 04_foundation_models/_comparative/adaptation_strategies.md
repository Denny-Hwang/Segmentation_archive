---
title: "Adaptation Strategies for Foundation Segmentation Models"
date: 2025-03-06
status: complete
tags: [adaptation, fine-tuning, adapter, lora, parameter-efficient]
difficulty: intermediate
---

# Adaptation Strategies

## Overview

Foundation segmentation models like SAM are trained on broad data (SA-1B) and perform well on general natural images. However, adapting them to specialized domains (medical, remote sensing, industrial inspection, etc.) requires additional training. This document compares the major adaptation strategies: full fine-tuning, LoRA, bottleneck adapters, visual prompt tuning, and head-only fine-tuning.

## Strategy Comparison Matrix

| Strategy | Trainable Params | Training Cost | Performance | Forgetting Risk | Multi-Domain |
|----------|-----------------|---------------|-------------|-----------------|-------------|
| Full fine-tuning | 100% (~636M) | Very high | Highest | High | Separate models |
| LoRA (rank 16) | ~1% (~6M) | Low | High | Low | Swappable weights |
| Adapters (d=64) | ~1.5% (~10M) | Low | High | Very low | Swappable modules |
| Prompt tuning | ~0.2% (~1.5M) | Very low | Moderate | Very low | Swappable tokens |
| Head-only | ~0.6% (~4M) | Very low | Low-Moderate | None | Swappable heads |

## Full Fine-Tuning

### Method

All parameters of the model (encoder + prompt encoder + decoder) are updated during training. This is the most straightforward adaptation approach.

### When to Use

- Large domain-specific datasets available (>50K samples)
- Maximum performance is required on the target domain
- The adapted model does not need to retain performance on the original domain
- Computational budget for training is not a constraint

### Implementation Details

- Learning rate: 1e-5 to 5e-5 (lower than training from scratch)
- Warmup: 1-5% of total steps
- Weight decay: 0.01-0.05
- Training epochs: 10-50 depending on dataset size

### Pros and Cons

**Pros:**
- Highest adaptation capacity
- All layers adjust to the new domain
- Straightforward implementation

**Cons:**
- Risk of catastrophic forgetting (model loses general capabilities)
- Requires large datasets to avoid overfitting
- Full model copy needed per domain (~2.5 GB per adaptation for ViT-H)
- Long training time

## LoRA (Low-Rank Adaptation)

### Method

LoRA factorizes weight updates as low-rank matrices. For a weight matrix W in the attention layers, the adapted weight is W' = W + BA where B is D x r and A is r x D, with rank r << D.

### Where to Apply

Typically applied to the attention projection matrices in the ViT encoder:
- Query projection (W_q)
- Value projection (W_v)
- Optionally: Key projection (W_k) and output projection (W_o)

Applying LoRA to Q and V only is the most common and cost-effective configuration.

### Rank Selection

| Rank | Parameters | Typical Performance Impact |
|------|-----------|---------------------------|
| 1 | ~0.4M | Minimal adaptation, small domain shift only |
| 4 | ~1.5M | Good for moderate domain shifts |
| 8 | ~3M | Strong adaptation for most domains |
| 16 | ~6M | Near full fine-tuning performance |
| 32 | ~12M | Rarely needed; diminishing returns |

### Key Advantage: Weight Merging

At inference time, the LoRA matrices can be merged into the original weights: W' = W + BA. This adds zero inference overhead, making LoRA the fastest PEFT method at deployment.

### Implementation

```python
# Pseudocode for LoRA applied to a linear layer
class LoRALinear(nn.Module):
    def __init__(self, original_linear, rank=8, alpha=16):
        self.original = original_linear  # frozen
        self.lora_A = nn.Linear(original_linear.in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, original_linear.out_features, bias=False)
        self.scaling = alpha / rank

    def forward(self, x):
        return self.original(x) + self.lora_B(self.lora_A(x)) * self.scaling
```

## Bottleneck Adapters

### Method

Small bottleneck MLP modules are inserted after the attention and FFN layers in each ViT block. The original weights are frozen.

### Design Choices

- **Bottleneck dimension:** 64-128 (vs. hidden dim 1280 for ViT-H)
- **Activation:** ReLU or GELU
- **Initialization:** Zero init for up-projection ensures no-op at start
- **Scale factor:** Learned, initialized to 0.1

### Key Advantage: Auxiliary Input Injection

Unlike LoRA, adapters can incorporate additional input signals:
- Depth maps for scene understanding
- Edge maps for boundary refinement
- Frequency features for camouflage detection
- Domain-specific prior information

This makes adapters particularly suitable for tasks where auxiliary information is available.

### Comparison to LoRA in Practice

On medical imaging benchmarks (adapting SAM ViT-B):

| Method | Params | Liver DSC | Kidney DSC | Brain DSC |
|--------|--------|-----------|------------|-----------|
| Full fine-tune | 91M | 0.91 | 0.89 | 0.83 |
| LoRA r=8 | 3M | 0.89 | 0.87 | 0.80 |
| Adapter d=64 | 8M | 0.90 | 0.88 | 0.81 |
| Head-only | 4M | 0.84 | 0.82 | 0.75 |

Adapters slightly outperform LoRA at the cost of more parameters and inference overhead.

## Visual Prompt Tuning (VPT)

### Method

Learnable tokens are prepended to the input sequence of each ViT layer. These tokens interact with the image tokens through self-attention, effectively steering the model's computation.

### Variants

- **VPT-Shallow:** Learnable tokens added only to the first layer
- **VPT-Deep:** Learnable tokens added to every layer (more effective, standard choice)

### Token Count

Typical configuration: 10-50 tokens per layer
- 10 tokens x 32 layers x 1280 dim = ~0.4M parameters
- 50 tokens x 32 layers x 1280 dim = ~2M parameters

### Limitations

- Cannot directly modify the feature computation; only influences it through attention
- Performance gap of 3-7 points compared to adapters/LoRA on large domain shifts
- More effective for subtle distribution shifts where the pretrained features are largely appropriate

## Head-Only Fine-Tuning

### Method

Only the mask decoder (and optionally the prompt encoder) is trained. The image encoder remains completely frozen.

### When This Works

- The target domain is visually similar to the pretraining data
- The encoder features are already suitable (e.g., natural images with different object categories)
- Very limited training data is available (< 1K samples)
- Training compute is extremely constrained

### When This Fails

- Large domain gap (medical, satellite, microscopy)
- The encoder produces features that are fundamentally misaligned with the target domain
- Fine-grained boundary accuracy is critical (frozen encoder may not capture domain-specific edges)

## Combined Strategies

### LoRA + Head Fine-Tuning

Apply LoRA to the encoder and fine-tune the decoder fully. This is a common effective combination that:
- Adapts encoder features with minimal parameter overhead
- Allows the decoder maximum flexibility to use the adapted features
- Typically performs within 1-2 points of full fine-tuning

### Adapters + Frozen Decoder

Use adapters in the encoder but keep the decoder frozen. This tests whether the adaptation is purely in the feature space:
- Works when the decoder architecture is general enough
- Limits adaptation if the decoder has task-specific biases

### Progressive Unfreezing

Start with head-only fine-tuning, then gradually unfreeze encoder layers from top to bottom:
1. Epoch 1-5: Only decoder
2. Epoch 5-10: Decoder + last 8 encoder blocks
3. Epoch 10-20: Decoder + all encoder blocks

This approach stabilizes training and often outperforms direct full fine-tuning on small datasets.

## Decision Guide

```
Is the domain gap large?
├── No → Head-only fine-tuning or LoRA rank 4
└── Yes
    ├── Large dataset (>50K)? → Full fine-tuning
    └── Small dataset (<50K)?
        ├── Need auxiliary inputs? → Adapters
        ├── Inference speed critical? → LoRA (mergeable)
        └── Multiple domains? → Adapters (swappable)
```

## Practical Tips

1. **Always start with head-only** as a baseline; it reveals how much the pretrained features already help
2. **LoRA rank 8 on Q and V** is a strong default that works for most domain shifts
3. **Learning rate matters more than method choice** for small performance differences
4. **Validate on a held-out set from the target domain** to detect overfitting early
5. **Combine data augmentation** with PEFT methods; they are complementary
6. **Monitor training loss and validation metrics together** to catch overfitting (common with small datasets and high-capacity methods)
