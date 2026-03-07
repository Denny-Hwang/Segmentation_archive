---
title: "Adapter Tuning for SAM"
date: 2025-03-06
status: complete
tags: [adapter, parameter-efficient, fine-tuning, transfer-learning]
difficulty: intermediate
---

# Adapter Tuning

## Overview

Adapter tuning is a parameter-efficient fine-tuning (PEFT) strategy that inserts small trainable modules (adapters) into a frozen pretrained model. In the context of SAM, adapters are added to the ViT image encoder to adapt its feature representations for new domains while preserving the original pretrained weights. This document covers the mechanics of adapter tuning, comparisons with alternative PEFT methods, and practical guidance for applying adapters to SAM.

## Bottleneck Adapter Architecture

### Standard Design

The most common adapter design follows a bottleneck pattern:

```
x (input, dimension D)
|
Linear_down: D -> d     (d << D, e.g., d=64 when D=1280)
|
Activation (ReLU or GELU)
|
Linear_up: d -> D
|
Scale: multiply by factor s
|
Output: x + s * adapter(x)    (residual connection)
```

### Key Hyperparameters

| Hyperparameter | Typical Range | Effect |
|----------------|--------------|--------|
| Bottleneck dim (d) | 32-256 | Capacity vs. efficiency tradeoff |
| Scale factor (s) | 0.1-1.0 | Controls adapter influence; small initial values ensure stable training |
| Activation | ReLU, GELU | GELU often slightly better for ViT |
| Dropout | 0.0-0.1 | Regularization for small datasets |

### Parameter Count

For a ViT-H with D=1280 and bottleneck d=64:
- Parameters per adapter: 2 x 1280 x 64 + 64 + 1280 = ~164K
- With 2 adapters per block and 32 blocks: 32 x 2 x 164K = ~10.5M
- This is approximately 1.6% of ViT-H's 632M parameters

## Where to Insert Adapters

### Post-Attention Insertion

Adapters placed after the multi-head self-attention (MHSA) layer modify the attention output before it enters the residual stream:

```
MHSA output -> LayerNorm -> Adapter -> + residual
```

This position allows the adapter to refine which spatial relationships the attention has captured, useful for domains where object boundaries and spatial patterns differ from natural images.

### Post-FFN Insertion

Adapters placed after the feed-forward network (FFN) modify the feature transformation:

```
FFN output -> LayerNorm -> Adapter -> + residual
```

This position adjusts the non-linear feature mixing, useful for adapting to different texture and appearance distributions.

### Both Positions (Recommended)

Using adapters at both positions (the standard approach from Houlsby et al.) provides the most capacity and typically yields the best results. The overhead is doubled but still small relative to the full model.

### Parallel vs. Sequential

- **Sequential (standard):** Adapter processes the layer's output, then adds to the residual
- **Parallel:** Adapter processes the layer's input in parallel with the main layer, outputs are summed

Parallel adapters can be slightly more efficient at inference (adapter and main layer run simultaneously) but provide similar accuracy.

## Comparison to Other PEFT Methods

### LoRA (Low-Rank Adaptation)

LoRA decomposes weight updates as low-rank matrices: W' = W + BA where B is D x r and A is r x D.

| Aspect | Adapters | LoRA |
|--------|----------|------|
| Modification type | New modules added | Existing weights modified |
| Inference overhead | Slight (extra layers) | None (weights merged) |
| Where applied | After attention/FFN | Attention Q, K, V, O matrices |
| Auxiliary input | Can incorporate extra signals | Cannot |
| Typical rank/dim | d=64-128 | r=4-16 |
| Parameter count (ViT-H) | ~8-12M | ~2-6M |
| Performance | Slightly higher | Competitive |

**When to use LoRA:**
- When inference latency is critical (merged weights add zero overhead)
- When parameter budget is very tight
- When no auxiliary information needs to be injected

**When to use adapters:**
- When auxiliary task information (depth, edges, etc.) should be incorporated
- When multiple adaptations need to be switched dynamically
- When slightly higher capacity is justified

### Visual Prompt Tuning (VPT)

VPT prepends learnable tokens to the input sequence of each ViT layer.

| Aspect | Adapters | VPT |
|--------|----------|-----|
| Parameters | ~8-12M | ~0.5-2M |
| Performance | Higher | Lower (2-5 points) |
| Mechanism | Transforms features | Adds context tokens |
| Depth | Per-layer modification | Per-layer token injection |

VPT is the most parameter-efficient but has limited representational capacity, making it unsuitable for large domain shifts.

### BitFit (Bias Tuning)

Only bias terms in the network are fine-tuned.

- Extremely parameter-efficient (~0.1% of parameters)
- Performance is significantly below adapters and LoRA (5-10 point gap)
- Best suited as a lightweight baseline or for very small datasets

### Head-Only Fine-Tuning

Only the mask decoder is trained; the encoder is completely frozen.

- No modification to the encoder at all
- Performance depends heavily on how well the pretrained features transfer
- Works well when the domain gap is small; fails for large shifts (medical, remote sensing)

## Training Strategies

### Learning Rate

Adapters typically require higher learning rates than full fine-tuning:
- Full fine-tuning: 1e-5 to 5e-5
- Adapter tuning: 1e-4 to 5e-4
- The adapter parameters are randomly initialized and need larger updates to learn effectively

### Initialization

- Down-projection: Random Gaussian (std 0.01)
- Up-projection: Zero initialization (ensures adapter outputs zero at the start, making it a no-op)
- Scale factor: Initialize to 0.1 and let it be learned

Zero initialization of the up-projection is critical: it ensures the model starts from the pretrained behavior and gradually incorporates adapter contributions.

### Warmup and Scheduling

- Linear warmup for 5-10% of training steps
- Cosine decay schedule to final learning rate of 1e-6
- This combination prevents early training instability from the randomly initialized adapters

### Data Augmentation

Since adapter training uses small datasets, strong augmentation is important:
- Random horizontal/vertical flips
- Random rotation (up to 30 degrees)
- Color jitter (for natural images)
- Random scaling/cropping
- Mixup or CutMix can further regularize

## Multi-Domain Adapter Switching

### Architecture for Multiple Domains

A key advantage of adapters is the ability to maintain a single frozen backbone with multiple adapter sets:

```
Frozen SAM backbone
  |-- Medical adapters    (swap in for medical images)
  |-- Remote sensing adapters (swap in for satellite images)
  |-- Camouflage adapters (swap in for COD)
```

At inference time, the appropriate adapter set is loaded based on the input domain. This requires:
- Storing only the adapter weights per domain (~8-12M per set)
- No duplication of the backbone (~632M shared)
- A domain identification mechanism (manual or automatic)

### Adapter Fusion

When input domains are mixed or unknown, adapter fusion combines multiple adapter outputs:
- Simple averaging of adapter outputs from multiple domains
- Learned attention weights over adapter outputs
- This can improve robustness but adds inference cost

## Practical Recommendations

| Scenario | Recommended Approach |
|----------|---------------------|
| Large domain shift, large dataset | Full fine-tuning or adapters with d=128 |
| Large domain shift, small dataset | Adapters with d=64 + strong augmentation |
| Small domain shift | LoRA rank 8 or head-only fine-tuning |
| Multiple domains needed | Adapters (swappable) |
| Inference latency critical | LoRA (merged weights) |
| Auxiliary information available | Adapters (can inject extra signals) |
