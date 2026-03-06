---
title: "Stable Diffusion U-Net Architecture"
date: 2025-03-06
status: planned
tags:
  - stable-diffusion
  - latent-diffusion
  - cross-attention
  - architecture
parent: unet_in_diffusion/review.md
---

# Stable Diffusion U-Net Architecture

## Overview

_TODO: Describe the specific U-Net architecture used in Stable Diffusion (Latent Diffusion Models) and how it differs from the original biomedical U-Net._

---

## Architecture Components

### Encoder Blocks

_TODO: ResNet blocks + spatial self-attention + cross-attention (for text conditioning)._

### Middle Block

_TODO: ResNet + self-attention + ResNet at the bottleneck._

### Decoder Blocks

_TODO: ResNet blocks + spatial self-attention + cross-attention, with skip connections from encoder._

---

## Key Modifications vs Original U-Net

| Component | Original U-Net | Stable Diffusion U-Net |
|-----------|---------------|----------------------|
| Convolution blocks | 2x (Conv + ReLU) | ResNet blocks with GroupNorm + SiLU |
| Attention | None | Self-attention + cross-attention |
| Conditioning | None | Cross-attention with CLIP text embeddings |
| Timestep | N/A | Sinusoidal embedding + MLP, injected into ResBlocks |
| Input space | Pixel space | Latent space (4-channel) |
| Output | Class probabilities | Predicted noise |

---

## Cross-Attention for Text Conditioning

_TODO: How text embeddings from CLIP are injected via cross-attention at multiple scales._

---

## Timestep Conditioning

_TODO: Sinusoidal encoding -> MLP -> scale and shift in each ResNet block._

---

## Channel and Resolution Progression

_TODO: Typical configuration: [320, 640, 1280, 1280] channels at resolutions [64, 32, 16, 8]._

---

## Parameter Count

_TODO: Approximately 860M parameters in the U-Net alone._

---

## Variants

| Model | U-Net Changes |
|-------|--------------|
| SD 1.x | Base architecture |
| SD 2.x | Deeper attention, different text encoder |
| SDXL | Larger U-Net, additional conditioning |

---

## Future: U-Net vs Transformer

_TODO: Discuss the DiT (Diffusion Transformer) architecture that replaces U-Net with a transformer._
