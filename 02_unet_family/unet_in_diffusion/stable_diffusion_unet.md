---
title: "U-Net Architecture in Stable Diffusion"
date: 2025-03-06
status: complete
tags: [stable-diffusion, latent-diffusion, cross-attention, time-embedding]
difficulty: advanced
---

# Stable Diffusion U-Net

## Overview

The U-Net in Stable Diffusion operates in the latent space of a variational autoencoder (VAE), processing 64×64×4 latent representations instead of full-resolution pixel images. It incorporates time step conditioning and cross-attention to text embeddings, enabling text-guided image generation.

## Architecture

The U-Net has 4 resolution levels (64→32→16→8 in latent space) with each level containing ResBlocks and SpatialTransformer blocks:

| Level | Resolution | Channels | ResBlocks | Attention |
|-------|-----------|----------|-----------|-----------|
| Down 1 | 64×64 | 320 | 2 | Optional |
| Down 2 | 32×32 | 640 | 2 | Yes |
| Down 3 | 16×16 | 1280 | 2 | Yes |
| Down 4 | 8×8 | 1280 | 2 | Yes |
| Mid | 8×8 | 1280 | 1 | Yes |
| Up 4-1 | (mirror) | (mirror) | 3 each | Yes |

## Time Step Embedding

The diffusion time step t is encoded using sinusoidal positional encoding, then projected through a 2-layer MLP:

```
t_emb = MLP(sinusoidal_embedding(t))  # → (batch, 1280)
```

This embedding conditions each ResBlock via Adaptive Group Normalization:
```
h = GroupNorm(h)
h = h * (1 + scale(t_emb)) + shift(t_emb)
```

where scale and shift are linear projections of the time embedding. This allows each ResBlock to behave differently at different noise levels.

## Cross-Attention to Text

Text prompts are encoded by a CLIP text encoder into a sequence of token embeddings (77 tokens × 768 dimensions). The SpatialTransformer block in each resolution level applies:

1. **Self-attention**: spatial features attend to themselves
2. **Cross-attention**: spatial features (queries) attend to text tokens (keys, values)
3. **FFN**: feedforward network

```
Q = W_q · spatial_features    # from image latents
K = W_k · text_embeddings     # from CLIP
V = W_v · text_embeddings     # from CLIP
output = softmax(QK^T / √d) · V
```

This cross-attention mechanism allows every spatial position to attend to relevant text tokens, enabling precise text-to-image alignment.

## Skip Connections

Same as segmentation U-Net: encoder features at each level are concatenated with decoder features. In diffusion U-Net, the decoder has 3 ResBlocks per level (vs 2 in encoder) to account for the additional channels from concatenation. FreeU showed that re-weighting these skip connections can improve generation quality.
