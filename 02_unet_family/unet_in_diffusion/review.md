---
title: "U-Net as Backbone in Denoising Diffusion Probabilistic Models"
date: 2025-03-06
status: planned
tags:
  - diffusion-models
  - denoising
  - generative
  - latent-diffusion
  - stable-diffusion
difficulty: advanced
---

# U-Net as Backbone in Denoising Diffusion Probabilistic Models

## Meta Information

| Field          | Details |
|----------------|---------|
| **Paper Title(s)** | Denoising Diffusion Probabilistic Models (DDPM); High-Resolution Image Synthesis with Latent Diffusion Models (LDM) |
| **Authors**       | Jonathan Ho, Ajay Jain, Pieter Abbeel (DDPM); Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, Bjorn Ommer (LDM) |
| **Year**          | 2020 (DDPM), 2022 (LDM/Stable Diffusion) |
| **Venue**         | NeurIPS 2020 (DDPM); CVPR 2022 (LDM) |
| **ArXiv ID**      | [2006.11239](https://arxiv.org/abs/2006.11239) (DDPM); [2112.10752](https://arxiv.org/abs/2112.10752) (LDM) |

## One-Line Summary

The U-Net architecture serves as the denoising backbone in diffusion models, predicting noise at each timestep through an encoder-decoder structure augmented with self-attention, cross-attention (for conditioning), and timestep embeddings -- far removed from its biomedical segmentation origins but leveraging the same multi-scale skip connection design.

---

## Motivation and Problem Statement

_TODO: Describe how diffusion models need a network that can predict noise at multiple scales and why the U-Net's multi-resolution design is naturally suited for this task._

---

## Key Contributions

- _TODO: U-Net as the epsilon-predictor in DDPM_
- _TODO: Addition of self-attention and cross-attention layers_
- _TODO: Timestep conditioning via sinusoidal embeddings_
- _TODO: Latent-space diffusion (LDM) for efficiency_

---

## Architecture Overview

_TODO: Describe the modified U-Net used in diffusion models. Reference [stable_diffusion_unet.md](./stable_diffusion_unet.md)._

---

## Method Details

### Differences from Segmentation U-Net

| Feature | Segmentation U-Net | Diffusion U-Net |
|---------|-------------------|-----------------|
| Input | Image | Noisy image/latent + timestep |
| Output | Segmentation mask | Predicted noise |
| Attention | None or simple gates | Self-attention + cross-attention |
| Conditioning | None | Text/class via cross-attention |
| Skip connections | Concatenation | Concatenation (same) |

### Timestep Embedding

_TODO: Sinusoidal positional encoding of the diffusion timestep, injected via FiLM or addition._

### Self-Attention and Cross-Attention

_TODO: Where attention layers are placed in the U-Net and what they attend to._

### Latent Diffusion

_TODO: Operating in the latent space of a VAE for computational efficiency._

---

## Experimental Results

_TODO: Image generation quality metrics (FID, IS) are outside standard segmentation metrics._

---

## Strengths

- _TODO_

---

## Weaknesses and Limitations

- _TODO_

---

## Connections to Other Work

| Related Paper | Relationship |
|---------------|-------------|
| U-Net (Ronneberger et al., 2015) | Original architecture |
| Attention U-Net (Oktay et al., 2018) | Attention in U-Net (segmentation context) |
| DDPM (Ho et al., 2020) | Introduced diffusion with U-Net backbone |
| Stable Diffusion (Rombach et al., 2022) | Latent diffusion with conditioned U-Net |
| DiT (Peebles & Xie, 2023) | Transformer replacing U-Net in diffusion |
| FreeU (Si et al., 2023) | Feature reweighting in diffusion U-Net |

---

## Open Questions

- _TODO: Will transformers (DiT) fully replace U-Net in diffusion models?_
