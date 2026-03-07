---
title: "CLIP Backbone in OMG-Seg"
date: 2025-03-06
status: complete
tags: [clip, vision-language, backbone, feature-extraction]
difficulty: advanced
---

# CLIP Backbone

## Overview

OMG-Seg uses CLIP (Contrastive Language-Image Pretraining) ViT-L/14 as its image encoder backbone. This choice is central to the model's ability to handle open-vocabulary segmentation and benefit from vision-language pretraining. This document explains how CLIP features are extracted, adapted for dense prediction, and how the vision-language alignment enables open-vocabulary capabilities.

## CLIP Background

### What Is CLIP

CLIP is a vision-language model trained by OpenAI on 400 million image-text pairs scraped from the internet. It learns a shared embedding space where images and text descriptions of those images are close together:

- **Image encoder:** ViT-L/14 (or ResNet variants) maps images to a 768-dimensional embedding
- **Text encoder:** Transformer maps text strings to the same 768-dimensional space
- **Training objective:** Contrastive loss that maximizes cosine similarity between matched image-text pairs and minimizes it for mismatched pairs

### Why CLIP for Segmentation

Traditional segmentation backbones (ResNet, Swin) are pretrained on ImageNet classification, which provides visual features but no language alignment. CLIP provides:

1. **Richer visual features:** Trained on 400M diverse image-text pairs vs. 1.3M ImageNet images
2. **Language-aligned representations:** Visual features naturally correspond to textual descriptions
3. **Open-vocabulary potential:** New categories can be specified via text without retraining
4. **Transfer strength:** CLIP features transfer well to diverse downstream tasks

## Multi-Scale Feature Extraction

### The Challenge

Standard CLIP extracts a single global image embedding (the CLS token after the final layer). This is insufficient for segmentation, which requires dense spatial predictions at pixel-level resolution.

### OMG-Seg's Approach

OMG-Seg extracts features from multiple intermediate layers of the CLIP ViT:

| Feature Level | ViT Layer | Resolution | Semantic Level |
|--------------|-----------|------------|----------------|
| F1 | Layer 6 | 1/14 | Low-level (edges, textures) |
| F2 | Layer 12 | 1/14 | Mid-level (parts, patterns) |
| F3 | Layer 18 | 1/14 | High-level (objects) |
| F4 | Layer 24 (final) | 1/14 | Semantic (categories) |

All feature maps have the same spatial resolution (1/14 of input) because ViT uses a fixed patch size, but they capture information at different semantic levels.

### Feature Pyramid Construction

The extracted multi-scale features are processed into a feature pyramid:

1. Each feature map is projected to a common channel dimension via 1x1 convolution
2. Features are progressively upsampled using transposed convolutions or bilinear interpolation
3. Adjacent scales are fused via element-wise addition
4. The result is a pyramid with features at 1/4, 1/8, 1/16, and 1/32 resolution

This is similar to an FPN (Feature Pyramid Network) but adapted for ViT's uniform-resolution intermediate features.

## Vision-Language Alignment for Segmentation

### How Open-Vocabulary Segmentation Works

The key insight is that CLIP's shared embedding space can be exploited for zero-shot category recognition:

1. **Region feature extraction:** For each predicted mask region, pool the pixel features to get a region embedding
2. **Text encoding:** Encode category names as text using CLIP's text encoder (e.g., "a photo of a {category}")
3. **Matching:** Compute cosine similarity between each region embedding and each text embedding
4. **Classification:** Assign each region to the category with the highest similarity

This process requires no training on the target categories. New categories are added simply by providing their text descriptions.

### Prompt Templates for Text Encoding

The text fed to CLIP's text encoder significantly affects classification accuracy. OMG-Seg uses ensemble prompt templates:

```
"a photo of a {category}"
"a photo of a {category} in the scene"
"there is a {category} in the scene"
"a photo of the {category}"
```

Text embeddings from multiple templates are averaged to produce more robust category representations.

### Maintaining CLIP Alignment During Fine-Tuning

A critical challenge is that fine-tuning the CLIP backbone on segmentation data can destroy the learned vision-language alignment. OMG-Seg addresses this through:

- **Frozen CLIP layers:** The deepest CLIP layers (closest to the original embedding space) are kept frozen
- **Lightweight adaptation:** Only added projection layers and the pixel decoder are fully trained
- **Feature distillation:** A regularization loss encourages adapted features to remain close to the original CLIP features
- **Gradual unfreezing:** If the backbone is fine-tuned, earlier layers are unfrozen first with a very small learning rate

## Feature Properties

### Comparison to ImageNet-Pretrained Backbones

| Property | ImageNet ViT | CLIP ViT |
|----------|-------------|----------|
| Training data | 1.3M images, 1K classes | 400M image-text pairs |
| Supervision | Class labels | Natural language |
| Vocabulary | Fixed 1,000 classes | Open (any text) |
| Feature transfer | Good for similar domains | Strong across diverse domains |
| Language alignment | None | Built-in |
| Segmentation quality | High (with fine-tuning) | High (with or without fine-tuning) |

### What CLIP Features Capture

Analysis of CLIP ViT features in the context of segmentation reveals:
- **Early layers (1-8):** Texture and edge information, similar to ImageNet-pretrained models
- **Middle layers (9-16):** Part-level grouping that corresponds to semantically meaningful regions
- **Late layers (17-24):** Category-level representations aligned with language descriptions
- **Attention maps:** CLIP's self-attention naturally segments salient objects even without segmentation training

## Practical Considerations

### Computational Cost

CLIP ViT-L/14 is a large backbone:

| Property | Value |
|----------|-------|
| Parameters | ~430M |
| FLOPs (224x224 input) | ~61 GFLOPs |
| FLOPs (1024x1024 input) | ~1.2 TFLOPs |
| Inference time (A100, 1024x1024) | ~120ms |

For high-resolution segmentation, the CLIP backbone is the computational bottleneck.

### Resolution Handling

CLIP ViT-L/14 was trained at 224x224 resolution. Segmentation requires higher resolution (512-1024). This is handled by:
- Interpolating CLIP's positional embeddings to the target resolution
- Fine-tuning with the higher resolution to adapt the position encoding
- Using a windowed attention variant for very high resolutions

### Alternatives to CLIP ViT-L

| Backbone | Open-Vocab | Speed | Segmentation Quality |
|----------|-----------|-------|---------------------|
| CLIP ViT-B/16 | Yes | 3x faster | 2-3 mIoU lower |
| CLIP ViT-L/14 | Yes | Baseline | Baseline |
| EVA-CLIP ViT-G | Yes | 2x slower | 1-2 mIoU higher |
| Swin-L (ImageNet) | No | Similar | Similar (closed-vocab) |

The choice depends on whether open-vocabulary capability is needed and the available computational budget.

## Impact on Segmentation Architecture Design

CLIP backbones have become standard for open-vocabulary segmentation models. The key lesson from OMG-Seg and related work is that vision-language pretraining provides a strictly richer feature space than vision-only pretraining: it matches or exceeds closed-vocabulary performance while additionally enabling open-vocabulary generalization.
