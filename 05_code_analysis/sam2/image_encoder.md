---
title: "SAM 2 - Image Encoder Analysis"
date: 2025-01-15
status: planned
parent: "sam2/repo_overview.md"
tags: [sam2, hiera, image-encoder, vision-transformer]
---

# SAM 2 Image Encoder

## Overview

TODO: Analyze the Hiera-based image encoder in `sam2/modeling/image_encoder.py`

## Hiera Architecture

### Hierarchical Vision Transformer
TODO: How Hiera produces multi-scale feature maps

### Windowed Attention
TODO: Analyze the windowed attention mechanism

### Feature Pyramid
TODO: What feature scales are produced and passed to the decoder

## Input Processing

TODO: Image preprocessing (resizing, normalization, padding)

## Output Format

TODO: Shape and structure of encoder outputs

## Model Variants

| Variant | Embedding Dim | Depth | Heads | Parameters |
|---------|--------------|-------|-------|-----------|
| Tiny | TODO | TODO | TODO | TODO |
| Small | TODO | TODO | TODO | TODO |
| Base+ | TODO | TODO | TODO | TODO |
| Large | TODO | TODO | TODO | TODO |

## Comparison with SAM 1 Encoder

TODO: Key differences from the original ViT-based SAM encoder
