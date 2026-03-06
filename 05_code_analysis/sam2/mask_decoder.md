---
title: "SAM 2 - Mask Decoder Analysis"
date: 2025-01-15
status: planned
parent: "sam2/repo_overview.md"
tags: [sam2, mask-decoder, transformer, segmentation-head]
---

# SAM 2 Mask Decoder

## Overview

TODO: Analyze the mask decoder in `sam2/modeling/mask_decoder.py`

## Architecture

### Transformer Decoder Layers
TODO: Number of layers, attention heads, hidden dim

### Two-Way Attention
TODO: How tokens attend to image embeddings and vice versa

### Multi-Mask Output
TODO: How multiple mask candidates are generated

## Mask Prediction Head

TODO: MLP that produces mask logits from decoder output tokens

## IoU Prediction Head

TODO: How the model predicts mask quality scores

## Ambiguity Resolution

TODO: How the model handles ambiguous prompts with multiple valid masks

## Comparison with SAM 1 Decoder

TODO: Key differences from the original SAM mask decoder
