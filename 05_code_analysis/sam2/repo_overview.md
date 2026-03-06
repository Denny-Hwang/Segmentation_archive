---
title: "SAM 2 Repository Overview"
date: 2025-01-15
status: planned
repo_url: "https://github.com/facebookresearch/sam2"
framework: PyTorch
tags: [sam2, segment-anything, foundation-model, pytorch, promptable]
---

# SAM 2 (facebookresearch/sam2)

## Repository Summary

| Field | Details |
|-------|---------|
| URL | https://github.com/facebookresearch/sam2 |
| License | Apache-2.0 |
| Framework | PyTorch |
| Primary Use Case | Promptable image and video segmentation |
| Key Strength | Foundation model -- zero-shot segmentation with points, boxes, or masks as prompts |

## Why This Repository

SAM 2 extends the Segment Anything Model to video, introducing a streaming memory architecture that tracks objects across frames. It represents the state of the art in promptable segmentation.

## Repository Structure

```
sam2/
├── sam2/
│   ├── modeling/
│   │   ├── sam2_base.py           # Base SAM2 model
│   │   ├── image_encoder.py       # Hiera-based image encoder
│   │   ├── memory_attention.py    # Memory attention module
│   │   ├── memory_encoder.py      # Memory encoder
│   │   ├── mask_decoder.py        # Mask decoder (from SAM)
│   │   ├── prompt_encoder.py      # Prompt encoder (from SAM)
│   │   └── position_encoding.py   # Positional encodings
│   ├── sam2_image_predictor.py    # Image prediction API
│   ├── sam2_video_predictor.py    # Video prediction API
│   └── automatic_mask_generator.py
├── configs/                       # Model configurations
├── checkpoints/                   # Model weights
└── notebooks/                     # Demo notebooks
```

## Key Architectural Components

- **Image Encoder**: Hiera vision transformer
- **Prompt Encoder**: Encodes points, boxes, and masks into embeddings
- **Mask Decoder**: Lightweight transformer that predicts masks from image and prompt embeddings
- **Memory Attention**: Cross-attention between current frame and memory bank
- **Memory Encoder**: Encodes past predictions into memory tokens

## Analysis Files

| File | Description | Status |
|------|-------------|--------|
| `image_encoder.md` | Hiera image encoder analysis | Planned |
| `memory_attention.md` | Memory attention mechanism for video | Planned |
| `mask_decoder.md` | Mask decoder architecture | Planned |
| `prompt_encoder.md` | Prompt encoding for points, boxes, masks | Planned |
