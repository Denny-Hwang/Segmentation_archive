---
title: "SAM 2 - Image Encoder Analysis"
date: 2025-01-15
status: planned
parent: "sam2/repo_overview.md"
tags: [sam2, hiera, image-encoder, vision-transformer]
---

# SAM 2 Image Encoder

## Overview

The SAM 2 image encoder (defined in `sam2/modeling/backbones/image_encoder.py` and `sam2/modeling/backbones/hieradet.py`) replaces SAM 1's plain ViT encoder with **Hiera** (Hierarchical Vision Transformer). Hiera produces multi-scale feature maps natively, eliminating the need for a separate neck/FPN module. The encoder processes images (or video frames) independently to produce feature embeddings that are consumed by the memory attention module and mask decoder.

## Hiera Architecture

### Hierarchical Vision Transformer

Hiera operates as a hierarchical transformer that progressively reduces spatial resolution while increasing channel depth, similar to how CNNs like ResNet work but using attention:

```python
class Hiera(nn.Module):
    def __init__(self, embed_dim=96, num_heads=1, stages=(2, 3, 16, 3), ...):
        # Patch embedding: image -> tokens
        self.patch_embed = PatchEmbed(
            kernel_size=(7, 7), stride=(4, 4), padding=(3, 3),
            in_chans=3, embed_dim=embed_dim
        )
        # Build stages with increasing channels and decreasing resolution
        # Stage 1: embed_dim,     resolution H/4 x W/4
        # Stage 2: embed_dim*2,   resolution H/8 x W/8
        # Stage 3: embed_dim*4,   resolution H/16 x W/16
        # Stage 4: embed_dim*8,   resolution H/32 x W/32
```

Between stages, **stride-2 pooling attention** reduces spatial dimensions by 2x while doubling channels. Within each stage, tokens maintain the same spatial resolution.

The key innovation over plain ViT: Hiera is **natively hierarchical** rather than operating at a single scale. This makes it more efficient (fewer tokens at deeper stages) and naturally produces multi-scale features.

### Windowed Attention

To reduce the quadratic cost of global self-attention, Hiera uses **windowed attention** for most transformer blocks:

```python
class HieraBlock(nn.Module):
    def forward(self, x):
        # Partition tokens into non-overlapping windows
        x = window_partition(x, window_size=8)  # Default 8x8 windows
        # Self-attention within each window (local attention)
        x = self.attn(x)  # O(window_size^2) per window instead of O(H*W)
        # Reverse partitioning
        x = window_unpartition(x)
        # FFN
        x = x + self.mlp(self.norm2(x))
        return x
```

Every few blocks (configurable), a **global attention** block is inserted that attends across all tokens, allowing information to flow between windows. This hybrid approach balances efficiency with global receptive field:
- Windowed blocks: O(n * w^2) where w=window_size, n=num_windows
- Global blocks: O((H*W)^2) but applied infrequently

### Feature Pyramid

Hiera outputs feature maps at multiple scales, which are processed by an FPN neck (`FpnNeck`):

```python
class FpnNeck(nn.Module):
    """Feature Pyramid Network neck for multi-scale features."""
    def __init__(self, d_model, backbone_channel_list, ...):
        # Lateral convolutions to project each scale to d_model channels
        self.convs = nn.ModuleList([
            nn.Conv2d(ch, d_model, kernel_size=1) for ch in backbone_channel_list
        ])
        # Top-down pathway with upsampling and addition

    def forward(self, features):
        # features: list of feature maps from Hiera stages
        # Returns: multi-scale features all projected to d_model channels
```

The FPN neck produces features at multiple scales (typically 1/4, 1/8, 1/16, 1/32 of input resolution), all with the same channel dimension (`d_model=256`).

## Input Processing

Images are preprocessed before encoding:

```python
# In sam2/utils/transforms.py
class SAM2Transforms:
    def __init__(self, resolution=1024, mask_threshold=0.0):
        self.resolution = resolution
        self.mean = [0.485, 0.456, 0.406]  # ImageNet mean
        self.std = [0.229, 0.224, 0.225]   # ImageNet std

    def __call__(self, image):
        # Resize longest side to self.resolution (1024)
        image = self.resize_longest_side(image)
        # Pad to square (1024x1024)
        image = self.pad_to_square(image)
        # Normalize with ImageNet statistics
        image = (image - self.mean) / self.std
        return image
```

- Input: RGB image of any size
- Resize: Longest side scaled to 1024 pixels (preserving aspect ratio)
- Padding: Zero-padded to 1024x1024 square
- Normalization: ImageNet mean/std (inherited from pretrained backbone)
- Output to encoder: `(B, 3, 1024, 1024)` float32 tensor

## Output Format

The image encoder produces:

```python
# After FpnNeck processing:
vision_features: Tensor  # Shape: (B, d_model, H/16, W/16) = (B, 256, 64, 64)
vision_pos_enc: List[Tensor]  # Positional encodings at each scale
backbone_fpn: List[Tensor]   # Multi-scale features: [(B,256,256,256), (B,256,128,128),
                              #                        (B,256,64,64), (B,256,32,32)]
```

The primary output (`vision_features`) at 1/16 scale is the main feature map used by the mask decoder. The multi-scale features from the FPN are used for high-resolution mask prediction.

## Model Variants

| Variant | Embedding Dim | Stages (blocks) | Heads | Parameters | Input Size |
|---------|--------------|-----------------|-------|-----------|------------|
| Tiny | 96 | (1, 2, 7, 2) | (1, 2, 4, 8) | ~28M | 1024x1024 |
| Small | 96 | (1, 2, 11, 2) | (1, 2, 4, 8) | ~46M | 1024x1024 |
| Base+ | 112 | (2, 3, 16, 3) | (2, 4, 8, 16) | ~80M | 1024x1024 |
| Large | 144 | (2, 6, 36, 4) | (2, 4, 8, 16) | ~213M | 1024x1024 |

All variants use:
- Patch size: 7x7 with stride 4 (initial 4x downsampling)
- Window size: 8x8 for windowed attention
- Global attention at select blocks within each stage
- FPN neck projecting to 256 channels

## Comparison with SAM 1 Encoder

| Aspect | SAM 1 (ViT) | SAM 2 (Hiera) |
|--------|-------------|---------------|
| Architecture | Plain ViT (ViT-B/L/H) | Hierarchical ViT (Hiera) |
| Feature scales | Single scale (1/16) | Multi-scale (1/4 to 1/32) |
| Attention | Global attention everywhere | Windowed + sparse global |
| Neck | Simple 2-layer neck | FPN neck with lateral connections |
| Parameters (Large) | ~307M (ViT-H) | ~213M (Hiera-L) |
| Speed | Slower (global attention) | Faster (windowed attention) |
| Pretraining | MAE on ImageNet | MAE on ImageNet, then SA-1B |
| Output dim | 256 | 256 |
| Video support | N/A (image only) | Frame-by-frame encoding for video |
| Positional encoding | Absolute learned | Absolute learned + window-relative |

The shift to Hiera brings three key advantages:
1. **Efficiency**: Windowed attention is much cheaper than global attention for high-resolution images
2. **Multi-scale features**: Native FPN-style outputs improve mask quality at object boundaries
3. **Smaller model**: Hiera-L has fewer parameters than ViT-H while achieving comparable or better performance
