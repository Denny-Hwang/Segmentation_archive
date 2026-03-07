---
title: "SAM 2 - Image Encoder Analysis"
date: 2025-01-15
status: complete
parent: "sam2/repo_overview.md"
tags: [sam2, hiera, image-encoder, vision-transformer]
---

# SAM 2 Image Encoder

## Overview

The image encoder in SAM 2 is built on **Hiera**, a hierarchical vision transformer that replaces the plain ViT (Vision Transformer) used in the original SAM. The encoder is defined in `sam2/modeling/backbones/hieradet.py` (Hiera with detection-oriented modifications) and wrapped by `ImageEncoder` in `sam2/modeling/backbones/image_encoder.py`. The key architectural change from SAM 1 is the shift from a single-scale ViT output to a multi-scale feature pyramid, which provides features at multiple resolutions for the mask decoder. This multi-scale design is critical for SAM 2's improved performance on objects of varying sizes and its ability to process video frames efficiently.

The `ImageEncoder` class composes two sub-modules: the Hiera backbone (called `trunk`) and an `FpnNeck` module that processes the backbone's multi-scale outputs into a standardized feature pyramid. The trunk produces features at 4 hierarchical scales (1/4, 1/8, 1/16, 1/32 of input resolution), and the FpnNeck refines and aligns these features for consumption by the mask decoder and memory attention modules.

## Hiera Architecture

### Hierarchical Vision Transformer

Hiera (Hierarchical ViT) produces multi-scale feature maps by progressively reducing spatial resolution and increasing channel dimension across stages, similar to how CNNs like ResNet work but using transformer blocks instead of convolutions. The architecture divides the transformer layers into stages, with spatial downsampling between stages achieved through patch merging (strided attention or pooling operations):

```python
# Simplified Hiera structure (from hieradet.py)
class Hiera(nn.Module):
    def __init__(self, embed_dim=96, num_heads=1, stages=(2, 3, 16, 3), ...):
        # Patch embedding: image -> tokens at 1/4 resolution
        self.patch_embed = PatchEmbed(kernel_size=7, stride=4, embed_dim=embed_dim)

        # Stage 0: embed_dim channels, 1/4 resolution
        # Stage 1: embed_dim*2 channels, 1/8 resolution
        # Stage 2: embed_dim*4 channels, 1/16 resolution
        # Stage 3: embed_dim*8 channels, 1/32 resolution
```

Each stage consists of multiple transformer blocks. Between stages, a spatial downsampling operation (implemented via strided pooling in the attention mechanism) halves the spatial dimensions while doubling the channel count. This produces a natural feature pyramid without the need for a separate FPN module. The `Hiera` class tracks which layers produce the multi-scale outputs via `self._out_feature_strides` and `self._out_feature_channels`.

### Windowed Attention

Hiera uses windowed (local) attention within each stage to reduce the quadratic computational cost of global self-attention. Instead of computing attention across all spatial tokens, attention is restricted to local windows of fixed size (typically 8x8 tokens):

```python
# Windowed attention (simplified)
class HieraBlock(nn.Module):
    def forward(self, x):
        # Partition tokens into windows
        x = window_partition(x, window_size=self.window_size)  # [B*num_windows, win_h*win_w, C]
        # Apply self-attention within each window
        x = self.attn(x)
        # Merge windows back
        x = window_unpartition(x, original_shape)
        return x
```

Periodically (every few blocks), global attention is applied to allow information flow across windows. In the Hiera-Det variant used by SAM 2, the last block of each stage uses global attention while earlier blocks use windowed attention. This combination achieves a good trade-off between computational efficiency (windowed attention is O(n) in image size) and representational power (global attention captures long-range dependencies). The window size decreases in later stages as the spatial resolution decreases, ensuring that each window covers a consistent receptive field in pixel space.

### Feature Pyramid

The FpnNeck module (in `sam2/modeling/backbones/image_encoder.py`) takes the multi-scale features from Hiera and produces a refined feature pyramid. It applies 1x1 convolutions to align channel dimensions, followed by top-down lateral connections with element-wise addition:

```python
class FpnNeck(nn.Module):
    def __init__(self, d_model, backbone_channel_list, ...):
        self.convs = nn.ModuleList()
        for dim in backbone_channel_list:
            self.convs.append(nn.Conv2d(dim, d_model, kernel_size=1))
        # Position encoding for each scale
        self.position_encoding = PositionEmbeddingSine(d_model // 2)

    def forward(self, xs):
        # xs: list of multi-scale features from Hiera
        out = [self.convs[i](xs[i]) for i in range(len(xs))]
        # Top-down pathway (FPN-style)
        for i in range(len(out) - 1, 0, -1):
            out[i-1] = out[i-1] + F.interpolate(out[i], scale_factor=2)
        # Add positional encoding to each level
        return [(feat, self.position_encoding(feat).to(feat.dtype)) for feat in out]
```

The FPN produces features at strides [4, 8, 16, 32] relative to the input image, all projected to the same channel dimension (`d_model`, typically 256). This uniform dimensionality simplifies downstream processing in the mask decoder and memory attention modules.

## Input Processing

Input images are resized to 1024x1024 pixels (the default resolution for SAM 2), then normalized using ImageNet statistics (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]). The patch embedding layer converts the 1024x1024 image into a grid of tokens at 1/4 resolution (256x256 tokens for stride-4 patches):

```python
# Input preprocessing (from sam2/utils/transforms.py)
def preprocess(image):
    # Resize longest side to 1024, pad to square
    image = resize_longest_side(image, target_length=1024)
    image = pad_to_square(image, pad_value=0)
    # Normalize with ImageNet stats
    image = (image - pixel_mean) / pixel_std
    return image  # Shape: [B, 3, 1024, 1024]
```

Unlike SAM 1 which used 16x16 non-overlapping patches, SAM 2's Hiera uses a convolutional patch embedding with kernel_size=7 and stride=4, producing overlapping patches that capture finer spatial detail from the start.

## Output Format

The image encoder produces a list of (feature, position_encoding) tuples, one per FPN level:

```python
# Output structure:
# [
#   (feat_stride4,  pos_stride4),   # [B, 256, 256, 256] - 1/4 resolution
#   (feat_stride8,  pos_stride8),   # [B, 256, 128, 128] - 1/8 resolution
#   (feat_stride16, pos_stride16),  # [B, 256, 64, 64]   - 1/16 resolution
#   (feat_stride32, pos_stride32),  # [B, 256, 32, 32]   - 1/32 resolution
# ]
```

The mask decoder typically uses the 1/16 resolution features as the primary input (same as SAM 1's ViT output resolution), while the other scales are used for multi-scale refinement. For video processing, these features are also fed into the memory encoder to create spatial memory representations.

## Model Variants

| Variant | Embedding Dim | Depth (stages) | Heads | Parameters | Notes |
|---------|--------------|----------------|-------|-----------|-------|
| Hiera-Tiny | 96 | 2+3+7+2 (14 total) | 1/2/4/8 | ~28M | Fastest, suitable for mobile/edge |
| Hiera-Small | 96 | 2+3+16+2 (23 total) | 1/2/4/8 | ~35M | Good speed/accuracy trade-off |
| Hiera-Base+ | 112 | 2+3+16+3 (24 total) | 2/4/8/16 | ~70M | Default for most applications |
| Hiera-Large | 144 | 2+6+36+4 (48 total) | 2/4/8/16 | ~214M | Highest accuracy, slowest |

The number of heads doubles at each stage, matching the doubling of channel dimensions. The Base+ variant is the recommended default, offering strong performance with manageable computational cost. The Large variant is used when maximum accuracy is needed and inference speed is less critical.

## Comparison with SAM 1 Encoder

The transition from SAM 1's ViT-based encoder to SAM 2's Hiera-based encoder represents a fundamental architectural shift:

SAM 1 used a plain ViT (ViT-H with 632M parameters) that processed 16x16 non-overlapping patches through 32 identical transformer blocks, producing a single-scale output at 1/16 resolution (64x64 tokens for 1024x1024 input). This single-scale output limited the decoder's ability to capture fine details and required the decoder to perform all upsampling from 1/16 to full resolution.

SAM 2's Hiera produces multi-scale features at 4 resolutions (1/4, 1/8, 1/16, 1/32), giving the decoder access to both fine-grained and coarse features. Hiera is also significantly more efficient: the Base+ variant has only ~70M parameters (vs ViT-H's 632M) while achieving better segmentation quality, thanks to the hierarchical design and windowed attention. The efficiency gain is critical for SAM 2's video processing capability, where the encoder must process many frames sequentially. Additionally, Hiera was pretrained with MAE (Masked Autoencoder) self-supervised learning, which provides strong visual representations without requiring labeled data for the pretraining stage.
