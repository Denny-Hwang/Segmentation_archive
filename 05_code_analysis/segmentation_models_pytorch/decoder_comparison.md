---
title: "SMP - Decoder Comparison"
date: 2025-01-15
status: planned
parent: "segmentation_models_pytorch/repo_overview.md"
tags: [smp, decoder, unet, fpn, deeplabv3, comparison]
---

# SMP Decoder Comparison

## Available Decoders

| Decoder | Paper | Skip Connections | Multi-Scale | Notes |
|---------|-------|-----------------|-------------|-------|
| U-Net | Ronneberger 2015 | Yes (concat) | Progressive upsampling | Standard baseline, most widely used |
| U-Net++ | Zhou 2018 | Yes (nested) | Dense nested connections | Re-designed skip paths with dense blocks at each level |
| FPN | Lin 2017 | Yes (add) | Top-down + lateral | Lightweight; originally for detection, adapted for segmentation |
| DeepLabV3 | Chen 2017 | No | ASPP (atrous spatial pyramid pooling) | Multi-scale context via dilated convolutions |
| DeepLabV3+ | Chen 2018 | Partial (low-level) | ASPP + low-level skip | Adds one skip connection from early encoder stage |
| PAN | Li 2018 | Yes (attention) | Feature Pyramid Attention | Attention-weighted skip connections |
| PSPNet | Zhao 2017 | No | PPM (Pyramid Pooling Module) | Global context via multi-scale pooling |
| Linknet | Chaurasia 2017 | Yes (add) | Addition-based skip | Fast; uses addition instead of concat to save memory |
| MAnet | Fan 2020 | Yes (attention) | Multi-scale attention | Position + channel attention on skip connections |

## Decoder Interface

All decoders inherit from `SegmentationModel` (defined in `base/model.py`), which enforces a standard structure:

```python
class SegmentationModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = ...           # Backbone feature extractor
        self.decoder = ...           # Specific decoder implementation
        self.segmentation_head = ... # Final conv to n_classes
        self.classification_head = ... # Optional auxiliary head

    def forward(self, x):
        features = self.encoder(x)
        decoder_output = self.decoder(*features)
        masks = self.segmentation_head(decoder_output)
        if self.classification_head is not None:
            labels = self.classification_head(features[-1])
            return masks, labels
        return masks
```

Every decoder must implement:
- `__init__(self, encoder_channels, decoder_channels, n_blocks, ...)` -- accepts encoder output channels to build compatible layers
- `forward(self, *features)` -- takes the list of encoder feature maps and returns a single feature map

The decoder receives features as `*features` (unpacked from the encoder's list), where `features[0]` is the original image and `features[-1]` is the deepest (most downsampled) feature map.

## Feature Aggregation Strategies

### Concatenation-Based (U-Net)

The U-Net decoder (`decoders/unet/decoder.py`) uses `DecoderBlock` which concatenates skip connections:

```python
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, ...):
        self.conv1 = Conv2dReLU(in_channels + skip_channels, out_channels, ...)
        self.conv2 = Conv2dReLU(out_channels, out_channels, ...)
        self.attention = attention_block(...)  # Optional attention (scse, etc.)

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        if skip is not None:
            x = torch.cat([x, skip], dim=1)  # Channel concatenation
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention(x)  # Optional squeeze-excitation
        return x
```

Concatenation doubles the channel count before the first convolution, preserving all information from both paths but at higher computational cost.

### Addition-Based (FPN, LinkNet)

**FPN** (`decoders/fpn/decoder.py`) uses lateral connections with element-wise addition:

```python
class FPNBlock(nn.Module):
    def __init__(self, pyramid_channels, skip_channels):
        self.skip_conv = nn.Conv2d(skip_channels, pyramid_channels, kernel_size=1)

    def forward(self, x, skip):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        skip = self.skip_conv(skip)  # Project to same channels via 1x1 conv
        x = x + skip                # Element-wise addition
        return x
```

**LinkNet** (`decoders/linknet/decoder.py`) similarly uses addition but with a transposed convolution for upsampling:

```python
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, ...):
        self.block = nn.Sequential(
            Conv2dReLU(in_channels, in_channels // 4, kernel_size=1),
            nn.ConvTranspose2d(in_channels // 4, in_channels // 4, kernel_size=3,
                               stride=2, padding=1, output_padding=1),
            Conv2dReLU(in_channels // 4, out_channels, kernel_size=1),
        )

    def forward(self, x, skip=None):
        x = self.block(x)
        if skip is not None:
            x = x + skip  # Addition
        return x
```

Addition is cheaper than concatenation and preserves spatial information, but can lose fine details when encoder and decoder features have very different characteristics.

### Attention-Based (MAnet)

The MAnet decoder (`decoders/manet/decoder.py`) applies dual attention (position + channel) to skip connections before merging:

```python
class PAB(nn.Module):  # Position Attention Block
    """Computes spatial attention map via query-key-value mechanism."""
    def forward(self, x):
        # Q, K, V projections via 1x1 conv
        # Attention = softmax(Q^T * K) * V
        # Output = gamma * attention + x (residual)

class MFAB(nn.Module):  # Multi-scale Feature Aggregation Block
    def forward(self, x, skip):
        x = F.interpolate(x, scale_factor=2)
        x = torch.cat([x, skip], dim=1)
        x = self.pab(x)  # Position attention
        x = self.cab(x)  # Channel attention
        return x
```

This allows the decoder to selectively weight which spatial locations and channels from the skip connection are most relevant, rather than treating all features equally.

## Output Head

### SegmentationHead

```python
class SegmentationHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 \
                     else nn.Identity()
        activation = nn.Identity()  # Raw logits by default
        super().__init__(conv2d, upsampling, activation)
```

- Applies a final convolution (typically 3x3) to map decoder features to `n_classes` channels
- Optional upsampling factor if the decoder output is smaller than the desired mask size
- Activation is `nn.Identity()` by default (logits), but can be set to sigmoid/softmax via the `activation` parameter

### ClassificationHead

```python
class ClassificationHead(nn.Sequential):
    def __init__(self, in_channels, classes, pooling="avg", dropout=0.2):
        pool = nn.AdaptiveAvgPool2d(1)  # Global average pooling
        flatten = nn.Flatten()
        dropout = nn.Dropout(p=dropout)
        linear = nn.Linear(in_channels, classes)
        super().__init__(pool, flatten, dropout, linear)
```

Optional auxiliary head that classifies the entire image, useful for multi-task learning. Takes the deepest encoder features and applies global pooling + linear projection.

## Performance Comparison

Approximate decoder parameter counts with a ResNet-34 encoder (encoder ~21.3M params):

| Decoder | Decoder Params | Total Params | Relative Speed | Best Use Case |
|---------|---------------|-------------|----------------|---------------|
| U-Net | ~8.6M | ~30M | 1.0x (baseline) | General purpose, medical imaging |
| U-Net++ | ~15.5M | ~37M | 0.6x | When nested features help (small objects) |
| FPN | ~0.6M | ~22M | 1.8x | Real-time, detection-style tasks |
| DeepLabV3 | ~15.4M | ~37M | 0.7x | Large context, scene parsing |
| DeepLabV3+ | ~5.1M | ~26M | 1.1x | Balanced accuracy/speed |
| PAN | ~0.5M | ~22M | 1.7x | Lightweight attention segmentation |
| PSPNet | ~5.1M | ~26M | 1.0x | Scene parsing, global context |
| LinkNet | ~0.6M | ~22M | 2.0x | Real-time segmentation |
| MAnet | ~12.4M | ~34M | 0.5x | High accuracy with attention |

Key tradeoffs:
- **FPN and LinkNet** are fastest with minimal decoder overhead, ideal for real-time applications
- **U-Net++ and MAnet** have the most decoder parameters, trading speed for accuracy
- **DeepLabV3** is expensive but captures global context via dilated convolutions
- All decoders benefit from pretrained encoders; the decoder is always trained from scratch
