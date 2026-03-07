---
title: "SMP - Pretrained Weights System"
date: 2025-01-15
status: planned
parent: "segmentation_models_pytorch/repo_overview.md"
tags: [smp, pretrained, weights, transfer-learning]
---

# SMP Pretrained Weights System

## Overview

SMP manages pretrained encoder weights through a registry of URL-based weight configurations stored alongside each encoder definition. When a user specifies `encoder_weights="imagenet"`, the system downloads the corresponding checkpoint, loads the state dict, and adapts it to the current model configuration (e.g., different input channels). Only encoders have pretrained weights -- decoders and segmentation heads are always initialized from scratch.

## Weight Sources

| Source | Training Data | Models |
|--------|--------------|--------|
| ImageNet-1k | 1.2M images, 1000 classes | All ResNet, VGG, DenseNet, EfficientNet, DPN, SE-Net, MobileNet variants |
| ImageNet-21k | 14M images, 21k classes | Select timm models (ViT, BEiT, Swin) |
| Noisy Student | ImageNet + 300M unlabeled images (pseudo-labeled) | EfficientNet-b0 through b7, EfficientNet-L2 |
| AdvProp | ImageNet with adversarial examples | EfficientNet-b0 through b8 |
| Instagram | 940M public Instagram images | ResNeXt-101 32x8d, 32x16d, 32x48d (WSL: weakly supervised learning) |
| SwAV | ImageNet (self-supervised, no labels) | ResNet-50 |
| SSL/SWSL | Semi-supervised / semi-weakly supervised | ResNet-18/50, ResNeXt variants |

## Weight Loading Mechanism

### Download and Caching

Weights are downloaded via `torch.utils.model_zoo.load_url()`, which:
1. Checks `~/.cache/torch/hub/checkpoints/` for cached files
2. Downloads from the URL specified in the encoder's `pretrained_settings` dict if not cached
3. Stores the downloaded file with a hash-based filename for deduplication

```python
# In get_encoder():
settings = encoders[name]["pretrained_settings"][weights]
encoder.load_state_dict(model_zoo.load_url(settings["url"]))
```

For `timm`-based encoders, weight loading delegates to `timm.create_model(pretrained=True)` which uses its own download/caching mechanism (typically `~/.cache/torch/hub/checkpoints/` as well).

### State Dict Adaptation

When the model's state dict has minor mismatches with the pretrained checkpoint:
- **Missing keys** (e.g., new layers not in pretrained model): Initialized randomly
- **Extra keys** (e.g., classification head in pretrained but not in encoder-only model): Silently ignored via `strict=False` in some paths
- **Shape mismatches**: Handled explicitly for the first conv layer (see Input Channel Handling below)

The classification head (`fc`, `classifier`) from pretrained models is discarded since SMP only uses the encoder as a feature extractor.

### Input Channel Handling

When `in_channels != 3`, SMP modifies the first convolutional layer via `encoder.set_in_channels()`:

```python
def set_in_channels(self, in_channels, pretrained=True):
    if in_channels == 3:
        return
    # Get the first conv layer
    self._in_channels = in_channels
    if pretrained:
        # Average the pretrained 3-channel weights across channel dim
        # Then repeat/tile to match new in_channels
        weight = self._first_conv.weight.data
        mean_weight = weight.mean(dim=1, keepdim=True)  # Average over RGB
        new_weight = mean_weight.repeat(1, in_channels, 1, 1)
        # For channels <= 3, slice the pretrained weights instead
    else:
        # Reinitialize the first conv with new in_channels
        self._first_conv = nn.Conv2d(in_channels, ...)
```

This strategy preserves pretrained knowledge for non-RGB inputs:
- **1-channel (grayscale)**: Averages RGB weights into a single channel weight
- **4-channel (RGBD)**: Copies RGB weights for first 3 channels, averages for the 4th
- **>3 channels**: Tiles the averaged weight across all input channels

## Custom Pretrained Weights

To use custom pretrained weights:

```python
# Method 1: Load state dict manually after model creation
model = smp.Unet(encoder_name="resnet34", encoder_weights=None)  # No pretrained
model.encoder.load_state_dict(torch.load("my_custom_encoder.pth"))

# Method 2: Register custom weights in the encoder registry
from segmentation_models_pytorch.encoders import encoders
encoders["resnet34"]["pretrained_settings"]["my_dataset"] = {
    "url": "file:///path/to/weights.pth",  # or http URL
    "mean": [0.5, 0.5, 0.5],
    "std": [0.5, 0.5, 0.5],
    "input_range": [0, 1],
    "num_classes": 10,
}
model = smp.Unet(encoder_name="resnet34", encoder_weights="my_dataset")

# Method 3: Use timm universal encoder for any timm-compatible checkpoint
model = smp.Unet(encoder_name="tu-resnet34")  # timm-universal prefix
```

## Initialization for Non-Pretrained Parts

When only the encoder is pretrained, the remaining components are initialized as follows:

**Decoder**: Uses default PyTorch initialization:
- `Conv2d` layers: Kaiming uniform (He initialization)
- `BatchNorm2d` layers: weight=1, bias=0
- Some decoders apply Xavier initialization via `initialize_decoder()`:

```python
def initialize_decoder(module):
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
```

**SegmentationHead**: Standard Kaiming initialization for the final convolution. Bias initialized to zero.

**ClassificationHead**: Linear layer uses default PyTorch initialization (Kaiming uniform).

This asymmetric initialization (pretrained encoder + random decoder) is standard practice and works well because:
1. The encoder provides strong feature representations from day one
2. The decoder learns to combine these features, which converges quickly
3. Using differential learning rates (lower for encoder, higher for decoder) can further improve results
