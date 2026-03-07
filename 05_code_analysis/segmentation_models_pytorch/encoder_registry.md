---
title: "SMP - Encoder Registry Analysis"
date: 2025-01-15
status: planned
parent: "segmentation_models_pytorch/repo_overview.md"
tags: [smp, encoder, registry, backbone, pretrained]
---

# SMP Encoder Registry

## Overview

The encoder registry in `segmentation_models_pytorch/encoders/__init__.py` implements a dictionary-based registry pattern that maps string names to encoder classes and their pretrained weight configurations. This is the backbone of SMP's modular architecture -- any encoder can be paired with any decoder through a standardized interface.

The central data structure is the `encoders` dict, populated by importing encoder-specific modules:

```python
from .resnet import resnet_encoders
from .dpn import dpn_encoders
from .vgg import vgg_encoders
# ... etc

encoders = {}
encoders.update(resnet_encoders)
encoders.update(dpn_encoders)
encoders.update(vgg_encoders)
# ...
```

## Registry Architecture

### Registration Mechanism

Each encoder family defines its own dictionary mapping encoder names to configuration dicts. For example, in `encoders/resnet.py`:

```python
resnet_encoders = {
    "resnet18": {
        "encoder": ResNetEncoder,
        "pretrained_settings": {
            "imagenet": {
                "url": "https://download.pytorch.org/models/resnet18-...",
                "input_space": "RGB",
                "input_range": [0, 1],
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
                "num_classes": 1000,
            }
        },
        "params": {
            "out_channels": (3, 64, 64, 128, 256, 512),
            "block": BasicBlock,
            "layers": [2, 2, 2, 2],
        },
    },
    # resnet34, resnet50, resnet101, resnet152...
}
```

The `get_encoder()` function retrieves and instantiates an encoder:

```python
def get_encoder(name, in_channels=3, depth=5, weights=None, output_stride=32):
    Encoder = encoders[name]["encoder"]
    params = encoders[name]["params"]
    encoder = Encoder(**params)
    if weights is not None:
        settings = encoders[name]["pretrained_settings"][weights]
        encoder.load_state_dict(model_zoo.load_url(settings["url"]))
    encoder.set_in_channels(in_channels)
    return encoder
```

### Encoder Interface Contract

Every encoder must inherit from `_base.EncoderMixin` and expose:

```python
class EncoderMixin:
    @property
    def out_channels(self):
        """List of channel counts at each feature level, including the stem."""
        return self._out_channels

    def set_in_channels(self, in_channels, pretrained=True):
        """Modify the first conv layer to accept in_channels != 3."""

    @property
    def output_stride(self):
        """Total downsampling factor (e.g., 32 for standard ResNet)."""
        return min(self._output_stride, 2 ** self._depth)
```

The critical contract is `out_channels` -- a tuple like `(3, 64, 64, 128, 256, 512)` where:
- Index 0: input image channels (identity skip)
- Index 1: stem/initial conv output
- Indices 2-5: feature maps at 1/4, 1/8, 1/16, 1/32 spatial resolution

The `forward()` method must return a **list of feature maps** at these scales.

### Feature Map Extraction

Encoders override `forward()` to collect intermediate feature maps:

```python
def forward(self, x):
    features = []
    features.append(x)              # Stage 0: original input
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    features.append(x)              # Stage 1: after stem
    x = self.maxpool(x)
    x = self.layer1(x)
    features.append(x)              # Stage 2: 1/4 resolution
    x = self.layer2(x)
    features.append(x)              # Stage 3: 1/8 resolution
    x = self.layer3(x)
    features.append(x)              # Stage 4: 1/16 resolution
    x = self.layer4(x)
    features.append(x)              # Stage 5: 1/32 resolution
    return features
```

The `depth` parameter controls how many stages are computed -- setting `depth=3` would skip `layer3` and `layer4`.

## Supported Encoder Families

| Family | Example Models | Source |
|--------|---------------|--------|
| ResNet | resnet18, resnet34, resnet50, resnet101, resnet152 | `torchvision.models.resnet` adapted |
| ResNeXt | resnext50_32x4d, resnext101_32x8d | `torchvision` adapted |
| DPN | dpn68, dpn92, dpn98, dpn107, dpn131 | Custom implementation |
| VGG | vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn | `torchvision.models.vgg` adapted |
| SE-Net | senet154, se_resnet50, se_resnext50_32x4d | `pretrainedmodels` package |
| DenseNet | densenet121, densenet161, densenet169, densenet201 | `torchvision.models.densenet` adapted |
| Inception | inceptionv4, inceptionresnetv2 | `pretrainedmodels` package |
| EfficientNet | efficientnet-b0 through efficientnet-b7 | Custom or `timm`-based |
| MobileNet | mobilenet_v2, timm-mobilenetv3_large | `torchvision` / `timm` |
| timm-universal | Any `timm` model via `TimmUniversalEncoder` | `timm` library wrapper |
| MixNet | timm-mixnet_s, timm-mixnet_m, timm-mixnet_l | `timm` wrapper |

## Pretrained Weight Loading

Weights are loaded via `torch.utils.model_zoo.load_url()` which downloads to `~/.cache/torch/hub/checkpoints/`. Each encoder entry specifies URLs and preprocessing parameters:

```python
pretrained_settings = {
    "imagenet": {
        "url": "https://...",
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "input_range": [0, 1],
    },
    "advprop": { ... },  # Some encoders have multiple pretrained sources
}
```

The preprocessing metadata (`mean`, `std`, `input_range`) is stored on the encoder instance and accessible via `encoder.pretrained_settings`, allowing data pipelines to use matching normalization.

## Adding a Custom Encoder

1. **Create the encoder module** inheriting from both the backbone and `EncoderMixin`:

```python
# segmentation_models_pytorch/encoders/my_encoder.py
from ._base import EncoderMixin

class MyEncoder(torch.nn.Module, EncoderMixin):
    def __init__(self, out_channels, depth=5, **kwargs):
        super().__init__()
        self._out_channels = out_channels
        self._depth = depth
        self._output_stride = 32
        # Define layers...

    def forward(self, x):
        features = []
        # Collect multi-scale features...
        return features

    def load_state_dict(self, state_dict, **kwargs):
        # Handle any key remapping if needed
        super().load_state_dict(state_dict, **kwargs)
```

2. **Define the registry entry**:

```python
my_encoders = {
    "my_encoder_small": {
        "encoder": MyEncoder,
        "pretrained_settings": {},
        "params": {"out_channels": (3, 32, 64, 128, 256, 512), ...},
    }
}
```

3. **Register in `encoders/__init__.py`**:

```python
from .my_encoder import my_encoders
encoders.update(my_encoders)
```

4. **Use**: `model = smp.Unet(encoder_name="my_encoder_small")`

## Key Code Paths

Tracing `smp.Unet(encoder_name="resnet34", encoder_weights="imagenet")`:

```
smp.Unet.__init__()
  │
  ├── get_encoder("resnet34", in_channels=3, depth=5, weights="imagenet")
  │     ├── encoders["resnet34"]["encoder"]  ->  ResNetEncoder class
  │     ├── ResNetEncoder(**params)  ->  instantiate with layers=[3,4,6,3], block=BasicBlock
  │     ├── load_state_dict(model_zoo.load_url(url))  ->  download & load ImageNet weights
  │     └── encoder.set_in_channels(3)  ->  adapt first conv (no-op for 3 channels)
  │
  ├── UnetDecoder(
  │     encoder_channels=(3, 64, 64, 128, 256, 512),
  │     decoder_channels=(256, 128, 64, 32, 16),
  │     n_blocks=5
  │   )
  │
  ├── SegmentationHead(in_channels=16, out_channels=n_classes, kernel_size=3)
  │
  └── ClassificationHead (optional, for auxiliary classification)
```

The forward pass:
```python
def forward(self, x):
    features = self.encoder(x)          # Returns list of 6 feature maps
    decoder_output = self.decoder(*features)  # Progressively upsamples
    masks = self.segmentation_head(decoder_output)  # Final conv
    return masks
```
