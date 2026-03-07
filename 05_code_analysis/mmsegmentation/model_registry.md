---
title: "MMSegmentation - Model Registry Analysis"
date: 2025-01-15
status: complete
parent: "mmsegmentation/repo_overview.md"
tags: [mmsegmentation, registry, models, decorator-pattern]
---

# MMSegmentation Model Registry

## Overview

MMSegmentation uses MMEngine's registry pattern to manage all model components (backbones, decode heads, losses, segmentors). The registry acts as a global dictionary that maps string names to Python classes, enabling config-driven model construction. When a config dict contains `type='ResNetV1c'`, the registry looks up the class registered under that name and instantiates it with the remaining config fields as constructor arguments. This pattern decouples configuration from implementation, allowing new model components to be added with a single decorator without modifying any existing code.

The registry system is defined in `mmseg/registry.py` and builds on MMEngine's `Registry` class. Multiple registries exist for different component types: `MODELS` for all model components (backbones, heads, segmentors, losses), `DATASETS` for dataset classes, `TRANSFORMS` for data augmentation operations, and several others. Each registry maintains its own namespace, so a class name can be registered in multiple registries without conflict.

## Registry Architecture

### MODELS Registry

The `MODELS` registry is the central registry for all model-related classes. Components register themselves using the `@MODELS.register_module()` decorator:

```python
# mmseg/registry.py
from mmengine.registry import Registry, MODELS as MMENGINE_MODELS

MODELS = Registry('models', parent=MMENGINE_MODELS, locations=['mmseg.models'])
```

The `parent=MMENGINE_MODELS` parameter establishes a parent-child relationship: if a class is not found in MMSeg's MODELS registry, it falls back to MMEngine's MODELS registry. This allows MMSegmentation to use common components (like optimizers, losses) registered in MMEngine without re-registering them.

```python
# Example: registering a custom backbone
from mmseg.registry import MODELS

@MODELS.register_module()
class MyCustomBackbone(nn.Module):
    def __init__(self, depth, num_stages=4, ...):
        super().__init__()
        # ... build layers ...

    def forward(self, x):
        outs = []
        for stage in self.stages:
            x = stage(x)
            outs.append(x)
        return tuple(outs)
```

The decorator reads the class name (`MyCustomBackbone`) and registers it in the `MODELS` registry. Alternatively, a custom name can be specified: `@MODELS.register_module(name='my_backbone')`. Once registered, the class can be instantiated from a config dict: `MODELS.build(dict(type='MyCustomBackbone', depth=50))`.

### Build from Config

The `MODELS.build(cfg)` method is the entry point for config-driven instantiation. It extracts the `type` field, looks up the corresponding class in the registry, and passes the remaining fields as keyword arguments:

```python
# How MODELS.build() works internally (simplified)
def build(cfg):
    cfg = cfg.copy()
    obj_type = cfg.pop('type')          # e.g., 'EncoderDecoder'
    obj_cls = self._module_dict[obj_type]  # Look up class in registry
    return obj_cls(**cfg)                # Instantiate with remaining kwargs
```

For nested configs, the build process is recursive. The `EncoderDecoder` class receives `backbone=dict(type='ResNetV1c', ...)` as a keyword argument and internally calls `MODELS.build(backbone_cfg)` to construct the backbone. This recursive build pattern means the entire model tree is constructed from a single top-level `MODELS.build(cfg.model)` call.

```python
# Recursive build inside EncoderDecoder.__init__()
class EncoderDecoder(BaseSegmentor):
    def __init__(self, backbone, decode_head, neck=None, auxiliary_head=None, ...):
        self.backbone = MODELS.build(backbone)          # Build backbone from dict
        if neck is not None:
            self.neck = MODELS.build(neck)               # Build neck from dict
        self.decode_head = MODELS.build(decode_head)     # Build decode head
        if auxiliary_head is not None:
            self.auxiliary_head = MODELS.build(auxiliary_head)
```

## Segmentor Types

### EncoderDecoder

The `EncoderDecoder` class (in `mmseg/models/segmentors/encoder_decoder.py`) is the standard segmentor that composes a backbone, optional neck, and one or more decode heads. It implements the full forward pass for training (with loss computation) and inference (with sliding window or whole-image prediction):

```python
@MODELS.register_module()
class EncoderDecoder(BaseSegmentor):
    def extract_feat(self, inputs):
        """Extract features from backbone (and neck if present)."""
        x = self.backbone(inputs)
        if self.with_neck:
            x = self.neck(x)
        return x

    def encode_decode(self, inputs, batch_img_metas):
        """Forward pass for inference."""
        x = self.extract_feat(inputs)
        seg_logits = self.decode_head.predict(x, batch_img_metas, self.test_cfg)
        return seg_logits

    def loss(self, inputs, data_samples):
        """Forward pass for training with loss computation."""
        x = self.extract_feat(inputs)
        losses = self.decode_head.loss(x, data_samples, self.train_cfg)
        if self.with_auxiliary_head:
            aux_losses = self.auxiliary_head.loss(x, data_samples, self.train_cfg)
            losses.update(aux_losses)
        return losses
```

The `test_cfg` controls inference behavior: `mode='whole'` processes the entire image at once, while `mode='slide'` uses sliding window inference with configurable `crop_size` and `stride`. Sliding window mode is essential for high-resolution images that would exceed GPU memory if processed whole.

### CascadeEncoderDecoder

The `CascadeEncoderDecoder` (in `mmseg/models/segmentors/cascade_encoder_decoder.py`) supports cascaded decode heads where each head refines the previous head's prediction. The primary use case is cascaded architectures like Cascade PSPNet, where a coarse prediction is iteratively refined:

```python
@MODELS.register_module()
class CascadeEncoderDecoder(EncoderDecoder):
    def __init__(self, backbone, decode_head, num_stages, ...):
        # decode_head is a list of head configs
        self.decode_head = nn.ModuleList()
        for head_cfg in decode_head:
            self.decode_head.append(MODELS.build(head_cfg))

    def encode_decode(self, inputs, batch_img_metas):
        x = self.extract_feat(inputs)
        prev_output = self.decode_head[0].predict(x, batch_img_metas)
        for head in self.decode_head[1:]:
            prev_output = head.predict(x, batch_img_metas, prev_output=prev_output)
        return prev_output
```

Each subsequent head receives the previous head's output as additional context, enabling iterative refinement. This pattern is less common than standard `EncoderDecoder` but provides meaningful improvements for some challenging segmentation tasks.

## Adding Custom Components

### Custom Backbone

To register a custom backbone, create a new file in `mmseg/models/backbones/` (or your own project directory) and use the `@MODELS.register_module()` decorator:

```python
# my_project/backbones/custom_backbone.py
from mmseg.registry import MODELS
from mmengine.model import BaseModule

@MODELS.register_module()
class CustomBackbone(BaseModule):
    def __init__(self, in_channels=3, base_channels=64, num_stages=4,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.stages = nn.ModuleList()
        ch = in_channels
        for i in range(num_stages):
            out_ch = base_channels * (2 ** i)
            self.stages.append(self._make_stage(ch, out_ch))
            ch = out_ch

    def forward(self, x):
        outs = []
        for stage in self.stages:
            x = stage(x)
            outs.append(x)
        return tuple(outs)  # Must return tuple of multi-scale features
```

The backbone must return a tuple of feature maps at different scales. The `init_cfg` parameter integrates with MMEngine's weight initialization system. To use this backbone in a config, ensure the module is importable (via `custom_imports` in the config or by placing it in the appropriate package).

### Custom Decode Head

Custom decode heads inherit from `BaseDecodeHead` which provides boilerplate for loss computation, input selection, and output formatting:

```python
# my_project/decode_heads/custom_head.py
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.registry import MODELS

@MODELS.register_module()
class CustomDecodeHead(BaseDecodeHead):
    def __init__(self, feature_strides, **kwargs):
        super().__init__(**kwargs)
        # self.in_channels, self.channels, self.num_classes
        # are set by BaseDecodeHead from kwargs
        self.conv = nn.Conv2d(self.in_channels, self.channels, 3, padding=1)
        self.cls_seg = nn.Conv2d(self.channels, self.num_classes, 1)

    def forward(self, inputs):
        # inputs: tuple of backbone outputs
        x = self._transform_inputs(inputs)  # Select by in_index, resize
        x = self.conv(x)
        x = self.cls_seg(x)
        return x
```

`BaseDecodeHead` provides `_transform_inputs()` which selects the appropriate backbone output based on `in_index` and handles multi-input concatenation. It also provides the `loss()` and `predict()` methods that compute the loss against ground truth and handle test-time resizing respectively.

## Available Model Inventory

MMSegmentation provides a comprehensive inventory of registered components:

| Category | Components | Examples |
|----------|-----------|----------|
| **Backbones** | 20+ | ResNet, ResNeXt, HRNet, SwinTransformer, MiT (SegFormer), BEiT, MAE, ConvNeXt, MobileNetV2/V3 |
| **Decode Heads** | 25+ | FCNHead, PSPHead, ASPPHead, UPerHead, SegformerHead, SETRHead, Mask2FormerHead, DAHead, OCRHead, DNLHead |
| **Losses** | 10+ | CrossEntropyLoss, DiceLoss, FocalLoss, LovaszLoss, TverskyLoss, OhemCrossEntropy |
| **Segmentors** | 3 | EncoderDecoder, CascadeEncoderDecoder, MultimodalEncoderDecoder |
| **Necks** | 3+ | FPN, Feature2Pyramid, MultiLevelNeck |

Each decode head corresponds to a specific paper: PSPHead implements Pyramid Scene Parsing (Zhao et al., 2017), ASPPHead implements Atrous Spatial Pyramid Pooling from DeepLab (Chen et al., 2017), UPerHead implements Unified Perceptual Parsing (Xiao et al., 2018), and SegformerHead implements the lightweight MLP head from SegFormer (Xie et al., 2021). The full list of registered components can be queried programmatically via `MODELS.module_dict.keys()`.
