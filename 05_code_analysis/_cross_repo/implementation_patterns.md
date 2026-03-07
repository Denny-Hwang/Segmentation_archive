---
title: "Cross-Repository Implementation Patterns"
date: 2025-01-15
status: complete
tags: [cross-repo, design-patterns, encoder-decoder, best-practices]
---

# Cross-Repository Implementation Patterns

## Purpose

This document identifies recurring implementation patterns observed across all analyzed segmentation repositories.

## Pattern 1: Encoder-Decoder Composition

### Description

Every segmentation repository separates the model into an encoder (feature extractor) and a decoder (spatial reconstruction), but the mechanism for composing these components varies significantly. The choice reflects each project's design philosophy -- simplicity vs. flexibility vs. full automation.

### Implementations

| Repository | Approach | Notes |
|-----------|----------|-------|
| Pytorch-UNet | Hardcoded composition | Encoder and decoder are defined together in `UNet.__init__()` with fixed module names (`inc`, `down1`-`down4`, `up1`-`up4`, `outc`). Changing the encoder requires modifying the class directly. Simplest to understand but least flexible. |
| SMP | Registry + factory | Encoders are loaded via `get_encoder(name, weights)` from a registry of 400+ pretrained backbones. Decoders are separate classes (`UnetDecoder`, `FPNDecoder`). The `smp.Unet(encoder_name='resnet34')` factory composes them. Highly flexible but requires understanding the registry API. |
| nnU-Net | Dynamic from plan | Architecture is generated at runtime from a `plans.json` file. `PlainConvUNet` or `ResidualEncoderUNet` is instantiated with dynamically computed parameters (depth, channels, kernel sizes). No manual architecture specification needed. |
| MMSegmentation | Config-driven builder | Components are specified as nested dicts with `type` keys, resolved via `MODELS.build()`. The `EncoderDecoder` segmentor composes backbone, neck, and decode_head from their respective configs. Maximum configurability but steep learning curve. |
| SAM 2 | Module composition | `SAM2Base` composes `ImageEncoder`, `MemoryAttention`, `MemoryEncoder`, `MaskDecoder`, and `PromptEncoder` as named sub-modules. Each module is a self-contained `nn.Module` with clear interfaces. Clean but tailored to SAM's specific architecture. |
| keras-unet-collection | Functional API | Models are built via factory functions (`models.unet_2d(...)`) using Keras's functional API. The encoder and decoder are constructed within the same function, connected by tensor operations. Easy to use but hard to modify internally. |

## Pattern 2: Skip Connection Strategies

Skip connections are the defining feature of U-Net-style architectures, and their implementation varies across repositories:

**Concatenation** is the most common approach, used by Pytorch-UNet, SMP's Unet decoder, nnU-Net, and keras-unet-collection's U-Net. Encoder features are concatenated with decoder features along the channel dimension (`torch.cat([x_skip, x_up], dim=1)` or `tf.concat`), doubling the channel count before a subsequent convolution reduces it. This preserves all information from both sources but increases memory usage temporarily.

**Addition** is used by SMP's FPN and Linknet decoders, and by MMSegmentation's FPN neck. Encoder and decoder features are projected to the same channel dimension via 1x1 convolutions, then summed element-wise. This is more memory-efficient (no channel doubling) but can lose information when encoder and decoder features conflict.

**Attention-gated connections** are used by SMP's MAnet and keras-unet-collection's Attention U-Net. A gating mechanism learns to weight encoder features based on the decoder's context, suppressing irrelevant spatial regions before concatenation. This adds a small number of parameters but can significantly improve performance by focusing the decoder on relevant encoder features.

**Nested/Dense connections** (U-Net++, both in SMP and keras-unet-collection) create intermediate processing nodes between every encoder-decoder pair at the same scale, forming a dense connectivity pattern that captures features at multiple semantic levels.

## Pattern 3: Multi-Scale Feature Handling

All repositories must handle the fundamental challenge of producing high-resolution segmentation maps from features extracted at multiple spatial scales:

**Feature Pyramid Networks (FPN)**: Used by SMP's FPN decoder, SAM 2's FpnNeck, and MMSegmentation's FPN neck. A top-down pathway with lateral connections produces features at multiple scales, all with the same channel dimension. The highest-resolution features are used for the final prediction.

**Progressive upsampling**: The classic U-Net approach (Pytorch-UNet, SMP's Unet, nnU-Net). Features are upsampled stage-by-stage, merging with skip connections at each scale. This allows gradual spatial refinement but processes features sequentially.

**Atrous Spatial Pyramid Pooling (ASPP)**: Used in SMP's DeepLabV3/DeepLabV3+ and MMSegmentation's ASPPHead. Multiple parallel dilated convolutions with different dilation rates capture context at multiple scales without reducing spatial resolution. The outputs are concatenated and projected.

**Pyramid Pooling Module (PPM)**: Used in SMP's PSPNet and MMSegmentation's PSPHead. The feature map is pooled at multiple fixed scales (1x1, 2x2, 3x3, 6x6), each pooled output is projected via 1x1 conv, then upsampled and concatenated with the original feature map.

## Pattern 4: Loss Function Composition

Every repository that includes training code combines multiple loss functions to handle the challenges of segmentation:

```python
# Pytorch-UNet: equal-weight CE + Dice
loss = cross_entropy_loss(pred, target) + dice_loss(pred, target)

# nnU-Net: CE + Dice with deep supervision weighting
loss = 0.5 * ce_loss + 0.5 * dice_loss  # at each supervision level
total_loss = sum(weight[i] * loss[i] for i in range(num_supervision_levels))

# SMP: configurable, default CE
loss = smp.losses.DiceLoss(mode='multiclass')(pred, target)

# MMSegmentation: config-driven with loss_weight
loss_decode = dict(type='CrossEntropyLoss', loss_weight=1.0)
# or: loss_decode=[dict(type='CrossEntropyLoss', loss_weight=1.0),
#                  dict(type='DiceLoss', loss_weight=0.5)]
```

The CE + Dice combination is nearly universal because CE provides good per-pixel gradients while Dice provides region-level optimization that is invariant to class imbalance. nnU-Net additionally uses deep supervision (computing loss at multiple decoder stages with decreasing weights), which provides gradient signal to earlier layers and helps with training stability.

## Pattern 5: Pretrained Encoder Integration

Loading pretrained backbones is critical for achieving competitive segmentation performance with limited training data:

**SMP**: Uses `timm` as a universal encoder provider. The `get_encoder()` function wraps any `timm` model, intercepts its forward pass to extract multi-scale features, and exposes a standardized `encoder.out_channels` attribute. This gives access to 400+ pretrained architectures with a single API.

**MMSegmentation**: Uses `open-mmlab://` URLs or local paths to load pretrained weights. The config's `pretrained` field triggers automatic download and loading. Weight adaptation (e.g., handling different input channels) is handled by custom init functions.

**nnU-Net**: Does not use pretrained encoders by default. The self-configuring philosophy assumes sufficient training data from the target domain. However, custom trainers can load pretrained weights via standard PyTorch `load_state_dict()`.

**SAM 2**: Ships with MAE-pretrained Hiera weights. The pretrained weights are loaded by the model builder from checkpoint files. No support for arbitrary pretrained backbones -- the architecture is fixed to Hiera variants.

**keras-unet-collection**: Integrates with `tf.keras.applications` via the `backbone` parameter. When set, the encoder is replaced with the specified pretrained model (e.g., `'ResNet50'`), and intermediate layers are extracted as multi-scale features.

## Pattern 6: Configuration Management

Configuration approaches range from minimal to elaborate:

**Argparse (Pytorch-UNet)**: Command-line arguments parsed with `argparse`. Simple and transparent but limited to flat key-value pairs. No support for nested configurations or inheritance.

**Python config files (MMSegmentation)**: Full Python files parsed by MMEngine's `Config` class. Supports inheritance, variable references, conditional logic, and arbitrary Python expressions. Most powerful but most complex.

**JSON plans (nnU-Net)**: Auto-generated `nnUNetPlans.json` files that fully specify the experiment. Users rarely write configs manually; the planner generates them. Plans can be edited post-generation for customization.

**Constructor arguments (SMP, keras-unet-collection)**: Model configuration via constructor/factory function arguments. Simple API but limited to predefined parameters. SMP adds flexibility via the encoder registry.

**YAML + Python (SAM 2)**: Uses Hydra/OmegaConf for configuration with YAML files defining model variants. Config composition via Hydra's override system.

## Pattern 7: Data Pipeline Abstraction

Data loading and augmentation show the widest variation across repositories:

**Custom Dataset classes (Pytorch-UNet, nnU-Net)**: `torch.utils.data.Dataset` subclasses with manual preprocessing in `__getitem__()`. Pytorch-UNet's `BasicDataset` loads images from disk and applies resize + normalize. nnU-Net's dataset loads preprocessed `.npz` files and applies on-the-fly augmentation via `batchgenerators` library.

**Config-driven pipelines (MMSegmentation)**: Transform pipelines defined as lists of dicts in config files. Each transform is a registered class instantiated at runtime. Highly composable and reproducible, but requires learning the transform registry API.

**Framework datasets (SMP)**: Does not provide dataset classes; users bring their own `Dataset` implementations. SMP focuses exclusively on models and losses, leaving data handling to the user.

**tf.data pipelines (keras-unet-collection)**: Users construct `tf.data.Dataset` pipelines externally. The library provides no data loading utilities.

## Summary Table

| Pattern | Pytorch-UNet | SMP | nnU-Net | SAM 2 | MMSeg | keras-unet |
|---------|-------------|-----|---------|-------|-------|------------|
| Encoder-Decoder | Hardcoded | Registry | Dynamic | Module | Config | Functional |
| Skip Connections | Concat | Varies | Concat | Cross-attn | Varies | Varies |
| Multi-Scale | Progressive | Varies | Progressive | FPN | Varies | Progressive |
| Loss Composition | CE+Dice | User choice | CE+Dice+DS | Focal+Dice+IoU | Config | User choice |
| Pretrained Encoder | No | timm | No (default) | MAE Hiera | mmcls/timm | tf.keras.apps |
| Configuration | argparse | Constructor | JSON plans | Hydra/YAML | Python files | Constructor |
| Data Pipeline | Custom Dataset | User-provided | batchgenerators | Custom | Config pipeline | User-provided |
