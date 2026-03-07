---
title: "keras-unet-collection - Model Comparison"
date: 2025-01-15
status: complete
parent: "keras_unet_collection/repo_overview.md"
tags: [keras, unet-variants, comparison, tensorflow]
---

# keras-unet-collection Model Comparison

## Available Models

| Model | Function | Paper | Key Feature |
|-------|----------|-------|-------------|
| U-Net | `models.unet_2d()` | Ronneberger 2015 | Baseline encoder-decoder |
| U-Net++ | `models.unet_plus_2d()` | Zhou 2018 | Nested skip connections |
| Attention U-Net | `models.att_unet_2d()` | Oktay 2018 | Attention gates on skip connections |
| R2U-Net | `models.r2_unet_2d()` | Alom 2018 | Recurrent + residual blocks |
| TransUNet | `models.transunet_2d()` | Chen 2021 | CNN encoder + ViT + CNN decoder |
| Swin-UNET | `models.swin_unet_2d()` | Cao 2022 | Pure Swin Transformer U-Net |
| V-Net | `models.vnet_2d()` | Milletari 2016 | Residual blocks + Dice loss |

## Common Interface

All models in keras-unet-collection share a consistent functional API with a common set of parameters. Each model is created through a factory function rather than direct class instantiation, following Keras's functional API pattern:

```python
from keras_unet_collection import models

# Common signature across all model functions:
model = models.unet_2d(
    input_size=(256, 256, 3),       # (H, W, C) input shape
    filter_num=[64, 128, 256, 512], # Channels per encoder/decoder stage
    n_labels=2,                      # Number of output classes
    stack_num_down=2,                # Conv blocks per encoder stage
    stack_num_up=2,                  # Conv blocks per decoder stage
    activation='ReLU',               # Activation function name
    output_activation='Softmax',     # Final activation ('Softmax' or 'Sigmoid')
    batch_norm=True,                 # Whether to use BatchNormalization
    pool=True,                       # True=MaxPool, False=strided conv
    unpool=True,                     # True=UpSampling2D, False=Conv2DTranspose
    backbone=None,                   # Optional pretrained backbone name
    weights='imagenet',              # Pretrained weights (if backbone is set)
    freeze_backbone=True,            # Whether to freeze backbone weights
    freeze_batch_norm=True,          # Whether to freeze BN layers in backbone
    name='unet',                     # Model name string
)
```

The `filter_num` list defines the network depth and width simultaneously: its length determines the number of encoder/decoder stages, and its values determine the channel count at each stage. The `backbone` parameter enables transfer learning by replacing the default encoder with a pretrained backbone from `tf.keras.applications` (e.g., `'VGG16'`, `'ResNet50'`, `'EfficientNetB0'`). When a backbone is specified, `filter_num` is ignored for the encoder and only controls the decoder channel counts.

The `pool`/`unpool` boolean parameters control the downsampling and upsampling strategy respectively. Setting `pool=False` uses strided convolutions for downsampling (learnable, but more expensive), and `unpool=False` uses transposed convolutions for upsampling (learnable, but can cause checkerboard artifacts).

## Architecture Comparison

### Encoder Design

All models share the same basic encoder pattern when no pretrained backbone is specified: a stack of convolutional blocks at each scale, with spatial downsampling between scales. The key differences are in the convolutional block design:

- **U-Net**: Standard Conv2D-BN-ReLU blocks, stacked `stack_num_down` times per stage.
- **U-Net++**: Same encoder as U-Net. The innovation is entirely in the skip connections and decoder.
- **Attention U-Net**: Same encoder as U-Net, with attention gates applied to skip connections during decoding.
- **R2U-Net**: Replaces standard conv blocks with recurrent residual convolutional units (RRCNN blocks). Each block applies the convolution multiple times (controlled by `recur_num`, default 2) with shared weights, and adds a residual connection. This increases the effective receptive field without adding parameters.
- **TransUNet**: Uses a CNN encoder (optionally pretrained) for early stages, then reshapes the feature map into a sequence of patches and processes them through a Vision Transformer. The ViT features are reshaped back to spatial dimensions for the decoder.
- **Swin-UNET**: Replaces the entire CNN encoder with Swin Transformer blocks. Spatial downsampling is achieved through patch merging layers rather than pooling.
- **V-Net**: Uses residual convolutional blocks (input + conv output) similar to ResNet, originally designed for volumetric (3D) medical image segmentation but adapted to 2D in this collection.

### Skip Connection Variants

The skip connection strategy is the primary differentiator between models in this collection:

**Standard concatenation** (U-Net, V-Net): Encoder features are concatenated with upsampled decoder features along the channel axis. Simple and effective, but treats all encoder features equally regardless of their relevance to the current decoder stage.

**Nested/Dense connections** (U-Net++): Intermediate dense blocks connect each encoder stage to each decoder stage through nested pathways. Instead of a single skip connection per stage, U-Net++ creates a dense web of connections with intermediate convolutional nodes. This is implemented via nested `for` loops that build a triangular grid of feature processing nodes:

```python
# U-Net++ nested connections (simplified logic)
# X[i][j] = ConvBlock(concat(UpSample(X[i+1][j-1]), X[i][0], X[i][1], ..., X[i][j-1]))
```

**Attention-gated connections** (Attention U-Net): Before concatenation, encoder features are passed through an attention gate that uses the decoder features as a gating signal. The gate learns to suppress irrelevant spatial regions in the encoder features, allowing the decoder to focus on salient areas:

```python
# Attention gate (simplified)
def attention_gate(x_skip, x_decoder, filters):
    theta_x = Conv2D(filters, 1)(x_skip)       # Encoder features
    phi_g = Conv2D(filters, 1)(x_decoder)        # Decoder features (gate signal)
    psi = Activation('relu')(theta_x + phi_g)
    psi = Conv2D(1, 1)(psi)
    psi = Activation('sigmoid')(psi)             # Attention coefficients [0, 1]
    return x_skip * psi                           # Weighted encoder features
```

**Recurrent + residual** (R2U-Net): Skip connections are standard concatenation, but the convolutional blocks themselves use recurrent processing (same conv applied multiple times with accumulated features) and residual connections.

### Decoder Design

All decoders follow the same general pattern: upsample, concatenate with skip connection, apply convolutional blocks. The differences lie in the skip connection strategy (described above) and the number/type of convolutions:

- **U-Net, Attention U-Net**: Simple UpSampling2D + concat + Conv blocks.
- **U-Net++**: Dense decoder with intermediate nodes at each level, supporting optional deep supervision where each decoder level produces a segmentation output.
- **TransUNet**: Standard U-Net decoder that progressively upsamples the ViT features, merging with CNN encoder features via concatenation.
- **Swin-UNET**: Swin Transformer blocks in the decoder with patch expanding layers for upsampling, mirroring the encoder structure.

## Parameter Counts

Approximate parameter counts for default configurations (`input_size=(256, 256, 3)`, `filter_num=[64, 128, 256, 512]`, `n_labels=2`):

| Model | Parameters | Relative Size | Notes |
|-------|-----------|--------------|-------|
| U-Net | ~7.8M | 1.0x (baseline) | Simplest architecture |
| U-Net++ | ~9.2M | 1.18x | Extra nested convolution nodes |
| Attention U-Net | ~8.7M | 1.12x | Attention gate parameters are small |
| R2U-Net | ~7.8M | 1.0x | Shared weights in recurrent blocks |
| TransUNet | ~105M | 13.5x | ViT adds significant parameters |
| Swin-UNET | ~27M | 3.5x | Swin Transformer parameters |
| V-Net | ~8.5M | 1.09x | Residual connections add minimal params |

R2U-Net achieves the same parameter count as U-Net because the recurrent convolutions share weights -- the same kernels are applied multiple times. TransUNet has dramatically more parameters due to the embedded Vision Transformer (typically ViT-B/16 with ~86M parameters). When using pretrained backbones (e.g., `backbone='ResNet50'`), the parameter count is dominated by the backbone regardless of the decoder architecture.

## Keras vs PyTorch Implementation Differences

Several notable differences arise from the TensorFlow/Keras vs PyTorch framework divide:

**Channel ordering**: Keras defaults to channels-last format (`NHWC`), while PyTorch uses channels-first (`NCHW`). The `input_size` parameter in keras-unet-collection expects `(H, W, C)`, whereas PyTorch models expect `(C, H, W)`. This affects all internal convolution operations and can cause confusion when porting models or comparing feature map shapes.

**Functional API vs Module-based**: keras-unet-collection uses Keras's functional API where models are defined by chaining layer calls on tensor objects, producing a `tf.keras.Model`. PyTorch implementations (e.g., milesial/Pytorch-UNet) use `nn.Module` subclasses with explicit `forward()` methods. The functional API approach makes it easier to create complex skip connection topologies (like U-Net++) but harder to add custom logic within the forward pass.

**Weight initialization**: Keras uses Glorot uniform (Xavier uniform) initialization by default for `Conv2D`, while PyTorch uses Kaiming uniform. Both are reasonable defaults, but they produce slightly different initial loss values and convergence behavior. The keras-unet-collection does not override default initialization.

**Batch normalization behavior**: In Keras, `BatchNormalization` uses `momentum=0.99` by default (the moving average update coefficient), while PyTorch's `BatchNorm2d` uses `momentum=0.1`. These represent the same concept but with complementary conventions: Keras's 0.99 equals PyTorch's 0.01 (they use `(1-momentum)` differently). This difference can affect training stability, especially when fine-tuning pretrained models.

**Pretrained backbone integration**: keras-unet-collection integrates with `tf.keras.applications` for pretrained backbones (VGG, ResNet, EfficientNet, etc.), while PyTorch equivalents use `torchvision.models` or `timm`. The Keras approach extracts intermediate layer outputs by name from the applications model, while PyTorch implementations typically use forward hooks or modify the backbone's forward method. The keras-unet-collection's `backbone` parameter provides a convenient one-line integration that is more streamlined than most PyTorch equivalents.
