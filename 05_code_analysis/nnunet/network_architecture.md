---
title: "nnU-Net - Network Architecture"
date: 2025-01-15
status: planned
parent: "nnunet/repo_overview.md"
tags: [nnunet, architecture, plain-conv-unet, residual-encoder]
---

# nnU-Net Network Architecture

## Available Architectures

### PlainConvUNet

Defined in `nnunetv2/architectures/PlainConvUNet.py` (or `dynamic_network_architectures`), the `PlainConvUNet` is a standard convolutional U-Net with dynamically configured depth and channel widths.

Each encoder stage consists of `n_conv_per_stage` blocks, where each block is:
```python
ConvDropoutNormNonlin:
    nn.Conv{2,3}d(in_ch, out_ch, kernel_size, padding)
    nn.Dropout{2,3}d(p)       # Usually p=0 (disabled)
    nn.InstanceNorm{2,3}d(out_ch, affine=True)
    nn.LeakyReLU(negative_slope=0.01, inplace=True)
```

The first convolution in each stage handles channel expansion; subsequent convolutions within the stage maintain the channel count. Downsampling between stages is done via strided convolution (not max pooling).

### ResidualEncoderUNet

The `ResidualEncoderUNet` replaces the plain convolutional blocks in the encoder with residual blocks:

```python
ResidualBlock:
    # Main path
    ConvDropoutNormNonlin(in_ch, out_ch, ...)
    ConvDropoutNormNonlin(out_ch, out_ch, ...)
    # Skip path
    identity (if in_ch == out_ch)
    OR nn.Conv(in_ch, out_ch, 1x1) (if channel mismatch)
    # Output = main + skip
```

Key differences from `PlainConvUNet`:
- Residual connections within each encoder stage enable deeper networks without degradation
- The decoder still uses plain convolutions (residual connections in the decoder showed no benefit)
- Slightly higher parameter count due to potential 1x1 projection convolutions
- Better gradient flow for very deep networks (6+ stages)

## Dynamic Architecture Generation

The architecture is constructed entirely from the experiment plan at runtime. The `get_network_from_plans()` function reads the plan and builds the network:

```python
def get_network_from_plans(plans, dataset_json, configuration, num_input_channels):
    plan = plans['configurations'][configuration]
    conv_kernel_sizes = plan['conv_kernel_sizes']      # e.g., [[3,3,3], [3,3,3], ...]
    pool_op_kernel_sizes = plan['pool_op_kernel_sizes'] # e.g., [[1,1,1], [2,2,2], ...]
    num_stages = len(conv_kernel_sizes)
    features_per_stage = compute_features(plan['UNet_base_num_features'],
                                          plan['unet_max_num_features'],
                                          num_stages)

    network = PlainConvUNet(
        input_channels=num_input_channels,
        n_stages=num_stages,
        features_per_stage=features_per_stage,
        conv_op=nn.Conv3d,  # or Conv2d for 2D
        kernel_sizes=conv_kernel_sizes,
        strides=pool_op_kernel_sizes,
        n_conv_per_stage=plan['n_conv_per_stage_encoder'],
        n_conv_per_stage_decoder=plan['n_conv_per_stage_decoder'],
        conv_bias=True,
        norm_op=nn.InstanceNorm3d,
        nonlin=nn.LeakyReLU,
        deep_supervision=True
    )
    return network
```

No architecture is hardcoded -- everything adapts to the dataset properties.

## Encoder Configuration

Typical configurations for a 3D medical imaging dataset:

| Stage | Channels | Kernel Size | Pool/Stride | Output Resolution |
|-------|----------|-------------|-------------|-------------------|
| 0 | 32 | 3x3x3 | 1x1x1 (no pool) | Full |
| 1 | 64 | 3x3x3 | 2x2x2 | 1/2 |
| 2 | 128 | 3x3x3 | 2x2x2 | 1/4 |
| 3 | 256 | 3x3x3 | 2x2x2 | 1/8 |
| 4 | 320 | 3x3x3 | 2x2x2 | 1/16 |
| 5 | 320 | 3x3x3 | 2x2x2 | 1/32 |

For anisotropic data, the kernel and stride in the anisotropic axis may be 1 instead of 2/3 at early stages, delaying pooling until the voxel spacing is approximately isotropic.

Downsampling uses **strided convolution** (stride=pool_kernel_size) rather than max pooling, which is a learned operation and avoids information loss.

## Decoder Configuration

The decoder mirrors the encoder with:
- **Upsampling**: `nn.ConvTranspose{2,3}d` with `kernel_size=stride=pool_kernel_size` (learned upsampling matching the encoder's stride)
- **Skip connections**: Concatenation along the channel dimension (same as standard U-Net)
- **Channel counts**: After concatenation, the decoder block reduces channels to match the corresponding encoder stage
- **n_conv_per_stage_decoder**: Usually 2 convolutions per decoder stage (configurable)

```python
# Decoder stage pseudocode:
x = ConvTranspose(x)                 # Upsample
x = torch.cat([x, skip], dim=1)      # Concatenate with encoder skip
x = ConvBlock(x)                     # Reduce channels
x = ConvBlock(x)                     # Refine features
```

## Normalization

nnU-Net uses **Instance Normalization** (`nn.InstanceNorm{2,3}d` with `affine=True`) by default, which is a deliberate departure from the BatchNorm used in most other implementations.

Rationale:
- Medical imaging typically uses small batch sizes (2-4) due to large 3D volumes
- BatchNorm statistics are unreliable with small batches, leading to training instability
- InstanceNorm normalizes per-sample, per-channel, so it is independent of batch size
- `affine=True` adds learnable scale and shift parameters (gamma, beta)

The activation function is `LeakyReLU(negative_slope=0.01)` rather than standard ReLU, providing small gradients for negative inputs to prevent dead neurons.

## Deep Supervision

nnU-Net applies deep supervision during training by computing loss at multiple decoder stages:

```python
# During training, the network returns predictions from all decoder levels:
seg_outputs = []
for i, decoder_stage in enumerate(self.decoder_stages):
    x = decoder_stage(x, skip)
    if self.deep_supervision:
        seg_outputs.append(self.seg_layers[i](x))  # 1x1 conv to n_classes
```

The total loss is a weighted sum:
```python
weights = [1.0, 0.5, 0.25, 0.125, ...]  # Halving at each deeper level
total_loss = sum(w * loss(pred, downsampled_target) for w, pred in zip(weights, seg_outputs))
```

- Targets are downsampled to match each prediction resolution
- Provides gradient signal at every decoder level, accelerating convergence
- Only the highest-resolution output is used at inference time
- Deep supervision is disabled during validation/inference (`model.deep_supervision = False`)

## Parameter Count Scaling

Parameter count scales with the planned configuration:

| Factor | Impact |
|--------|--------|
| Base features (32 vs 64) | ~4x total parameters |
| Max features cap (320 vs 512) | ~1.5-2x at deepest stages |
| Number of stages (5 vs 7) | ~1.3x per additional stage |
| 2D vs 3D convolutions | 3D has 3x kernel parameters (3x3x3 vs 3x3) |
| n_conv_per_stage (2 vs 3) | Linear scaling per stage |

Typical parameter counts:
- **3D_fullres (6 stages, base=32, max=320)**: ~30M parameters
- **2D (6 stages, base=32, max=512)**: ~25M parameters
- **ResidualEncoderUNet (same config)**: ~35M parameters (extra projection convolutions)

## Comparison with Standard U-Net

| Aspect | Standard U-Net | nnU-Net |
|--------|---------------|---------|
| Normalization | BatchNorm | InstanceNorm (affine=True) |
| Skip Connections | Concatenation | Concatenation (same) |
| Depth | Fixed (4-5 stages) | Dynamic (3-7 stages, data-dependent) |
| Kernel Sizes | Fixed 3x3 | Per-stage, data-dependent (may be 1xNxN for anisotropic) |
| Deep Supervision | No | Yes (weighted multi-scale loss) |
| Downsampling | MaxPool | Strided convolution (learned) |
| Upsampling | Bilinear or ConvTranspose | ConvTranspose (always) |
| Activation | ReLU | LeakyReLU(0.01) |
| Base Channels | 64 | 32 (doubles per stage, capped) |
| Dropout | Sometimes at bottleneck | Configurable per stage (usually disabled) |
| Architecture | Hardcoded | Fully dynamic from plan |
