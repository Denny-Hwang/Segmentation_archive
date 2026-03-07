---
title: "Pytorch-UNet - Module Breakdown"
date: 2025-01-15
status: planned
parent: "unet_pytorch/repo_overview.md"
tags: [unet, modules, pytorch, components]
---

# Pytorch-UNet Module Breakdown

## Module Inventory

| Module | File | Parameters | Description |
|--------|------|-----------|-------------|
| `DoubleConv` | `unet_parts.py` | ~221K (64->128 case) | Two consecutive Conv-BN-ReLU blocks |
| `Down` | `unet_parts.py` | Same as DoubleConv (pool has no params) | MaxPool followed by DoubleConv |
| `Up` | `unet_parts.py` | ~1.5M (bilinear, 1024->512 case) | Upsample + skip concatenation + DoubleConv |
| `OutConv` | `unet_parts.py` | 64 * n_classes + n_classes | 1x1 convolution for final prediction |
| `UNet` | `unet_model.py` | ~31M (bilinear) / ~17.3M (transposed) | Full U-Net assembled from parts |

## DoubleConv

### Code Analysis

Defined in `unet/unet_parts.py`, `DoubleConv` builds a `nn.Sequential` with the following layers:

```python
nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False)
nn.BatchNorm2d(mid_channels)
nn.ReLU(inplace=True)
nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False)
nn.BatchNorm2d(out_channels)
nn.ReLU(inplace=True)
```

- **Kernel size**: 3x3 throughout, which captures local spatial context efficiently
- **Padding**: `padding=1` (same-padding) preserves spatial dimensions, unlike the original U-Net paper which used valid convolutions
- **Bias**: Disabled (`bias=False`) because BatchNorm's learnable shift parameter (beta) makes the convolution bias redundant
- **mid_channels**: Optional parameter defaulting to `out_channels`; set to `in_channels // 2` in the bilinear upsampling `Up` block to reduce computation

### Design Decisions

**BatchNorm** was chosen over alternatives for several practical reasons:
- BatchNorm was the dominant normalization technique when this implementation was created
- Works well with reasonably sized mini-batches (batch size >= 8)
- Provides regularization effect during training, reducing need for dropout
- GroupNorm or InstanceNorm would be better choices for very small batch sizes (e.g., medical imaging with large 3D volumes), but this 2D implementation typically allows larger batches

## Down

### Code Analysis

```python
class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
```

- **MaxPool2d(2)**: kernel_size=2, stride=2 (stride defaults to kernel_size). Halves spatial dimensions.
- **No learnable parameters** in the pooling layer itself; all parameters come from the subsequent `DoubleConv`
- The order (pool first, then convolve) is computationally efficient -- convolutions operate on smaller feature maps

## Up

### Bilinear vs Transposed Convolution

The `Up` block supports two upsampling modes controlled by the `bilinear` flag:

| Aspect | `bilinear=True` | `bilinear=False` |
|--------|-----------------|------------------|
| Upsampling op | `nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)` | `nn.ConvTranspose2d(in_ch, in_ch//2, kernel_size=2, stride=2)` |
| Learnable upsampling | No | Yes |
| Channel reduction before concat | None (done in DoubleConv via `mid_channels`) | Yes (in ConvTranspose2d) |
| Artifacts | Smoother output | Can produce checkerboard artifacts |
| Parameter count | Lower (no upsample params) | Higher |
| `DoubleConv` mid_channels | `in_channels // 2` | `out_channels` (default) |

In the `UNet.__init__`, when `bilinear=True`, a `factor=2` is used to halve the bottleneck channel count (512 instead of 1024), keeping the model smaller.

### Skip Connection Concatenation

The forward pass handles dimension mismatches through explicit padding:

```python
def forward(self, x1, x2):
    x1 = self.up(x1)
    diffY = x2.size()[2] - x1.size()[2]
    diffX = x2.size()[3] - x1.size()[3]
    x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                     diffY // 2, diffY - diffY // 2])
    x = torch.cat([x2, x1], dim=1)
    return self.conv(x)
```

`x2` is the skip connection from the encoder, `x1` is the upsampled decoder feature. The padding ensures they match spatially before channel-wise concatenation. This approach is robust to arbitrary input sizes, unlike cropping which would discard information.

## OutConv

### Code Analysis

```python
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
```

A **1x1 convolution** is used for the final layer because:
- It acts as a per-pixel linear classifier across the 64 feature channels
- No spatial context is needed at this stage (already captured by preceding layers)
- It maps from the feature dimension (64) to the number of output classes
- No activation function is applied -- raw logits are output, allowing the loss function to apply softmax/sigmoid
- `bias=True` (default) is used here since there is no subsequent BatchNorm

## Parameter Count Summary

Parameter counts for the default configuration (`n_channels=3, n_classes=2, bilinear=True`):

| Component | Parameters | % of Total |
|-----------|-----------|------------|
| Encoder (inc + down1-3) | ~2.77M | 16.0% |
| Bottleneck (down4) | ~4.72M | 27.3% |
| Decoder (up1-4) | ~9.68M | 56.0% |
| Output Head (outc) | ~130 | <0.1% |
| **Total** | **~17.3M** | 100% |

When `bilinear=False`, the total rises to approximately 31M parameters because the transposed convolution adds learnable upsampling weights and the bottleneck uses 1024 channels instead of 512.
