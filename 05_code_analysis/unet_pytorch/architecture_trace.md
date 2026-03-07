---
title: "Pytorch-UNet - Architecture Trace"
date: 2025-01-15
status: planned
parent: "unet_pytorch/repo_overview.md"
tags: [unet, architecture, forward-pass, pytorch]
---

# Pytorch-UNet Architecture Trace

## Forward Pass Flow

```
Input (B, C_in, H, W)
  │
  ├── inc: DoubleConv ──────────────────────────── skip1
  │       (B, C_in, H, W) -> (B, 64, H, W)
  │
  ├── down1: Down ──────────────────────────────── skip2
  │       (B, 64, H, W) -> (B, 128, H/2, W/2)
  │
  ├── down2: Down ──────────────────────────────── skip3
  │       (B, 128, H/2, W/2) -> (B, 256, H/4, W/4)
  │
  ├── down3: Down ──────────────────────────────── skip4
  │       (B, 256, H/4, W/4) -> (B, 512, H/8, W/8)
  │
  ├── down4: Down (bottleneck)
  │       (B, 512, H/8, W/8) -> (B, 1024, H/16, W/16)
  │
  ├── up1: Up ← skip4
  │       (B, 1024, H/16, W/16) -> (B, 512, H/8, W/8)
  │
  ├── up2: Up ← skip3
  │       (B, 512, H/8, W/8) -> (B, 256, H/4, W/4)
  │
  ├── up3: Up ← skip2
  │       (B, 256, H/4, W/4) -> (B, 128, H/2, W/2)
  │
  ├── up4: Up ← skip1
  │       (B, 128, H/2, W/2) -> (B, 64, H, W)
  │
  └── outc: OutConv
          (B, 64, H, W) -> (B, n_classes, H, W)
```

## DoubleConv Block Detail

The `DoubleConv` block (defined in `unet/unet_parts.py`) applies two consecutive convolution layers, each followed by batch normalization and ReLU activation:

```python
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
```

Key details:
- `kernel_size=3` with `padding=1` produces **same-padding** convolutions (output H/W = input H/W)
- `bias=False` because BatchNorm already has a learnable bias (beta parameter)
- `mid_channels` defaults to `out_channels` but can be overridden (used in the bilinear upsampling path)
- `inplace=True` on ReLU saves memory by modifying tensors in-place

## Down Block Detail

The `Down` block applies max pooling followed by `DoubleConv`:

```python
class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),       # kernel_size=2, stride=2 (default)
            DoubleConv(in_channels, out_channels)
        )
```

- `MaxPool2d(2)` halves both spatial dimensions (H/2, W/2)
- The pooling happens **before** the convolutions, which is the standard U-Net pattern
- Stride defaults to `kernel_size` when not specified, so stride=2

## Up Block Detail

The `Up` block supports two upsampling modes:

```python
class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2,
                                          kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Pad x1 to match x2 spatial dimensions
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                         diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)  # Concatenate along channel dim
        return self.conv(x)
```

- **Bilinear mode**: Uses `nn.Upsample` (no learnable parameters), then feeds concatenated features through `DoubleConv` with reduced `mid_channels` to compensate for missing learned upsampling
- **Transposed convolution mode**: `ConvTranspose2d` halves channels during upsampling, then `DoubleConv` processes the concatenated result
- Skip connection uses **concatenation** along the channel dimension (`dim=1`)

## Padding and Cropping Strategy

Unlike the original U-Net paper which used valid (unpadded) convolutions, this implementation uses `padding=1` to maintain spatial dimensions through convolutions. However, spatial mismatches can still occur when input dimensions are odd. The `Up.forward()` method handles this with dynamic padding:

```python
diffY = x2.size()[2] - x1.size()[2]
diffX = x2.size()[3] - x1.size()[3]
x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                 diffY // 2, diffY - diffY // 2])
```

This asymmetric padding ensures that upsampled features (`x1`) match the encoder skip connection (`x2`) exactly, even when dimensions differ by 1 pixel. The padding is applied with zero values by default.

## Shape Verification Script

```python
import torch
from unet import UNet

model = UNet(n_channels=3, n_classes=2, bilinear=True)
x = torch.randn(1, 3, 572, 572)

# Hook to print shapes at each stage
def trace_shapes(model, x):
    x1 = model.inc(x)
    print(f"inc:   {x.shape} -> {x1.shape}")        # [1,3,572,572] -> [1,64,572,572]
    x2 = model.down1(x1)
    print(f"down1: {x1.shape} -> {x2.shape}")       # [1,64,572,572] -> [1,128,286,286]
    x3 = model.down2(x2)
    print(f"down2: {x2.shape} -> {x3.shape}")       # [1,128,286,286] -> [1,256,143,143]
    x4 = model.down3(x3)
    print(f"down3: {x3.shape} -> {x4.shape}")       # [1,256,143,143] -> [1,512,71,71]
    x5 = model.down4(x4)
    print(f"down4: {x4.shape} -> {x5.shape}")       # [1,512,71,71] -> [1,1024,35,35]
    x = model.up1(x5, x4)
    print(f"up1:   {x5.shape} + skip -> {x.shape}") # [1,1024,35,35] + skip -> [1,512,71,71]
    x = model.up2(x, x3)
    print(f"up2:   -> {x.shape}")                    # -> [1,256,143,143]
    x = model.up3(x, x2)
    print(f"up3:   -> {x.shape}")                    # -> [1,128,286,286]
    x = model.up4(x, x1)
    print(f"up4:   -> {x.shape}")                    # -> [1,64,572,572]
    logits = model.outc(x)
    print(f"outc:  -> {logits.shape}")               # -> [1,2,572,572]

trace_shapes(model, x)
```

Note: When `bilinear=True`, the bottleneck outputs 512 channels (not 1024) because the `factor` variable in `UNet.__init__` halves the channel count at the deepest level.
