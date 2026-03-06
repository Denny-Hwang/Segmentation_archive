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

TODO: Trace the exact sequence of Conv2d -> BatchNorm2d -> ReLU -> Conv2d -> BatchNorm2d -> ReLU

## Down Block Detail

TODO: Trace MaxPool2d -> DoubleConv

## Up Block Detail

TODO: Trace Upsample/ConvTranspose2d -> concat with skip -> DoubleConv

## Padding and Cropping Strategy

TODO: Document how spatial dimension mismatches are handled in skip connections

## Shape Verification Script

TODO: Add a script that instantiates the model and prints shapes at each stage
