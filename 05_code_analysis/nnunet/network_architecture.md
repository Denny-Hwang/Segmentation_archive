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
TODO: Analyze the standard convolutional U-Net used in nnU-Net

### ResidualEncoderUNet
TODO: Analyze the residual variant

## Dynamic Architecture Generation

TODO: How the architecture is built dynamically from the experiment plan

## Encoder Configuration

TODO: Number of stages, channels per stage, convolution kernel sizes

## Decoder Configuration

TODO: Upsampling method, skip connections, channel counts

## Normalization

TODO: Instance normalization vs batch normalization choices

## Deep Supervision

TODO: Analyze the deep supervision mechanism

## Parameter Count Scaling

TODO: How parameter count scales with the planned configuration

## Comparison with Standard U-Net

| Aspect | Standard U-Net | nnU-Net |
|--------|---------------|---------|
| Normalization | BatchNorm | TODO |
| Skip Connections | Concatenation | TODO |
| Depth | Fixed (4-5) | TODO |
| Kernel Sizes | 3x3 | TODO |
| Deep Supervision | No | TODO |
