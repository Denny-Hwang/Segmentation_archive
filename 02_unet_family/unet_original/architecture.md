---
title: "U-Net Architecture Details"
date: 2025-03-06
status: complete
tags: [u-net, encoder-decoder, skip-connections, architecture]
difficulty: beginner
---

# U-Net Architecture

## Overview

U-Net (Ronneberger et al., 2015) is a symmetric encoder-decoder architecture with skip connections. The architecture consists of a contracting path (encoder), a bottleneck, and an expanding path (decoder). The encoder progressively reduces spatial resolution while increasing feature channels, and the decoder reverses this while incorporating high-resolution features from the encoder via skip connections.

## Contracting Path (Encoder)

The encoder follows a standard CNN pattern with four downsampling blocks:

| Block | Input Size | Operations | Output Size |
|-------|-----------|------------|-------------|
| Block 1 | 572×572×1 | 2× (3×3 conv + ReLU) | 568×568×64 → pool → 284×284×64 |
| Block 2 | 284×284×64 | 2× (3×3 conv + ReLU) | 280×280×128 → pool → 140×140×128 |
| Block 3 | 140×140×128 | 2× (3×3 conv + ReLU) | 136×136×256 → pool → 68×68×256 |
| Block 4 | 68×68×256 | 2× (3×3 conv + ReLU) | 64×64×512 → pool → 32×32×512 |

Each block applies two 3×3 unpadded convolutions, each followed by ReLU activation. A 2×2 max pooling with stride 2 halves the spatial dimensions. Channel dimensions double at each block: 64 → 128 → 256 → 512.

## Bottleneck

The bottleneck consists of two 3×3 convolutions with 1024 channels at the lowest spatial resolution (32×32). This is where the most abstract, semantically rich features are computed. The bottleneck has the largest receptive field relative to the input image.

## Expanding Path (Decoder)

The decoder mirrors the encoder with four upsampling blocks:

| Block | Input | Up-conv | Concat Skip | 2× Conv | Output |
|-------|-------|---------|-------------|---------|--------|
| Block 5 | 32×32×1024 | 64×64×512 | 64×64×1024 | 2×(3×3) | 60×60×512 |
| Block 6 | 60×60×512 | 120×120×256 | 120×120×512 | 2×(3×3) | 116×116×256 |
| Block 7 | 116×116×256 | 232×232×128 | 232×232×256 | 2×(3×3) | 228×228×128 |
| Block 8 | 228×228×128 | 456×456×64 | 456×456×128 | 2×(3×3) | 452×452×64 |

Each block: (1) 2×2 up-convolution (transposed conv) halving channels; (2) center-crop and concatenate corresponding encoder features; (3) two 3×3 convolutions with ReLU.

## Final Layer

A 1×1 convolution maps the 64-channel feature map to the desired number of classes. In the original paper, 2 output channels were used for binary segmentation (cell vs background).

## Skip Connections

Skip connections are the defining innovation of U-Net. They concatenate encoder features at each resolution level with the corresponding decoder features. This allows the decoder to access both high-level semantic information (from the bottleneck path) and low-level spatial details (from the encoder). The center-cropping handles the size mismatch caused by unpadded convolutions.

## Parameter Count

The original U-Net has approximately 31 million parameters. Modern implementations typically use padded convolutions (padding=1) to maintain spatial dimensions, eliminating the need for center-cropping and producing output at the same resolution as the input.

## Key Design Choices

- **No padding**: Original U-Net uses valid (unpadded) convolutions, reducing spatial dimensions at each conv layer. Modern implementations use same padding.
- **Channel doubling**: Each encoder block doubles channels (64→128→256→512→1024), providing progressively richer feature representations.
- **Concatenation over addition**: Skip connections use concatenation rather than element-wise addition, preserving both encoder and decoder features without information loss.
