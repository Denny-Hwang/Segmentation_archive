---
title: "RSU Block Analysis"
date: 2025-03-06
status: planned
tags:
  - RSU-block
  - nested-u-structure
  - multi-scale
parent: u2net/review.md
---

# RSU Block (Residual U-Block) Analysis

## Overview

_TODO: Explain the RSU-L-C_in-C_mid block -- a small U-Net with L levels, input channels C_in, and mid channels C_mid, plus a residual connection._

---

## RSU Block Architecture

### Structure

_TODO: Diagram showing the mini U-Net structure within each RSU block._

1. _TODO: Input convolution to map to mid channels_
2. _TODO: Encoder path with L-1 downsampling steps_
3. _TODO: Bottleneck_
4. _TODO: Decoder path with L-1 upsampling steps_
5. _TODO: Skip connections within the RSU block_
6. _TODO: Residual connection from input to output_

### RSU-L Variants

| Block | Levels (L) | Downsampling | Use Case |
|-------|-----------|-------------|----------|
| RSU-7 | 7 | 6 pooling ops | Shallow stages (high res) |
| RSU-6 | 6 | 5 pooling ops | _TODO_ |
| RSU-5 | 5 | 4 pooling ops | _TODO_ |
| RSU-4 | 4 | 3 pooling ops | _TODO_ |
| RSU-4F | 4 (dilated) | No pooling | Deep stages (low res) |

---

## RSU-4F: The Dilated Variant

_TODO: At the deepest stages, spatial resolution is too small for pooling -- RSU-4F uses dilated convolutions instead._

---

## Why RSU Over Plain Convolution Blocks?

_TODO: Comparison of receptive fields and multi-scale feature capture._

| Block Type | Multi-Scale? | Parameters | Receptive Field |
|-----------|-------------|------------|-----------------|
| Plain 2x conv | No | Baseline | Small |
| Inception-like | Yes | Higher | Medium |
| RSU block | Yes | Moderate | Large |

---

## Parameter Efficiency

_TODO: Analyze parameter count of RSU blocks vs alternative multi-scale blocks._

---

## Ablation Studies

_TODO: Document results from removing the residual connection, reducing L, etc._
