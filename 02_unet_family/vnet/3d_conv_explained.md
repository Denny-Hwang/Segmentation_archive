---
title: "3D Convolutions in V-Net"
date: 2025-03-06
status: complete
tags: [3d-convolution, volumetric, computational-cost, v-net]
difficulty: intermediate
---

# 3D Convolutions Explained

## Overview

3D convolutions extend standard 2D convolutions to process volumetric data (depth × height × width). They are essential for medical imaging where spatial context along the z-axis (slice direction) carries important anatomical information. V-Net was one of the first architectures to use 3D convolutions throughout the entire network.

## Kernel Dimensions

A 3D convolution kernel has shape (D×H×W) where D is the depth dimension. V-Net uses 5×5×5 kernels, which means each output voxel is computed from a 5×5×5 neighborhood in the input volume. The kernel slides in three spatial dimensions:

- **2D convolution**: kernel = K×K, slides over (H, W)
- **3D convolution**: kernel = K×K×K, slides over (D, H, W)

## Computational Cost

3D convolutions are significantly more expensive than 2D:

| Operation | Parameters | FLOPs per output voxel |
|-----------|-----------|----------------------|
| 2D conv (3×3, C_in→C_out) | 9 · C_in · C_out | 9 · C_in |
| 3D conv (3×3×3, C_in→C_out) | 27 · C_in · C_out | 27 · C_in |
| 3D conv (5×5×5, C_in→C_out) | 125 · C_in · C_out | 125 · C_in |

A 3×3×3 kernel has 3× the parameters and FLOPs of a 3×3 kernel. A 5×5×5 kernel (V-Net) has ~14× the cost. This makes 3D networks significantly more memory and compute intensive, typically requiring smaller batch sizes and lower-resolution inputs.

## 3D Batch Normalization

Standard batch normalization is extended to 3D by computing mean and variance across the batch and all three spatial dimensions (D, H, W). The normalization parameters are per-channel, same as 2D. PyTorch: `nn.BatchNorm3d(num_features)`.

## 3D Pooling and Upsampling

- **3D Max Pooling**: `nn.MaxPool3d(2)` reduces each dimension by 2×, so a 64×64×32 volume becomes 32×32×16
- **3D Transposed Convolution**: `nn.ConvTranspose3d(C_in, C_out, 2, stride=2)` doubles each spatial dimension
- **3D Trilinear Interpolation**: `F.interpolate(x, scale_factor=2, mode='trilinear')` for smooth upsampling

## Memory Considerations

A single 3D feature map with C=64 channels at 128×128×64 resolution requires 64×128×128×64×4 bytes ≈ 256 MB in float32. This makes 3D networks memory-constrained — typical training uses batch size 1-2 with gradient accumulation or mixed precision. Techniques like gradient checkpointing and patch-based training are commonly used.

## 2D vs 3D Trade-offs

| Aspect | 2D (slice-by-slice) | 3D (volumetric) |
|--------|---------------------|-----------------|
| Z-axis context | None | Full |
| Memory | Low | High (~8-16×) |
| Training speed | Fast | Slow |
| Batch size | Large (16-32) | Small (1-2) |
| Best for | Thick slices, 2D images | Thin slices, isotropic |

nnU-Net automatically decides between 2D and 3D processing based on dataset properties (spacing, memory constraints).
