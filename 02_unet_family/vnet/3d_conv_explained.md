---
title: "3D Convolutions Explained"
date: 2025-03-06
status: planned
tags:
  - 3d-convolution
  - volumetric
  - kernels
parent: vnet/review.md
---

# 3D Convolutions Explained

## Overview

_TODO: Explain the transition from 2D to 3D convolutional operations and why they are necessary for volumetric medical data._

---

## 2D vs 3D Convolutions

### 2D Convolution Recap

_TODO: Kernel of size (k, k) slides over (H, W) spatial dimensions._

### 3D Convolution

_TODO: Kernel of size (k, k, k) slides over (D, H, W) spatial dimensions. Describe the parameter increase._

---

## Parameter Comparison

| Kernel | 2D Parameters | 3D Parameters | Ratio |
|--------|--------------|--------------|-------|
| 3x3 vs 3x3x3 | 9 * C_in * C_out | 27 * C_in * C_out | 3x |
| 5x5 vs 5x5x5 | 25 * C_in * C_out | 125 * C_in * C_out | 5x |

---

## Memory and Compute Implications

_TODO: Discuss the cubic growth of memory and computation with 3D kernels._

---

## 3D Pooling and Upsampling

_TODO: 3D max pooling, 3D transposed convolutions, trilinear interpolation._

---

## Practical Considerations

- _TODO: Patch-based training for GPU memory constraints_
- _TODO: Anisotropic vs isotropic voxel spacing_
- _TODO: 2.5D approaches as a compromise_

---

## How V-Net Uses 3D Convolutions

_TODO: Specific kernel sizes, stride patterns, and channel progression used in V-Net._
