---
title: "Cross-Repository Dependency Map"
date: 2025-01-15
status: planned
tags: [cross-repo, dependencies, compatibility, versions]
---

# Dependency Map

## Purpose

This document tracks shared dependencies across all analyzed repositories, highlighting version requirements and compatibility considerations.

## Core Dependencies

| Dependency | Pytorch-UNet | SMP | nnU-Net | SAM 2 | MMSeg | keras-unet |
|-----------|-------------|-----|---------|-------|-------|------------|
| Python | >=3.8 | >=3.8 | >=3.9 | >=3.10 | >=3.8 | >=3.6 |
| PyTorch | >=1.13 | >=1.9 | >=2.0 | >=2.3.1 | >=1.8 | N/A |
| TensorFlow | N/A | N/A | N/A | N/A | N/A | >=2.6 |
| torchvision | >=0.14 | >=0.10 | >=0.15 | >=0.18 | >=0.9 | N/A |
| numpy | >=1.21 | >=1.20 | >=1.22 | >=1.24 | >=1.20 | >=1.19 |
| OpenCV | Not required | Not required | Not required | >=4.7 | >=4.5 (mmcv dep) | Not required |
| Pillow | >=8.0 | >=8.0 | Not required | >=9.0 | >=8.0 | Not required |
| albumentations | Not included | Recommended | Not used | Not used | Not used | Not used |
| timm | N/A | >=0.9 (optional) | Not required | >=0.9 | >=0.9 (some backbones) | N/A |
| mmengine | N/A | N/A | N/A | N/A | >=0.7.0 | N/A |
| mmcv | N/A | N/A | N/A | N/A | >=2.0.0 | N/A |
| scipy | Not required | Not required | >=1.9 | Not required | Not required | Not required |
| batchgenerators | N/A | N/A | >=0.25 | N/A | N/A | N/A |
| wandb | >=0.12 | Not required | Not required | Not required | Not required | Not required |
| tqdm | >=4.60 | Not required | >=4.60 | Not required | Not required | Not required |
| SimpleITK | N/A | N/A | >=2.2 | N/A | N/A | N/A |
| hydra-core | N/A | N/A | N/A | >=1.3 | N/A | N/A |

## Version Compatibility Notes

**PyTorch version conflicts**:
- SAM 2 requires PyTorch >= 2.3.1 with CUDA 12.1+ for its custom CUDA kernels. This is the most restrictive requirement and may conflict with older CUDA drivers.
- nnU-Net v2 requires PyTorch >= 2.0 for `torch.compile()` support (optional but recommended).
- MMSegmentation requires specific mmcv versions that must match the PyTorch version exactly. Use the [mmcv compatibility table](https://mmcv.readthedocs.io/en/latest/get_started/installation.html) to find the correct mmcv version.

**NumPy version conflicts**:
- NumPy 2.0+ breaks backward compatibility with some older packages. nnU-Net and SAM 2 generally work with NumPy 2.0, but `batchgenerators` (nnU-Net dependency) may require NumPy <2.0 in older versions.

**TensorFlow vs PyTorch**:
- `keras-unet-collection` is the only TensorFlow-based repo. It cannot share a CUDA context with PyTorch repos on the same GPU simultaneously (though they can coexist in the same environment).

**mmcv/mmengine pinning**:
- MMSegmentation has strict version pinning: `mmengine >= 0.7.0`, `mmcv >= 2.0.0`. These must be installed from the OpenMMLab channel, not PyPI generic packages. Use: `pip install -U openmim && mim install mmengine mmcv`.

**timm versioning**:
- SMP and MMSegmentation use `timm` for additional encoder backbones. Breaking API changes between timm 0.x and 1.x can cause import errors. SMP's `timm` encoder adapter handles both versions, but older SMP versions may not.

## GPU Requirements

| Repository | Min VRAM (Training) | Min VRAM (Inference) | Notes |
|-----------|-------------------|---------------------|-------|
| Pytorch-UNet | 4 GB (scale=0.5) / 8 GB (scale=1.0) | 2 GB | Scales with input resolution and batch size |
| SMP | 4-8 GB (ResNet encoder) / 12-16 GB (EfficientNet-B7) | 2-4 GB | Encoder choice dominates VRAM |
| nnU-Net | 8-12 GB (auto-configured) | 4-8 GB | Plans target ~8 GB reference GPU; uses patch-based training |
| SAM 2 | 16-24 GB (Hiera-B+) / 32+ GB (Hiera-L) | 6-8 GB (image) / 8-12 GB (video) | Video mode adds memory bank overhead |
| MMSegmentation | 8-16 GB (varies by model) | 4-8 GB | Slide inference for large images; SyncBN requires multi-GPU |

**Memory-saving techniques**:
- AMP/mixed precision: Reduces training VRAM by ~30-40% (Pytorch-UNet, nnU-Net, SAM 2)
- Gradient checkpointing: Trades compute for memory (available in SMP, nnU-Net)
- Patch-based training: nnU-Net trains on patches, not full images, bounding memory to patch size
- Sliding window inference: MMSegmentation, nnU-Net process large images in overlapping tiles

## Shared Virtual Environment Strategy

**Feasibility: Partial -- separate environments recommended for full compatibility.**

A shared environment is possible for a subset of repos, but not all six:

| Group | Repos | Python | PyTorch | Compatible? |
|-------|-------|--------|---------|-------------|
| Group A | Pytorch-UNet + SMP + nnU-Net | 3.10 | 2.3+ | Yes (no conflicts) |
| Group B | SAM 2 | 3.10 | 2.3.1+ CUDA 12.1 | Yes (can share with Group A if CUDA matches) |
| Group C | MMSegmentation | 3.10 | 2.1+ | Requires mmcv/mmengine; can coexist with Group A/B |
| Group D | keras-unet-collection | 3.10 | N/A (TensorFlow) | Must be separate or use `tf` + `torch` side-by-side |

**Recommended setup**:

```bash
# Environment 1: PyTorch-based (covers 5 repos)
conda create -n seg-pytorch python=3.10
conda activate seg-pytorch
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
pip install segmentation-models-pytorch nnunetv2
pip install -e ./sam2  # SAM 2 from source
pip install -U openmim && mim install mmengine mmcv  # For MMSeg
pip install -e ./mmsegmentation

# Environment 2: TensorFlow (keras-unet-collection)
conda create -n seg-keras python=3.10
conda activate seg-keras
pip install tensorflow keras-unet-collection
```

**Key conflict**: TensorFlow and PyTorch can coexist in one environment, but they compete for GPU memory. If you only need inference (not training) from keras-unet-collection, installing both in one environment works. For training, use separate environments to avoid GPU memory contention.

**Docker alternative**: Use separate Docker containers per repo for full isolation. This is the most reliable approach for reproducibility and avoids all dependency conflicts. nnU-Net and MMSegmentation both provide official Docker images.
