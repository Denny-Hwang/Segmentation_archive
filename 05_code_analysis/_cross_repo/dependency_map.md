---
title: "Cross-Repository Dependency Map"
date: 2025-01-15
status: complete
tags: [cross-repo, dependencies, compatibility, versions]
---

# Dependency Map

## Purpose

This document tracks shared dependencies across all analyzed repositories, highlighting version requirements and compatibility considerations.

## Core Dependencies

| Dependency | Pytorch-UNet | SMP | nnU-Net | SAM 2 | MMSeg | keras-unet |
|-----------|-------------|-----|---------|-------|-------|------------|
| Python | >=3.6 | >=3.8 | >=3.9 | >=3.10 | >=3.8 | >=3.6 |
| PyTorch | >=1.13 | >=1.9 | >=2.0 | >=2.3.1 | >=1.8 | N/A |
| TensorFlow | N/A | N/A | N/A | N/A | N/A | >=2.3 |
| torchvision | >=0.14 | >=0.10 | >=0.15 | >=0.18 | >=0.9 | N/A |
| numpy | >=1.20 | >=1.20 | >=1.22 | >=1.24 | >=1.20 | >=1.18 |
| OpenCV | Not required | Not required | Not required | >=4.7 | >=4.5 (mmcv) | Not required |
| Pillow | >=8.0 | >=8.0 | Not direct | >=9.4 | Via mmcv | >=7.0 |
| albumentations | Not included | Not required | Not direct | Not required | Not required | Not required |
| timm | N/A | >=0.9 | >=0.9 | >=0.9 | Optional | N/A |
| mmengine | N/A | N/A | N/A | N/A | >=0.7.1 | N/A |
| mmcv | N/A | N/A | N/A | N/A | >=2.0 | N/A |

### Additional Repository-Specific Dependencies

**Pytorch-UNet**: `wandb` (logging), `tqdm` (progress bars). Both are soft dependencies -- `wandb` can be disabled, `tqdm` is used for training progress display.

**SMP**: `efficientnet-pytorch` (for EfficientNet encoders when not using timm), `pretrainedmodels` (for legacy encoder support). The `timm` integration is the modern recommended path and subsumes both.

**nnU-Net**: `batchgenerators` (data augmentation library from MIC-DKFZ), `SimpleITK` (medical image I/O for NIfTI, NRRD, MHA formats), `scipy` (connected component analysis in postprocessing), `scikit-learn` (optional, for some evaluation metrics), `acvl-utils` (utility functions from the same research group). The `batchgenerators` dependency is critical and not interchangeable with other augmentation libraries.

**SAM 2**: `hydra-core` and `omegaconf` (configuration management), `iopath` (I/O path abstraction), `tqdm`. SAM 2 has notably few dependencies, reflecting its focus on inference rather than training pipelines.

**MMSegmentation**: `mmengine` (the OpenMMLab engine), `mmcv` (the OpenMMLab computer vision library). These two dependencies are substantial and have their own complex dependency trees. `mmcv` includes custom CUDA operators and requires a specific build matching the PyTorch and CUDA versions.

**keras-unet-collection**: `tensorflow` (>=2.3), `keras` (bundled with TF 2.x). Optionally uses `patchify` for patch-based processing. Minimal dependency footprint.

## Version Compatibility Notes

The most challenging dependency conflict across these repositories is the **PyTorch version requirement**. SAM 2 requires PyTorch >=2.3.1 (for `torch.compile` and latest `nn.Module` features), while MMSegmentation's `mmcv` may not have prebuilt binaries for the latest PyTorch version. When `mmcv` lags behind PyTorch releases, users must either build `mmcv` from source or wait for an updated release. This conflict makes running SAM 2 and MMSegmentation in the same environment difficult.

The **timm** library is used by SMP, nnU-Net (v2), and SAM 2. Version mismatches can cause model loading failures because `timm` occasionally renames or restructures model classes between minor versions. Pin `timm` to a specific version when using multiple repositories: `timm==0.9.12` is a widely compatible choice.

**numpy** 2.0 (released June 2024) introduced breaking changes that affected several repositories. nnU-Net and SAM 2 updated to support numpy 2.0, but older versions of SMP and MMSegmentation may fail with numpy 2.0. If encountering `AttributeError` related to numpy dtypes, pin `numpy<2.0`.

**CUDA compatibility** is a cross-cutting concern. Each PyTorch version supports specific CUDA versions, and `mmcv` custom ops must be compiled against the same CUDA version. The safest approach is to install PyTorch via the official CUDA-specific channels (`pytorch-cuda=12.1`) and then install `mmcv` from the matching prebuilt wheel.

**Python 3.12+** may cause issues with older versions of `mmengine` and `batchgenerators` due to removed deprecated features. nnU-Net v2 officially supports Python 3.9-3.11. SAM 2 supports Python 3.10+.

## GPU Requirements

| Repository | Min VRAM (Training) | Min VRAM (Inference) | Notes |
|-----------|-------------------|---------------------|-------|
| Pytorch-UNet | 4 GB (scale=0.5, batch=1) | 2 GB | Full resolution (scale=1.0) needs 8-12 GB |
| SMP | 4-8 GB | 2-4 GB | Depends on encoder (ResNet34: 4GB, EfficientNet-B7: 12GB) |
| nnU-Net | 8 GB (default target) | 4 GB | Planner adapts to GPU; 3D_fullres most demanding |
| SAM 2 | 16 GB (fine-tuning) | 6 GB (Hiera-B+) | Large model needs A100 for training; inference on consumer GPUs |
| MMSegmentation | 8-16 GB | 2-8 GB | Varies widely by model; Swin-L + UPerHead needs 16GB |

nnU-Net is notable for automatically adapting to available GPU memory through its planning algorithm. The 8GB target can be overridden with `--gpu_memory_target` to utilize larger GPUs. For Pytorch-UNet, the `--scale` parameter is the primary memory control: scale=0.5 halves each dimension, reducing memory by ~4x compared to scale=1.0.

SAM 2 inference with the Hiera-B+ model requires approximately 6GB VRAM for single-image processing and 8-10GB for video processing (due to the memory bank). The Hiera-T (Tiny) variant can run on 4GB GPUs for inference.

## Shared Virtual Environment Strategy

Running all repositories in a single Python environment is **not recommended** due to incompatible version requirements. The primary conflicts are:

1. **SAM 2 requires PyTorch >=2.3.1** while mmcv prebuilt binaries may not support this version
2. **MMSegmentation requires mmcv and mmengine** which are large, opinionated packages that may conflict with other dependencies
3. **keras-unet-collection requires TensorFlow** which conflicts with PyTorch's CUDA allocator on GPU (they compete for GPU memory and CUDA context)
4. **numpy 2.0 compatibility** varies across repositories

The recommended approach is to use **separate virtual environments** grouped by compatibility:

```bash
# Environment 1: PyTorch-based (SAM 2, nnU-Net, SMP, Pytorch-UNet)
conda create -n seg_pytorch python=3.11
pip install torch==2.3.1 torchvision==0.18.1
pip install segmentation-models-pytorch nnunetv2 sam2

# Environment 2: MMSegmentation
conda create -n seg_mmseg python=3.10
pip install torch==2.1.0 torchvision==0.16.0
pip install mmengine mmcv==2.1.0 mmsegmentation

# Environment 3: TensorFlow (keras-unet-collection)
conda create -n seg_keras python=3.10
pip install tensorflow==2.15 keras-unet-collection
```

If a single environment is necessary (e.g., for comparison experiments), the most compatible combination is **Pytorch-UNet + SMP + nnU-Net** (all pure PyTorch, no framework conflicts). SAM 2 can often be added to this environment if the PyTorch version is >=2.3.1. MMSegmentation and keras-unet-collection should always be in separate environments.

For CI/CD or automated benchmarking, use Docker containers with separate images for each environment group. This provides full isolation and reproducibility.
