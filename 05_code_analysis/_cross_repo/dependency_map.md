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
| Python | TODO | TODO | TODO | TODO | TODO | TODO |
| PyTorch | TODO | TODO | TODO | TODO | TODO | N/A |
| TensorFlow | N/A | N/A | N/A | N/A | N/A | TODO |
| torchvision | TODO | TODO | TODO | TODO | TODO | N/A |
| numpy | TODO | TODO | TODO | TODO | TODO | TODO |
| OpenCV | TODO | TODO | TODO | TODO | TODO | TODO |
| Pillow | TODO | TODO | TODO | TODO | TODO | TODO |
| albumentations | TODO | TODO | TODO | TODO | TODO | TODO |
| timm | N/A | TODO | TODO | TODO | TODO | N/A |
| mmengine | N/A | N/A | N/A | N/A | TODO | N/A |
| mmcv | N/A | N/A | N/A | N/A | TODO | N/A |

## Version Compatibility Notes

TODO: Document known version conflicts and resolution strategies

## GPU Requirements

| Repository | Min VRAM (Training) | Min VRAM (Inference) | Notes |
|-----------|-------------------|---------------------|-------|
| Pytorch-UNet | TODO | TODO | |
| SMP | TODO | TODO | |
| nnU-Net | TODO | TODO | |
| SAM 2 | TODO | TODO | |
| MMSegmentation | TODO | TODO | |

## Shared Virtual Environment Strategy

TODO: Whether it is feasible to run all repos in a single environment, or if separate environments are needed
