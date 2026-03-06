---
title: "nnU-Net Repository Overview"
date: 2025-01-15
status: planned
repo_url: "https://github.com/MIC-DKFZ/nnUNet"
framework: PyTorch
tags: [nnunet, medical-segmentation, self-configuring, pytorch]
---

# nnU-Net (MIC-DKFZ/nnUNet)

## Repository Summary

| Field | Details |
|-------|---------|
| URL | https://github.com/MIC-DKFZ/nnUNet |
| License | Apache-2.0 |
| Framework | PyTorch |
| Primary Use Case | Medical image segmentation (2D, 3D) |
| Key Strength | Self-configuring pipeline -- automatically adapts to any dataset |

## Why This Repository

nnU-Net is the dominant framework for medical image segmentation, consistently winning or placing highly in segmentation challenges. Its key innovation is the automatic configuration of preprocessing, architecture, and postprocessing based on dataset properties.

## Repository Structure (v2)

```
nnunetv2/
├── experiment_planning/
│   ├── plan_and_preprocess_api.py
│   ├── experiment_planners/
│   └── dataset_fingerprint/
├── preprocessing/
│   ├── preprocessors/
│   └── resampling/
├── training/
│   ├── nnUNetTrainer/
│   ├── loss/
│   └── lr_scheduler/
├── architecture/
│   ├── PlainConvUNet.py
│   └── ResidualEncoderUNet.py
├── postprocessing/
├── inference/
├── evaluation/
└── utilities/
```

## Key Concepts

- **Dataset fingerprint**: Automatic analysis of dataset statistics
- **Experiment planner**: Determines architecture, patch size, batch size from fingerprint
- **Three configurations**: 2D, 3D full-resolution, 3D low-resolution + 3D cascade
- **Postprocessing**: Automatic selection of optimal postprocessing strategy

## Analysis Files

| File | Description | Status |
|------|-------------|--------|
| `experiment_planner.md` | How nnU-Net automatically configures experiments | Planned |
| `network_architecture.md` | Architecture details (PlainConvUNet, ResidualEncoderUNet) | Planned |
| `preprocessing_pipeline.md` | Automated preprocessing pipeline | Planned |
| `postprocessing.md` | Postprocessing and ensembling | Planned |
