---
title: "Experiments - Environment Setup and Overview"
date: 2025-01-15
status: in-progress
description: "Practical segmentation experiments with reproducible configurations"
---

# Experiments

## Purpose

This section contains practical, reproducible experiments for image segmentation. Each experiment directory includes a description, configuration, and instructions for reproduction.

## Environment Setup

### 1. Create Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Verify GPU Availability

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
```

### 4. Recommended Hardware

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU VRAM | 8 GB | 16+ GB |
| RAM | 16 GB | 32+ GB |
| Storage | 50 GB | 200+ GB (for datasets) |

## Experiment Index

| Directory | Description | Status |
|-----------|-------------|--------|
| `unet_baseline/` | U-Net baseline on a standard dataset | Planned |
| `unet_variants_comparison/` | Compare U-Net, U-Net++, Attention U-Net | Planned |
| `transformer_vs_cnn/` | SegFormer vs DeepLabV3+ comparison | Planned |
| `sam2_finetuning/` | Fine-tuning SAM 2 on a custom dataset | Planned |

## Common Utilities

The `_common/` directory provides shared code used across experiments:

| File | Description |
|------|-------------|
| `metrics.py` | IoU, Dice coefficient, pixel accuracy |
| `visualization.py` | Mask overlay, training curve plotting |
| `augmentation.py` | Standard augmentation pipelines via albumentations |
| `callbacks.py` | Early stopping, model checkpoint, logging callbacks |

## Experiment Conventions

1. Each experiment has its own `README.md` with goals, methodology, and expected results
2. Configurations are stored in `config.yaml` files
3. Results are logged to TensorBoard (`runs/` directory, gitignored)
4. Model checkpoints are saved to `checkpoints/` (gitignored)
5. All experiments use the common utilities from `_common/`

## Running an Experiment

```bash
cd 06_experiments/<experiment_name>
python train.py --config config.yaml
```
