---
title: "Experiment: U-Net Baseline"
date: 2025-01-15
status: planned
tags: [experiment, unet, baseline, semantic-segmentation]
---

# U-Net Baseline Experiment

## Objective

Establish a reproducible U-Net baseline on a standard segmentation benchmark. This serves as the reference point for all subsequent experiments.

## Dataset

- **Name**: Oxford-IIIT Pet Dataset (or DRIVE retinal vessels -- choose one)
- **Task**: Binary segmentation (foreground/background)
- **Split**: 80% train / 10% validation / 10% test
- **Preprocessing**: Resize to 256x256, normalize to [0, 1]

## Model

- **Architecture**: U-Net (encoder-decoder with skip connections)
- **Encoder depth**: 4 downsampling stages
- **Base filters**: 64 (doubling at each stage)
- **Upsampling**: Bilinear interpolation + 1x1 conv

## Training Configuration

See `config.yaml` for full configuration. Key settings:

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Learning Rate | 1e-4 |
| Batch Size | 8 |
| Epochs | 100 |
| Loss | BCE + Dice |
| Scheduler | ReduceLROnPlateau |
| Early Stopping | Patience 15 epochs |

## Evaluation Metrics

- IoU (Intersection over Union)
- Dice Coefficient
- Pixel Accuracy

## Expected Results

| Metric | Target |
|--------|--------|
| IoU | > 0.75 |
| Dice | > 0.85 |
| Pixel Accuracy | > 0.90 |

## How to Run

```bash
# From this directory
python train.py --config config.yaml

# Monitor training
tensorboard --logdir runs/
```

## Files

| File | Description |
|------|-------------|
| `config.yaml` | Experiment configuration |
| `train.py` | Training script (TODO) |
| `evaluate.py` | Evaluation script (TODO) |
