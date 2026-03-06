---
title: "Experiment: U-Net Variants Comparison"
date: 2025-01-15
status: planned
tags: [experiment, unet, unet++, attention-unet, comparison]
---

# U-Net Variants Comparison Experiment

## Objective

Compare multiple U-Net variants under identical training conditions to measure the impact of architectural changes (nested skip connections, attention gates, etc.) on segmentation quality.

## Models Compared

| Model | Key Modification | Reference |
|-------|-----------------|-----------|
| U-Net | Baseline encoder-decoder | Ronneberger 2015 |
| U-Net++ | Nested, dense skip connections | Zhou 2018 |
| Attention U-Net | Attention gates on skip connections | Oktay 2018 |
| U-Net + ResNet34 encoder | Pretrained encoder backbone | - |

## Dataset

- Same dataset and splits as the U-Net baseline experiment
- Identical augmentation pipeline across all models

## Controlled Variables

- Same optimizer (Adam, lr=1e-4)
- Same batch size (8)
- Same loss function (BCE + Dice)
- Same number of training epochs (100)
- Same random seed (42)

## Evaluation

### Metrics
- IoU, Dice, Pixel Accuracy (per model)
- Parameter count, FLOPs, inference time

### Analysis
- Statistical significance testing (paired t-test on fold results)
- Qualitative comparison of segmentation outputs on difficult cases

## Expected Outcome

TODO: Fill in after running experiments

## How to Run

```bash
python run_comparison.py --config config.yaml
```

## Files

| File | Description |
|------|-------------|
| `config.yaml` | Comparison experiment configuration (TODO) |
| `run_comparison.py` | Script to train all variants sequentially (TODO) |
| `analyze_results.py` | Results analysis and visualization (TODO) |
