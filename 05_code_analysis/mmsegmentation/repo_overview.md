---
title: "MMSegmentation Repository Overview"
date: 2025-01-15
status: planned
repo_url: "https://github.com/open-mmlab/mmsegmentation"
framework: "PyTorch (MMEngine)"
tags: [mmsegmentation, openmmlab, config-system, segmentation-toolbox]
---

# MMSegmentation (open-mmlab/mmsegmentation)

## Repository Summary

| Field | Details |
|-------|---------|
| URL | https://github.com/open-mmlab/mmsegmentation |
| License | Apache-2.0 |
| Framework | PyTorch via MMEngine/MMCV |
| Primary Use Case | Comprehensive semantic segmentation toolbox |
| Key Strength | 40+ models, config-driven experimentation, extensive benchmark support |

## Why This Repository

MMSegmentation is the most comprehensive segmentation toolbox, supporting virtually every major architecture (FCN, PSPNet, DeepLab, U-Net, SegFormer, Mask2Former, etc.) with standardized training, evaluation, and benchmarking.

## Repository Structure

```
mmseg/
├── configs/                    # Config files for all models
│   ├── _base_/                 # Base configs (datasets, schedules, models)
│   ├── deeplabv3/
│   ├── segformer/
│   └── ...
├── mmseg/
│   ├── models/
│   │   ├── backbones/          # Encoder networks
│   │   ├── decode_heads/       # Decoder heads
│   │   ├── segmentors/         # Full model wrappers
│   │   └── losses/             # Loss functions
│   ├── datasets/               # Dataset classes
│   ├── engine/                 # Training hooks and optimizers
│   └── evaluation/             # Metrics
└── tools/
    ├── train.py
    └── test.py
```

## Key Design Patterns

- **Config inheritance system**: Configs inherit from base configs
- **Registry pattern**: Models, datasets, and transforms registered via decorators
- **Runner abstraction**: MMEngine runner handles the training loop

## Analysis Files

| File | Description | Status |
|------|-------------|--------|
| `config_system.md` | Config inheritance and override system | Planned |
| `model_registry.md` | How models are registered and built | Planned |
