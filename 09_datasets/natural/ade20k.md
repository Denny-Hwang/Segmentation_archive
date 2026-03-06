---
id: ade20k
name: ADE20K
domain: natural
modality: rgb
task: semantic
classes: 150
size: "25,574 images"
license: BSD 3-Clause
---

# ADE20K

## Overview

| Field | Details |
|---|---|
| **Name** | ADE20K (MIT Scene Parsing Benchmark) |
| **Source** | [MIT CSAIL](https://groups.csail.mit.edu/vision/datasets/ADE20K/) |
| **Size** | 20,210 training + 2,000 validation + 3,352 testing images |
| **Classes** | 150 semantic categories |
| **Modality** | RGB natural images |
| **Common Use** | Semantic segmentation benchmarking, scene parsing |

## Description

ADE20K is a large-scale scene parsing dataset developed by MIT CSAIL. It covers a broad range of indoor and outdoor scenes with dense pixel-level annotations for 150 semantic categories. The dataset is notable for its diversity in scenes, object categories, and annotation density.

ADE20K serves as the primary benchmark for many semantic segmentation architectures, especially transformer-based methods. The standard evaluation metric is mean Intersection over Union (mIoU) on the validation set.

## Download Instructions

1. Visit the [ADE20K dataset page](https://groups.csail.mit.edu/vision/datasets/ADE20K/)
2. Download the dataset zip file
3. Alternatively, use MMSegmentation or HuggingFace Datasets:

```python
from datasets import load_dataset
dataset = load_dataset("scene_parse_150")
```

## Key Papers Using This Dataset

- **SegFormer** (Xie et al., 2021) - Efficient transformer segmentation
- **Mask2Former** (Cheng et al., 2022) - Universal image segmentation
- **OneFormer** (Jain et al., 2023) - Multi-task segmentation
- **BEiT** (Bao et al., 2021) - Pre-training for vision transformers
- **UPerNet** (Xiao et al., 2018) - Unified perceptual parsing
