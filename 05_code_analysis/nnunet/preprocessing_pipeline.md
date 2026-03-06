---
title: "nnU-Net - Preprocessing Pipeline"
date: 2025-01-15
status: planned
parent: "nnunet/repo_overview.md"
tags: [nnunet, preprocessing, resampling, normalization]
---

# nnU-Net Preprocessing Pipeline

## Pipeline Overview

```
Raw Data (any format)
    │
    ├── Dataset conversion to nnU-Net format
    ├── Fingerprint extraction
    ├── Resampling to target spacing
    ├── Intensity normalization
    ├── Cropping to non-zero region
    │
    └── Preprocessed .npy / .npz files
```

## Resampling

### Spacing Determination
TODO: How target spacing is chosen

### Interpolation Methods
TODO: What interpolation is used for images vs labels

## Normalization

### Modality-Specific Normalization
TODO: CT (clipping + z-score) vs MRI (z-score per image) vs other

### Normalization Parameters
TODO: Where normalization parameters are stored

## Cropping

TODO: How non-zero region cropping works

## Data Format

### Input Format
TODO: Document expected input file structure

### Output Format
TODO: Document preprocessed data format

## Custom Preprocessing

TODO: How to add custom preprocessing steps
