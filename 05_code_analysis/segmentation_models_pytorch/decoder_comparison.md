---
title: "SMP - Decoder Comparison"
date: 2025-01-15
status: planned
parent: "segmentation_models_pytorch/repo_overview.md"
tags: [smp, decoder, unet, fpn, deeplabv3, comparison]
---

# SMP Decoder Comparison

## Available Decoders

| Decoder | Paper | Skip Connections | Multi-Scale | Notes |
|---------|-------|-----------------|-------------|-------|
| U-Net | Ronneberger 2015 | Yes (concat) | TODO | |
| U-Net++ | Zhou 2018 | Yes (nested) | TODO | |
| FPN | Lin 2017 | Yes (add) | TODO | |
| DeepLabV3 | Chen 2017 | No | TODO (ASPP) | |
| DeepLabV3+ | Chen 2018 | Partial | TODO | |
| PAN | Li 2018 | TODO | TODO | |
| PSPNet | Zhao 2017 | No | TODO (PPM) | |
| Linknet | Chaurasia 2017 | Yes (add) | TODO | |
| MAnet | Fan 2020 | Yes (attention) | TODO | |

## Decoder Interface

TODO: Analyze the common decoder interface defined in `base/model.py`

## Feature Aggregation Strategies

### Concatenation-Based (U-Net)
TODO: Analyze implementation

### Addition-Based (FPN, LinkNet)
TODO: Analyze implementation

### Attention-Based (MAnet)
TODO: Analyze implementation

## Output Head

TODO: Analyze `SegmentationHead` and `ClassificationHead`

## Performance Comparison

TODO: Compare parameter counts and FLOPs for each decoder with the same encoder
