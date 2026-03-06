---
title: "Road Extraction by Deep Residual U-Net"
date: 2025-03-06
status: planned
tags:
  - residual-learning
  - road-extraction
  - remote-sensing
difficulty: intermediate
---

# Road Extraction by Deep Residual U-Net (ResUNet)

## Meta Information

| Field          | Details |
|----------------|---------|
| **Paper Title**   | Road Extraction by Deep Residual U-Net |
| **Authors**       | Zhengxin Zhang, Qingjie Liu, Yunhong Wang |
| **Year**          | 2018 |
| **Venue**         | IEEE Geoscience and Remote Sensing Letters |
| **ArXiv ID**      | [1711.10684](https://arxiv.org/abs/1711.10684) |

## One-Line Summary

ResUNet combines the U-Net encoder-decoder architecture with residual learning by replacing standard convolution blocks with residual units, improving gradient flow and enabling deeper networks for road extraction from remote sensing imagery.

---

## Motivation and Problem Statement

_TODO: Describe the challenges of road extraction (thin, elongated structures) and how deeper networks with residual learning can help._

---

## Key Contributions

- _TODO: Integration of residual blocks into U-Net_
- _TODO: Batch normalization before activation_
- _TODO: Application to road extraction from aerial imagery_

---

## Architecture Overview

_TODO: U-Net with each double-convolution block replaced by a residual unit. Reference [residual_vs_plain_comparison.md](./residual_vs_plain_comparison.md)._

---

## Method Details

### Residual Unit Design

_TODO: BN -> ReLU -> Conv -> BN -> ReLU -> Conv with identity shortcut._

### Encoder Path

_TODO: Residual units with strided convolution for downsampling._

### Decoder Path

_TODO: Upsampling followed by residual units._

---

## Experimental Results

| Dataset | Metric | ResUNet | U-Net |
|---------|--------|---------|-------|
| Massachusetts Roads | _TODO_ | _TODO_ | _TODO_ |

---

## Strengths

- _TODO_

---

## Weaknesses and Limitations

- _TODO_

---

## Connections to Other Work

| Related Paper | Relationship |
|---------------|-------------|
| U-Net (Ronneberger et al., 2015) | Base architecture |
| ResNet (He et al., 2016) | Residual learning framework |
| R2U-Net (Alom et al., 2018) | Combines recurrent + residual with U-Net |

---

## Open Questions

- _TODO_
