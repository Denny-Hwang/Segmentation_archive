---
title: "UNet 3+: A Full-Scale Connected UNet for Medical Image Segmentation"
date: 2025-03-06
status: planned
tags:
  - full-scale-skip
  - multi-scale-features
  - deep-supervision
  - classification-guided
difficulty: intermediate
---

# UNet 3+: A Full-Scale Connected UNet for Medical Image Segmentation

## Meta Information

| Field          | Details |
|----------------|---------|
| **Paper Title**   | UNet 3+: A Full-Scale Connected UNet for Medical Image Segmentation |
| **Authors**       | Huimin Huang, Lanfen Lin, Ruofeng Tong, Hongjie Hu, Qiaowei Zhang, Yutaro Iwamoto, Xianhua Han, Yen-Wei Chen, Jian Wu |
| **Year**          | 2020 |
| **Venue**         | ICASSP 2020 |
| **ArXiv ID**      | [2004.08790](https://arxiv.org/abs/2004.08790) |

## One-Line Summary

UNet 3+ introduces full-scale skip connections that combine features from all encoder and decoder levels at each decoder node, along with classification-guided module and deep supervision to reduce false positives in organ segmentation.

---

## Motivation and Problem Statement

_TODO: Describe the limitations of U-Net (same-scale skips) and UNet++ (nested but still local) in capturing full-scale multi-resolution context._

---

## Key Contributions

- _TODO: Full-scale skip connections from all encoder/decoder levels_
- _TODO: Classification-guided module to suppress non-organ predictions_
- _TODO: Full-scale deep supervision_
- _TODO: Fewer parameters than UNet++_

---

## Architecture Overview

_TODO: Each decoder node aggregates features from ALL encoder levels AND all preceding decoder levels. Reference [full_scale_skip.md](./full_scale_skip.md)._

---

## Method Details

### Full-Scale Skip Connections

_TODO: Reference [full_scale_skip.md](./full_scale_skip.md)._

### Classification-Guided Module

_TODO: Binary classification head that predicts whether the target organ exists in the image._

### Deep Supervision

_TODO: Supervision at each decoder level with the classification guidance._

---

## Experimental Results

| Dataset | Metric | UNet 3+ | UNet++ | U-Net |
|---------|--------|---------|--------|-------|
| Liver (LiTS) | Dice | _TODO_ | _TODO_ | _TODO_ |
| Spleen | Dice | _TODO_ | _TODO_ | _TODO_ |

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
| U-Net (Ronneberger et al., 2015) | Same-scale skip connections |
| UNet++ (Zhou et al., 2018) | Nested dense skip connections |
| FPN (Lin et al., 2017) | Multi-scale feature pyramid concept |

---

## Open Questions

- _TODO_
