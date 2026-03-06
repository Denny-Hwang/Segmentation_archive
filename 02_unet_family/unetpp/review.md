---
title: "UNet++: A Nested U-Net Architecture for Medical Image Segmentation"
date: 2025-03-06
status: planned
tags:
  - nested-architecture
  - dense-skip-connections
  - deep-supervision
  - pruning
difficulty: intermediate
---

# UNet++: A Nested U-Net Architecture for Medical Image Segmentation

## Meta Information

| Field          | Details |
|----------------|---------|
| **Paper Title**   | UNet++: A Nested U-Net Architecture for Medical Image Segmentation |
| **Authors**       | Zongwei Zhou, Md Mahfuzur Rahman Siddiquee, Nima Tajbakhsh, Jianming Liang |
| **Year**          | 2018 |
| **Venue**         | DLMIA 2018 / MICCAI Workshop |
| **ArXiv ID**      | [1807.10165](https://arxiv.org/abs/1807.10165) |

## One-Line Summary

UNet++ redesigns the skip connections of U-Net as nested, dense convolutional blocks that bridge the semantic gap between encoder and decoder feature maps, and enables model pruning at inference through deep supervision.

---

## Motivation and Problem Statement

_TODO: Describe the semantic gap between encoder and decoder features in standard U-Net skip connections._

---

## Key Contributions

- _TODO: Nested dense skip pathways_
- _TODO: Deep supervision for flexible inference_
- _TODO: Architecture pruning without retraining_
- _TODO: Consistent improvement across multiple segmentation tasks_

---

## Architecture Overview

_TODO: Describe the dense blocks between encoder and decoder nodes. Reference [nested_skip_analysis.md](./nested_skip_analysis.md)._

---

## Method Details

### Nested Skip Pathways

_TODO: Reference [nested_skip_analysis.md](./nested_skip_analysis.md)._

### Deep Supervision

_TODO: Reference [deep_supervision.md](./deep_supervision.md)._

### Pruning at Inference

_TODO: Reference [pruning_at_inference.md](./pruning_at_inference.md)._

---

## Experimental Results

| Dataset | Metric | UNet++ | U-Net | Improvement |
|---------|--------|--------|-------|-------------|
| Cell nuclei | _TODO_ | _TODO_ | _TODO_ | _TODO_ |
| Colon polyp | _TODO_ | _TODO_ | _TODO_ | _TODO_ |
| Liver | _TODO_ | _TODO_ | _TODO_ | _TODO_ |
| Lung nodule | _TODO_ | _TODO_ | _TODO_ | _TODO_ |

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
| DenseNet (Huang et al., 2017) | Dense connectivity inspiration |
| UNet 3+ (Huang et al., 2020) | Full-scale skip extension |

---

## Open Questions

- _TODO_
