---
title: "Pruning at Inference in UNet++"
date: 2025-03-06
status: planned
tags:
  - pruning
  - inference-efficiency
  - model-compression
parent: unetpp/review.md
---

# Pruning at Inference in UNet++

## Overview

_TODO: Explain how deep supervision allows UNet++ to be pruned at inference time by removing deeper layers, trading accuracy for speed without retraining._

---

## How Pruning Works

### The Key Insight

_TODO: Because each output X^(0,j) is trained with its own supervision, any of these outputs can serve as the final prediction._

### Pruning Levels

_TODO: L1 uses only X^(0,1), L2 uses X^(0,1) and X^(0,2), etc._

| Pruning Level | Layers Used | Speed | Accuracy |
|--------------|------------|-------|----------|
| L1 (most pruned) | _TODO_ | Fastest | Lowest |
| L2 | _TODO_ | Fast | _TODO_ |
| L3 | _TODO_ | Moderate | _TODO_ |
| L4 (full model) | _TODO_ | Slowest | Highest |

---

## Accuracy vs Speed Tradeoff

_TODO: Present the tradeoff curve from the paper._

---

## When to Use Pruning

- _TODO: Real-time inference requirements_
- _TODO: Edge deployment with limited compute_
- _TODO: Screening vs diagnostic accuracy_

---

## Comparison with Other Compression Techniques

| Technique | Requires Retraining? | Granularity |
|-----------|---------------------|-------------|
| UNet++ pruning | No | Layer-level |
| Knowledge distillation | Yes | Model-level |
| Weight pruning | Often | Weight-level |
| Quantization | Sometimes | Precision-level |

---

## Limitations

_TODO: Discuss when pruning degrades performance unacceptably._
