---
title: "Deep Supervision in UNet++"
date: 2025-03-06
status: planned
tags:
  - deep-supervision
  - auxiliary-loss
  - training-strategy
parent: unetpp/review.md
---

# Deep Supervision in UNet++

## Overview

_TODO: Explain how deep supervision applies loss functions to outputs at multiple decoder depths, not just the final output._

---

## What Is Deep Supervision?

_TODO: Auxiliary classification/segmentation heads at intermediate layers, each contributing to the total loss._

---

## Deep Supervision in UNet++

### Supervision Points

_TODO: Each full-resolution output (X^(0,1), X^(0,2), X^(0,3), X^(0,4)) has a 1x1 conv + sigmoid producing a segmentation map._

### Combined Loss

_TODO: Total loss is the average of losses from all supervision points._

---

## Benefits

- _TODO: Faster convergence_
- _TODO: Better gradient flow to shallow layers_
- _TODO: Enables pruning at inference_

---

## Comparison with Other Deep Supervision Approaches

| Method | Where Supervision Is Applied |
|--------|----------------------------|
| Deeply Supervised Nets (Lee et al., 2015) | Hidden layers |
| HED (Xie & Tu, 2015) | Side outputs at each scale |
| UNet++ | Nested decoder outputs |

---

## Training Hyperparameters

_TODO: Loss weighting strategy, effect of number of supervision points._
