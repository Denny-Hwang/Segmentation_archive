---
title: "SMP - Pretrained Weights System"
date: 2025-01-15
status: planned
parent: "segmentation_models_pytorch/repo_overview.md"
tags: [smp, pretrained, weights, transfer-learning]
---

# SMP Pretrained Weights System

## Overview

TODO: How SMP manages pretrained encoder weights from multiple sources (ImageNet, Noisy Student, etc.)

## Weight Sources

| Source | Training Data | Models |
|--------|--------------|--------|
| ImageNet-1k | 1.2M images, 1000 classes | TODO |
| ImageNet-21k | TODO | TODO |
| Noisy Student | TODO | TODO |
| AdvProp | TODO | TODO |

## Weight Loading Mechanism

### Download and Caching
TODO: Where weights are cached and how downloads are triggered

### State Dict Adaptation
TODO: How pretrained weights are adapted when the model structure differs

### Input Channel Handling
TODO: How the system handles `in_channels != 3`

## Custom Pretrained Weights

TODO: How to use custom pretrained weights with SMP

## Initialization for Non-Pretrained Parts

TODO: How decoder weights are initialized when only the encoder is pretrained
