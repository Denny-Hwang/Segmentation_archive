---
title: "SMP - Encoder Registry Analysis"
date: 2025-01-15
status: planned
parent: "segmentation_models_pytorch/repo_overview.md"
tags: [smp, encoder, registry, backbone, pretrained]
---

# SMP Encoder Registry

## Overview

TODO: Analyze how `segmentation_models_pytorch/encoders/__init__.py` implements the encoder registry pattern.

## Registry Architecture

### Registration Mechanism
TODO: How new encoders are added to the registry

### Encoder Interface Contract
TODO: What methods/attributes every encoder must expose

### Feature Map Extraction
TODO: How multi-scale features are extracted for decoder consumption

## Supported Encoder Families

| Family | Example Models | Source |
|--------|---------------|--------|
| ResNet | resnet18, resnet50, resnet101 | TODO |
| EfficientNet | efficientnet-b0 through b7 | TODO |
| timm models | Universal adapter for timm | TODO |
| TODO | | |

## Pretrained Weight Loading

TODO: How pretrained weights are downloaded and applied

## Adding a Custom Encoder

TODO: Step-by-step guide derived from code analysis

## Key Code Paths

TODO: Trace the code path from `smp.Unet(encoder_name="resnet34")` to model instantiation
