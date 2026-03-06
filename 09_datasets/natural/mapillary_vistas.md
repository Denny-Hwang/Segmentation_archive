---
id: mapillary_vistas
name: Mapillary Vistas
domain: natural
modality: rgb
task: semantic_panoptic
classes: "66 categories, 37 instance classes"
size: "25,000 images"
license: CC BY-NC-SA 4.0
---

# Mapillary Vistas

## Overview

| Field | Details |
|---|---|
| **Name** | Mapillary Vistas Dataset |
| **Source** | [Mapillary Research](https://www.mapillary.com/dataset/vistas) |
| **Size** | 25,000 high-resolution street-level images |
| **Classes** | 66 semantic categories, 37 instance-specific classes |
| **Modality** | RGB street-level images (variable resolution) |
| **Common Use** | Large-scale street scene parsing, panoptic segmentation |

## Description

Mapillary Vistas is a large-scale street-level image dataset for semantic and panoptic segmentation. Unlike Cityscapes which focuses on German cities, Mapillary Vistas covers images from all over the world across different weather conditions, seasons, and times of day. The images are captured at varying resolutions using diverse camera equipment.

The dataset provides 66 fine-grained semantic categories and 37 instance-level classes, making it one of the most detailed street scene parsing benchmarks.

## Download Instructions

1. Visit the [Mapillary Vistas page](https://www.mapillary.com/dataset/vistas)
2. Sign up and accept the research license
3. Download the training, validation, and test splits
4. Images come with semantic, instance, and panoptic annotations

## Key Papers Using This Dataset

- **Mask2Former** (Cheng et al., 2022) - Universal image segmentation
- **OneFormer** (Jain et al., 2023) - Multi-task universal segmentation
- **Panoptic-DeepLab** (Cheng et al., 2020) - Bottom-up panoptic segmentation
- **SegFormer** (Xie et al., 2021) - Evaluated on diverse driving datasets
