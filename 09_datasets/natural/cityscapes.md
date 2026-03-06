---
id: cityscapes
name: Cityscapes
domain: natural
modality: rgb
task: semantic_instance
classes: 30
size: "5,000 fine + 20,000 coarse annotations"
license: Custom (research only)
---

# Cityscapes

## Overview

| Field | Details |
|---|---|
| **Name** | Cityscapes |
| **Source** | [cityscapes-dataset.com](https://www.cityscapes-dataset.com/) |
| **Size** | 5,000 finely annotated + 20,000 coarsely annotated images |
| **Classes** | 30 classes (19 used for evaluation) |
| **Modality** | RGB street-level images (2048x1024) |
| **Common Use** | Urban scene segmentation, autonomous driving research |

## Description

Cityscapes is a benchmark for pixel-level and instance-level semantic labeling of urban street scenes. Images were captured from a moving vehicle across 50 different cities in Germany and neighboring countries. The dataset provides fine-grained polygon annotations for 30 classes grouped into 8 categories.

The fine set (5,000 images) is split into 2,975 training, 500 validation, and 1,525 test images. The standard evaluation uses mIoU over 19 classes.

## Download Instructions

1. Register at [cityscapes-dataset.com](https://www.cityscapes-dataset.com/)
2. Login and navigate to the downloads section
3. Download `gtFine_trainvaltest.zip` (annotations) and `leftImg8bit_trainvaltest.zip` (images)
4. Extract and organize by split (train/val/test)

## Key Papers Using This Dataset

- **DeepLab v3+** (Chen et al., 2018) - Atrous spatial pyramid pooling
- **PSPNet** (Zhao et al., 2017) - Pyramid pooling module
- **SegFormer** (Xie et al., 2021) - Lightweight transformer segmentation
- **Mask2Former** (Cheng et al., 2022) - Universal segmentation
- **HRNet** (Sun et al., 2019) - High-resolution representation learning
