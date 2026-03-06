---
id: sa1b
name: SA-1B (Segment Anything 1 Billion)
domain: specialized
modality: rgb
task: promptable
classes: N/A
size: "11M images, 1.1B masks"
license: Custom (Meta license)
---

# SA-1B - Segment Anything 1 Billion Masks

## Overview

| Field | Details |
|---|---|
| **Name** | SA-1B (Segment Anything 1 Billion) |
| **Source** | [Meta AI](https://segment-anything.com/dataset/index.html) |
| **Size** | 11 million images, 1.1 billion masks |
| **Classes** | Class-agnostic (no semantic labels) |
| **Modality** | RGB images (diverse sources) |
| **Common Use** | Foundation model training, promptable segmentation pre-training |

## Description

SA-1B is the largest segmentation dataset ever created, built by Meta AI for training the Segment Anything Model (SAM). It contains over 1.1 billion automatically generated masks across 11 million licensed and privacy-respecting images. Masks were produced using a data engine that iteratively improved SAM through manual, semi-automatic, and fully automatic annotation stages.

The masks are class-agnostic (no semantic labels attached), making this dataset suitable for training foundation models that can segment any object when given a prompt (point, box, or mask).

## Download Instructions

1. Visit [segment-anything.com/dataset](https://segment-anything.com/dataset/index.html)
2. Review and accept the Meta license terms
3. Download image tar files and mask JSON files (total ~25 TB)
4. Masks are stored in JSON format with RLE encoding

> Note: The full dataset is extremely large. Consider downloading a subset for experimentation.

## Key Papers Using This Dataset

- **SAM** (Kirillov et al., 2023) - Segment Anything Model (original training dataset)
- **SAM 2** (Ravi et al., 2024) - Extended to video with additional SA-V data
- **EfficientSAM** (Xiong et al., 2024) - Efficient distillation from SAM
- **FastSAM** (Zhao et al., 2023) - Real-time segment anything
