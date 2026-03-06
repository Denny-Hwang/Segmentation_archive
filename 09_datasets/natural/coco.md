---
id: coco
name: COCO (Common Objects in Context)
domain: natural
modality: rgb
task: instance_panoptic
classes: "80 things + 53 stuff"
size: "330K images, 1.5M object instances"
license: CC BY 4.0
---

# COCO - Common Objects in Context

## Overview

| Field | Details |
|---|---|
| **Name** | COCO (Common Objects in Context) |
| **Source** | [cocodataset.org](https://cocodataset.org/) |
| **Size** | 330K images, 1.5M object instances |
| **Classes** | 80 thing categories + 53 stuff categories (panoptic: 133 total) |
| **Modality** | RGB natural images |
| **Common Use** | Instance segmentation, panoptic segmentation, object detection |

## Description

COCO is one of the most widely used benchmarks in computer vision. For segmentation, it supports instance segmentation (detecting and delineating individual objects), stuff segmentation (amorphous regions like sky, grass), and panoptic segmentation (unified thing + stuff). Annotations are provided as polygons and RLE-encoded masks.

Standard evaluation metrics include AP (average precision) at various IoU thresholds for instance segmentation, and PQ (panoptic quality) for panoptic segmentation.

## Download Instructions

1. Visit [cocodataset.org/#download](https://cocodataset.org/#download)
2. Download the desired split images (train2017, val2017)
3. Download the corresponding annotations (instances, panoptic, stuff)
4. Use the `pycocotools` package to load and visualize annotations

```bash
pip install pycocotools
```

## Key Papers Using This Dataset

- **Mask R-CNN** (He et al., 2017) - Instance segmentation baseline
- **Mask2Former** (Cheng et al., 2022) - Universal segmentation with masked attention
- **OneFormer** (Jain et al., 2023) - Multi-task universal segmentation
- **SAM** (Kirillov et al., 2023) - Segment Anything Model (trained on SA-1B, evaluated on COCO)
- **Panoptic FPN** (Kirillov et al., 2019) - Feature pyramid for panoptic segmentation
