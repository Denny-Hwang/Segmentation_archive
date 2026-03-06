---
id: lits
name: LiTS - Liver Tumor Segmentation Challenge
domain: medical
modality: ct
task: semantic
classes: 3
size: "201 CT scans"
license: Custom (challenge terms)
---

# LiTS - Liver Tumor Segmentation Challenge

## Overview

| Field | Details |
|---|---|
| **Name** | LiTS (Liver Tumor Segmentation Challenge) |
| **Source** | [CodaLab](https://competitions.codalab.org/competitions/17094) |
| **Size** | 201 contrast-enhanced abdominal CT scans (131 train, 70 test) |
| **Classes** | 3: background, liver, tumor |
| **Modality** | Contrast-enhanced abdominal CT |
| **Common Use** | Liver and tumor segmentation, cascaded segmentation approaches |

## Description

The LiTS dataset was created for the ISBI 2017 Liver Tumor Segmentation Challenge. It contains 201 contrast-enhanced abdominal CT scans with pixel-level annotations for liver parenchyma and liver tumors. The data was collected from six clinical sites with varying CT scanners and imaging protocols, making it a challenging dataset for generalization.

The dataset is commonly used to benchmark cascaded segmentation approaches where the liver is segmented first, followed by tumor segmentation within the liver region.

## Download Instructions

1. Visit the [LiTS CodaLab competition page](https://competitions.codalab.org/competitions/17094)
2. Register for the competition
3. Navigate to the Participate tab to access the data
4. Download training volumes and segmentation labels (NIfTI format)

## Key Papers Using This Dataset

- **H-DenseUNet** (Li et al., 2018) - Hybrid densely connected UNet for liver tumor segmentation
- **nnU-Net** (Isensee et al., 2021) - Self-configuring framework
- **Attention U-Net** (Oktay et al., 2018) - Originally evaluated on pancreas and liver
- **UNet++** (Zhou et al., 2018) - Nested U-Net with dense skip connections
