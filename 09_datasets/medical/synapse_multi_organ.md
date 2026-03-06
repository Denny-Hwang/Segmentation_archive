---
id: synapse_multi_organ
name: Synapse Multi-Organ Segmentation
domain: medical
modality: CT
task: semantic
classes: 13
size: "30 cases (3779 axial slices)"
license: Custom (Synapse terms)
---

# Synapse Multi-Organ Segmentation

## Overview

| Field | Details |
|---|---|
| **Name** | Synapse Multi-Organ CT Segmentation |
| **Source** | [Synapse Platform](https://www.synapse.org/#!Synapse:syn3193805) |
| **Size** | 30 abdominal CT cases (3779 axial slices) |
| **Classes** | 13 organs (aorta, gallbladder, spleen, left kidney, right kidney, liver, stomach, pancreas, etc.) |
| **Modality** | Contrast-enhanced abdominal CT |
| **Common Use** | Multi-organ segmentation benchmarking, TransUNet / Swin-Unet evaluation |

## Description

The Synapse multi-organ segmentation dataset is derived from the MICCAI 2015 Multi-Atlas Abdomen Labeling Challenge. It contains 30 abdominal CT scans with voxel-level annotations for 13 organs. This dataset has become a standard benchmark for evaluating transformer-based medical segmentation architectures.

The standard split uses 18 cases for training and 12 for testing, following the protocol established by TransUNet.

## Download Instructions

1. Create an account on [Synapse.org](https://www.synapse.org/)
2. Navigate to the [dataset page](https://www.synapse.org/#!Synapse:syn3193805)
3. Accept the data use terms
4. Download the CT volumes and label files

> Preprocessed versions are often available in TransUNet and Swin-Unet repository READMEs.

## Key Papers Using This Dataset

- **TransUNet** (Chen et al., 2021) - Hybrid CNN-Transformer for medical image segmentation
- **Swin-Unet** (Cao et al., 2021) - Pure transformer U-shaped architecture
- **UCTransNet** (Wang et al., 2022) - Channel-wise cross fusion transformer
- **HiFormer** (Heidari et al., 2023) - Hierarchical multi-scale feature fusion
