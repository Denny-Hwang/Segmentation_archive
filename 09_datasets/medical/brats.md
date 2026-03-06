---
id: brats
name: Brain Tumor Segmentation Challenge (BraTS)
domain: medical
modality: brain_mri
task: semantic
classes: 4
size: "2000+ multi-modal MRI scans"
license: Custom (challenge terms)
---

# BraTS - Brain Tumor Segmentation Challenge

## Overview

| Field | Details |
|---|---|
| **Name** | BraTS (Brain Tumor Segmentation Challenge) |
| **Source** | [CBICA / UPenn](https://www.med.upenn.edu/cbica/brats/) |
| **Size** | 2000+ multi-modal MRI scans (varies by year) |
| **Classes** | 4: background, necrotic core, peritumoral edema, GD-enhancing tumor |
| **Modality** | Multi-modal brain MRI (T1, T1ce, T2, FLAIR) |
| **Common Use** | Brain tumor segmentation, 3D medical segmentation benchmarking |

## Description

The Brain Tumor Segmentation (BraTS) challenge provides multi-institutional, pre-operative multi-modal MRI scans of glioma patients. Each case includes four MRI modalities (T1, T1-contrast enhanced, T2, FLAIR) with expert annotations of tumor sub-regions.

BraTS has been running annually since 2012 and is one of the most established medical image segmentation challenges. The dataset grows each year with new cases and increasingly difficult evaluation criteria.

## Download Instructions

1. Visit the [BraTS challenge page](https://www.med.upenn.edu/cbica/brats/)
2. Register for the challenge on Synapse or the CBICA portal
3. Accept the data use agreement
4. Download the multi-modal NIfTI volumes and segmentation labels

## Key Papers Using This Dataset

- **3D U-Net** (Cicek et al., 2016) - Volumetric segmentation from sparse annotation
- **nnU-Net** (Isensee et al., 2021) - Top performer across multiple BraTS editions
- **V-Net** (Milletari et al., 2016) - Volumetric CNN with Dice loss
- **TransBTS** (Wang et al., 2021) - Transformer for brain tumor segmentation
