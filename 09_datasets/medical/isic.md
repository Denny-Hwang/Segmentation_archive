---
id: isic
name: ISIC Skin Lesion Analysis
domain: medical
modality: dermoscopy
task: semantic
classes: "2-8 (varies by year)"
size: "25,000+ images"
license: CC BY-NC 4.0
---

# ISIC - International Skin Imaging Collaboration

## Overview

| Field | Details |
|---|---|
| **Name** | ISIC Skin Lesion Analysis Dataset |
| **Source** | [ISIC Archive](https://www.isic-archive.com/) |
| **Size** | 25,000+ dermoscopic images (cumulative across challenge years) |
| **Classes** | 2 (binary lesion segmentation) to 8 (multi-class diagnosis) |
| **Modality** | Dermoscopic images |
| **Common Use** | Skin lesion segmentation, dermatology AI benchmarking |

## Description

The ISIC Archive is a large-scale public repository of dermoscopic images maintained by the International Skin Imaging Collaboration. Various challenge editions (2016-2020) have provided training sets with pixel-level segmentation masks for skin lesion boundary delineation.

ISIC 2018 Task 1 is the most commonly used benchmark for segmentation, containing 2,594 training images with binary lesion masks.

## Download Instructions

1. Visit the [ISIC Archive](https://www.isic-archive.com/)
2. Browse or search for specific challenge datasets
3. For challenge data, visit the respective challenge pages (e.g., ISIC 2018)
4. Download images and ground truth masks in standard formats (JPEG/PNG)

## Key Papers Using This Dataset

- **U-Net** (Ronneberger et al., 2015) - Adapted for skin lesion segmentation
- **Attention U-Net** (Oktay et al., 2018) - Attention gating for lesion focus
- **DoubleU-Net** (Jha et al., 2020) - Stacked U-Net architecture
- **FAT-Net** (Wu et al., 2022) - Feature adaptive transformers
