---
id: kvasir_seg
name: Kvasir-SEG
domain: medical
modality: endoscopy
task: binary
classes: 2
size: "1,000 images"
license: CC BY 4.0
---

# Kvasir-SEG

## Overview

| Field | Details |
|---|---|
| **Name** | Kvasir-SEG |
| **Source** | [SimulaMet / Datasets](https://datasets.simula.no/kvasir-seg/) |
| **Size** | 1,000 polyp images with masks |
| **Classes** | 2: background, polyp |
| **Modality** | Gastrointestinal endoscopy |
| **Common Use** | Polyp segmentation, medical binary segmentation benchmarking |

## Description

Kvasir-SEG is an open-access dataset of gastrointestinal polyp images and corresponding segmentation masks. The images were collected from colonoscopy videos at Vestre Viken Health Trust in Norway. Each image has a corresponding pixel-level ground truth mask annotated and verified by experienced gastroenterologists.

The dataset is notable for its open license (CC BY 4.0) and has become widely used for benchmarking polyp segmentation methods.

## Download Instructions

1. Visit the [Kvasir-SEG dataset page](https://datasets.simula.no/kvasir-seg/)
2. Download the zip archive directly (no registration required)
3. Extract the images and masks directories
4. Images are in JPEG format, masks in PNG format

## Key Papers Using This Dataset

- **PraNet** (Fan et al., 2020) - Parallel reverse attention for polyp segmentation
- **HarDNet-MSEG** (Huang et al., 2021) - Efficient polyp segmentation
- **Polyp-PVT** (Dong et al., 2021) - Pyramid vision transformer for polyps
- **ColonFormer** (Duc et al., 2022) - Transformer-based colonoscopy segmentation
