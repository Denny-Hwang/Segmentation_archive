---
id: remote_sensing
name: Remote Sensing Segmentation Datasets
domain: specialized
modality: satellite
task: semantic
classes: varies
size: varies
license: varies
---

# Remote Sensing Segmentation Datasets

## Overview

| Field | Details |
|---|---|
| **Name** | Remote Sensing Segmentation (collection) |
| **Source** | Various (see individual datasets below) |
| **Size** | Varies by dataset |
| **Classes** | Varies (land cover, building, road, etc.) |
| **Modality** | Satellite / aerial imagery (RGB, multispectral) |
| **Common Use** | Land cover mapping, building extraction, change detection |

## Description

Remote sensing segmentation encompasses several specialized datasets for segmenting satellite and aerial imagery. These datasets present unique challenges including large image sizes, class imbalance, varying spatial resolutions, and multi-spectral inputs.

## Notable Datasets

### ISPRS Vaihingen & Potsdam
- **Task**: Urban land cover classification
- **Classes**: 6 (impervious surfaces, building, low vegetation, tree, car, clutter)
- **Source**: [ISPRS WG III/4](https://www.isprs.org/education/benchmarks/UrbanSemLab/)

### DeepGlobe Land Cover
- **Task**: Land cover classification
- **Classes**: 7 (urban, agriculture, rangeland, forest, water, barren, unknown)
- **Source**: [DeepGlobe Challenge](http://deepglobe.org/)

### SpaceNet
- **Task**: Building footprint extraction, road network mapping
- **Source**: [SpaceNet on AWS](https://spacenet.ai/)

### LoveDA
- **Task**: Domain adaptive land cover segmentation
- **Classes**: 7
- **Source**: [GitHub](https://github.com/Junjue-Wang/LoveDA)

## Download Instructions

1. Visit the respective dataset websites listed above
2. Most require registration or challenge participation
3. SpaceNet data is hosted on AWS and can be downloaded via the AWS CLI
4. ISPRS data requires an academic request form

## Key Papers Using These Datasets

- **U-Net** (Ronneberger et al., 2015) - Widely adapted for remote sensing
- **DeepLab v3+** (Chen et al., 2018) - Applied to land cover classification
- **HRNet** (Sun et al., 2019) - High-resolution representations for dense prediction
- **SegFormer** (Xie et al., 2021) - Applied to aerial image segmentation
