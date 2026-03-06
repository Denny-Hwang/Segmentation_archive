---
id: pascal_voc
name: PASCAL VOC 2012
domain: natural
modality: rgb
task: semantic_instance
classes: 21
size: "11,530 images"
license: Custom (Flickr terms)
---

# PASCAL VOC 2012

## Overview

| Field | Details |
|---|---|
| **Name** | PASCAL Visual Object Classes 2012 |
| **Source** | [Oxford VGG](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/) |
| **Size** | 11,530 images (1,464 train, 1,449 val, 1,456 test for segmentation) |
| **Classes** | 21 (20 object classes + background) |
| **Modality** | RGB natural images |
| **Common Use** | Semantic segmentation, classic benchmark for new architectures |

## Description

PASCAL VOC 2012 is one of the foundational benchmarks in semantic segmentation. The segmentation task contains pixel-level annotations for 20 object categories plus background. Although smaller than newer datasets, it remains an important benchmark and is often augmented with the SBD (Semantic Boundaries Dataset) to increase training data to ~10,582 images.

The standard evaluation metric is mIoU across all 21 classes on the validation or test set.

## Download Instructions

1. Visit the [PASCAL VOC 2012 page](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/)
2. Download `VOCtrainval_11-May-2012.tar`
3. For augmented training data, download the [SBD dataset](http://home.bharathh.info/pubs/codes/SBD/)
4. Extract and use the `SegmentationClass` directory for semantic labels

## Key Papers Using This Dataset

- **FCN** (Long et al., 2015) - Fully convolutional networks (seminal work)
- **DeepLab** (Chen et al., 2014-2018) - Atrous convolution series
- **PSPNet** (Zhao et al., 2017) - Pyramid pooling module
- **U-Net** (Ronneberger et al., 2015) - Encoder-decoder with skip connections
- **SegFormer** (Xie et al., 2021) - Transformer-based segmentation
