---
title: "OMG-Seg: Is One Model Good Enough For All Segmentation?"
date: 2025-03-06
status: complete
tags: [universal-segmentation, clip, unified-architecture, multi-dataset]
difficulty: advanced
---

# OMG-Seg

## Paper Overview

**Title:** OMG-Seg: Is One Model Good Enough For All Segmentation?
**Authors:** Xiangtai Li, Haobo Yuan, Wei Li, Henghui Ding, Size Wu, Wenwei Zhang, Yining Li, Kai Chen, Chen Change Loy
**Venue:** CVPR 2024

OMG-Seg proposes a unified segmentation model that handles all major segmentation tasks -- semantic segmentation, instance segmentation, panoptic segmentation, and open-vocabulary segmentation -- within a single architecture. The model uses a CLIP backbone for vision-language alignment combined with task-specific heads, achieving competitive or state-of-the-art results across multiple benchmarks simultaneously.

## Motivation

The segmentation field has produced specialized architectures for each task type:
- Semantic segmentation: FCN, DeepLab, SegFormer
- Instance segmentation: Mask R-CNN, SOLO, Mask2Former
- Panoptic segmentation: Panoptic FPN, MaskFormer, Mask2Former
- Open-vocabulary segmentation: LSeg, OpenSeg, SAN

This proliferation of models is inefficient. OMG-Seg asks whether a single model can perform all tasks well, reducing deployment complexity and leveraging shared visual representations.

## Architecture

### Overview

OMG-Seg consists of three components:

1. **CLIP-based image encoder** (shared across all tasks)
2. **Pixel decoder** (shared feature refinement)
3. **Task-specific query decoders** (one per task type)

### CLIP Image Encoder

- Uses CLIP ViT-L/14 as the backbone
- Extracts multi-scale features from intermediate ViT layers (not just the final layer)
- The CLIP encoder provides two key benefits:
  - Strong visual features learned from 400M image-text pairs
  - Aligned vision-language embedding space enabling open-vocabulary capabilities

### Pixel Decoder

A feature pyramid-style decoder that refines the multi-scale features from CLIP into high-resolution feature maps:
- Takes features from multiple ViT layers (e.g., layers 6, 12, 18, 24)
- Progressively upsamples and fuses features
- Produces feature maps at 1/4, 1/8, 1/16, and 1/32 resolution

### Task-Specific Decoders

Each segmentation task has a lightweight query-based decoder:

**Semantic segmentation head:**
- Learnable class queries (one per category)
- Cross-attention between queries and pixel features
- Produces per-class probability maps

**Instance segmentation head:**
- Learnable object queries (N queries for up to N instances)
- Cross-attention and self-attention layers
- Produces per-instance masks and classification scores

**Panoptic segmentation head:**
- Combines thing queries (instances) and stuff queries (semantic regions)
- Merging strategy to produce non-overlapping panoptic maps

**Open-vocabulary head:**
- Uses CLIP text embeddings as classification weights
- Region features are matched against text embeddings in the shared CLIP space
- Can recognize categories not seen during training

## Training

### Multi-Dataset Training

OMG-Seg is trained on multiple datasets simultaneously:

| Dataset | Task | Categories |
|---------|------|------------|
| COCO | Panoptic, Instance | 133 classes |
| ADE20K | Semantic | 150 classes |
| Cityscapes | Panoptic, Semantic | 19 classes |
| COCO-Stuff | Semantic | 171 classes |

### Training Strategy

- Each training batch samples from a single dataset
- Dataset-specific heads are activated based on the batch's task
- The shared encoder and pixel decoder receive gradients from all tasks
- Category embedding alignment ensures consistent class representations across datasets

### Loss Functions

- Mask loss: Combination of binary cross-entropy and dice loss
- Classification loss: Cross-entropy for closed-vocabulary, cosine similarity for open-vocabulary
- Hungarian matching for instance-level tasks (matching predicted queries to ground truth)

## Key Results

### Panoptic Segmentation

| Method | COCO PQ | ADE20K PQ |
|--------|---------|-----------|
| Mask2Former (Swin-L) | 57.8 | 48.1 |
| OneFormer (Swin-L) | 58.0 | 49.0 |
| **OMG-Seg (CLIP ViT-L)** | **58.3** | **49.5** |

### Instance Segmentation

| Method | COCO AP |
|--------|---------|
| Mask2Former (Swin-L) | 50.1 |
| **OMG-Seg** | **49.8** |

### Open-Vocabulary Segmentation

| Method | ADE20K (open-vocab) mIoU |
|--------|--------------------------|
| SAN (CLIP ViT-L) | 33.3 |
| FC-CLIP | 34.1 |
| **OMG-Seg** | **35.2** |

### Single Model, All Tasks

The key result is that a single OMG-Seg model achieves competitive performance across all task types simultaneously, rather than requiring separate models per task.

## Comparison to Other Unified Models

| Model | Semantic | Instance | Panoptic | Open-Vocab | Single Model |
|-------|----------|----------|----------|------------|-------------|
| Mask2Former | Yes | Yes | Yes | No | Yes |
| OneFormer | Yes | Yes | Yes | No | Yes |
| X-Decoder | Yes | Yes | Yes | Yes | Yes |
| **OMG-Seg** | **Yes** | **Yes** | **Yes** | **Yes** | **Yes** |

OMG-Seg is distinguished by achieving the strongest overall performance across all four task types in a single model.

## Strengths

- True unification: one model handles all segmentation tasks without task-specific retraining
- CLIP backbone provides open-vocabulary capability and strong transfer learning
- Multi-dataset training improves generalization beyond any single dataset
- Efficient at deployment: only one model needs to be loaded regardless of the task
- Modular design allows adding new task heads without retraining the backbone

## Limitations

- CLIP ViT-L backbone is computationally expensive (~430M parameters)
- Open-vocabulary performance still lags behind closed-vocabulary on in-distribution data
- Multi-dataset training requires careful balancing to avoid one task dominating
- The model does not handle video segmentation or 3D segmentation
- Performance on each individual task is competitive but not always state-of-the-art compared to the best specialized model for that task

## Impact

OMG-Seg advanced the trend toward universal segmentation models, showing that the traditional separation of segmentation into distinct tasks with separate models is unnecessary. The CLIP-based architecture influenced subsequent work on vision-language segmentation models.

## Citation

```
Li, X., et al. "OMG-Seg: Is One Model Good Enough For All Segmentation?" CVPR 2024.
```
