---
title: "Zero-Shot Segmentation Evaluation"
date: 2025-03-06
status: complete
tags: [zero-shot, evaluation, generalization, benchmark]
difficulty: intermediate
---

# Zero-Shot Segmentation Evaluation

## Overview

Zero-shot evaluation measures a model's ability to segment objects in images from domains, datasets, or categories it has never seen during training. For foundation segmentation models like SAM and SAM 2, zero-shot performance is a primary indicator of generalization quality. This document covers evaluation methodology, standard benchmarks, metrics, and comparative results.

## What "Zero-Shot" Means for Segmentation

### Definition

In the context of foundation segmentation models, zero-shot refers to applying the model to a target dataset without any fine-tuning or adaptation on that dataset. The model uses only its pretraining knowledge (e.g., from SA-1B) and the prompt provided at inference time.

### Important Distinctions

- **Zero-shot with prompts:** The model receives prompts (points, boxes, masks) at inference time but was never trained on the target dataset. This is the standard evaluation mode for SAM.
- **Zero-shot without prompts:** The model segments without any user guidance (automatic mode). Only possible with models that have automatic mask generation.
- **Open-vocabulary zero-shot:** The model recognizes categories specified by text that were not in the training label set. Relevant for CLIP-based models like OMG-Seg.

## Evaluation Methodology

### Prompt Generation for Fair Comparison

When evaluating zero-shot segmentation, prompts must be generated consistently:

**Point prompts:**
- Center of mass of the ground truth mask
- Random point inside the ground truth mask
- Multiple random interior points (e.g., 1, 3, 5, 10 points)

**Box prompts:**
- Tight bounding box of the ground truth mask
- Bounding box with added jitter (e.g., +/- 10% padding) to simulate detector output
- Boxes from an actual object detector (Grounding DINO, Faster R-CNN)

**Oracle evaluation:**
- Use the ground truth mask as the prompt to establish an upper bound
- Use the best of N random prompts (oracle point selection)

### Multi-Mask Handling

SAM and SAM 2 output multiple mask candidates (typically 3). Evaluation protocols:
- **Default:** Use the mask with the highest predicted IoU score
- **Oracle:** Use the mask with the highest actual IoU against ground truth
- **All masks:** Report metrics for each granularity level separately

## Standard Benchmarks

### Natural Image Benchmarks

| Benchmark | Images | Task | Primary Metric |
|-----------|--------|------|---------------|
| COCO val2017 | 5,000 | Instance segmentation | AP, AR |
| LVIS v1 val | 19,809 | Long-tail instance seg | AP, AR |
| ADE20K | 2,000 | Semantic segmentation | mIoU |
| Cityscapes | 500 | Urban scene parsing | mIoU |
| PASCAL VOC 2012 | 1,449 | Semantic/instance seg | mIoU, AP |
| BSDS500 | 200 | Edge/boundary detection | ODS, OIS |

### Medical Imaging Benchmarks

| Benchmark | Images | Modality | Primary Metric |
|-----------|--------|----------|---------------|
| BTCV | 30 volumes | CT (abdominal) | DSC |
| ACDC | 100 volumes | MRI (cardiac) | DSC |
| Synapse | 30 volumes | CT (multi-organ) | DSC, HD95 |
| ISIC 2018 | 2,594 | Dermoscopy | DSC, IoU |
| Kvasir-SEG | 1,000 | Endoscopy | DSC, IoU |

### Specialized Domain Benchmarks

| Benchmark | Domain | Images | Primary Metric |
|-----------|--------|--------|---------------|
| iSAID | Remote sensing | 2,806 | mIoU |
| DeepGlobe | Satellite | 803 | mIoU |
| COD10K | Camouflage | 2,026 | S-measure, MAE |
| CHAMELEON | Camouflage | 76 | S-measure |
| MVTec AD | Industrial defect | 5,354 | AUROC, AP |

## Metrics

### Mask Quality Metrics

**IoU (Intersection over Union):**
The standard metric. IoU = |prediction AND ground_truth| / |prediction OR ground_truth|.

**Dice Similarity Coefficient (DSC):**
DSC = 2 * |prediction AND ground_truth| / (|prediction| + |ground_truth|). Numerically related to IoU but more forgiving of small errors.

**Average Recall (AR@k):**
Measures the fraction of ground truth objects that are detected with IoU above a threshold, averaged over thresholds from 0.5 to 0.95. AR@1000 allows up to 1000 proposals per image.

### Boundary Metrics

**Boundary IoU:**
IoU computed only within a narrow band (e.g., 2 pixels) around the ground truth boundary. Measures boundary accuracy specifically.

**Hausdorff Distance (HD95):**
The 95th percentile of distances between predicted and ground truth boundaries. Common in medical imaging.

### Detection-Style Metrics

**AP (Average Precision):**
Precision-recall curve area, used for instance segmentation. AP50 uses IoU threshold 0.5; AP75 uses 0.75.

## SAM Zero-Shot Results

### COCO

| Prompt | AR@100 | AR@1000 |
|--------|--------|---------|
| Automatic (grid) | 47.2 | 69.7 |
| 1 center point | 54.3 | -- |
| GT bounding box | 76.8 | -- |

### 23 Diverse Datasets (from SAM paper)

SAM was evaluated across 23 datasets spanning different domains:

| Prompt Setting | Mean mIoU |
|---------------|-----------|
| 1 point (center) | 60.6 |
| 1 point (oracle) | 67.8 |
| 3 points | 69.2 |
| Ground truth box | 75.3 |
| Oracle (best of 3 masks) | 73.0 |

### Domain-Specific Zero-Shot Performance

| Domain | SAM mIoU (1 point) | SAM mIoU (box) |
|--------|--------------------|--------------------|
| Natural images | 72.5 | 83.1 |
| Medical (CT/MRI) | 38.2 | 55.8 |
| Remote sensing | 45.6 | 62.3 |
| Camouflaged objects | 42.1 | 58.7 |
| Underwater | 51.3 | 68.9 |

Performance degrades significantly on out-of-distribution domains, motivating adaptation methods.

## SAM 2 vs. SAM Zero-Shot Comparison

### Image Segmentation

| Benchmark | SAM (ViT-H) | SAM 2 (Hiera-L) |
|-----------|-------------|-----------------|
| COCO AR@1000 | 69.7 | 71.2 |
| LVIS AR@1000 | 75.4 | 76.8 |
| SA-23 mIoU (box) | 75.3 | 76.1 |

SAM 2 slightly improves over SAM on image benchmarks while being significantly faster.

### Video Segmentation (Zero-Shot)

| Benchmark | SAM (per-frame) | SAM 2 |
|-----------|----------------|-------|
| DAVIS 2017 J&F | 63.2 (no propagation) | 82.5 |
| YouTube-VOS J&F | 55.8 (no propagation) | 81.2 |

The gap on video benchmarks is dramatic because SAM lacks any temporal mechanism.

## Evaluation Best Practices

### Reporting Guidelines

When reporting zero-shot results, always specify:
1. **Prompt type** (point, box, mask) and how prompts were generated
2. **Multi-mask selection strategy** (highest predicted IoU vs. oracle)
3. **Image preprocessing** (resolution, normalization)
4. **Model variant** (ViT-B, ViT-L, ViT-H, Hiera-B+, Hiera-L)
5. **Whether any part of the model was fine-tuned** (zero-shot means completely frozen)

### Common Pitfalls

- **Using ground truth boxes as prompts** overstates practical performance (detectors produce noisy boxes)
- **Oracle mask selection** gives an upper bound, not realistic performance
- **Evaluating only on easy subsets** (large, centered objects) inflates results
- **Not accounting for model inference resolution** (higher resolution generally helps)
- **Comparing models at different resolutions or prompt types** is misleading

### Statistical Significance

- Report standard deviation across multiple runs (if prompt generation involves randomness)
- Use paired statistical tests when comparing two models on the same dataset
- Be cautious about small dataset benchmarks (e.g., CHAMELEON with 76 images) where a few samples can swing results
