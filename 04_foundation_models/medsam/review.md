---
title: "Segment Anything in Medical Images"
date: 2025-03-06
status: planned
tags: [foundation-model, medical-segmentation, domain-adaptation, sam]
difficulty: intermediate
---

# MedSAM

## Meta Information

| Field | Value |
|-------|-------|
| **Paper Title** | Segment Anything in Medical Images |
| **Authors** | Ma, J., He, Y., Li, F., Han, L., You, C., Wang, B. |
| **Year** | 2024 |
| **Venue** | Nature Communications |
| **arXiv** | N/A |
| **Difficulty** | Intermediate |

## One-Line Summary

MedSAM fine-tunes SAM on a large-scale medical image dataset spanning multiple modalities and anatomies, demonstrating that domain-specific adaptation significantly improves segmentation performance on medical images.

## Motivation and Problem Statement

<!-- Why SAM needs adaptation for medical images -->

## Architecture Overview

<!-- How MedSAM modifies or retains SAM's architecture -->

### Key Components

- **Medical Adaptation**: See [medical_adaptation.md](medical_adaptation.md)

## Technical Details

### Training Data

<!-- Large-scale medical image dataset used for fine-tuning -->

### Fine-Tuning Strategy

<!-- Which components are fine-tuned and how -->

### Modality Coverage

<!-- CT, MRI, ultrasound, X-ray, endoscopy, etc. -->

### Prompt Strategy for Medical Images

<!-- How prompts (bounding boxes) are used in the medical context -->

## Experiments and Results

### Datasets and Modalities

<!-- Range of medical imaging benchmarks evaluated -->

### Key Results

<!-- Main quantitative results -->

### Comparison with Vanilla SAM

<!-- How much adaptation improves over zero-shot SAM -->

## Strengths

<!-- List key strengths of the approach -->

## Limitations

<!-- List key limitations -->

## Connections

<!-- How this work relates to SAM and other medical segmentation papers -->

## References

<!-- Key references cited in the paper -->
