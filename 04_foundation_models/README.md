---
title: "Foundation Models for Segmentation"
date: 2025-03-06
status: planned
tags: [foundation-models, sam, zero-shot, promptable-segmentation]
difficulty: advanced
---

# 04 - Foundation Models for Segmentation

## Overview

This section covers foundation models for image segmentation -- large-scale models trained on massive datasets that exhibit strong zero-shot and few-shot generalization. The Segment Anything Model (SAM) family is the central focus, along with domain-specific adaptations (MedSAM) and unified architectures (OMG-Seg).

## Table of Contents

| Model | Authors | Year | Venue | Key Contribution |
|-------|---------|------|-------|------------------|
| [SAM](sam/review.md) | Kirillov et al. | 2023 | ICCV | Promptable segmentation foundation model trained on SA-1B |
| [SAM 2](sam2/review.md) | Ravi et al. | 2024 | arXiv | Extension to video with streaming memory architecture |
| [MedSAM](medsam/review.md) | Ma et al. | 2024 | Nature Communications | SAM adapted for universal medical image segmentation |
| [MedSAM-2](medsam2/review.md) | Zhu et al. | 2024 | arXiv | SAM 2 adapted for 3D medical volumes as video |
| [SAM-Adapter](sam_adapter/review.md) | Chen et al. | 2023 | arXiv | Parameter-efficient adaptation of SAM via adapters |
| [OMG-Seg](omg_seg/review.md) | Li et al. | 2024 | CVPR | Unified segmentation with CLIP backbone |

## Comparative Analyses

- [Foundation vs Specialized Models](_comparative/foundation_vs_specialized.md) - When foundation models outperform task-specific models
- [Zero-Shot Evaluation](_comparative/zero_shot_evaluation.md) - Zero-shot segmentation capabilities and limitations
- [Adaptation Strategies](_comparative/adaptation_strategies.md) - Methods for adapting foundation models to new domains

## Key Themes

1. **Promptable Segmentation**: Using points, boxes, masks, or text as prompts for interactive segmentation
2. **Scale of Training Data**: Impact of billion-scale annotated datasets (SA-1B, SA-V)
3. **Domain Adaptation**: Transferring foundation models to specialized domains like medical imaging
4. **Video Extension**: Extending image segmentation models to temporal sequences
5. **Parameter-Efficient Fine-Tuning**: Adapting large models with minimal trainable parameters
