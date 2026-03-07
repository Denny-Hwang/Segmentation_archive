---
title: "Universal Segmentation in Mask2Former"
date: 2025-03-06
status: complete
tags: [universal-segmentation, panoptic, instance, semantic, unified-architecture]
difficulty: advanced
---

# Universal Segmentation

## Overview

Mask2Former demonstrates that a single architecture can achieve state-of-the-art on semantic, instance, and panoptic segmentation without task-specific modifications. The key insight: all segmentation tasks can be formulated as mask classification — predicting a set of binary masks, each with a class label.

## Task Formulation

All three tasks are reformulated as predicting N (class, binary mask) pairs:

- **Semantic**: Each query predicts one of K classes. Multiple queries with the same class have their masks merged via argmax.
- **Instance**: Each query predicts a thing class + instance mask. Ranked by confidence, non-overlapping via NMS.
- **Panoptic**: Thing queries predict instances, stuff queries predict semantic regions. K_thing + K_stuff + 1 (no-object) classes.

The architecture and loss are identical across tasks — only the class space and post-processing differ.

## Shared Architecture Across Tasks

Everything is shared: backbone (Swin-L/ResNet), pixel decoder (multi-scale deformable), transformer decoder (9 layers, masked attention), prediction heads (class logits + mask embeddings). Only the classification head output dimension and post-processing differ per task.

## Training Strategy

Hungarian matching assigns predictions to ground truth using combined classification + mask costs. For semantic segmentation, connected components of each class serve as matching targets. For panoptic, things and stuff are handled uniformly — each gets a binary mask target, with "no-object" assigned to unmatched queries.

## Performance Across Tasks

| Task | Dataset | Mask2Former | Best Specialized |
|------|---------|------------|-----------------|
| Semantic | ADE20K | 57.8 mIoU | 56.0 (SegFormer) |
| Instance | COCO | 50.1 AP | 49.3 (Cascade M-RCNN) |
| Panoptic | COCO | 57.8 PQ | 55.1 (Panoptic SegFormer) |

One architecture outperforms task-specific models on all tasks. Gains are largest on panoptic, where previous methods struggled with the thing-stuff dichotomy.

## Comparison with Other Universal Approaches

| Method | Training | Single Model? | Performance |
|--------|----------|--------------|-------------|
| Panoptic FPN | Task-specific | No | Moderate |
| MaskFormer | Task-specific | No | Good |
| Mask2Former | Task-specific | No (3 models) | SOTA |
| OneFormer | Joint | Yes (1 model) | SOTA+ |

Mask2Former trains separate models per task despite using the same architecture. OneFormer later showed that joint training achieves comparable results with a single model.

## Implementation Notes

Post-processing per task:
- **Semantic**: argmax over query predictions weighted by mask logits per pixel
- **Instance**: confidence threshold → mask NMS → top-K instances
- **Panoptic**: assign each pixel to highest-confidence query, merge stuff, filter small instances
