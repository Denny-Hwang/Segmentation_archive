---
title: "Query-Text Contrastive Learning in OneFormer"
date: 2025-03-06
status: complete
tags: [contrastive-learning, query-text, text-supervision, universal-segmentation]
difficulty: advanced
---

# Query-Text Contrastive Learning

## Overview

Query-text contrastive learning is a training-time regularization technique in OneFormer that aligns object query representations with text embeddings of their assigned class names. By bringing queries closer to their corresponding text descriptions in a shared embedding space, this loss provides additional semantic supervision beyond pixel-level mask losses, helping queries develop more discriminative and class-aware representations.

## Text Representation

Text embeddings are generated using a frozen pre-trained CLIP text encoder. For each class in the dataset, a text prompt template is used: "a photo of a {class_name}". The CLIP text encoder produces a d-dimensional embedding for each class. These text embeddings are computed once and cached, adding no runtime cost during training. For classes with multi-word names (e.g., "traffic light"), the full name is used directly in the template.

## Contrastive Objective

The contrastive loss follows the InfoNCE formulation. For each matched query-ground truth pair, the query's output representation (after the transformer decoder) is projected to a shared embedding space and aligned with the text embedding of its assigned class:

`L_contrast = -log(exp(sim(q_i, t_pos) / τ) / Σ_j exp(sim(q_i, t_j) / τ))`

where `q_i` is the projected query representation, `t_pos` is the text embedding of the assigned class, `t_j` are text embeddings of all classes (positive + negatives), `sim` is cosine similarity, and `τ = 0.07` is the temperature. A linear projection layer maps query features (from the transformer decoder) to the CLIP embedding dimension before computing similarity.

## Role of Language in Segmentation

The contrastive loss serves multiple purposes: (1) it provides a class-level supervisory signal that complements the pixel-level mask loss, helping queries understand what category they should represent; (2) it transfers semantic knowledge from CLIP's large-scale vision-language pretraining into the segmentation model; (3) it acts as a regularizer that prevents query collapse (multiple queries collapsing to the same representation). The text supervision is particularly beneficial for rare classes where limited pixel-level examples make learning difficult.

## Impact on Performance

| Configuration | ADE20K PQ | ADE20K mIoU |
|--------------|-----------|-------------|
| Without contrastive loss | 49.0 | 57.2 |
| With contrastive loss | 49.8 | 58.0 |
| Δ | +0.8 | +0.8 |

The improvement is consistent across tasks and datasets, with slightly larger gains on datasets with more classes (ADE20K: 150 classes) where the semantic disambiguation from text is most valuable.

## Comparison with Other Text-Guided Methods

| Method | Text Source | Training Cost | Inference Cost |
|--------|-----------|---------------|---------------|
| OneFormer contrastive | CLIP (frozen) | Low (cached) | None |
| Open-vocabulary seg | CLIP (frozen/tuned) | Moderate | Moderate |
| Language-guided seg | Custom text encoder | High | High |

OneFormer's approach is uniquely efficient: the CLIP text encoder is frozen and text embeddings are precomputed, so the contrastive loss adds minimal training cost and zero inference cost. Unlike open-vocabulary segmentation methods that use text at inference time, OneFormer only uses text during training for regularization.

## Implementation Notes

The projection layer is a single `nn.Linear(d_query, d_clip)` mapping query features to CLIP's embedding dimension (typically 512 or 768). The contrastive loss weight is 0.5 relative to the mask and classification losses. Only matched queries (assigned to a ground truth via Hungarian matching) participate in the contrastive loss — unmatched queries are excluded. Text embeddings are stored as a fixed `nn.Embedding` layer (no gradients) of shape (num_classes, d_clip).
