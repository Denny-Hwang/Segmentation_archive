---
title: "OMG-Seg: Is One Model Good Enough For All Segmentation?"
date: 2025-03-06
status: planned
tags: [universal-segmentation, clip, unified-architecture, multi-dataset]
difficulty: advanced
---

# OMG-Seg

## Meta Information

| Field | Value |
|-------|-------|
| **Paper Title** | OMG-Seg: Is One Model Good Enough For All Segmentation? |
| **Authors** | Li, X., Yuan, H., Li, W., Ding, H., Wu, S., Zhang, W., Li, Y., Chen, K., Loy, C.C. |
| **Year** | 2024 |
| **Venue** | CVPR |
| **arXiv** | [2401.10229](https://arxiv.org/abs/2401.10229) |
| **Difficulty** | Advanced |

## One-Line Summary

OMG-Seg presents a unified segmentation model that handles image-level, video-level, and interactive segmentation using a CLIP backbone and shared decoder, achieving competitive performance across all tasks.

## Motivation and Problem Statement

The segmentation landscape has become increasingly fragmented, with different model architectures and training paradigms for semantic segmentation, instance segmentation, panoptic segmentation, video object segmentation, and interactive segmentation. Models like Mask2Former excel at image-level panoptic segmentation, SAM dominates interactive segmentation, and dedicated VOS methods lead video segmentation benchmarks. This specialization creates practical problems: deploying multiple models is expensive, maintaining separate codebases is burdensome, and the knowledge learned for one task does not benefit others.

OMG-Seg asks a fundamental question: can a single model achieve competitive performance across all segmentation tasks simultaneously? The key insight is that all segmentation tasks ultimately require the same core capabilities -- identifying objects, delineating their boundaries, and (for some tasks) classifying them. The differences lie primarily in the output format (semantic labels, instance IDs, temporal tracks) and the conditioning signal (dataset-specific categories, user prompts, video context). By using a CLIP backbone that provides semantically rich features and a unified decoder that can produce different output types, OMG-Seg aims to be one model for all segmentation.

## Architecture Overview

OMG-Seg follows an encoder-decoder architecture with three main components: a CLIP-based image encoder that extracts multi-scale features with built-in semantic understanding, a unified transformer decoder that processes object queries to produce mask and class predictions, and task-specific output formatting that converts the unified decoder's output into the appropriate format for each task (semantic labels, instance masks, panoptic maps, video tracks, or interactive masks). The CLIP backbone provides a crucial advantage over traditional backbones: its features are aligned with text embeddings, enabling open-vocabulary classification without task-specific category heads.

### Key Components

- **CLIP Backbone**: See [clip_backbone.md](clip_backbone.md)

## Technical Details

### CLIP-Based Feature Extraction

OMG-Seg uses a CLIP ViT-L/14 model as its image encoder, extracting multi-scale feature maps by tapping into intermediate transformer layers. Unlike standard ViT usage (which produces a single-scale output), OMG-Seg extracts features from layers 6, 12, 18, and 24 (for a 24-layer ViT-L), creating a feature pyramid analogous to the multi-scale features produced by hierarchical backbones like Swin Transformer. These multi-scale features are projected to a common channel dimension (256) using 1x1 convolutions and fused through a Feature Pyramid Network (FPN) to produce the final multi-scale feature maps fed to the decoder.

The CLIP features carry semantic information from the vision-language pretraining, meaning that features for "dog" and "cat" regions already occupy different parts of the embedding space corresponding to their text descriptions. This semantic organization provides a strong prior for category classification, enabling OMG-Seg to perform open-vocabulary segmentation by matching region features against text embeddings of category names.

### Unified Decoder

The unified decoder follows the Mask2Former architecture: a stack of transformer decoder layers that process a set of learnable object queries against the multi-scale image features. Each query attends to the image features through cross-attention and to other queries through self-attention, progressively refining its representation to correspond to a specific object or region in the image. After the decoder layers, each query produces two outputs: a binary mask (through dot product with pixel-level features) and a class prediction (through matching against CLIP text embeddings or a learned classifier).

The decoder uses 100 object queries by default, with 6 transformer decoder layers. Masked attention (where each query's cross-attention is restricted to its predicted mask region from the previous layer) improves convergence and quality. The same decoder architecture and weights are shared across all tasks, with the only difference being how the outputs are interpreted and formatted.

### Task Routing

Different segmentation tasks are handled by different output formatting strategies applied to the same decoder output:

- **Semantic segmentation**: Each query's mask is assigned the query's predicted class, and overlapping predictions are resolved by taking the highest-confidence class per pixel.
- **Instance segmentation**: Each query corresponds to a distinct object instance, with masks filtered by confidence score and processed through NMS.
- **Panoptic segmentation**: Queries are partitioned into "stuff" (background regions) and "thing" (countable objects) based on their predicted class, and assembled into a panoptic map.
- **Video segmentation**: Queries are associated across frames using a lightweight temporal tracking module, producing temporally consistent instance tracks.
- **Interactive segmentation**: User prompts (points, boxes) are encoded as additional conditioning tokens that bias specific queries toward the prompted region.

This routing mechanism requires no task-specific parameters -- the same decoder weights produce all outputs, with only the post-processing differing.

### Multi-Dataset Training

OMG-Seg is trained on a mixture of datasets spanning all target tasks: COCO for panoptic/instance/semantic segmentation, ADE20K for semantic segmentation, YouTube-VOS and DAVIS for video segmentation, and SA-1B-derived data for interactive segmentation. The multi-dataset training uses a sampling strategy that balances the different tasks within each batch, ensuring the model sees examples from all tasks during each training epoch.

A key challenge is the inconsistency of annotation vocabularies across datasets (e.g., COCO has 133 categories while ADE20K has 150, with partial overlap). OMG-Seg handles this through CLIP-based open-vocabulary classification: rather than using dataset-specific classification heads, the model matches region features against text embeddings of category names, allowing seamless generalization across different label vocabularies. During training, the class loss is computed only against the categories present in each dataset.

### Loss Function

The training loss follows Mask2Former's formulation: a combination of binary cross-entropy loss and dice loss for mask prediction, plus cross-entropy loss for class prediction, applied to the bipartite-matched pairs of predictions and ground-truth objects. Hungarian matching is used to assign predicted queries to ground-truth objects, minimizing the total matching cost (which combines mask and class costs). The loss is computed at each decoder layer (deep supervision) with decreasing weights for earlier layers.

For video training, the loss is computed per frame with an additional temporal consistency term that penalizes large changes in query assignments between consecutive frames. The total training objective is a weighted sum of the per-task losses, with task weights tuned to balance performance across all tasks.

## Experiments and Results

### Image Segmentation

On COCO panoptic segmentation, OMG-Seg achieves 57.7 PQ (Panoptic Quality), competitive with Mask2Former (57.8 PQ) that was specifically designed for this task. On ADE20K semantic segmentation, OMG-Seg achieves 50.1 mIoU, compared to Mask2Former's 56.1 mIoU -- a larger gap indicating that the multi-task training introduces some dilution for semantic segmentation. On COCO instance segmentation, OMG-Seg achieves 46.5 AP, compared to Mask2Former's 50.1 AP. These results demonstrate that a single model can approach task-specific performance on image segmentation benchmarks.

### Video Segmentation

On YouTube-VOS 2019, OMG-Seg achieves 73.8 J&F, competitive with dedicated VOS methods but below SAM 2's 81.2 J&F. On DAVIS 2017 val, OMG-Seg achieves 74.2 J&F. The video segmentation performance is respectable but clearly below the state of the art, reflecting the fact that OMG-Seg's temporal modeling (lightweight tracking module) is less sophisticated than SAM 2's streaming memory architecture or specialized VOS methods.

### Interactive Segmentation

On interactive segmentation benchmarks, OMG-Seg achieves competitive performance with SAM for box-prompted segmentation (within 2-3 IoU points) but lags behind SAM for point-prompted segmentation (approximately 5 IoU points lower). This gap is expected because SAM's architecture and training are specifically optimized for prompt-based interaction, while OMG-Seg's interactive mode is one of several capabilities.

### Key Results

The headline result is that OMG-Seg achieves competitive performance across all six segmentation tasks with a single set of model weights. No single-task specialist achieves the same breadth: SAM cannot do semantic segmentation, Mask2Former cannot do video segmentation, and VOS methods cannot do panoptic segmentation. The unified model's total parameter count (approximately 300M) is substantially less than the sum of specialized models it replaces (approximately 1.2B total). On the geometric mean of performance across all tasks, OMG-Seg matches or exceeds any combination of two specialized models.

## Strengths

- **True task unification**: A single model with shared weights handles six distinct segmentation tasks, demonstrating that task-specific architectures are not necessary for competitive performance.
- **Open-vocabulary classification**: The CLIP backbone enables classification of novel categories not seen during training, simply by providing text descriptions.
- **Efficient deployment**: One model replaces multiple specialized models, reducing storage, maintenance, and inference infrastructure requirements.
- **Knowledge sharing across tasks**: Multi-task training provides implicit regularization and allows knowledge transfer between related tasks (e.g., instance segmentation benefits from semantic segmentation training).
- **Strong image segmentation**: Near state-of-the-art performance on panoptic segmentation with COCO, demonstrating that generality does not require large accuracy sacrifices.

## Limitations

- **Video segmentation gap**: The lightweight temporal module falls significantly short of dedicated memory-based VOS methods like SAM 2, suggesting that temporal reasoning requires more specialized architectural support.
- **Semantic segmentation dilution**: Multi-task training slightly harms semantic segmentation performance compared to single-task training (approximately 6 mIoU points below Mask2Former on ADE20K).
- **Interactive segmentation gap**: Point-prompted segmentation is notably weaker than SAM, indicating that the promptable interface is not as refined as a purpose-built interactive model.
- **Training complexity**: Multi-dataset training with different annotation formats, loss functions, and sampling strategies is complex to implement and tune.
- **CLIP backbone limitations**: CLIP features, while semantically rich, may lack the fine-grained spatial detail needed for precise boundary delineation, particularly for small objects.

## Connections

OMG-Seg builds on Mask2Former (Cheng et al. 2022) for its decoder design, CLIP (Radford et al. 2021) for its backbone and open-vocabulary capabilities, and SAM (Kirillov et al. 2023) for its interactive segmentation inspiration. OneFormer (Jain et al. 2023) pursued a similar goal of unifying image segmentation tasks but did not address video or interactive segmentation. X-Decoder (Zou et al. 2023) also explored multi-task segmentation with vision-language features. Compared to SAM 2, OMG-Seg offers broader task coverage (semantic and panoptic segmentation) but weaker video and interactive performance. The contrasting approaches of SAM (prompting-first) and OMG-Seg (unification-first) represent two distinct philosophies for building general segmentation systems.

## References

- Cheng, B., et al. "Masked-attention Mask Transformer for Universal Image Segmentation." CVPR 2022 (Mask2Former).
- Radford, A., et al. "Learning Transferable Visual Models From Natural Language Supervision." ICML 2021 (CLIP).
- Kirillov, A., et al. "Segment Anything." ICCV 2023 (SAM).
- Jain, J., et al. "OneFormer: One Transformer to Rule Universal Image Segmentation." CVPR 2023.
- Zou, X., et al. "Generalized Decoding for Pixel, Image, and Language." CVPR 2023 (X-Decoder).
- Lin, T.-Y., et al. "Feature Pyramid Networks for Object Detection." CVPR 2017 (FPN).
