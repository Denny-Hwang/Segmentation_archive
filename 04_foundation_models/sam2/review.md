---
title: "SAM 2: Segment Anything in Images and Videos"
date: 2025-03-06
status: complete
tags: [foundation-model, video-segmentation, streaming-memory, sa-v]
difficulty: advanced
---

# SAM 2

## Paper Overview

**Title:** SAM 2: Segment Anything in Images and Videos
**Authors:** Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang Hu, Chaitanya Ryali, Tengyu Ma, Haitham Khedr, Roman Radle, Chloe Rolland, Laura Gustafson, Eric Mintun, Junting Pan, Kalyan Vasudev Alwala, Nicolas Carion, Chao-Yuan Wu, Ross Girshick, Piotr Dollar, Christoph Feichtenhofer
**Venue:** arXiv 2024 (Meta FAIR)

SAM 2 extends the Segment Anything paradigm from static images to video, creating a unified model that handles both image and video segmentation with a promptable, streaming architecture. It introduces a memory mechanism that allows the model to track and segment objects across video frames while maintaining the interactive prompting interface from SAM.

## Key Contributions

1. A unified architecture for promptable segmentation in both images and videos
2. A streaming memory mechanism for temporal propagation without requiring access to all frames simultaneously
3. The SA-V dataset: the largest video segmentation dataset to date (50.9K videos, 642.6K masklets)
4. State-of-the-art results on video object segmentation benchmarks while being faster than prior methods

## Architecture

### Image Encoder (Hiera)

SAM 2 replaces SAM's ViT-H with Hiera, a hierarchical vision transformer:
- Hiera is pretrained with MAE and produces multi-scale features
- The hierarchical structure is more efficient than plain ViT for video processing
- Feature maps at multiple resolutions enable better handling of objects at different scales
- The image encoder processes each frame independently (no temporal fusion at this stage)

### Prompt Encoder

Identical to SAM's prompt encoder, supporting points, boxes, and masks. Prompts can be provided on any frame in the video (not just the first frame), enabling flexible interaction patterns.

### Memory Architecture

The central innovation of SAM 2 is the streaming memory system consisting of:

- **Memory Encoder:** Produces memory representations from predicted masks and image features for frames that have been processed
- **Memory Bank:** Stores memories from recent frames (FIFO buffer, up to 6 frames) and all prompted frames (frames where the user provided explicit prompts)
- **Memory Attention Module:** Cross-attention mechanism that conditions the current frame's features on stored memories, enabling temporal propagation

### Mask Decoder

An upgraded version of SAM's decoder that:
- Attends to both the current frame embedding and memory-conditioned features
- Produces mask predictions and occlusion scores
- The occlusion head predicts whether the object is visible in the current frame, enabling graceful handling of objects leaving and re-entering the scene

## Training

### Data

- SA-V dataset (50.9K videos) for video training
- SA-1B dataset (11M images) for image training
- Combined training on both data sources, treating images as single-frame videos

### Training Protocol

- 8-frame training sequences sampled from videos
- Simulated interactive prompting during training: the model receives corrective clicks on frames with the worst predictions
- Multi-phase training with increasing data complexity

## Key Results

### Video Object Segmentation

| Benchmark | Metric | SAM 2 | Previous SOTA |
|-----------|--------|-------|--------------|
| DAVIS 2017 (val) | J&F | 82.5 | 79.5 (XMem) |
| SA-V (test) | J&F | 76.0 | -- |
| MOSE | J&F | 73.8 | 68.9 |
| LVOS v2 | J&F | 75.3 | 67.2 |

### Image Segmentation

SAM 2 also improves over SAM on image-only benchmarks, achieving 6x faster inference than SAM with ViT-H while matching or exceeding its mask quality. This is largely due to the more efficient Hiera backbone.

### Interactive Video Segmentation

With 3 interactive clicks across frames, SAM 2 matches or exceeds the performance of methods that require manual annotation on every frame, demonstrating the effectiveness of its temporal propagation.

## Comparison to SAM

| Aspect | SAM | SAM 2 |
|--------|-----|-------|
| Scope | Images only | Images + Videos |
| Backbone | ViT-H (MAE) | Hiera (MAE) |
| Memory | None | Streaming memory bank |
| Occlusion handling | N/A | Explicit occlusion head |
| Speed | ~8 img/s (ViT-H) | ~44 FPS (video) |
| Training data | SA-1B (images) | SA-1B + SA-V |

## Strengths

- Unified image and video segmentation eliminates the need for separate models
- Streaming architecture enables processing arbitrarily long videos with bounded memory
- Interactive prompting on any frame allows intuitive user workflows
- The occlusion head enables robust handling of disappearing and reappearing objects
- Faster than SAM for image segmentation due to the Hiera backbone

## Limitations

- Performance degrades on very long videos with many occlusions and appearance changes
- The FIFO memory bank has a fixed size, meaning very old context may be lost
- Does not explicitly model object interactions or scene-level semantics
- Requires initial user prompts; fully automatic video segmentation is not addressed
- The SA-V dataset, while large, is still biased toward certain video types

## Impact

SAM 2 established a new paradigm for interactive video segmentation and enabled practical tools for video annotation, editing, and analysis. Its streaming architecture influenced subsequent work on memory-efficient video models, and the SA-V dataset became a benchmark for video segmentation research.

## Citation

```
Ravi, N., et al. "SAM 2: Segment Anything in Images and Videos." arXiv 2024.
```
