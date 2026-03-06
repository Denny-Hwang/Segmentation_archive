---
title: "SAM 2: Segment Anything in Images and Videos"
date: 2025-03-06
status: planned
tags: [foundation-model, video-segmentation, streaming-memory, sa-v]
difficulty: advanced
---

# SAM 2

## Meta Information

| Field | Value |
|-------|-------|
| **Paper Title** | SAM 2: Segment Anything in Images and Videos |
| **Authors** | Ravi, N., Gabeur, V., Hu, Y.-T., Hu, R., Ryali, C., Ma, T., Khedr, H., Radle, R., Rolland, C., Gustafson, L., Mintun, E., Pan, J., Alwala, K.V., Carion, N., Wu, C.-Y., Girshick, R., Dollar, P., Feichtenhofer, C. |
| **Year** | 2024 |
| **Venue** | arXiv |
| **arXiv** | [2408.00714](https://arxiv.org/abs/2408.00714) |
| **Difficulty** | Advanced |

## One-Line Summary

SAM 2 extends SAM to video by introducing a streaming memory architecture that propagates segmentation across frames, trained on the SA-V dataset with 35.5M masks on 50.9K videos.

## Motivation and Problem Statement

<!-- Why was this work needed? What gap does it address compared to SAM? -->

## Architecture Overview

<!-- High-level description: image encoder, memory attention, memory encoder, memory bank, prompt encoder, mask decoder -->

### Key Components

- **Streaming Memory**: See [streaming_memory.md](streaming_memory.md)
- **Video Segmentation**: See [video_segmentation.md](video_segmentation.md)
- **SA-V Dataset**: See [sav_dataset.md](sav_dataset.md)

## Technical Details

### Image Encoder

<!-- Hiera backbone for efficient image encoding -->

### Memory Attention Module

<!-- How current frame attends to memory of past frames -->

### Memory Encoder and Memory Bank

<!-- How frame predictions are stored and managed -->

### Prompt Encoder and Mask Decoder

<!-- Shared components with SAM -->

### Training Strategy

<!-- Training on images and videos jointly -->

## Experiments and Results

### Video Segmentation Benchmarks

<!-- Performance on video object segmentation tasks -->

### Image Segmentation

<!-- Maintaining image segmentation performance -->

### Key Results

<!-- Main quantitative results -->

## Strengths

<!-- List key strengths of the approach -->

## Limitations

<!-- List key limitations -->

## Connections

<!-- How this work relates to SAM and other papers -->

## References

<!-- Key references cited in the paper -->
