---
id: sav
name: SA-V (Segment Anything Video)
domain: specialized
modality: video
task: video_segmentation
classes: N/A
size: "50.9K videos, 642.6K masklets"
license: Custom (Meta license)
---

# SA-V - Segment Anything Video

## Overview

| Field | Details |
|---|---|
| **Name** | SA-V (Segment Anything Video) |
| **Source** | [Meta AI](https://ai.meta.com/datasets/segment-anything-video/) |
| **Size** | 50.9K videos, 642.6K spatio-temporal masklets |
| **Classes** | Class-agnostic (no semantic labels) |
| **Modality** | Video |
| **Common Use** | Video object segmentation, SAM 2 training |

## Description

SA-V is the video segmentation dataset created by Meta AI for training SAM 2. It contains approximately 50,900 videos with 642,600 spatio-temporal masklets (temporally consistent mask annotations). The videos span diverse real-world scenarios and were annotated using an interactive data engine similar to the one used for SA-1B.

SA-V extends the Segment Anything paradigm from images to videos, enabling promptable segmentation that propagates across video frames.

## Download Instructions

1. Visit the [SA-V dataset page](https://ai.meta.com/datasets/segment-anything-video/)
2. Review and accept the Meta license terms
3. Download the video files and corresponding masklet annotations
4. Annotations include per-frame masks linked across time

## Key Papers Using This Dataset

- **SAM 2** (Ravi et al., 2024) - Primary training dataset for video segmentation
- Various video object segmentation works that benchmark against SAM 2
