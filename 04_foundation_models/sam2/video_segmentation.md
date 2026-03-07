---
title: "Video Segmentation in SAM 2"
date: 2025-03-06
status: complete
tags: [video-segmentation, temporal-consistency, object-tracking, vos]
difficulty: advanced
---

# Video Segmentation in SAM 2

## Overview

SAM 2 approaches video segmentation as a frame-by-frame prediction task augmented with a memory mechanism. Rather than processing all frames simultaneously (which is memory-prohibitive for long videos), SAM 2 processes one frame at a time and uses a memory bank to propagate information from previously seen frames to the current one.

## Frame-by-Frame Processing Pipeline

For each frame in the video, SAM 2 executes the following steps:

1. **Image encoding:** The Hiera backbone encodes the current frame into a multi-scale feature representation, independent of other frames
2. **Memory conditioning:** The memory attention module cross-attends to stored memories from previous frames, injecting temporal context into the current frame's features
3. **Prompt encoding (if applicable):** If the user provides a prompt on the current frame, it is encoded and passed to the decoder
4. **Mask decoding:** The decoder produces a mask prediction, an IoU score, and an occlusion score for the current frame
5. **Memory encoding:** The current frame's prediction and features are encoded into a memory representation and added to the memory bank

## Temporal Propagation

### How Context Flows Across Frames

Temporal propagation in SAM 2 occurs exclusively through the memory attention mechanism. When processing frame t:

- The model retrieves stored memories from frames t-1, t-2, ..., t-N (recent frames) and any prompted frames
- Cross-attention allows the current frame's features to query these memories
- Spatial alignment between frames is handled implicitly by the attention mechanism (no explicit optical flow or warping)

This design means SAM 2 does not require explicit motion estimation. The model learns to match corresponding regions across frames through attention, which is more flexible than flow-based approaches.

### Propagation Behavior

- **Forward propagation:** The default mode. Prompts on an early frame propagate forward to segment subsequent frames.
- **Bidirectional propagation:** Users can prompt any frame (not just the first), and the model propagates both forward and backward. In practice, backward propagation is implemented by reversing the frame order and running the model again.
- **Multi-prompt refinement:** Users can correct errors on intermediate frames. Each correction updates the memory bank, and subsequent frames benefit from the improved context.

## Occlusion Handling

### The Occlusion Prediction Head

A key challenge in video segmentation is handling objects that leave and re-enter the frame. SAM 2 includes an explicit occlusion head that predicts the probability that the target object is not visible in the current frame.

**Behavior when occlusion is detected:**
- The model still processes the frame and updates the memory bank
- The predicted mask is suppressed (not output) for occluded frames
- When the object reappears, the memory bank contains pre-occlusion information that helps re-identify it

### Occlusion Training

During training, video clips are sampled to include frames where target objects are absent. The model learns to distinguish between:
- Object visible and segmented correctly
- Object partially occluded (output partial mask)
- Object fully occluded (output empty mask with high occlusion score)

## Object Tracking Capabilities

### Single Object Tracking

Given a prompt (point, box, or mask) on one frame, SAM 2 tracks the specified object through subsequent frames. This is analogous to semi-supervised video object segmentation (VOS).

### Multi-Object Tracking

SAM 2 handles multiple objects by running independent inference streams per object, each with its own memory bank. The masks from all streams are merged at the output stage. While this is simple and avoids inter-object interference, it scales linearly in computation with the number of objects.

### Handling Appearance Changes

The memory bank helps handle gradual appearance changes (e.g., rotation, deformation, lighting changes) because recent memories capture the evolving appearance. However, sudden drastic changes (e.g., clothing change in a movie cut) can cause tracking failure since the memory contains outdated appearance information.

## Interactive Video Segmentation Workflow

### Typical User Interaction

1. User provides a prompt on frame 1 (e.g., clicks on an object)
2. SAM 2 propagates the segmentation through the video
3. User reviews the results and identifies frames with errors
4. User provides corrective prompts (additional points or boxes) on error frames
5. SAM 2 re-runs from the corrective frame onward with the updated memory
6. Steps 3-5 repeat until quality is satisfactory

### Efficiency

- The image encoder runs once per frame and features are cached
- Re-propagation after a correction only needs to re-run the memory attention and decoder (not the encoder)
- In practice, 1-3 corrective interactions achieve high-quality segmentation for most videos

## Benchmarks and Evaluation

### Standard VOS Benchmarks

SAM 2 is evaluated on semi-supervised VOS benchmarks where the first-frame mask is given:

| Benchmark | Setting | J&F |
|-----------|---------|-----|
| DAVIS 2017 val | 1-click prompt | 78.4 |
| DAVIS 2017 val | Ground-truth mask | 82.5 |
| YouTube-VOS 2019 | GT mask | 81.2 |
| MOSE (complex scenarios) | GT mask | 73.8 |

### Comparison to Specialized VOS Methods

SAM 2 outperforms specialized VOS methods (XMem, DeAOT, Cutie) while being a more general model that also handles images and interactive prompting. This suggests that the foundation model approach with massive data can match or exceed task-specific architectures.

## Limitations for Video

- No explicit motion model means the system can struggle with very fast object motion
- The fixed-size FIFO memory limits long-range context (beyond ~6 frames of implicit history)
- Per-object inference for multi-object tracking is computationally redundant since all objects share the same video
- Scene cuts (e.g., in movies) break temporal continuity assumptions
- Very small or thin objects may be lost during propagation due to the 64x64 feature resolution
