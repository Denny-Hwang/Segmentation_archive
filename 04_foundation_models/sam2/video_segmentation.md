---
title: "Video Segmentation in SAM 2"
date: 2025-03-06
status: planned
tags: [video-segmentation, temporal-consistency, object-tracking, vos]
difficulty: advanced
---

# Video Segmentation in SAM 2

## Overview

SAM 2 extends promptable segmentation from static images to video by introducing a streaming memory mechanism that propagates masks across frames. Rather than processing each frame independently, SAM 2 conditions every frame's prediction on a memory bank of previous frames' features and predictions. Users can provide prompts (points, boxes, masks) on any frame in the video, and the model propagates the resulting segmentation both forward and backward in time. This transforms interactive segmentation from a per-image task into an interactive video annotation task where a single prompt can segment an object across an entire video clip.

## Promptable Video Segmentation Task

The promptable video segmentation task is defined as follows: given a video of T frames and one or more prompts on one or more frames, produce a segmentation mask for the specified object(s) on every frame. Prompts can be provided on a single initial frame (e.g., a bounding box on frame 1) or iteratively across multiple frames (e.g., a corrective click on frame 50 where the mask has drifted). The model tracks prompted objects across frames, handling appearance changes, camera motion, and partial occlusion. This definition generalizes both semi-supervised VOS (where a mask is given on frame 1) and interactive VOS (where a user provides clicks on multiple frames).

SAM 2 supports multi-object tracking by maintaining separate memory streams per object. Each object gets its own set of memory tokens and mask predictions, allowing the model to track multiple objects simultaneously without confusion. The maximum number of simultaneously tracked objects depends on GPU memory but typically supports 10-20 objects in practice.

## Temporal Consistency

SAM 2 achieves temporal consistency through the memory attention mechanism rather than explicit temporal smoothing. Because each frame's features are conditioned on memories from previous frames, the model implicitly learns to produce masks that are consistent with past predictions. The memory tokens encode both the visual appearance of the target object and its predicted spatial extent, so the decoder receives a strong prior about where the object should be and what it looks like.

In practice, SAM 2 produces notably smoother mask sequences than frame-by-frame SAM application. Quantitatively, the temporal stability (measured as the average IoU between consecutive frames' masks) is 3-5 points higher for SAM 2 compared to per-frame SAM. The model also naturally handles gradual appearance changes -- for example, a person turning around or a car changing viewing angle -- because the memory bank accumulates diverse appearance observations over time.

## Interactive Video Annotation

A key application of SAM 2 is interactive video annotation, where a human annotator segments objects in video with minimal effort. The workflow proceeds as follows: (1) the annotator provides a prompt on one frame, (2) SAM 2 propagates the mask across the video, (3) the annotator inspects the results and provides corrective prompts on frames where the mask is incorrect, (4) SAM 2 re-propagates from the corrected frames. This loop typically converges in 2-4 interactions for most objects.

Experiments show that SAM 2 requires approximately 8.4 interactions to annotate an object across a full SA-V video, compared to approximately 25.2 interactions needed when using SAM frame-by-frame -- a 3x reduction. The annotator time savings are even larger because SAM 2 processes intermediate frames automatically. For large-scale annotation projects, this efficiency gain is transformative: the SA-V dataset itself was annotated using SAM 2 in a model-in-the-loop pipeline.

## Semi-Supervised VOS Performance

On the standard semi-supervised VOS setting (mask provided on frame 1, propagate to all subsequent frames), SAM 2 achieves competitive or superior results to specialized VOS methods. On DAVIS 2017 val, SAM 2 Large achieves 82.5 J&F, compared to XMem at 81.2 and DeAOT at 80.5. On YouTube-VOS 2019 val, SAM 2 achieves 81.2 J&F. On the more challenging MOSE benchmark (featuring complex multi-object scenes with heavy occlusion), SAM 2 achieves 73.8 J&F, outperforming prior methods by 2-3 points.

The key advantage over specialized VOS methods is that SAM 2 does not require first-frame mask initialization -- it can be prompted with points or boxes instead. This makes it more practical for real-world applications where generating a precise first-frame mask is itself a laborious task. When initialized with a ground-truth first-frame mask (the standard VOS protocol), SAM 2 matches or exceeds specialized methods that were specifically designed for this setting.

## Comparison with Specialized VOS Methods

Compared to XMem, DeAOT, and Cutie (specialized VOS methods), SAM 2 offers several advantages: it handles both images and video, supports interactive prompting on any frame, and processes frames approximately 6x faster. However, specialized methods still have edges in certain scenarios. XMem's sensory memory module is more sophisticated for very long videos (>1000 frames) where SAM 2's FIFO memory may drop important early frames. DeAOT's hierarchical propagation better handles heavily occluded multi-object scenes. Cutie achieves slightly higher J&F on YouTube-VOS 2019 (82.1 vs. 81.2) due to its object-level memory reading mechanism.

The practical trade-off is clear: for most applications, SAM 2's generality and interactive capabilities outweigh the 1-2 point accuracy gap on specific benchmarks. For specialized production pipelines targeting a specific VOS benchmark, task-specific methods may still be preferred.

## Real-Time Capabilities

SAM 2 processes video at approximately 44 frames per second on a single A100 GPU using the Hiera-B+ encoder, and approximately 24 FPS with the larger Hiera-L encoder. This includes image encoding, memory attention, and mask decoding for each frame. The streaming design ensures constant per-frame cost regardless of video length, as the memory bank has a fixed capacity (6 recent frames + 2 prompted frames). Memory attention adds approximately 5ms per frame compared to image-only processing.

For real-time interactive applications, SAM 2 supports an "online" mode where frames are processed as they arrive. The user can provide prompts at any time, and the model incorporates them into the memory bank and adjusts subsequent predictions. This mode is suitable for video annotation tools and live segmentation applications. For offline processing, SAM 2 can process a full video in a single forward pass, propagating prompts bidirectionally.

## Implementation Notes

The official SAM 2 codebase provides a `SAM2VideoPredictor` class for video inference. The typical workflow is: (1) initialize with a video or image directory, (2) add prompts on specific frames using `add_new_points()` or `add_new_mask()`, (3) call `propagate_in_video()` to generate masks for all frames. Memory management is handled automatically. For multi-object tracking, assign unique object IDs and provide prompts for each object separately. Batch processing of multiple videos is supported but requires separate predictor instances. GPU memory usage scales linearly with the number of tracked objects but is constant with respect to video length.
