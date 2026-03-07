---
title: "Streaming Memory Architecture in SAM 2"
date: 2025-03-06
status: planned
tags: [streaming-memory, memory-attention, temporal, video]
difficulty: advanced
---

# Streaming Memory Architecture

## Overview

The streaming memory architecture is the core innovation that enables SAM 2 to extend promptable segmentation from images to video. It works by maintaining a bounded memory bank of past frame representations that condition the processing of each new frame. As each frame is processed, the image encoder extracts features, the memory attention module integrates information from stored memories, and the mask decoder produces predictions. The current frame's features and predicted mask are then compressed into a memory token and added to the memory bank. This design allows SAM 2 to process arbitrarily long videos with constant per-frame computational cost, as the memory bank never exceeds a fixed size.

## Memory Bank Design

The memory bank is a fixed-capacity buffer that stores two types of memory: recent memories and prompted memories. Recent memories follow a FIFO (first-in, first-out) policy with a capacity of 6 frames, meaning that as new frames are processed, the oldest unprompted memories are discarded. Prompted memories -- frames on which the user provided an explicit prompt (click, box, or mask) -- are stored separately with a capacity of 2 and are never evicted by the FIFO policy. This ensures that the model always retains the user's explicit guidance even in very long videos.

Each memory entry consists of a set of spatial memory tokens (at the feature map resolution, typically 64x64) and a compact object pointer token (a single 256-dimensional vector summarizing the object's appearance). The total memory footprint per frame is approximately 0.5 MB, making the full memory bank (8 frames maximum) consume roughly 4 MB -- negligible compared to the model weights and intermediate activations.

## Memory Encoder

The memory encoder compresses a frame's representation and predicted mask into a memory token suitable for storage. It takes as input the frame's image features (from the image encoder) and the predicted mask (output of the mask decoder), downsamples the mask to the feature resolution, and concatenates them channel-wise. This concatenated representation is processed through a lightweight convolutional network (3 convolutional layers with skip connections) that produces the spatial memory tokens. Separately, an MLP pool produces the object pointer token by global average pooling over the mask region.

The memory encoder has approximately 1.2M parameters, making it a negligible fraction of the total model size. The encoding process takes less than 1ms per frame, adding virtually no latency to the pipeline. The key design choice is that the memory encodes both appearance (from image features) and shape (from the mask), allowing the memory attention module to recall both what the object looks like and where it was in previous frames.

## Memory Attention

The memory attention module is a stack of transformer layers that allow the current frame's image features to attend to the stored memory tokens. Specifically, each layer performs self-attention among the current frame's tokens, followed by cross-attention where current frame tokens are queries and memory tokens are keys and values. This mechanism enables the model to retrieve relevant spatial and appearance information from past frames.

The cross-attention operates over all stored memory tokens simultaneously, including both recent and prompted frames. The model learns to weight different memories based on their relevance to the current frame -- for example, assigning higher attention weight to a memory frame where the object had a similar appearance or position. The memory attention module consists of 4 transformer layers with 8 attention heads and adds approximately 5.5M parameters. Processing time is roughly 5ms per frame, regardless of video length (since the memory bank has fixed size).

## Memory Selection Strategy

The memory selection strategy balances recency, prompt relevance, and computational cost. The 6-slot FIFO buffer for recent frames ensures the model always has context from the immediate temporal neighborhood. The 2-slot prompted memory buffer ensures user guidance persists throughout the video. When both buffers are full, the oldest FIFO entry is evicted when a new non-prompted frame is added, and the oldest prompted entry is evicted only when a new prompted frame exceeds the prompt buffer capacity.

Ablation experiments show that the number of recent memory frames has a significant impact on performance. Reducing from 6 to 1 recent frame drops J&F on DAVIS 2017 by approximately 3 points. Increasing beyond 6 provides diminishing returns (less than 0.5 points improvement with 12 frames) while doubling the memory attention cost. The 2-slot prompted memory buffer is also critical: removing it (relying only on FIFO) drops performance by approximately 2 points on interactive benchmarks where prompts are spread across the video.

## Handling Occlusion and Reappearance

One of the most challenging aspects of video segmentation is handling objects that become fully occluded and later reappear. SAM 2 addresses this through two mechanisms. First, the mask decoder produces an occlusion score alongside its mask predictions, indicating when the target object is not visible in the current frame. When the occlusion score exceeds a threshold, the model outputs an empty mask and does not add the frame to the memory bank, preventing corrupted memories from polluting the bank.

Second, when an occluded object reappears, the memory attention mechanism can retrieve its appearance from pre-occlusion memories that remain in the memory bank. Because prompted frames are retained indefinitely and recent frames from just before the occlusion may still be in the FIFO buffer, the model has access to the object's last known appearance. Experiments show that SAM 2 can recover objects after occlusions of up to approximately 20-30 frames (roughly 1 second at 30fps), though performance degrades for longer occlusions as pre-occlusion memories may be evicted from the FIFO buffer.

## Computational Efficiency

The streaming design ensures that per-frame inference cost is O(1) with respect to video length. The dominant cost is the image encoder (approximately 15ms per frame for Hiera-L), followed by memory attention (approximately 5ms), mask decoding (approximately 2ms), and memory encoding (approximately 1ms). Total per-frame inference is approximately 23ms for SAM 2 Large, corresponding to roughly 44 FPS. Memory usage is also constant: approximately 4 MB for the memory bank plus approximately 500 MB for model weights and per-frame activations.

Compared to offline VOS methods that process all frames simultaneously (requiring O(T) memory for a T-frame video), SAM 2's streaming approach is far more scalable. For a 10-minute video at 30fps (18,000 frames), offline methods would require prohibitive GPU memory, while SAM 2 processes each frame in constant time and memory. The trade-off is that SAM 2's memory bank has limited temporal range, which can be a disadvantage for very long videos with infrequent prompts.

## Comparison with Non-Streaming Approaches

Offline VOS methods like STM and STCN process videos by storing features from all past frames and performing space-time attention over the entire history. This provides maximum temporal context but scales linearly in both memory and computation with video length. XMem introduced a hierarchical memory (sensory, working, long-term) that compresses older memories progressively, partially mitigating the scaling problem but still requiring O(T) total memory.

SAM 2's streaming approach is most similar to online VOS methods but differs in its fixed-capacity memory bank and its integration with the promptable segmentation framework. The key trade-off is temporal context: offline methods can use information from any frame in the video, while SAM 2 is limited to its 8-frame memory bank. In practice, this limitation is rarely problematic because most tracking failures can be corrected with a single prompt on the error frame, which permanently enters the prompted memory buffer.

## Implementation Notes

The memory bank is implemented as a dictionary mapping frame indices to memory tensors. During inference, the `propagate_in_video()` method iterates through frames sequentially, calling the memory encoder after each frame and updating the bank. For bidirectional propagation (when a prompt is given on a middle frame), the video is processed in two passes: forward from the prompted frame to the end, and backward from the prompted frame to the beginning. The memory bank is reset between passes. Multi-object tracking shares the image encoder computation but maintains separate memory banks per object, with memory attention performed independently for each object's tokens.
