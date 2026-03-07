---
title: "Streaming Memory Architecture in SAM 2"
date: 2025-03-06
status: complete
tags: [streaming-memory, memory-attention, temporal, video]
difficulty: advanced
---

# Streaming Memory Architecture

## Overview

The streaming memory architecture is the central technical contribution of SAM 2. It enables the model to process arbitrarily long videos with bounded memory and computational cost per frame. The architecture consists of three components: a memory encoder, a memory bank, and a memory attention module.

## Memory Encoder

### Purpose

The memory encoder produces compact representations of past frames that capture both spatial features and predicted mask information. These representations are stored in the memory bank and later retrieved by the memory attention module.

### Architecture

The memory encoder takes two inputs for each frame:
1. The image encoder output (multi-scale features from Hiera)
2. The predicted mask for that frame (after decoding)

The mask is first downsampled and passed through a lightweight convolutional network to produce a mask embedding. This embedding is then combined with the image features via element-wise summation. The result is a per-frame memory representation with spatial dimensions (typically 64x64) and 256 channels.

### What the Memory Captures

Each memory entry encodes:
- **Spatial layout:** Where objects and boundaries are located in that frame
- **Appearance features:** What the objects look like (texture, color, shape)
- **Mask context:** Which pixels belong to the target object, providing explicit segmentation guidance
- **Positional information:** Spatial positions via positional encodings, enabling spatial alignment across frames

## Memory Bank

### Structure

The memory bank maintains two types of stored memories:

#### FIFO Memory (Recent Frames)
- Stores memories from the N most recent frames (default N=6)
- Follows a first-in-first-out replacement policy
- Captures the recent temporal context and handles gradual appearance changes
- Does not require any special selection criterion; simply the latest frames

#### Prompted Memory (User-Annotated Frames)
- Stores memories from all frames where the user provided explicit prompts
- These are never evicted from the bank, regardless of how old they are
- Captures the user's intent and provides anchor points for the segmentation
- Typically 1-3 entries for most interactive workflows

### Memory Bank Operations

| Operation | Description |
|-----------|-------------|
| Insert | Add a new memory after processing a frame |
| Evict | Remove the oldest FIFO memory when capacity is reached |
| Retrieve | Return all stored memories for the attention module |
| Protect | Mark prompted memories as non-evictable |

### Memory Capacity

The bounded size of the memory bank (6 FIFO + all prompted frames) means:
- Computational cost per frame is roughly constant regardless of video length
- Very old unprompted frames are forgotten, which can hurt for objects that reappear after long absences
- The system can process videos of arbitrary length without running out of memory

## Memory Attention Module

### Architecture

The memory attention module is a cross-attention mechanism inserted between the image encoder and the mask decoder. It conditions the current frame's features on the stored memories.

**Attention structure:**
- **Queries:** Current frame features (64x64 spatial tokens)
- **Keys and Values:** Concatenated memory entries from the memory bank (each 64x64 spatial tokens)
- Multiple attention heads allow the model to attend to different aspects of the memories simultaneously

### Attention Flow

```
Current frame features (Q)
         |
         v
  [Memory Attention]  <--- Memory bank entries (K, V)
         |
         v
Memory-conditioned features
         |
         v
     Mask Decoder
```

### Spatial Alignment

A key challenge is aligning spatial locations across frames when objects move. The memory attention module handles this implicitly:

- Each spatial position in the current frame can attend to any spatial position in the memory frames
- The attention weights learn to match corresponding object regions across frames
- No explicit optical flow or spatial warping is required
- Positional encodings provide spatial reference, but the attention is free to attend non-locally

This implicit alignment is more flexible than explicit flow-based methods because it can handle:
- Non-rigid deformation
- Partial occlusions
- Appearance changes
- Scene structure changes

### Self-Attention Integration

Before cross-attending to memories, the current frame's features undergo self-attention. This allows the model to:
1. First reason about the spatial structure of the current frame
2. Then incorporate temporal context from memories
3. The combined representation goes to the mask decoder

## Design Decisions and Tradeoffs

### Why Streaming (Not Global)?

Alternative approaches process all frames simultaneously (e.g., using 3D convolutions or global attention over the full video). SAM 2 uses streaming because:

| Aspect | Streaming | Global |
|--------|-----------|--------|
| Memory usage | O(1) per frame | O(T) for T frames |
| Latency | Real-time capable | Must wait for full video |
| Video length | Unlimited | Limited by GPU memory |
| Long-range context | Limited by bank size | Full context |

The streaming design was chosen to support interactive use cases where users need real-time feedback.

### FIFO vs. Learned Selection

The FIFO memory replacement policy is simpler than learned selection strategies (e.g., selecting the most informative frames to keep). The authors found that FIFO performs well in practice because:
- Recent frames are almost always the most relevant for temporal continuity
- Prompted frames (which are always kept) provide long-range anchors
- Learned selection adds complexity without consistent improvement

### Memory Size Sensitivity

Experiments with different FIFO buffer sizes:

| FIFO Size | DAVIS J&F | Notes |
|-----------|-----------|-------|
| 1 | 79.2 | Only previous frame; struggles with occlusions |
| 3 | 81.1 | Good for short-range propagation |
| 6 | 82.5 | Default; best tradeoff |
| 12 | 82.3 | Diminishing returns; more compute |

## Connections to Other Architectures

### Relation to XMem

XMem (2022) also uses a multi-store memory architecture for VOS with sensory memory, working memory, and long-term memory. SAM 2 simplifies this to two stores (FIFO and prompted) while achieving better results, suggesting that the large-scale training data compensates for architectural complexity.

### Relation to Transformers in Video

The memory attention module can be viewed as a form of temporal attention restricted to a sliding window of frames plus anchors. This is related to sparse attention patterns in efficient transformers but applied specifically to the video segmentation setting.

## Practical Considerations

### GPU Memory Usage

For a typical video at 1024x1024 resolution:
- Image encoder: ~2 GB per frame (but only one frame at a time)
- Memory bank (6 entries): ~150 MB
- Total inference: ~4 GB GPU memory regardless of video length

### Throughput

- Real-time capable at ~44 FPS for mask decoding (after image encoding)
- Image encoding is the bottleneck at ~6 FPS for Hiera-L
- For interactive use, image features can be precomputed and cached
