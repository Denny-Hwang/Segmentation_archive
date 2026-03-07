---
title: "SAM 2 - Memory Attention Analysis"
date: 2025-01-15
status: complete
parent: "sam2/repo_overview.md"
tags: [sam2, memory-attention, video-segmentation, temporal]
---

# SAM 2 Memory Attention

## Overview

The memory attention mechanism is SAM 2's key innovation that enables video object segmentation. Defined in `sam2/modeling/memory_attention.py`, the `MemoryAttention` module conditions the current frame's image features on information from previously processed frames. It takes the image encoder's output for the current frame and attends to a bank of memory tokens derived from past frames' features and predictions. This allows SAM 2 to maintain temporal coherence when tracking objects across video frames, handling occlusions, appearance changes, and re-identification after disappearance.

The `MemoryAttention` module consists of a stack of transformer layers, each containing self-attention on the current frame's tokens, cross-attention from current tokens to memory tokens, and a feed-forward MLP. The output is a set of memory-conditioned features that replace the raw image encoder features when passed to the mask decoder. For single-image inference (no video context), the memory attention module can be bypassed or operates with an empty memory bank, making SAM 2 backward-compatible with SAM 1's image-only mode.

## Memory Bank Architecture

### Memory Types

SAM 2 maintains several types of memory in its memory bank, each serving a distinct purpose:

**Spatial memory features**: Dense feature maps from previously processed frames, produced by the memory encoder (`sam2/modeling/memory_encoder.py`). These capture where objects were located in past frames and what they looked like at the feature level. Each spatial memory is a downsampled feature map (typically at 1/16 resolution) that encodes both the image content and the predicted mask for that frame:

```python
# Memory encoder combines image features with mask prediction
class MemoryEncoder(nn.Module):
    def forward(self, pix_feat, masks, skip_mask_sigmoid=False):
        # pix_feat: image features from encoder [B, C, H, W]
        # masks: predicted mask logits [B, 1, H, W]
        masks = F.sigmoid(masks) if not skip_mask_sigmoid else masks
        # Fuse image features with mask via element-wise operations
        fused = self.mask_downsampler(masks) + pix_feat
        memory = self.fuser(fused)  # Conv layers to refine
        return memory  # [B, C, H/16, W/16]
```

**Object pointers**: Compact per-object embeddings that serve as object-level summaries. These are produced by the mask decoder's output tokens and capture high-level object identity information. Object pointers enable the model to re-identify objects even after occlusion or significant appearance change.

**Temporal position encoding**: Each memory entry is tagged with a temporal position encoding that indicates how many frames ago it was created. This allows the attention mechanism to weight recent memories more heavily than distant ones and helps the model understand temporal ordering.

### Memory Encoding

When a frame is processed, its features and predicted mask are encoded into memory by the `MemoryEncoder` module. The encoding process fuses the image features with the predicted segmentation mask, creating a representation that captures both "what the image looks like" and "where the object is." This fused representation is then stored in the memory bank for future frames to attend to:

```python
# After processing frame t:
memory_features = memory_encoder(image_features_t, predicted_mask_t)
object_pointer = mask_decoder_output_token  # From decoder
memory_bank.add(memory_features, object_pointer, frame_index=t)
```

The mask information is critical: it tells future frames not just what the scene looked like, but specifically where the tracked object was. Without the mask fusion, the memory would only contain generic scene features and the model would struggle to distinguish the tracked object from the background.

### Memory Selection

The memory bank has a fixed maximum size (typically 6-8 recent frames plus a few "prompted" frames). When the bank is full, older memories are discarded in a FIFO manner, with one exception: frames where the user provided explicit prompts (clicks, boxes) are retained as "permanent" memories because they contain the strongest signal about the object identity:

```python
# Memory management (simplified from sam2/sam2_video_predictor.py)
class MemoryBank:
    def __init__(self, max_memories=7):
        self.max_memories = max_memories
        self.memories = []

    def add(self, features, pointer, frame_idx, is_prompted=False):
        if len(self.memories) >= self.max_memories:
            # Remove oldest non-prompted memory
            non_prompted = [m for m in self.memories if not m.is_prompted]
            if non_prompted:
                self.memories.remove(non_prompted[0])
        self.memories.append(Memory(features, pointer, frame_idx, is_prompted))
```

## Cross-Attention Mechanism

The core of memory attention is cross-attention from current frame tokens to memory tokens. Each `MemoryAttentionLayer` in the stack performs three operations sequentially:

```python
class MemoryAttentionLayer(nn.Module):
    def forward(self, curr, memory, pos_enc, memory_pos_enc):
        # 1. Self-attention on current frame tokens
        curr = self.self_attn(curr, curr, curr)

        # 2. Cross-attention: current attends to memory
        curr = self.cross_attn_image(
            query=curr + pos_enc,
            key=memory + memory_pos_enc,
            value=memory
        )

        # 3. Feed-forward MLP
        curr = self.mlp(curr)
        return curr
```

The cross-attention allows each spatial location in the current frame to selectively attend to relevant locations in past frames. For example, a pixel on the boundary of a tracked object will attend strongly to boundary pixels in previous frames, gathering information about the object's shape and appearance at that location. The temporal position encoding ensures the model can distinguish between memories from 1 frame ago (highly relevant) and memories from 100 frames ago (less reliable due to potential appearance change).

The attention uses standard multi-head attention with `num_heads=8` and queries/keys/values projected to `d_model=256` (matching the FPN output dimension). Layer normalization is applied before each sub-layer (pre-norm convention), and residual connections are used throughout.

## Streaming Inference

For video processing, SAM 2 uses a streaming inference approach that processes frames sequentially, maintaining the memory bank as a sliding window over the video:

```python
# Streaming video inference (simplified from SAM2VideoPredictor)
class SAM2VideoPredictor:
    def propagate_in_video(self, video_frames, prompts):
        memory_bank = MemoryBank()

        for frame_idx, frame in enumerate(video_frames):
            # 1. Encode current frame
            image_features = self.image_encoder(frame)

            # 2. Condition on memory (if available)
            if memory_bank.is_empty():
                conditioned_features = image_features
            else:
                memory_tokens = memory_bank.get_all_memories()
                conditioned_features = self.memory_attention(
                    curr=image_features, memory=memory_tokens
                )

            # 3. Get prompts for this frame (if any)
            frame_prompts = prompts.get(frame_idx, None)

            # 4. Decode mask
            masks, iou_scores, obj_ptrs = self.mask_decoder(
                conditioned_features, frame_prompts
            )

            # 5. Update memory bank
            memory = self.memory_encoder(image_features, masks)
            memory_bank.add(memory, obj_ptrs, frame_idx,
                          is_prompted=(frame_prompts is not None))

            yield frame_idx, masks
```

This streaming approach means memory usage is constant regardless of video length (bounded by the memory bank size), making it practical for processing arbitrarily long videos. Each frame requires one image encoder pass plus one memory attention pass, giving roughly 2x the computational cost of single-image inference.

## Memory Management

The memory bank uses a bounded FIFO strategy with priority retention for prompted frames. The default configuration retains the 6 most recent frames plus up to 2 prompted frames. This means for a 1000-frame video, only 8 frames' worth of memory features are stored at any time, keeping GPU memory usage manageable (approximately 50-100MB for the memory bank at 1/16 resolution).

When multiple objects are tracked simultaneously, each object maintains its own set of object pointers, but spatial memory features are shared across objects. This is efficient because the image features are the same regardless of which object is being tracked -- only the mask-fused component differs. The mask decoder handles multi-object tracking by processing each object's prompts and object pointers independently, reusing the same memory-conditioned image features.

Memory is not updated for every frame. When the model's confidence is very low (IoU prediction below a threshold), the frame's memory may be skipped to avoid polluting the memory bank with unreliable information. This is particularly important for handling occlusion: when an object is fully occluded, the model should not add the occluded frame to memory, as the features would not contain useful object information.

## Ablation: Image vs Video Mode

SAM 2 operates differently depending on whether it is processing a single image or a video sequence:

**Image mode** (`SAM2ImagePredictor`): The memory attention module is effectively bypassed. Image features from the encoder are passed directly to the mask decoder without memory conditioning. The model behaves identically to SAM 1 in this mode, using only the current frame's features and the provided prompts to generate masks. No memory bank is initialized, and no temporal reasoning occurs.

**Video mode** (`SAM2VideoPredictor`): The full memory attention pipeline is active. The model maintains a memory bank, conditions each frame's features on past frame memories, and produces temporally coherent mask predictions. The first frame (or any frame with user prompts) initializes the memory bank, and subsequent frames use memory attention to propagate the segmentation. The model can handle both forward and backward propagation in a video, processing frames in both temporal directions from prompted frames to maximize coverage.

The architectural elegance of this design is that the same model weights serve both use cases. The memory attention layers add no computational overhead in image mode (they simply pass through the input), and in video mode they add a modest ~15% overhead per frame compared to image-only processing. This dual-mode capability means users can deploy a single model for both interactive image segmentation and video object tracking.
