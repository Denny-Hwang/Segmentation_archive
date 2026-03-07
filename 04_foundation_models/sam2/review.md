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

SAM was designed exclusively for single images and had no mechanism for maintaining temporal consistency across video frames. Applying SAM frame-by-frame to video produced flickering, inconsistent masks because each frame was processed independently without knowledge of previous predictions. Dedicated video object segmentation (VOS) methods like XMem and DeAOT addressed temporal consistency but required per-video optimization or were limited to specific object categories. There was a clear need for a unified model that could handle both images and videos with the same promptable interface.

SAM 2 addresses this by introducing a streaming memory architecture that conditions each frame's prediction on a bank of memories from previously processed frames. This allows the model to track objects across time, handle occlusions, and maintain consistent segmentation. Critically, SAM 2 subsumes SAM's image capability -- when applied to a single image (zero memory frames), it reduces to image segmentation and actually outperforms the original SAM.

## Architecture Overview

SAM 2 consists of six components: an image encoder (Hiera), a memory attention module, a prompt encoder, a mask decoder, a memory encoder, and a memory bank. For each frame, the image encoder extracts features, the memory attention module conditions these features on stored memories from past frames, and the prompt encoder + mask decoder generate mask predictions. The memory encoder then compresses the current frame's prediction into a memory representation that is added to the memory bank for use by future frames. This streaming design processes one frame at a time with constant memory cost, regardless of video length.

### Key Components

- **Streaming Memory**: See [streaming_memory.md](streaming_memory.md)
- **Video Segmentation**: See [video_segmentation.md](video_segmentation.md)
- **SA-V Dataset**: See [sav_dataset.md](sav_dataset.md)

## Technical Details

### Image Encoder

SAM 2 replaces SAM's ViT-H backbone with Hiera, a hierarchical vision transformer that produces multi-scale features more efficiently. Hiera was pre-trained using MAE on images and then on video with temporal masking. The encoder processes each frame independently at 1024x1024 resolution and produces feature maps at multiple scales (1/4, 1/8, 1/16, 1/32). The Hiera-B+ variant (used in SAM 2 base) has 80M parameters, while Hiera-L (used in SAM 2 large) has 214M parameters. Compared to SAM's ViT-H (632M), this represents a 3-8x reduction in encoder parameters while maintaining or improving segmentation quality.

### Memory Attention Module

The memory attention module is the core architectural innovation of SAM 2. It takes the current frame's image features and performs cross-attention to a set of memory tokens from previous frames. Specifically, the module uses stacked transformer blocks where the current frame's tokens serve as queries and the concatenated memory tokens serve as keys and values. This allows the model to retrieve relevant spatial and appearance information from past frames. The module adds roughly 5.5M parameters on top of the image encoder.

### Memory Encoder and Memory Bank

After generating a mask prediction for a frame, the memory encoder compresses the frame's features and predicted mask into a compact memory representation. This is done by downsampling the mask to the feature resolution and concatenating it with the image features, followed by lightweight convolutional layers that produce memory tokens. The memory bank stores these tokens using a FIFO strategy with a maximum of 6 recent frames, plus up to 2 prompted frames that are retained regardless of recency. This bounded memory ensures constant computational cost per frame.

### Prompt Encoder and Mask Decoder

The prompt encoder and mask decoder share the same design as SAM, accepting points, boxes, and masks as prompts. One key addition is that prompts can be provided on any frame in the video, not just the first frame. When a user provides a prompt on frame t, the model generates a mask for that frame and then propagates it both forward and backward in time using the memory mechanism. The mask decoder still predicts 3 candidate masks with IoU scores for ambiguity resolution.

### Training Strategy

SAM 2 is trained jointly on images and videos. Image training uses SA-1B data from SAM, while video training uses the new SA-V dataset. The training simulates interactive annotation by sampling 8-frame clips and providing simulated prompts (clicks on errors) during training. The model is trained end-to-end with a combination of focal loss and dice loss, with a batch size of 128 clips across 256 GPUs. Training takes approximately 60 hours on this hardware. Joint image-video training is critical: training on video alone degrades image performance, and training on images alone provides no temporal reasoning.

## Experiments and Results

### Video Segmentation Benchmarks

SAM 2 achieves state-of-the-art results on multiple video object segmentation benchmarks. On DAVIS 2017 (val), SAM 2 Large achieves a J&F score of 82.5, outperforming the previous best interactive method. On SA-V test, SAM 2 achieves 76.0 J&F. On YouTube-VOS 2019, SAM 2 achieves competitive J&F of 81.2. Notably, SAM 2 achieves these results while being approximately 6x faster than previous state-of-the-art methods, processing video at roughly 44 frames per second on a single A100 GPU.

### Image Segmentation

Despite being designed for video, SAM 2 also outperforms the original SAM on image segmentation. On the 37-dataset zero-shot benchmark, SAM 2 achieves higher average IoU than SAM across all prompt types. Specifically, SAM 2 Large achieves 2.0 points higher IoU with 1-point prompts and 1.5 points higher with box prompts compared to SAM ViT-H, while using only 1/3 the encoder parameters.

### Key Results

The most striking result is the efficiency-accuracy trade-off: SAM 2 Large uses 214M encoder parameters (vs. SAM's 632M) while achieving better image segmentation and adding video capability. On interactive video segmentation, SAM 2 requires 3.0x fewer interactions to reach the same mask quality as SAM applied frame-by-frame, demonstrating the value of temporal propagation. The model supports real-time interactive video annotation at approximately 44 FPS, making it practical for large-scale video annotation workflows.

## Strengths

SAM 2 unifies image and video segmentation in a single architecture, eliminating the need for separate models. The streaming memory design scales to arbitrarily long videos with constant per-frame cost. The model provides state-of-the-art video segmentation while simultaneously improving over SAM on images. Interactive video annotation is significantly more efficient thanks to temporal propagation, reducing human annotation effort by 3x. The Hiera backbone provides a better efficiency-accuracy trade-off than SAM's ViT-H.

## Limitations

SAM 2 can lose track of objects during prolonged full occlusion (more than approximately 20-30 frames of invisibility). The FIFO memory bank may drop critical frames in very long videos if the object undergoes significant appearance changes early on. Like SAM, SAM 2 does not produce semantic labels. Performance on very small objects (under 32x32 pixels) degrades compared to larger objects. The model also struggles with deformable objects that undergo extreme shape changes between frames, such as liquids or smoke.

## Connections

SAM 2 directly builds on SAM (Kirillov et al. 2023), inheriting its prompt interface and mask decoder design while adding temporal reasoning. The Hiera backbone comes from Ryali et al. 2023. The streaming memory design draws inspiration from XMem (Cheng et al. 2022) and other memory-based VOS methods, but integrates memory into a promptable framework. MedSAM-2 (Zhu et al. 2024) adapts SAM 2 for medical volumetric segmentation by treating 3D volumes as video. OMG-Seg (Li et al. 2024) offers an alternative unified approach using CLIP features.

## References

- Kirillov et al., "Segment Anything," ICCV 2023 (SAM predecessor).
- Ryali et al., "Hiera: A Hierarchical Vision Transformer without the Bells-and-Whistles," ICML 2023 (Hiera backbone).
- Cheng et al., "XMem: Long-Term Video Object Segmentation with an Atkinson-Shiffrin Memory Model," ECCV 2022 (memory-based VOS).
- Yang et al., "Decoupling Features in Hierarchical Propagation for Video Object Segmentation," NeurIPS 2022 (DeAOT).
- Oh et al., "Video Object Segmentation using Space-Time Memory Networks," ICCV 2019 (STM, foundational memory-based VOS).
