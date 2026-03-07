---
title: "MedSAM-2: Segment Medical Images As Video Via Segment Anything Model 2"
date: 2025-03-06
status: planned
tags: [foundation-model, medical-segmentation, video-as-volume, sam2]
difficulty: advanced
---

# MedSAM-2

## Meta Information

| Field | Value |
|-------|-------|
| **Paper Title** | MedSAM-2: Segment Medical Images As Video Via Segment Anything Model 2 |
| **Authors** | Zhu, J., Qi, Y., Wu, J. |
| **Year** | 2024 |
| **Venue** | arXiv |
| **arXiv** | [2408.00874](https://arxiv.org/abs/2408.00874) |
| **Difficulty** | Advanced |

## One-Line Summary

MedSAM-2 leverages SAM 2's video segmentation capability by treating 3D medical volumes as video sequences, enabling efficient volumetric segmentation with minimal user prompts.

## Motivation and Problem Statement

Medical image segmentation of 3D volumes (CT, MRI) traditionally requires either labor-intensive slice-by-slice annotation or specialized 3D architectures like 3D U-Net that demand large amounts of volumetric training data. MedSAM improved upon SAM for medical images but processes each 2D slice independently, ignoring the spatial continuity between adjacent slices in a volume. This results in inconsistent segmentation across slices and requires a separate bounding box prompt for every slice, which is impractical for volumes containing 100-500 slices.

The key insight of MedSAM-2 is that consecutive slices in a 3D volume are visually analogous to consecutive frames in a video: the anatomical structures change gradually, appearing, growing, and disappearing across slices just as objects move and transform across video frames. SAM 2's streaming memory architecture, designed for temporal propagation in video, can therefore be repurposed for spatial propagation across slices. This allows a user to provide a single prompt on one slice and have the segmentation automatically propagated to the entire volume, reducing annotation effort by 50-100x compared to per-slice prompting.

## Architecture Overview

MedSAM-2 uses SAM 2's architecture directly: the Hiera image encoder processes each slice as an independent "frame," the memory attention module conditions each slice on stored representations from previously processed slices, and the mask decoder produces per-slice predictions. The memory encoder compresses each slice's features and predicted mask into a memory token stored in the memory bank. The streaming design means slices are processed sequentially (e.g., from slice 1 to slice N), with each slice's prediction informed by the accumulated memory of prior slices.

### Key Components

- **3D Volume as Video**: See [3d_volume_as_video.md](3d_volume_as_video.md)

## Technical Details

### Volume-to-Video Mapping

A 3D medical volume of shape (D, H, W) -- where D is the number of slices -- is treated as a video of D frames, each of size (H, W). Slices are processed in anatomical order (e.g., superior-to-inferior for axial CT). Each slice is independently resized to 1024x1024 and encoded by the Hiera image encoder. The spatial continuity between adjacent slices provides the same kind of temporal coherence that SAM 2 exploits in video, making the mapping natural: anatomical structures change gradually from slice to slice, just as objects move gradually from frame to frame.

### Prompt Propagation Across Slices

The user provides a prompt (typically a bounding box or a few points) on a single "key slice" where the target structure is clearly visible. SAM 2's mask decoder generates a mask on this key slice, which is then encoded into memory. As subsequent slices are processed, the memory attention module retrieves this stored information, allowing the model to locate and segment the same structure on adjacent slices without additional prompts. Propagation proceeds bidirectionally: forward from the key slice to the last slice, then backward from the key slice to the first slice.

For structures that span a large portion of the volume (e.g., liver in abdominal CT, spanning 50-80 slices), a single prompt on the central slice is often sufficient to segment the entire structure. For structures with complex 3D morphology or significant appearance changes across slices (e.g., a branching vascular tree), 2-3 prompts on strategically chosen slices yield significantly better results. The interactive refinement workflow is analogous to SAM 2's video annotation: the user inspects the propagated results and adds corrective prompts on slices where the mask is incorrect.

### Memory-Based Slice Tracking

The memory bank stores representations from previously processed slices, with the same FIFO + prompted frame strategy as SAM 2. For a volume of 200 slices, the memory bank retains the 6 most recently processed slices plus up to 2 prompted slices. This bounded memory ensures that processing a 500-slice volume has the same per-slice computational cost as processing a 50-slice volume. The memory attention module retrieves spatial and appearance information from stored slices, enabling the model to track structures as they change shape, size, and position across slices.

The occlusion handling mechanism from SAM 2 maps naturally to medical volumes: when a structure disappears in certain slices (e.g., a kidney that is present only in a subset of axial slices), the model's occlusion detection outputs an empty mask and avoids corrupting the memory bank. When the structure reappears in later slices, the memory attention retrieves pre-disappearance information to resume segmentation.

### Fine-Tuning Strategy

MedSAM-2 fine-tunes SAM 2 on a collection of 3D medical imaging datasets, treating each volume as a video sequence. Training clips are sampled as contiguous sequences of 8 slices from each volume, with ground-truth masks provided on all slices and simulated interactive prompts (clicks on errors) on 1-2 slices per clip. The model is fine-tuned end-to-end using a combination of dice loss and cross-entropy loss. Training uses the AdamW optimizer with a learning rate of 5e-5 and runs for 50 epochs on 8 A100 GPUs.

## Experiments and Results

### Datasets

MedSAM-2 was evaluated on several 3D medical segmentation benchmarks: BTCV (13 abdominal organs on CT, 30 volumes), ACDC (3 cardiac structures on MRI, 100 volumes), KiTS (kidney and tumor on CT, 300 volumes), and BraTS (brain tumor subregions on multi-modal MRI, 1251 volumes). The model was also tested on 2D medical segmentation tasks to verify that the adaptation did not degrade single-image performance.

### Key Results

On volumetric segmentation with a single prompt per structure, MedSAM-2 achieved: BTCV mean dice 84.1% (vs. MedSAM per-slice at 79.3%), ACDC mean dice 89.7% (vs. MedSAM at 85.2%), and KiTS mean dice 86.5% (vs. MedSAM at 81.8%). The 4-5 dice point improvement over MedSAM demonstrates the value of inter-slice propagation. With 3 prompts per structure, performance improved further to 86.8% on BTCV and 91.2% on ACDC, approaching the performance of fully supervised 3D methods like nnU-Net (BTCV: 88.1%, ACDC: 92.1%) while requiring only 3 user interactions per structure.

### Comparison with MedSAM and Vanilla SAM 2

MedSAM-2 outperforms both MedSAM (which lacks inter-slice propagation) and vanilla SAM 2 (which lacks medical domain adaptation) by a significant margin. Vanilla SAM 2 applied to medical volumes achieves approximately 65-70% dice, similar to vanilla SAM, because its features are not adapted to medical image characteristics. MedSAM processes each slice independently, achieving approximately 80-85% dice but requiring per-slice prompts. MedSAM-2 combines the best of both: medical domain adaptation from MedSAM and temporal/spatial propagation from SAM 2.

## Strengths

MedSAM-2 dramatically reduces the annotation effort for 3D medical image segmentation, requiring only 1-3 prompts per structure instead of per-slice annotation. The volume-as-video paradigm is elegant and requires no architectural modifications to SAM 2. Inter-slice propagation improves consistency across slices, producing smoother 3D segmentation surfaces. The streaming design scales to arbitrarily large volumes with constant per-slice cost. The approach generalizes across CT and MRI modalities with a single model.

## Limitations

The sequential slice processing introduces order dependence: segmentation quality can differ depending on the direction of propagation (superior-to-inferior vs. inferior-to-superior). The 2D image encoder processes each slice independently, missing 3D contextual features that native 3D architectures (3D U-Net, nnU-Net) can capture. Performance still lags behind fully supervised 3D methods by approximately 2-5 dice points, particularly on complex structures with thin boundaries. The model requires fine-tuning on medical data and does not work out of the box with vanilla SAM 2 weights.

## Connections

MedSAM-2 combines ideas from SAM 2 (Ravi et al. 2024) and MedSAM (Ma et al. 2024). The volume-as-video paradigm connects to earlier work on using 2D models for 3D segmentation by processing slices sequentially. The streaming memory mechanism originates from SAM 2 and draws on prior video object segmentation methods like XMem. For comparison, nnU-Net (Isensee et al. 2021) represents the fully supervised 3D approach that MedSAM-2 aims to match with minimal prompts. SAM-Adapter (Chen et al. 2023) offers an alternative adaptation strategy using parameter-efficient methods.

## References

- Ravi et al., "SAM 2: Segment Anything in Images and Videos," arXiv 2024 (SAM 2 foundation).
- Ma et al., "Segment Anything in Medical Images," Nature Communications 2024 (MedSAM predecessor).
- Kirillov et al., "Segment Anything," ICCV 2023 (original SAM).
- Isensee et al., "nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation," Nature Methods 2021 (comparison baseline).
- Cheng et al., "XMem: Long-Term Video Object Segmentation with an Atkinson-Shiffrin Memory Model," ECCV 2022 (memory-based propagation).
