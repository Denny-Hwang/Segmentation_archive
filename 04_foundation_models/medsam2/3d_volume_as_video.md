---
title: "3D Volume as Video in MedSAM-2"
date: 2025-03-06
status: planned
tags: [3d-segmentation, video-as-volume, slice-propagation, volumetric]
difficulty: advanced
---

# 3D Volume as Video

## Overview

The volume-as-video paradigm is the central conceptual contribution of MedSAM-2: it reframes 3D medical image segmentation as a video object segmentation problem by treating each 2D slice of a volumetric scan as a video frame. This mapping exploits a fundamental similarity between the two domains -- both involve sequences of 2D images where objects change gradually from one element to the next. In video, objects move and deform over time; in a medical volume, anatomical structures grow, shrink, and change shape across spatial slices. By applying SAM 2's streaming memory architecture to this reframed problem, MedSAM-2 achieves inter-slice propagation without requiring any 3D convolutional operations or architectural modifications.

## Conceptual Mapping

The mapping between video and medical volumes is straightforward: time steps correspond to spatial positions along the slice axis, and object motion corresponds to anatomical change across slices. In an axial CT scan, for example, the liver appears as a small cross-section in superior slices, expands to its maximum extent in the mid-abdomen, and disappears in inferior slices -- analogous to an object entering, growing, and leaving a video frame. Camera motion in video corresponds to slight misalignment between slices due to patient breathing or motion artifacts.

This analogy holds well for most anatomical structures but has limits. Video objects typically maintain consistent appearance (texture, color) across frames, whereas medical structures can change dramatically in appearance across slices (e.g., a vertebral body transitions from cortical bone to cancellous bone to disc space). Additionally, the "frame rate" equivalent in medical volumes (inter-slice spacing) can vary significantly: thin-slice CT at 1mm spacing produces gradual transitions, while thick-slice acquisitions at 5mm may show abrupt changes between slices.

## Slice Ordering and Direction

Slices are ordered along the anatomical axis of the volume, typically the axial (head-to-foot) direction for CT and MRI. The choice of propagation direction matters: processing slices from superior to inferior may work well for structures that appear at the top of the volume, while inferior-to-superior may be better for pelvic structures. MedSAM-2 handles this by performing bidirectional propagation from the prompted slice: first processing forward (increasing slice index) to the end, then backward (decreasing slice index) to the beginning.

For non-axial orientations (sagittal, coronal), the same principle applies but with different anatomical axes. Some structures are better visualized in specific planes; for example, spinal structures are better segmented in the sagittal plane. MedSAM-2 allows the user to choose the propagation axis, though axial ordering is the default. Multi-axis propagation (processing the same volume along multiple axes and fusing results) has been explored in concurrent work and can improve performance by approximately 1-2 dice points at the cost of 2-3x computation.

## Prompt on a Single Slice

The user provides a prompt on a single "key slice" where the target structure is clearly visible and well-defined. This is typically a slice near the center of the structure's extent, where the cross-sectional area is largest and boundaries are most distinct. The prompt can be a bounding box (most common), one or more foreground/background points, or a coarse mask from a prior segmentation. The mask decoder generates a prediction on this key slice, and the memory encoder stores the slice's features and mask in the memory bank.

Choosing the right key slice significantly affects propagation quality. A key slice at the edge of a structure (where the cross-section is small and boundaries are ambiguous) produces poorer propagation than a central slice. In practice, clinicians naturally select informative slices because they scroll through the volume and identify the slice where the target structure is most clearly visible. Automated key slice selection (choosing the slice with the largest predicted mask area) has been explored and produces results within 1-2 dice points of human selection.

## Propagation Mechanism

Propagation from the key slice to adjacent slices uses SAM 2's memory attention mechanism. For each new slice, the Hiera encoder extracts image features, and the memory attention module performs cross-attention between these features and the stored memory tokens from previously processed slices. This cross-attention allows the model to "find" the target structure in the new slice by matching it against the stored appearance and location information. The mask decoder then produces a prediction, which is encoded into memory for use by subsequent slices.

The propagation is remarkably robust to gradual changes in structure appearance and position. For a liver segmentation starting from a central axial slice, the model successfully tracks the structure as it shrinks in superior slices (where only a small dome is visible) and as it develops lobar divisions in inferior slices. Typical propagation can handle 30-50 slices from the key slice before significant quality degradation, depending on the complexity of the structure. For very large structures spanning 100+ slices, adding 1-2 additional prompts at strategic locations ensures high-quality segmentation throughout.

## Handling Anatomical Variation Across Slices

Anatomical structures in medical volumes undergo changes that differ from typical video object motion. Structures can appear abruptly (e.g., a rib entering the field of view), bifurcate (e.g., the aorta splitting into iliac arteries), merge with adjacent structures (e.g., when two organs are in contact), or disappear entirely (e.g., the boundary slices of an organ). MedSAM-2 handles appearance and disappearance through SAM 2's occlusion detection mechanism, which outputs an empty mask when the target structure is not present.

Bifurcation and merging are more challenging because they fundamentally change the topology of the structure. When a vessel bifurcates, the model may track only one branch or merge both branches into a single mask. Similarly, when two structures that were separate in one slice merge in the next, the model may incorrectly segment both as a single object. These topological changes are the primary failure mode of the volume-as-video approach, and handling them typically requires additional prompts at the bifurcation point.

## Comparison with Direct 3D Segmentation

Native 3D segmentation architectures (3D U-Net, V-Net, nnU-Net) process the entire volume simultaneously using 3D convolutions that capture inter-slice context directly. These approaches achieve the highest segmentation quality (e.g., 88.1% dice on BTCV with nnU-Net) because they model 3D spatial relationships explicitly. However, they require dense volumetric annotations for training (every voxel labeled), large GPU memory (a 3D U-Net processing a 512x512x512 volume requires 24+ GB), and cannot easily incorporate interactive prompts.

The volume-as-video approach trades some accuracy for dramatically reduced annotation requirements and interactive capability. MedSAM-2 achieves 84-87% dice with 1-3 prompts versus nnU-Net's 88% with full supervision. The 2D processing means each slice requires only standard GPU memory (~2GB), enabling processing of arbitrarily large volumes. The interactive prompt interface allows rapid adaptation to new structures without retraining. For clinical workflows where speed and flexibility matter more than maximum accuracy, the volume-as-video approach offers a compelling trade-off.

## Implementation Notes

To process a 3D volume, slices are extracted along the chosen axis and saved as individual 2D images. Each slice is resized to 1024x1024 with appropriate intensity normalization (windowing for CT, per-volume normalization for MRI). The SAM 2 video predictor is initialized with the slice sequence, and prompts are added on the key slice(s). Bidirectional propagation is performed by calling `propagate_in_video()` twice: once forward, once backward from the prompted slice. The resulting per-slice masks are stacked to reconstruct the 3D segmentation volume. Post-processing (connected component analysis, hole filling) is applied to the 3D volume to improve consistency. Total processing time for a 200-slice CT volume is approximately 10-15 seconds on an A100 GPU.
