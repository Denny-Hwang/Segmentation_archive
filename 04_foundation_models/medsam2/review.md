---
title: "MedSAM-2: Segment Medical Images As Video Via Segment Anything Model 2"
date: 2025-03-06
status: complete
tags: [foundation-model, medical-segmentation, video-as-volume, sam2]
difficulty: advanced
---

# MedSAM-2

## Paper Overview

**Title:** MedSAM-2: Segment Medical Images As Video Via Segment Anything Model 2
**Authors:** Various (building on SAM 2 by Meta FAIR and MedSAM by Ma et al.)
**Context:** Extends SAM 2's video segmentation capabilities to 3D medical image volumes

MedSAM-2 adapts SAM 2 for 3D medical image segmentation by exploiting a key insight: a stack of 2D medical image slices (from CT or MRI) can be treated as frames of a video. This allows SAM 2's streaming memory architecture and temporal propagation to be directly applied to volumetric medical data, propagating annotations from a few labeled slices to the entire volume.

## Motivation

### Limitations of 2D Medical Segmentation

MedSAM and similar 2D approaches process each slice independently, which means:
- No inter-slice consistency: adjacent slices may produce contradictory segmentations
- Every slice requires a prompt (bounding box), which is burdensome for volumes with 100+ slices
- Volumetric context (e.g., the 3D shape of an organ) is entirely ignored

### The Video-Volume Analogy

3D medical volumes share key properties with videos:
- **Sequential structure:** Slices are ordered along the z-axis, just like video frames in time
- **Temporal coherence:** Adjacent slices show gradually changing anatomy, similar to gradual motion in video
- **Object persistence:** The same anatomical structure appears across many consecutive slices
- **Appearance evolution:** The shape and size of a structure changes smoothly across slices

This analogy makes SAM 2's streaming memory architecture directly applicable.

## Architecture

MedSAM-2 uses SAM 2's architecture with adaptations for medical imaging:

### Image Encoder
- Hiera backbone (same as SAM 2)
- Fine-tuned on medical imaging data or used with adapter layers
- Processes each slice independently, producing per-slice feature maps

### Memory Architecture
- SAM 2's streaming memory system is applied across slices instead of video frames
- The FIFO memory bank stores representations from recently processed slices
- Prompted memories correspond to slices where the radiologist provided annotations

### Mask Decoder
- Produces per-slice segmentation masks conditioned on memory from adjacent slices
- Occlusion prediction is repurposed to detect slices where the target structure is absent (e.g., above or below an organ)

## Workflow

### One-Prompt Volumetric Segmentation

The standard MedSAM-2 workflow:

1. **Initial annotation:** The radiologist provides a prompt (box or points) on a single representative slice (typically the slice where the target structure is largest)
2. **Forward propagation:** MedSAM-2 processes slices sequentially from the prompted slice toward the last slice, propagating the segmentation via memory
3. **Backward propagation:** The model processes slices in reverse from the prompted slice toward the first slice
4. **Merge:** Forward and backward predictions are combined into the complete 3D segmentation

### Few-Prompt Refinement

For higher accuracy:
1. Annotate 2-3 slices distributed across the volume
2. Run propagation from each annotated slice
3. Use the memory from all annotated slices to improve intermediate predictions
4. Optionally review and correct slices with low confidence

## Key Results

### Comparison to MedSAM (2D)

| Task | MedSAM (per-slice box) | MedSAM-2 (1 slice) | MedSAM-2 (3 slices) |
|------|----------------------|--------------------|--------------------|
| Liver (CT) | 0.91 DSC | 0.88 DSC | 0.92 DSC |
| Kidney (CT) | 0.89 DSC | 0.86 DSC | 0.90 DSC |
| Brain tumor (MRI) | 0.82 DSC | 0.78 DSC | 0.84 DSC |
| Cardiac (MRI) | 0.85 DSC | 0.80 DSC | 0.86 DSC |

With just 3 annotated slices, MedSAM-2 matches or exceeds MedSAM's per-slice annotation performance while requiring dramatically less user input.

### Annotation Efficiency

For a typical abdominal CT with 200 slices:
- **MedSAM:** 200 bounding boxes required (~15 minutes)
- **MedSAM-2 (1 prompt):** 1 bounding box required (~5 seconds)
- **MedSAM-2 (3 prompts):** 3 bounding boxes required (~15 seconds)

The reduction in annotation burden is approximately 60-200x.

## Technical Considerations

### Slice Spacing and Propagation

Medical volumes have variable slice spacing (e.g., 1mm for high-resolution CT, 5mm for standard MRI). This affects propagation quality:
- **Thin slices (< 2mm):** Strong inter-slice similarity; propagation works well
- **Thick slices (> 5mm):** Large anatomical changes between slices; propagation is less reliable
- MedSAM-2 handles this better than naive interpolation because it uses learned features rather than simple spatial proximity

### Bidirectional vs. Unidirectional

Bidirectional propagation (from the prompted slice in both directions) is essential for medical volumes:
- Unidirectional propagation accumulates errors over many slices
- Bidirectional propagation ensures slices near the prompt have the best quality
- Multiple prompts at different positions further reduce error accumulation

### Memory Bank Adaptation

The FIFO memory bank size may need adjustment for medical applications:
- Organs with gradual shape changes: 6 frames (default) works well
- Organs with rapid shape changes (e.g., branching vessels): larger banks may help
- Very long volumes (300+ slices): prompted memories at regular intervals prevent drift

## Strengths

- Dramatic reduction in annotation burden compared to per-slice methods
- Leverages SAM 2's proven streaming architecture without major architectural changes
- Handles variable-size structures through learned propagation
- Natural handling of structure appearance/disappearance via the occlusion head
- Works across CT and MRI modalities

## Limitations

- Propagation quality degrades for structures with highly variable appearance across slices
- Anisotropic resolution (common in MRI) makes the video analogy less applicable in some directions
- Cannot propagate across anatomical discontinuities (e.g., separate vertebrae)
- Requires fine-tuning or adaptation for optimal medical performance
- Single-object segmentation per propagation pass (multi-organ requires multiple runs)

## Impact

MedSAM-2 demonstrated that the video-volume analogy is practically effective for medical image segmentation, opening a pathway to efficient 3D medical annotation using foundation models designed for video.

## Citation

```
"MedSAM-2: Segment Medical Images As Video Via Segment Anything Model 2." 2024.
```
