---
title: "3D Volume as Video in MedSAM-2"
date: 2025-03-06
status: complete
tags: [3d-segmentation, video-as-volume, slice-propagation, volumetric]
difficulty: advanced
---

# 3D Volume as Video

## Core Concept

The central insight of MedSAM-2 is that a 3D medical image volume can be reframed as a video sequence. Each axial slice of a CT or MRI scan corresponds to a video frame, and the progression through slices along the z-axis corresponds to temporal progression in a video. This reframing allows video segmentation architectures (specifically SAM 2) to be applied directly to volumetric medical segmentation.

## The Analogy in Detail

### Mapping Between Domains

| Video Concept | Medical Volume Equivalent |
|---------------|--------------------------|
| Frame at time t | Slice at position z |
| Temporal progression | Spatial progression along z-axis |
| Object motion | Anatomical shape change across slices |
| Object appearance | Structure entering the imaging plane |
| Object disappearance | Structure exiting the imaging plane |
| Occlusion | Overlapping structures in a slice |
| Gradual motion | Smooth anatomical variation between adjacent slices |
| Scene cut | Anatomical discontinuity (rare) |

### Why the Analogy Works

The analogy is effective because both domains share these properties:

1. **Local coherence:** Adjacent frames/slices are highly similar, with only incremental changes
2. **Object persistence:** The same entity exists across many consecutive frames/slices
3. **Gradual transformation:** Objects change shape, size, and position smoothly
4. **Bounded extent:** Objects appear at some point and disappear at another

### Where the Analogy Breaks Down

The analogy is imperfect in several ways:

- **No true motion:** Anatomical structures do not "move" between slices; they change cross-sectional shape. This means motion-based reasoning is irrelevant.
- **Anisotropic resolution:** Medical volumes often have different resolution within a slice (e.g., 0.5mm) versus between slices (e.g., 3-5mm). This is unlike video where spatial resolution is uniform.
- **Branching structures:** Blood vessels and airways branch in 3D, causing a single structure to split into two in adjacent slices. Video objects rarely split.
- **No camera motion:** Unlike video, there is no viewpoint change. All slices are parallel planes through the same volume.

## Propagation Across Slices

### Forward Propagation

Starting from a prompted slice z_0, forward propagation processes slices z_0+1, z_0+2, ..., z_N:

1. The prompted slice z_0 generates an initial memory entry (features + mask)
2. Slice z_0+1 is encoded and conditioned on the memory from z_0
3. The predicted mask for z_0+1 is added to the memory bank
4. This continues slice by slice until the end of the volume

Each subsequent slice benefits from the accumulated memory of previous slices, allowing the model to track how the structure evolves.

### Backward Propagation

Backward propagation processes slices z_0-1, z_0-2, ..., z_1 in reverse order:

1. The same prompted slice z_0 provides the starting memory
2. Slices are processed in descending order
3. The memory bank accumulates in the reverse direction

Combining forward and backward passes yields a complete volumetric segmentation from a single annotated slice.

### Bidirectional Benefits

| Strategy | Avg. DSC (abdominal CT) | Slices with > 5 DSC drop |
|----------|------------------------|--------------------------|
| Forward only | 0.82 | 18% |
| Backward only | 0.81 | 20% |
| Bidirectional | 0.88 | 7% |

Bidirectional propagation significantly reduces error accumulation, particularly for slices far from the prompted slice.

## Handling Anatomical Challenges

### Structure Appearance and Disappearance

Medical structures have finite extent along the z-axis. For example, the liver spans approximately slices 50-200 in a typical abdominal CT. The occlusion head from SAM 2 is repurposed to handle this:

- When propagation reaches a slice where the structure is absent, the occlusion head activates
- The model outputs an empty mask with high occlusion probability
- Propagation continues (in case the structure reappears), but the empty mask is not included in the final segmentation

### Branching and Splitting

When a structure branches (e.g., a blood vessel forking), the model must track multiple branches simultaneously. This is handled through:
- The memory attention mechanism attending to the pre-branch region
- The mask decoder producing a mask that covers multiple branches if they all derive from the prompted structure
- In practice, fine vessels and small branches are often lost during propagation

### Size and Shape Changes

Organs change cross-sectional shape dramatically across slices. For example, the kidney appears as a small circle at its superior pole, grows to a large bean shape in the middle, and shrinks again at the inferior pole. The memory mechanism handles this because:
- Recent FIFO memories capture the current size/shape
- The attention mechanism learns to interpolate between stored sizes
- The prompted memory provides an anchor for the expected appearance

## Multi-Axis Propagation

### Beyond Axial Slices

While axial slicing is most common, medical volumes can also be sliced along sagittal and coronal planes. MedSAM-2 can propagate along any axis:

| Axis | Slice Plane | Typical Use |
|------|------------|-------------|
| Axial | Horizontal (z) | Most common; default for CT |
| Sagittal | Left-right (x) | Spine, brain midline |
| Coronal | Front-back (y) | Abdominal organs, lungs |

### Multi-Axis Fusion

For improved accuracy, predictions from multiple propagation axes can be fused:
1. Run propagation along all three axes independently
2. Average or vote across the three predictions per voxel
3. This reduces axis-specific errors and improves 3D consistency

In practice, multi-axis fusion improves DSC by 1-3 points over single-axis propagation but requires 3x the computation.

## Comparison to Native 3D Approaches

### 3D U-Net and Similar

Native 3D architectures (3D U-Net, V-Net, nnU-Net with 3D configuration) process the entire volume at once using 3D convolutions:

| Aspect | MedSAM-2 (slice-as-video) | Native 3D (e.g., nnU-Net) |
|--------|--------------------------|--------------------------|
| Memory usage | O(1) per slice | O(V) for volume V |
| GPU memory | ~4 GB | ~12-24 GB |
| Annotation needed | 1-3 slices | Full volume |
| Multi-organ | Per-organ propagation | All organs simultaneously |
| Training data needed | Pre-trained + adaptation | 50-500 volumes per organ |
| Volumetric consistency | Enforced by propagation | Enforced by 3D convolutions |

### Strengths of the Video Approach

- Far less annotation required at inference time (1 vs. all slices)
- Lower GPU memory enables processing of high-resolution volumes
- Foundation model pretraining provides strong priors even with limited medical data
- Interactive refinement is natural (add prompts on problematic slices)

### Weaknesses of the Video Approach

- Sequential processing prevents full 3D context from informing each prediction
- Error accumulation across many slices can degrade distant predictions
- No explicit 3D shape prior (the model does not know what a kidney should look like in 3D)
- Slower inference than a single 3D forward pass when processing all slices sequentially

## Practical Recommendations

1. **Prompt placement:** Annotate the slice where the target structure has the largest cross-section for the most reliable propagation
2. **Multiple prompts:** For structures spanning 100+ slices, provide prompts every 30-50 slices
3. **Bidirectional propagation:** Always use bidirectional propagation for clinical applications
4. **Thick-slice volumes:** For volumes with spacing > 5mm, consider interpolating to thinner spacing before propagation
5. **Quality review:** Always review propagated masks on extreme slices (where the structure first appears/disappears) since these are most error-prone
