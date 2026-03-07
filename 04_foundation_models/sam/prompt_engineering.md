---
title: "Prompt Engineering for SAM"
date: 2025-03-06
status: complete
tags: [prompt-engineering, points, boxes, masks, text-prompts]
difficulty: intermediate
---

# Prompt Engineering for SAM

## Overview

SAM's promptable interface accepts diverse input types to specify what should be segmented. The choice of prompt type, placement, and combination strategy has a major effect on segmentation quality. This document covers practical guidance for each prompt modality and strategies for combining them.

## Point Prompts

### Single Point

A single foreground point is the simplest prompt. The user clicks on the object of interest, and SAM returns up to three masks at different granularity levels (subpart, part, whole object). The mask with the highest predicted IoU is selected by default.

**Best practices:**
- Click near the center of the object for the most reliable results
- Avoid clicking near object boundaries where ambiguity is highest
- Single points near the edge of an object often yield partial masks

### Foreground and Background Points

Adding background points (negative prompts) helps SAM disambiguate when multiple objects overlap or when the initial mask bleeds into surrounding regions.

**Strategy:**
1. Place a foreground point on the target object
2. Observe the initial mask output
3. Place background points on incorrectly included regions
4. Iterate until the mask converges

### Multi-Point Foreground

Multiple foreground points on the same object improve segmentation when:
- The object has disconnected parts (e.g., a person partially occluded)
- The object is elongated or irregular in shape
- A single point yields only a subpart rather than the full object

Empirically, 3-5 well-placed foreground points approach the quality of a bounding box prompt on most objects.

## Box Prompts

### Single Bounding Box

A bounding box provides the strongest single-prompt signal for SAM. The model interprets the box as specifying the spatial extent of the target object.

**Guidelines:**
- The box should tightly enclose the object with minimal padding
- Overly loose boxes may cause the model to segment the wrong object or merge nearby objects
- Boxes are encoded as two corner points with special learned embeddings

### Performance Comparison

On COCO, box-prompted SAM achieves significantly higher mask quality than single-point prompts:

| Prompt Type | mIoU (COCO val) |
|-------------|-----------------|
| 1 point (center) | ~70 |
| 3 points | ~78 |
| 1 box (ground truth) | ~85 |
| 1 box + 1 center point | ~87 |

### Generating Box Prompts Automatically

In many pipelines, box prompts are derived from an upstream object detector (e.g., Grounding DINO, YOLO). This creates a detect-then-segment paradigm where detection quality directly limits segmentation quality.

## Mask Prompts

### Dense Mask Input

SAM accepts a coarse mask as a dense prompt. This is useful for:
- Iterative refinement: feed the previous output mask back as input to improve boundaries
- Correcting errors: provide an approximate mask from another model and let SAM refine it
- Region specification: use a rough painted region to indicate the area of interest

### Encoding

Dense masks are downsampled to 256x256 via two convolutional layers (stride-2 each with output channels 4 and 16), then a 1x1 convolution projects to 256 channels. The result is added element-wise to the image embedding.

### Iterative Refinement

The most powerful use of mask prompts is iterative refinement:

1. Start with a point or box prompt to get an initial mask
2. Feed the predicted mask back as a dense prompt (optionally with additional point corrections)
3. Repeat for 1-3 iterations

Each iteration typically improves boundary quality by 1-3 IoU points, with diminishing returns after 2-3 rounds.

## Text Prompts

### CLIP-Based Text Encoding

SAM supports text prompts through CLIP text embeddings. A text string (e.g., "the dog") is encoded by CLIP's text encoder and used as a sparse prompt token.

**Current limitations:**
- Text prompts were not included in the initial public release
- Performance is below point and box prompts due to the inherent ambiguity of language
- Works best for salient, unambiguous objects ("the large red car")
- Struggles with spatial references ("the cup on the left")

### Grounded SAM

The community-developed Grounded SAM pipeline combines Grounding DINO (open-vocabulary detection) with SAM to achieve text-prompted segmentation through an indirect route: text -> detection boxes -> SAM segmentation. This approach is more robust than direct text prompting.

## Multi-Prompt Combinations

### Point + Box

Combining a box with interior points yields the best overall results. The box constrains the spatial extent while points resolve internal ambiguity (e.g., which object inside the box to segment).

### Point + Mask

Starting from a coarse mask and adding corrective points allows fine-grained interactive editing. This is the typical workflow in annotation tools built on SAM.

### Batched Prompts

SAM can process multiple independent prompts against the same image embedding in a single forward pass. This is critical for efficiency in:
- Automatic mask generation (grid of point prompts)
- Instance segmentation (one box per detected object)
- Interactive tools (multiple objects being annotated simultaneously)

## Automatic Mask Generation

SAM includes an automatic mode that generates masks without any user prompts:

1. A regular grid of points (32x32 by default) is placed over the image
2. Each point generates 3 candidate masks
3. Masks are filtered by predicted IoU (threshold ~0.88) and stability score
4. Non-maximum suppression removes duplicates
5. The result is a complete segmentation of the image into object masks

This mode generates high-quality masks for downstream use but is computationally expensive (~2-4 seconds per image on GPU).

## Practical Recommendations

| Scenario | Recommended Prompt | Notes |
|----------|-------------------|-------|
| Interactive annotation | Box + corrective points | Best quality-effort tradeoff |
| Automated pipeline | Box from detector | Grounding DINO + SAM |
| Full image segmentation | Automatic grid | Slow but comprehensive |
| Refinement of existing masks | Mask + points | Iterative improvement |
| Quick single-object selection | Single center point | Fast but lower quality |
