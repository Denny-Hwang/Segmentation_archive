---
title: "SA-1B Dataset"
date: 2025-03-06
status: complete
tags: [dataset, sa-1b, data-engine, annotation, billion-scale]
difficulty: intermediate
---

# SA-1B Dataset

## Overview

SA-1B (Segment Anything 1 Billion) is the largest segmentation dataset ever created, containing over 1.1 billion mask annotations across 11 million licensed, privacy-preserving images. It was built using a novel data engine that co-evolved the SAM model and the annotation process across three phases of increasing automation.

## Dataset Statistics

| Property | Value |
|----------|-------|
| Total images | 11,000,000 |
| Total masks | 1,100,000,000 |
| Avg. masks per image | ~100 |
| Median image resolution | 3300 x 4950 pixels |
| Image source | Licensed from photography providers |
| Mask format | COCO RLE encoding |
| Geographic coverage | 200+ countries |

## The Three-Phase Data Engine

The core innovation behind SA-1B is a data engine that iteratively improves both the model and the dataset through three phases.

### Phase 1: Assisted-Manual Annotation

**Duration:** Initial phase
**Masks produced:** ~4.3 million masks from 120K images

In this phase, professional annotators used a SAM-powered interactive tool (similar to a browser-based annotation app) to label masks:

1. Annotators clicked foreground/background points to indicate objects
2. SAM generated mask predictions in real time
3. Annotators refined the masks using additional point clicks or brush tools
4. Every visually distinct object was labeled, including parts and subparts

The SAM model was retrained 6 times during this phase using the growing pool of annotations. Mean annotation time dropped from 34 seconds to 14 seconds per mask as the model improved.

### Phase 2: Semi-Automatic Annotation

**Duration:** Second phase
**Masks produced:** ~5.9 million masks from 180K images

In this phase, SAM was used to pre-generate confident masks automatically, and annotators focused on labeling additional objects that the model missed:

1. SAM detected prominent objects automatically
2. Annotators were shown the pre-generated masks and asked to annotate any remaining unannotated objects
3. This increased the diversity of annotated objects, since annotators focused on harder cases

The model was retrained 5 times during this phase. The average number of masks per image increased from 44 (phase 1) to 72 (phase 2) as annotators filled in gaps.

### Phase 3: Fully Automatic Annotation

**Duration:** Final phase, producing the vast majority of the dataset
**Masks produced:** ~1.1 billion masks from 11M images

In the fully automatic phase, SAM generated all masks without human intervention:

1. A 32x32 grid of point prompts was applied to each image
2. For each point, SAM predicted a set of candidate masks
3. Confident, stable, non-duplicate masks were retained after filtering
4. NMS (non-maximum suppression) removed overlapping predictions

Quality was ensured through:
- IoU prediction thresholding (selecting only high-confidence masks)
- Stability filtering (checking that masks remain consistent under small perturbations)
- Manual quality studies comparing automatic masks to professional annotations

## Quality Assessment

### Comparison to Professional Annotations

A human study compared 500 randomly sampled automatic masks to professional re-annotations of the same images:

| Metric | Automatic Masks | Professional Masks |
|--------|----------------|--------------------|
| Mean IoU (vs. consensus) | 90.0% | 91.0% |
| Rating: "Good" or better | 94% | 97% |

The gap between automatic and professional quality is remarkably small, validating the fully automatic pipeline.

### Mask Properties

- Masks span a wide range of sizes, from tiny parts (< 100 pixels) to image-spanning objects
- The dataset contains masks for both "things" (countable objects) and "stuff" (amorphous regions)
- Concavity and boundary complexity vary widely, capturing simple shapes (circles) and complex ones (tree branches)

## Diversity

### Geographic Diversity

Images were sourced to ensure representation across geographic regions:
- Coverage spans countries across all inhabited continents
- The top-3 represented countries account for less than 25% of images
- This mitigates geographic bias common in datasets like COCO and ImageNet

### Content Diversity

SA-1B covers an extremely broad range of visual content:
- Indoor and outdoor scenes
- Natural and man-made objects
- Close-up and wide-angle perspectives
- Various lighting conditions and weather
- Text, symbols, and graphical elements

## Privacy and Ethical Considerations

- All images are licensed from professional photography providers (not scraped from the web)
- Faces and license plates were blurred using detection models
- No personally identifiable information is included in metadata
- The dataset was analyzed for potential biases across geographic and demographic dimensions
- Images depicting harmful content were filtered

## Comparison to Other Datasets

| Dataset | Images | Annotations | Type |
|---------|--------|-------------|------|
| COCO | 330K | 2.5M instances | Instance masks |
| ADE20K | 25K | 450K instances | Semantic + instance |
| OpenImages V7 | 9M | 16M boxes | Boxes (some masks) |
| **SA-1B** | **11M** | **1.1B masks** | **Class-agnostic masks** |

SA-1B is approximately 400x larger than COCO in mask count. However, it provides class-agnostic masks (no category labels), which differentiates it from traditional datasets.

## Usage and Impact

### As Pretraining Data

SA-1B serves as the pretraining data source for SAM and its derivatives. Models pretrained on SA-1B learn a general-purpose visual segmentation prior that transfers well to downstream tasks.

### Data Engine as a Methodology

The three-phase data engine concept has influenced subsequent dataset creation efforts, including SA-V (for video) and domain-specific adaptations. The key insight is that model and data can be co-developed in a virtuous cycle.

### Access

SA-1B is publicly available for research under a permissive license. Images are provided at high resolution with corresponding mask annotations in COCO RLE format.
