---
title: "SA-V Dataset"
date: 2025-03-06
status: complete
tags: [dataset, sa-v, video-annotation, data-engine]
difficulty: intermediate
---

# SA-V Dataset

## Overview

SA-V (Segment Anything in Video) is the largest video segmentation dataset at the time of its release. It was developed alongside SAM 2 using a data engine approach analogous to the SA-1B pipeline but adapted for video content. The dataset provides dense mask annotations (masklets) that track objects across video frames.

## Dataset Statistics

| Property | Value |
|----------|-------|
| Total videos | 50,900 |
| Total masklets | 642,600 |
| Avg. masklets per video | ~12.6 |
| Avg. video length | ~14 seconds |
| Total frames annotated | ~35.5 million |
| Resolution | Mixed (up to 1080p) |
| Frame rate | Mixed (primarily 24-30 FPS) |

A "masklet" refers to a complete spatiotemporal mask track: the segmentation of a single object across all frames in which it appears. Each masklet includes per-frame binary masks with temporal correspondence.

## Annotation Pipeline

### Phase 1: Interactive Annotation with SAM

In the initial phase, annotators used SAM (image-only) as an interactive tool to annotate individual frames:

1. Annotators selected objects to track in a video
2. For each object, they annotated key frames using SAM-assisted point/box prompting
3. Masks were interpolated or manually adjusted for intermediate frames
4. Quality review ensured temporal consistency

This phase was labor-intensive but produced high-quality seed annotations for model training.

### Phase 2: Model-Assisted Video Annotation

Once an early version of SAM 2 was trained, it was deployed as the annotation tool:

1. Annotators provided a prompt on a single frame
2. SAM 2 propagated the mask across the entire video
3. Annotators reviewed the propagation and provided corrective prompts on error frames
4. The corrected masklets were added to the dataset

This phase dramatically improved annotation efficiency because annotators only needed to correct errors rather than annotate each frame independently.

### Phase 3: Automatic Annotation with Quality Filtering

The final phase used SAM 2 to generate masklets automatically:

1. Object candidates were identified using automatic mask generation on key frames
2. SAM 2 propagated each candidate through the video
3. Quality filters removed unstable or incoherent tracks
4. Human spot-checks validated random samples

This phase produced the bulk of the dataset volume.

## Comparison to Existing Video Segmentation Datasets

| Dataset | Videos | Objects | Annotation Density | Year |
|---------|--------|---------|-------------------|------|
| DAVIS 2017 | 90 | 376 | Every frame | 2017 |
| YouTube-VOS | 4,453 | 7,755 | Every 5th frame | 2018 |
| MOSE | 2,149 | 5,200 | Every frame | 2023 |
| BURST | 2,914 | 16,089 | Every frame | 2023 |
| **SA-V** | **50,900** | **642,600** | **Every frame** | **2024** |

SA-V is approximately 10x larger than previous video segmentation datasets in video count and nearly 100x larger in annotated object tracks.

## Video Content Characteristics

### Diversity

SA-V videos cover a wide range of scenarios:
- Indoor and outdoor environments
- Animals, vehicles, people, tools, furniture, and more
- Various motion types: static, slow, fast, deformable, rigid
- Different camera behaviors: static, panning, handheld, drone footage

### Challenging Scenarios

The dataset intentionally includes challenging cases:
- **Occlusion:** Objects passing behind other objects and reappearing
- **Fast motion:** Objects with significant displacement between frames
- **Similar objects:** Multiple instances of the same object class in close proximity
- **Scale changes:** Objects moving toward or away from the camera
- **Deformation:** Non-rigid objects changing shape (e.g., animals, cloth)
- **Crowded scenes:** Many objects with frequent interactions

## Masklet Quality

### Annotation Consistency

Quality metrics from human evaluation on sampled masklets:
- Temporal consistency (smooth boundaries across frames): 92% rated good or better
- Boundary accuracy (alignment with actual object edges): 89% rated good or better
- Identity consistency (same object tracked throughout): 95% rated good or better

### Automatic vs. Manual Masklets

| Source | Fraction of Dataset | Avg. Quality Score |
|--------|--------------------|--------------------|
| Manual (Phase 1) | ~5% | 4.5/5.0 |
| Model-assisted (Phase 2) | ~15% | 4.3/5.0 |
| Automatic (Phase 3) | ~80% | 4.0/5.0 |

The automatic masklets are slightly lower quality on average but provide essential scale for training.

## Dataset Splits

| Split | Videos | Masklets |
|-------|--------|----------|
| Train | 45,000 | ~570K |
| Val | 2,950 | ~36K |
| Test | 2,950 | ~36K |

The test set annotations are held out for benchmarking purposes.

## Role in SAM 2 Training

SA-V is combined with SA-1B during SAM 2 training:
- Images from SA-1B are treated as single-frame videos
- Video clips from SA-V provide the temporal supervision signal
- The combined training ensures the model works well on both images and videos
- Training alternates between image and video batches

## Access and License

SA-V is publicly released for research purposes alongside the SAM 2 code and model weights. Videos are provided with corresponding masklet annotations in a frame-by-frame mask format.
