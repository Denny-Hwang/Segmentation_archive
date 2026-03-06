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

<!-- Why treat 3D volumes as video? What problem does this solve? -->

## Architecture Overview

<!-- How SAM 2's architecture is applied to medical volumes -->

### Key Components

- **3D Volume as Video**: See [3d_volume_as_video.md](3d_volume_as_video.md)

## Technical Details

### Volume-to-Video Mapping

<!-- How 3D slices are treated as video frames -->

### Prompt Propagation Across Slices

<!-- How a prompt on one slice propagates to the full volume -->

### Memory-Based Slice Tracking

<!-- How SAM 2's memory mechanism tracks structures across slices -->

### Fine-Tuning Strategy

<!-- Adaptation details for medical data -->

## Experiments and Results

### Datasets

<!-- Medical imaging benchmarks evaluated -->

### Key Results

<!-- Main quantitative results -->

### Comparison with MedSAM and Vanilla SAM 2

<!-- Performance comparison with predecessors -->

## Strengths

<!-- List key strengths of the approach -->

## Limitations

<!-- List key limitations -->

## Connections

<!-- How this work relates to SAM 2, MedSAM, and other papers -->

## References

<!-- Key references cited in the paper -->
