---
title: "SA-V Dataset"
date: 2025-03-06
status: planned
tags: [dataset, sa-v, video-annotation, data-engine]
difficulty: intermediate
---

# SA-V Dataset

## Overview

SA-V (Segment Anything in Video) is a large-scale video segmentation dataset containing 50,900 videos with 642,600 spatio-temporal mask annotations (called "masklets"). The dataset was built using a model-in-the-loop annotation pipeline powered by SAM 2 itself, following the same data-engine philosophy as SA-1B for images. SA-V contains 35.5 million individual frame-level masks across all masklets, making it the largest video segmentation dataset by a significant margin. The videos are diverse in content, covering indoor and outdoor scenes, various object categories, and a wide range of motion patterns.

## Data Engine for Video

The SA-V data engine adapted the three-phase approach from SA-1B to video annotation. In the first phase (SAM per frame), annotators used SAM to segment objects on individual key frames, and masks were linked across frames manually. In the second phase (SAM 2 with correction), an early version of SAM 2 propagated masks across frames, and annotators provided corrective prompts on frames where propagation failed. In the third phase (SAM 2 automatic + verification), SAM 2 generated masklets automatically across entire videos, and annotators verified and filtered the results.

Each phase produced progressively more data with less per-mask effort. The first phase averaged approximately 37 seconds per masklet, the second phase reduced this to approximately 12 seconds, and the third phase (verification only) required approximately 4 seconds per masklet. The model was retrained between phases, improving its propagation quality and reducing annotation effort in subsequent phases. This virtuous cycle between model improvement and data collection mirrors the SA-1B data engine but operates in the temporal domain.

## Dataset Statistics

SA-V contains 50,900 videos averaging approximately 14 seconds in length at 24fps. The dataset includes 642,600 masklets with a median of 10 masklets per video. The total number of individual frame-level masks is 35.5 million. Videos are sourced from a licensed provider and span a wide range of scenes: approximately 35% indoor, 65% outdoor. Object categories include people (28%), animals (15%), vehicles (12%), household objects (18%), and others (27%). The average masklet spans approximately 75% of the video frames it appears in, reflecting that most objects are visible for the majority of the clip.

Resolution varies across videos, with a median resolution of approximately 1280x720. Frame rates range from 24 to 60fps, with the majority at 24fps or 30fps. The dataset is split into train (47,500 videos), val (1,200 videos), and test (2,200 videos) sets with no overlap in source content.

## Masklet Annotations

A "masklet" is a spatio-temporal mask annotation that tracks a single object across multiple frames. Unlike frame-level masks in SA-1B, masklets encode temporal continuity: each masklet links the same physical object across all frames where it is visible. Masklets handle appearance changes, partial occlusion (marked with reduced mask area), and full occlusion (marked with absent frames). The average masklet contains masks on 250 frames, and the quality of individual frame masks was rated at 92% by human evaluators.

Masklets are stored as sequences of per-frame run-length encoded (RLE) masks with frame indices indicating which frames contain the object. Frames where the object is fully occluded are explicitly marked as "absent" rather than omitted, allowing the dataset to serve as ground truth for occlusion detection. Approximately 12% of masklets contain at least one full-occlusion event.

## Quality and Diversity

Annotation quality was assessed through a human evaluation study where independent raters scored mask quality on a 1-5 scale. SA-V masklets received an average score of 4.3 out of 5, with 92% of individual frame masks rated as "good" or "excellent." The most common quality issues were slight boundary imprecision on deformable objects and occasional identity confusion in multi-object scenes with similar-looking objects.

Content diversity was ensured through geographic and semantic stratification during video sourcing. Videos span 47 countries across 6 continents. Scene types include urban streets, natural landscapes, indoor spaces, sports events, and close-up footage. The challenge level varies significantly: approximately 30% of videos contain challenging phenomena such as fast motion, heavy occlusion, appearance change, or scale change, making SA-V suitable for evaluating robustness.

## Comparison with Other Video Segmentation Datasets

SA-V is dramatically larger than existing video segmentation datasets. DAVIS 2017 contains only 90 videos with 376 object annotations. YouTube-VOS 2019 has 4,453 videos with approximately 7,800 objects across 94 categories. MOSE provides 2,149 videos focused on complex multi-object scenes. UVO (Unidentified Video Objects) offers 10,337 videos but with sparser annotations. SA-V exceeds all of these by at least 5x in video count and by over 80x in total mask count, providing an order-of-magnitude scaling improvement.

Unlike category-specific datasets (YouTube-VOS annotates only 94 categories), SA-V annotates all salient objects regardless of category, following the class-agnostic philosophy of SA-1B. This design choice is critical for training a foundation model: category-specific datasets bias models toward known classes, while SA-V's class-agnostic annotations encourage the model to segment any object.

## Impact on Model Performance

Ablation experiments demonstrate that SA-V training data is essential for SAM 2's video performance. Training on image data alone (SA-1B) produces a model with no temporal reasoning capability. Adding existing video datasets (DAVIS + YouTube-VOS + MOSE) provides temporal reasoning but limited generalization. Adding SA-V training data improves J&F on DAVIS 2017 by approximately 4 points and on MOSE by approximately 6 points compared to training without SA-V.

The scale of SA-V also matters: training on a 10% subset of SA-V achieves approximately 2 points lower J&F than the full dataset, indicating that the model benefits from the full data volume. The diversity of SA-V is equally important: models trained on SA-V generalize better to out-of-distribution videos (e.g., medical or industrial) than models trained on category-specific datasets of comparable size.

## Implementation Notes

SA-V is available for download through the SAM 2 project page. Videos are stored as JPEG frame sequences (for efficient random frame access) rather than compressed video files. Masklets are stored in JSON format with RLE-encoded masks per frame. The full dataset requires approximately 8 TB of storage. Subsets are available for researchers with limited resources. The dataset can be loaded using the provided `sa_v_dataset` dataloader class, which handles frame sampling, masklet decoding, and augmentation for training. For evaluation, the standard protocol evaluates J&F on the val and test sets using the official evaluation server.
