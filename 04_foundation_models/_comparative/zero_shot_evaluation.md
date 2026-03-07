---
title: "Zero-Shot Segmentation Evaluation"
date: 2025-03-06
status: planned
tags: [zero-shot, evaluation, generalization, benchmark]
difficulty: intermediate
---

# Zero-Shot Segmentation Evaluation

## Overview

Zero-shot evaluation measures a foundation model's ability to segment objects in images and videos from domains it was never explicitly trained on, without any fine-tuning or adaptation. This evaluation paradigm is central to assessing foundation models because their core value proposition is broad generalization. A model that achieves high accuracy on its training distribution but fails on unseen domains provides limited practical value compared to a model that maintains reasonable performance across diverse visual conditions. SAM, SAM 2, and MedSAM have all been evaluated zero-shot, revealing both the strengths and limitations of current foundation models for segmentation.

## Evaluation Protocol

The standard zero-shot evaluation protocol applies the model to benchmark datasets that were not used during training, using the model's original weights without any fine-tuning. For promptable models like SAM and SAM 2, prompts must be provided at test time because the models cannot generate class-specific masks without guidance. The two most common prompt protocols are: (1) ground-truth prompts, where bounding boxes or center-of-mass points derived from ground-truth masks are used as prompts, measuring the model's segmentation quality given perfect prompt information; and (2) automatic prompts, where a grid of points or a detector's bounding boxes are used, measuring the model's performance in a fully automatic pipeline.

For video models like SAM 2, the standard protocol provides a prompt (typically a ground-truth mask or bounding box) on the first frame and evaluates propagation quality across subsequent frames without additional prompts. Interactive evaluation protocols allow simulated corrective clicks on error frames and measure how many interactions are needed to reach a target quality threshold.

## Datasets for Zero-Shot Evaluation

A comprehensive zero-shot evaluation spans multiple visual domains to assess generalization breadth. Common benchmarks include: natural images (COCO val with 5,000 images and 36,781 instances; LVIS val with 19,809 images and 244,707 instances across 1,203 categories), medical imaging (BTCV with 30 CT volumes; ACDC with 100 cardiac MRI volumes; ISIC 2018 with 2,594 dermoscopy images), remote sensing (iSAID with 2,806 satellite images; SpaceNet with building segmentation), industrial inspection (MVTec with 5,354 images of manufacturing defects), and specialized domains (DRAM with 1,000 document images; DOORS with 500 camouflaged insect images).

For video evaluation, standard benchmarks include DAVIS 2017 (30 val videos, 59 objects), YouTube-VOS 2019 (507 val videos, 65 categories), and MOSE (742 val videos with complex multi-object scenes). These video benchmarks test temporal consistency, occlusion handling, and long-term tracking in addition to per-frame segmentation quality.

## Metrics

The primary metric for zero-shot segmentation is Intersection-over-Union (IoU), also known as Jaccard index, which measures the overlap between predicted and ground-truth masks. Mean IoU (mIoU) averages across all test samples. For instance segmentation, Average Precision (AP) at multiple IoU thresholds (AP50, AP75, AP) is standard. For video segmentation, J&F combines the Jaccard index (J, region similarity) with the F-measure (F, boundary accuracy), providing a balanced assessment of both region and boundary quality.

Additional metrics include: boundary IoU (measuring accuracy specifically near object boundaries), dice score (equivalent to F1 score, commonly used in medical imaging), predicted IoU calibration (measuring whether the model's confidence scores correlate with actual quality), and Number of Clicks (NoC, the number of interactive clicks needed to reach a target IoU threshold, typically 85% or 90%).

## SAM Zero-Shot Performance

SAM demonstrates strong zero-shot performance on natural images and progressively weaker performance on specialized domains. On COCO val with ground-truth box prompts, SAM ViT-H achieves approximately 78% IoU. On LVIS (which tests rare and unusual categories), performance drops to approximately 72% IoU. On the 23-dataset benchmark reported in the original paper, SAM achieves approximately 70% average IoU with single-point prompts and approximately 75% with box prompts.

On specialized domains, zero-shot SAM performance degrades substantially. On medical imaging (averaged across CT, MRI, and ultrasound), SAM achieves approximately 55-65% dice with box prompts. On remote sensing (satellite imagery), SAM achieves approximately 50-60% IoU. On camouflaged object detection (COD10K), SAM achieves only 42% IoU. On industrial defect detection (MVTec), SAM achieves approximately 45-55% IoU. These results indicate that SAM's natural image features transfer poorly to domains with fundamentally different visual characteristics.

## SAM 2 Zero-Shot Performance

SAM 2 improves over SAM on both image and video zero-shot benchmarks. On the 37-dataset zero-shot image benchmark, SAM 2 Large achieves approximately 2.0 IoU points higher than SAM ViT-H with single-point prompts and 1.5 points higher with box prompts. This improvement comes despite SAM 2's encoder (Hiera-L, 214M parameters) being significantly smaller than SAM's (ViT-H, 632M parameters), suggesting that the Hiera architecture and joint image-video training provide more efficient feature learning.

On video benchmarks, SAM 2 achieves strong zero-shot performance: 82.5 J&F on DAVIS 2017 and 81.2 J&F on YouTube-VOS 2019 using first-frame mask prompts. On the more challenging MOSE benchmark (complex multi-object scenes), SAM 2 achieves 73.8 J&F. Without any prompts beyond the first frame, SAM 2 maintains an average J&F above 75% for videos up to 200 frames, demonstrating robust temporal propagation. However, on very long videos (500+ frames) without corrective prompts, performance gradually degrades as the memory bank loses early frame information.

## Domain-Specific Challenges

Several domain characteristics cause significant zero-shot performance degradation. Low contrast between target and background is the primary challenge in medical imaging, where adjacent tissues may differ by only a few intensity values. SAM's features, trained on high-contrast natural images, produce weak activation patterns for these subtle boundaries, resulting in over-segmentation or missed regions.

Unusual spatial scales present challenges in satellite imagery (where objects like buildings span 10-50 pixels) and microscopy (where cellular structures span 5-20 pixels). SAM's training distribution is dominated by medium and large objects, leading to poor recall on very small structures. Non-standard image statistics affect domains like CT (Hounsfield units), MRI (arbitrary intensity scales), and SAR radar (speckle noise patterns), where the pixel value distributions differ fundamentally from natural RGB images. Deformable and amorphous objects (smoke, liquids, clouds) challenge SAM's learned shape priors, which favor relatively compact, convex shapes.

## Prompt Sensitivity

Zero-shot performance is heavily influenced by prompt quality, particularly for point prompts. On COCO val, shifting a single-point prompt by 10 pixels from the object center reduces IoU by approximately 3-5 points on average, with larger drops for small objects (up to 15 points for objects under 32x32 pixels). Box prompt sensitivity is lower: enlarging the ground-truth box by 10% reduces IoU by approximately 1-2 points. Loose boxes (50% larger than tight) reduce IoU by approximately 5-8 points.

On specialized domains, prompt sensitivity is generally higher than on natural images because the model's visual features are less reliable, making it more dependent on prompt precision. On medical images, a 10-pixel shift of a point prompt reduces dice by approximately 5-8 points, compared to 3-5 on natural images. Box prompts remain the most robust prompt type across all domains, which is why MedSAM exclusively uses box prompts. Using multiple points (3-5 per object) significantly reduces sensitivity: the standard deviation of IoU across different point placements drops by approximately 50% with 3 points compared to 1.

## Comparison Across Models

| Model | Natural Images (IoU) | Medical (Dice) | Remote Sensing (IoU) | Industrial (IoU) | Video (J&F) |
|-------|---------------------|----------------|---------------------|------------------|-------------|
| SAM ViT-H | 78% (box) | 62% (box) | 55% (box) | 50% (box) | N/A |
| SAM ViT-B | 72% (box) | 58% (box) | 50% (box) | 45% (box) | N/A |
| SAM 2 Large | 80% (box) | 65% (box) | 58% (box) | 52% (box) | 82.5 (DAVIS) |
| SAM 2 Base | 76% (box) | 61% (box) | 54% (box) | 48% (box) | 78.3 (DAVIS) |
| MedSAM | 70% (box) | 87% (box) | 48% (box) | 44% (box) | N/A |
| OMG-Seg | 74% (auto) | 55% (auto) | 52% (auto) | 46% (auto) | 74.2 (DAVIS) |

Note: MedSAM's natural image performance drops compared to SAM because fine-tuning on medical data causes partial forgetting of natural image features. OMG-Seg uses automatic prompts (no user interaction), making direct comparison with promptable models approximate. All numbers are approximate averages across multiple benchmarks within each domain.

## Improving Zero-Shot Performance

Several strategies can improve zero-shot performance without full fine-tuning. Multi-prompt inference uses multiple prompts (e.g., 5 points or a box + points) to provide more information, typically improving IoU by 5-10 points over single-point prompts. Test-time augmentation applies multiple augmentations (flips, scales) and averages the predictions, improving IoU by 1-3 points. Prompt ensembling generates multiple masks from different prompts and selects the most consistent one, improving robustness by reducing prompt sensitivity.

For domain-specific improvement, the recommended path is: (1) first evaluate zero-shot performance to establish a baseline, (2) try multi-prompt and test-time augmentation for quick gains, (3) if performance is still insufficient, apply parameter-efficient adaptation (adapters or LoRA) with minimal labeled data, (4) only resort to full fine-tuning if the domain gap is very large and sufficient data is available. This progressive approach minimizes effort while systematically closing the performance gap between zero-shot and fully supervised performance.
