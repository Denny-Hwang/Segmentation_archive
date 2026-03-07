---
title: "SA-1B Dataset"
date: 2025-03-06
status: planned
tags: [dataset, sa-1b, data-engine, annotation, billion-scale]
difficulty: intermediate
---

# SA-1B Dataset

## Overview

SA-1B (Segment Anything 1 Billion) is the largest segmentation dataset ever created, containing approximately 11 million high-resolution, licensed images with over 1.1 billion automatically generated segmentation masks. The dataset was produced through an iterative data engine that co-evolved with the SAM model itself, progressively scaling from human-annotated masks to fully automatic mask generation. SA-1B represents a fundamental shift in segmentation dataset philosophy: rather than carefully curating a small number of perfectly annotated images, it prioritizes massive scale and diversity, relying on automated quality filtering to maintain acceptable annotation quality across a billion masks.

The images in SA-1B were sourced from a licensed image provider and span a broad range of natural scenes, objects, and contexts. Each image has an average resolution of approximately 3300x4950 pixels, significantly higher than most existing segmentation datasets. The dataset is released publicly under a permissive license for research purposes, with faces and license plates blurred to protect privacy.

## Data Engine

### Manual Annotation Phase

The first phase of the data engine involved professional annotators using a browser-based interactive annotation tool powered by an early version of SAM. Annotators would click foreground and background points, and the model would predict masks in real time (within 50 milliseconds), which the annotators could then accept or refine. This SAM-assisted workflow was substantially faster than traditional polygon-based annotation: annotators could label approximately 12 masks per minute, compared to roughly 3-4 masks per minute with polygon tools on COCO-style tasks.

During this phase, 120,000 images were annotated, producing approximately 4.3 million masks. The SAM model was retrained six times on the growing set of annotations during this phase, with each retraining cycle improving the model's predictions and thus the speed and quality of subsequent annotation. This phase established the initial training signal and familiarized annotators with the interactive workflow.

### Semi-Automatic Phase

In the semi-automatic phase, SAM first automatically detected high-confidence masks, and human annotators were asked to annotate additional objects that the model had missed. This strategy was designed to increase the diversity of annotated objects, particularly for less salient or less common categories. By directing human attention specifically to the "gaps" in SAM's coverage, this phase ensured that the training data captured a broader range of visual concepts than the model could detect on its own.

This phase annotated an additional 180,000 images, bringing the total to approximately 5.9 million masks when combined with the automatic masks from the model. The average number of masks per image increased from 44 (in the manual phase) to 72, as annotators focused on previously missed objects. SAM was retrained using these masks, yielding further improvements in the model's automatic detection capability.

### Fully Automatic Phase

The final phase eliminated human annotators entirely, using SAM to generate masks automatically at scale. A grid of 32x32 points (1,024 points) was applied to each image, and SAM predicted masks for every point. The resulting masks were then filtered using two criteria: a confidence score threshold (predicted IoU > 0.88) and a stability score (requiring that the mask remain consistent when the prediction logit threshold is perturbed by ±1). Duplicate masks were removed via non-maximum suppression based on mask IoU. This pipeline was applied to the full set of 11 million images, producing the final 1.1 billion masks.

The fully automatic pipeline was run on a cluster of 256 A100 GPUs. Each image took approximately 50 seconds to process with the ViT-H encoder, including all 1,024 point prompts and post-processing. The entire generation process required roughly 14 GPU-years of compute. Quality filtering removed approximately 40% of initial predictions, with the remaining masks exhibiting high average quality as validated by human raters.

## Dataset Statistics

SA-1B contains 11,084,029 images and 1,101,254,522 masks. The average number of masks per image is approximately 100, with significant variation (the distribution is roughly log-normal, ranging from a few masks for simple scenes to several hundred for complex scenes). The median image resolution is approximately 3300x4950 pixels. The dataset covers a diverse range of geographic regions, with images sourced from over 190 countries, providing substantially better geographic representation than datasets like COCO or ImageNet that are biased toward Western contexts.

The masks themselves span a wide range of scales, from small objects (< 32x32 pixels) to large background regions covering most of the image. Object category diversity is also high, though the dataset does not include semantic labels -- all masks are class-agnostic binary segmentations. Temporal diversity is limited since all images are static photographs without video content.

## Quality Analysis

Meta conducted extensive quality analysis of SA-1B masks by comparing them with professionally annotated ground truth on a random sample of 500 images. Human raters evaluated mask quality on a 1-10 scale, and the automatically generated masks received a median rating of 7.5 out of 10, compared to 8.0 for professional manual annotations. In terms of IoU agreement with professional annotations, SA-1B masks achieved approximately 90% IoU on well-defined objects, with lower agreement on ambiguous boundaries and small objects.

A separate analysis compared SA-1B mask quality across the three data engine phases. Masks from the fully automatic phase showed slightly lower boundary precision than human-annotated masks but substantially better coverage (more objects detected per image). The authors argue that the increased scale and diversity of the automatic masks more than compensate for the modest reduction in per-mask quality, as evidenced by the strong downstream performance of SAM models trained on the full dataset.

## Responsible AI Considerations

The SA-1B dataset includes several responsible AI measures. All human faces in the images were automatically detected and blurred using a face detection model, and license plates were similarly obscured. The images were sourced from a licensed provider with consent mechanisms, avoiding the scraping of personal social media content. Geographic and income-level diversity analyses showed that SA-1B is substantially more balanced than COCO and ImageNet: the dataset's representation of low-income and middle-income countries is significantly higher, reducing geographic bias in trained models.

Fairness analyses of SAM's performance showed comparable segmentation quality across images from different geographic regions and across images containing people of different perceived gender and skin tone. However, the authors acknowledge that SA-1B, like any large-scale dataset, may contain biases that are not fully captured by the performed analyses. The class-agnostic nature of the annotations mitigates some bias risks associated with category labeling, but does not eliminate potential biases in which objects tend to be segmented in different cultural contexts.

## Impact on Model Performance

Scaling experiments demonstrated a clear positive relationship between training data volume and SAM's generalization performance. Training on 10x fewer masks resulted in a measurable drop in zero-shot transfer quality, particularly on out-of-distribution domains. The transition from the manual-phase dataset (~4.3M masks) to the full SA-1B dataset (~1.1B masks) improved zero-shot single-point mIoU by approximately 3-5 points on held-out benchmarks. Interestingly, the benefit of scale showed diminishing returns: the improvement from 100M to 1.1B masks was smaller than the improvement from 10M to 100M masks, suggesting that further scaling alone may have limited additional benefit.

The diversity of SA-1B was arguably as important as its scale. Models trained on a subset of SA-1B with restricted geographic or content diversity performed worse on out-of-distribution benchmarks, even when the total mask count was held constant. This finding supports the hypothesis that broad coverage of the visual world, rather than sheer volume of annotations on a narrow distribution, is the key driver of generalization.

## Comparison with Other Segmentation Datasets

SA-1B dwarfs all prior segmentation datasets in scale. COCO contains approximately 164K images with 1.5M instance masks (750x fewer masks than SA-1B). ADE20K has 25K images with approximately 450K segmented instances. OpenImages v7 contains 2.7M images with 16M instance masks. Even Mapillary Vistas, a large-scale street-scene dataset, has only 25K images. The table below summarizes key comparisons:

| Dataset | Images | Masks | Avg Masks/Image | Semantic Labels | Open Access |
|---------|--------|-------|-----------------|-----------------|-------------|
| SA-1B | 11M | 1.1B | ~100 | No | Yes |
| COCO | 164K | 1.5M | ~9 | Yes (80 cats) | Yes |
| ADE20K | 25K | 450K | ~18 | Yes (150 cats) | Yes |
| OpenImages v7 | 2.7M | 16M | ~6 | Yes (350 cats) | Yes |
| LVIS | 164K | 2M | ~12 | Yes (1203 cats) | Yes |

SA-1B's primary trade-off is the absence of semantic category labels. While COCO, ADE20K, and LVIS provide rich category annotations, SA-1B's masks are entirely class-agnostic. This makes SA-1B unsuitable for directly training semantic or panoptic segmentation models, but ideal for training general-purpose promptable segmentation models like SAM.

## Implementation Notes

SA-1B is publicly available for download through Meta's research platform. The dataset is distributed as a collection of image files and corresponding JSON annotation files, with each annotation containing a list of Run-Length Encoded (RLE) masks and associated metadata (predicted IoU, stability score, crop bounding box). Total download size is approximately 12 TB. Due to its scale, processing SA-1B typically requires distributed storage and parallel data loading. The official SAM repository includes data loaders compatible with SA-1B's format, and the dataset can be used with standard PyTorch DataLoader pipelines with appropriate preprocessing (resizing images to 1024x1024 and converting RLE masks to binary tensors).
