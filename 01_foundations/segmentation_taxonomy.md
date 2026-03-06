---
title: "Segmentation Taxonomy"
date: 2025-03-06
status: in-progress
tags: [segmentation, taxonomy, semantic, instance, panoptic, video, interactive, open-vocabulary]
difficulty: beginner
---

# Segmentation Taxonomy

Segmentation is the task of partitioning an image (or video) into meaningful regions. What counts as "meaningful" depends on the task definition. This document provides a structured taxonomy of segmentation tasks organized along three orthogonal axes: **what** is being predicted, **where** (image vs. video), and **how** the user interacts with the system.

---

## 1. The Core Trichotomy: Semantic, Instance, and Panoptic

### 1.1 Semantic Segmentation

**Definition.** Assign every pixel in the image a class label from a fixed set $\mathcal{C} = \{c_1, c_2, \dots, c_K\}$. The output is a dense label map $\hat{Y} \in \mathcal{C}^{H \times W}$.

**Key characteristics:**

- Does **not** distinguish between different instances of the same class. Two adjacent cars receive the same label.
- The label set typically includes a *background* or *void* class for pixels that do not belong to any category of interest.
- Formally, the model learns a mapping $f_\theta : \mathbb{R}^{H \times W \times 3} \rightarrow [0,1]^{H \times W \times K}$, where the output represents per-pixel class probabilities.

**Canonical datasets:** PASCAL VOC 2012 (21 classes), Cityscapes (19 classes), ADE20K (150 classes), COCO-Stuff (171 classes).

**Typical metrics:** mean Intersection over Union (mIoU), pixel accuracy.

---

### 1.2 Instance Segmentation

**Definition.** Detect every object instance in the image and produce a binary mask for each. The output is a set of tuples $\{(m_i, c_i, s_i)\}_{i=1}^{N}$ where $m_i \in \{0,1\}^{H \times W}$ is the mask, $c_i$ is the class label, and $s_i$ is the confidence score.

**Key characteristics:**

- Distinguishes individual object instances: two adjacent cars get separate masks.
- Only applies to *countable* ("thing") categories (e.g., person, car, dog). Amorphous "stuff" categories (e.g., sky, road) are typically ignored.
- Can be approached top-down (detect then segment, e.g., Mask R-CNN) or bottom-up (embed pixels then cluster, e.g., associative embedding).

**Canonical datasets:** COCO (80 thing classes), LVIS (1203 classes, long-tail), Cityscapes.

**Typical metrics:** mask AP (Average Precision), AP at different IoU thresholds (AP50, AP75).

---

### 1.3 Panoptic Segmentation

**Definition.** Unify semantic and instance segmentation into a single coherent output. Every pixel is assigned both a semantic class label and an instance ID. For "stuff" classes (uncountable regions like sky), all pixels of that class share a single ID. For "thing" classes (countable objects), each instance gets a unique ID.

Formally, the output is a map $P : \{1, \dots, H\} \times \{1, \dots, W\} \rightarrow \mathcal{C} \times \mathbb{N}$ such that each pixel $(x, y)$ maps to a tuple $(c, id)$.

**Key characteristics:**

- No pixel is left unlabeled and no pixel is assigned to two segments -- the output is a non-overlapping partition of the image.
- Introduced by Kirillov et al. (2019) to bridge the gap between the semantic and instance segmentation communities.
- Evaluated with the Panoptic Quality (PQ) metric, which jointly measures recognition and segmentation quality.

**Canonical datasets:** COCO Panoptic (133 classes: 80 things + 53 stuff), ADE20K Panoptic, Cityscapes Panoptic.

**Typical metrics:** Panoptic Quality (PQ), Segmentation Quality (SQ), Recognition Quality (RQ).

---

### 1.4 Comparison Table

| Property | Semantic | Instance | Panoptic |
|----------|----------|----------|----------|
| Pixel-level labels | Yes | Only for detected objects | Yes (all pixels) |
| Distinguishes instances | No | Yes | Yes (for things) |
| Handles "stuff" classes | Yes | No | Yes |
| Handles "thing" classes | Yes (no instance separation) | Yes | Yes |
| Overlapping masks allowed | N/A (dense map) | Yes | No |
| Primary metric | mIoU | AP | PQ |

---

## 2. Image vs. Video Segmentation

### 2.1 Image Segmentation

All three tasks above (semantic, instance, panoptic) are typically defined on single, static images. The model processes one frame independently and produces a segmentation map.

### 2.2 Video Segmentation

Video segmentation extends the spatial problem to the spatio-temporal domain. Several sub-tasks exist:

#### Video Object Segmentation (VOS)

Given one or more object masks in the first frame (semi-supervised setting), propagate those masks through the entire video sequence. The challenge is maintaining identity across frames despite occlusion, deformation, and appearance change.

- **Semi-supervised VOS:** The ground-truth mask is given for the first frame. The model must track and segment the object(s) through subsequent frames. (Datasets: DAVIS 2017, YouTube-VOS)
- **Unsupervised VOS:** No annotation is given. The model must automatically identify and segment the primary salient object(s). (Dataset: DAVIS 2016 unsupervised)

#### Video Instance Segmentation (VIS)

Simultaneously detect, segment, and track all object instances across every frame in a video clip. Each instance must be assigned a consistent identity throughout the clip.

- Introduced by Yang et al. (2019). Dataset: YouTube-VIS.
- Metrics: video-level AP, computed by matching predicted tracklets to ground-truth tracklets using spatio-temporal IoU.

#### Video Panoptic Segmentation (VPS)

The video extension of panoptic segmentation. Every pixel in every frame gets a semantic label and an instance ID. Instance IDs must be temporally consistent.

- Dataset: Cityscapes-VPS, VIPSeg.
- Metric: Video Panoptic Quality (VPQ).

#### Video Semantic Segmentation (VSS)

Assign a semantic class to every pixel in every frame. Temporal consistency is desirable but not strictly enforced by standard metrics. Efficient architectures exploit temporal redundancy (e.g., key-frame propagation, feature warping with optical flow).

---

## 3. Interaction Paradigms

### 3.1 Fully Automatic Segmentation

The standard setting: a trained model receives an image and produces segmentation output with no human intervention at inference time. All the tasks described above can be fully automatic.

### 3.2 Interactive Segmentation

The user provides guidance to help the model produce or refine a segmentation mask. Interaction modalities include:

- **Click-based:** Positive and negative clicks indicate foreground/background. The model iteratively refines its prediction. (Classic: GrabCut; modern: RITM, SimpleClick, Segment Anything)
- **Bounding box:** A bounding box around the object of interest constrains the segmentation. (e.g., Mask R-CNN with ground-truth boxes, SAM with box prompts)
- **Scribble-based:** The user draws rough strokes on foreground and background regions. (e.g., ScribbleSup, interactive GrabCut)
- **Text / language-based:** Natural language descriptions guide segmentation. (Overlaps with referring segmentation; see below.)

**Key design principles for interactive systems:**
- Fast inference (the user is waiting).
- Graceful refinement (each additional interaction should improve the result, not degrade it).
- Minimal number of interactions needed to reach a satisfactory mask.

### 3.3 Referring Segmentation

Given a natural language expression (e.g., "the man in the red shirt on the left"), produce a segmentation mask for the referred entity. This requires joint vision-language understanding.

- **Referring Image Segmentation:** Single image + text expression -> binary mask. (Datasets: RefCOCO, RefCOCO+, RefCOCOg)
- **Referring Video Object Segmentation (R-VOS):** Video + text expression -> mask track. (Dataset: Refer-YouTube-VOS)

### 3.4 Open-Vocabulary Segmentation

Traditional segmentation models are limited to a fixed set of classes seen during training. Open-vocabulary segmentation removes this constraint by leveraging vision-language models (e.g., CLIP) to segment classes described by arbitrary text at inference time.

**Sub-tasks:**

- **Open-vocabulary semantic segmentation:** Segment any class described by text, including classes never seen during training. (Methods: LSeg, OpenSeg, SAN, SED, FC-CLIP)
- **Open-vocabulary instance segmentation:** Detect and segment instances of any text-described class. (Methods: Detic, OV-DETR)
- **Open-vocabulary panoptic segmentation:** Full panoptic output for arbitrary class vocabularies. (Methods: ODISE, MasQCLIP)

**Key technical approaches:**

1. **Frozen CLIP backbone + segmentation head:** Use the pretrained CLIP visual encoder as a feature extractor and train only the segmentation-specific modules.
2. **Region-text alignment:** Align segment/region embeddings with text embeddings from a language model, typically using contrastive learning.
3. **Training with image-level labels:** Use datasets with image-level class labels (e.g., ImageNet-21K) to learn associations between visual regions and text, then transfer to pixel-level prediction.

**Evaluation protocol:** Models are typically evaluated on classes split into "base" (seen during training) and "novel" (unseen) categories. The key metric is mIoU on novel classes.

---

## 4. Other Notable Task Variants

| Task | Description |
|------|-------------|
| **Part segmentation** | Segment object parts (e.g., head, torso, limbs of a person). Datasets: Pascal-Part, PartImageNet. |
| **Scene parsing** | Dense semantic segmentation with a very large label set covering the full scene. Dataset: ADE20K (150 classes). |
| **Amodal segmentation** | Predict the full extent of objects, including occluded regions. |
| **3D segmentation** | Segment 3D point clouds or voxel grids. Datasets: ScanNet, S3DIS, SemanticKITTI. |
| **Medical image segmentation** | Segment anatomical structures or lesions in medical scans (CT, MRI, X-ray, histopathology). Often treated as a specialized domain due to unique challenges (3D volumes, class imbalance, limited data). |
| **Few-shot segmentation** | Segment novel classes given only a few annotated examples (support set). |
| **Zero-shot segmentation** | Segment classes with no pixel-level annotations, relying on class embeddings or language descriptions. Closely related to open-vocabulary segmentation. |

---

## 5. Summary

The segmentation landscape can be navigated along three axes:

1. **Task definition** (what to predict): Semantic vs. Instance vs. Panoptic.
2. **Temporal domain** (where): Single image vs. Video (with temporal consistency requirements).
3. **Interaction / vocabulary** (how): Fully automatic, interactive (click/box/scribble), referring (language-guided), or open-vocabulary (arbitrary classes at test time).

Modern foundation models such as SAM (Segment Anything) and its successors are beginning to blur these boundaries by providing a single model that can handle multiple task types through different prompting strategies.

---

## References

1. Long, J., Shelhamer, E., & Darrell, T. (2015). Fully Convolutional Networks for Semantic Segmentation. CVPR.
2. He, K., Gkioxari, G., Dollar, P., & Girshick, R. (2017). Mask R-CNN. ICCV.
3. Kirillov, A., He, K., Girshick, R., & Dollar, P. (2019). Panoptic Segmentation. CVPR.
4. Yang, L., Fan, Y., & Xu, N. (2019). Video Instance Segmentation. ICCV.
5. Kirillov, A., Mintun, E., Ravi, N., et al. (2023). Segment Anything. ICCV.
6. Ghiasi, G., Gu, X., Cui, Y., & Lin, T.-Y. (2022). Scaling Open-Vocabulary Image Segmentation with Image-Level Labels. ECCV.
7. Xu, J., De Mello, S., Liu, S., Byeon, W., Breuel, T., Kautz, J., & Wang, X. (2023). Open-Vocabulary Panoptic Segmentation with Text-to-Image Diffusion Models. CVPR.
