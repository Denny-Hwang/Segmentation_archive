---
title: "Segment Anything"
date: 2025-03-06
status: planned
tags: [foundation-model, promptable-segmentation, zero-shot, sa-1b]
difficulty: advanced
---

# SAM (Segment Anything Model)

## Meta Information

| Field | Value |
|-------|-------|
| **Paper Title** | Segment Anything |
| **Authors** | Kirillov, A., Mintun, E., Ravi, N., Mao, H., Rolland, C., Gustafson, L., Xiao, T., Whitehead, S., Berg, A.C., Lo, W.-Y., Dollar, P., Girshick, R. |
| **Year** | 2023 |
| **Venue** | ICCV |
| **arXiv** | [2304.02643](https://arxiv.org/abs/2304.02643) |
| **Difficulty** | Advanced |

## One-Line Summary

SAM is a promptable segmentation foundation model trained on 1 billion masks (SA-1B dataset) that can segment any object given points, boxes, masks, or text prompts with strong zero-shot generalization.

## Motivation and Problem Statement

Prior to SAM, segmentation models were typically designed for specific tasks (semantic, instance, or panoptic segmentation) and required extensive task-specific training data with carefully curated annotations. This paradigm meant that each new segmentation application demanded a new dataset and a new model, severely limiting scalability. The computer vision community lacked a foundation model analogous to GPT-3 in NLP or CLIP in vision-language understanding -- one that could generalize broadly across segmentation tasks with minimal task-specific supervision.

SAM was motivated by the observation that large-scale pretraining with a flexible prompting interface could unify diverse segmentation tasks under a single model. The authors aimed to build a "foundation model for segmentation" that could segment any object in any image when given an appropriate prompt, enabling zero-shot transfer to new domains and tasks without additional fine-tuning. This required simultaneously solving three challenges: designing a promptable architecture, building a data engine capable of producing annotations at unprecedented scale, and training a model with sufficient generalization capacity.

## Architecture Overview

SAM follows a three-component architecture: a heavyweight image encoder that processes the input image once, a lightweight prompt encoder that handles diverse prompt types, and a fast mask decoder that combines image and prompt embeddings to produce segmentation masks. This design intentionally separates the computationally expensive image encoding from the lightweight prompt-conditioned decoding, enabling real-time interactive segmentation where a single image embedding can be reused across multiple prompts.

### Key Components

- **Prompt Engineering**: See [prompt_engineering.md](prompt_engineering.md)
- **SA-1B Dataset**: See [sa1b_dataset.md](sa1b_dataset.md)

## Technical Details

### Image Encoder (ViT)

SAM uses a Vision Transformer (ViT) pretrained with Masked Autoencoders (MAE) as its image encoder. The default model employs a ViT-H (Huge) backbone with 632M parameters, processing input images at 1024x1024 resolution and producing 64x64 spatial feature embeddings with 256 channels. The ViT architecture applies a patch embedding layer (with 16x16 patches), followed by a sequence of transformer blocks with multi-head self-attention and MLP layers. After the transformer blocks, the features pass through a neck consisting of two convolutional layers (with layer normalization) to reduce the channel dimension to 256.

The authors also provide ViT-B (Base) and ViT-L (Large) variants offering different accuracy-efficiency trade-offs. The MAE pretraining is critical because it provides the encoder with strong visual representations learned from a large corpus of natural images, enabling effective feature extraction even for objects and domains not explicitly encountered during SAM's supervised training phase.

### Prompt Encoder

The prompt encoder maps different types of user inputs into a common embedding space compatible with the mask decoder. SAM supports four prompt types: points, bounding boxes, free-form masks, and text. Sparse prompts (points and boxes) are represented as positional encodings summed with learned embeddings indicating the prompt type. Points are encoded using positional encodings at their coordinates, with separate learned embeddings for foreground (positive) and background (negative) points. Bounding boxes are encoded as a pair of points (top-left and bottom-right corners), each with its own learned embedding.

Dense prompts (masks) are processed through a small convolutional network (two 2x2 stride-2 convolutions with output channels of 4 and 16, followed by a 1x1 convolution to produce a 256-dimensional output) that downscales the mask to 64x64 and maps it to the image embedding space. Text prompts are encoded using CLIP's text encoder, though this capability was not extensively explored in the released model. The encoder is designed to be lightweight so that prompt processing does not become a bottleneck during interactive use.

### Mask Decoder

The mask decoder is a modified transformer decoder that maps the image embedding, prompt embeddings, and a set of output tokens to segmentation masks. It consists of two transformer decoder layers, each performing self-attention among the tokens (prompt tokens and output tokens), followed by cross-attention from tokens to image embeddings, a point-wise MLP, and cross-attention from the image embedding to the tokens. After the decoder layers, the image embedding is upsampled via two transposed convolutional layers (2x upsampling each) to produce a spatial output at 256x256 resolution. Each output token is then projected to a set of dynamic linear classifier weights via an MLP, and the classifier is applied to the upsampled image embedding to produce the final mask logits.

The decoder also includes a learned IoU prediction head -- a small MLP that estimates the quality (intersection-over-union) of each predicted mask, enabling automatic selection of the best output mask. The entire decoder is lightweight, with only about 4M parameters, making it fast enough for real-time interaction.

### Ambiguity-Aware Output

A single prompt (e.g., a point click) can be ambiguous -- it might refer to a part, a subpart, or a whole object. SAM addresses this by predicting multiple masks simultaneously (by default three masks corresponding to whole, part, and subpart levels), each with an associated IoU confidence score. During training, only the mask with the lowest loss against the ground truth is backpropagated, following the approach used in prior multi-hypothesis prediction methods. At inference time, the model can either return all three masks and their confidence scores (for interactive applications where the user selects) or automatically select the mask with the highest predicted IoU (for automatic segmentation pipelines).

This multi-mask prediction mechanism is particularly important for single-point prompts, where ambiguity is highest. When more informative prompts are provided (e.g., a bounding box), the model can be configured to predict a single mask, as the prompt typically resolves the ambiguity sufficiently.

### Training Strategy

SAM's training relied on a novel iterative data engine that co-evolved the model and its training dataset through three phases. In the first (assisted-manual) phase, professional annotators used a SAM-assisted interactive tool to label masks, with a preliminary SAM model providing initial predictions that annotators refined. In the second (semi-automatic) phase, SAM automatically detected confident masks, and annotators focused on labeling additional objects that the model missed, increasing object diversity. In the third (fully automatic) phase, SAM generated masks on 11 million images using a grid of point prompts (32x32 points per image), with confidence and stability filtering to ensure quality.

The model was trained using a combination of focal loss and dice loss, supervised with the masks produced by the data engine. Training used the AdamW optimizer with a learning rate warm-up and linear decay schedule. Data augmentation included random horizontal flipping and large-scale jittering. The iterative nature of the data engine meant that as the model improved, it generated better automatic annotations, which in turn led to an even better model in the next training iteration.

## Experiments and Results

### Zero-Shot Transfer

SAM demonstrated strong zero-shot transfer capabilities across 23 diverse segmentation datasets spanning domains not seen during training. The evaluation covered edge detection (BSDS500), object proposal generation, instance segmentation, and text-to-mask prediction. On zero-shot single-point segmentation, SAM achieved an average mIoU of approximately 60-70% across diverse benchmarks, with performance improving substantially with additional prompt points. On zero-shot edge detection using automatic mask generation, SAM produced reasonable edge maps on BSDS500 without any edge-specific training.

The zero-shot transfer results were particularly noteworthy because SAM was never explicitly trained on any of these downstream evaluation datasets. The model's ability to generalize to novel visual concepts and domains validated the foundation model approach to segmentation, suggesting that scale and diversity of training data could substitute for task-specific supervision.

### Key Results

SAM's key quantitative achievements include: (1) on the LVIS dataset for zero-shot instance segmentation, SAM's automatically generated masks achieved higher quality than ViTDet predictions on 16 out of 23 evaluated datasets; (2) single-point segmentation achieved a median mIoU of approximately 75% when allowing the model to predict three masks and selecting the best; (3) the automatic mask generation pipeline could process a single image in approximately 50 seconds on an A100 GPU using the ViT-H encoder, producing an average of roughly 100 masks per image; (4) on interactive segmentation benchmarks, SAM was competitive with or superior to prior interactive segmentation methods that were specifically trained for those tasks.

### Comparison with Supervised Models

When compared to supervised task-specific models, SAM showed mixed results depending on the domain and evaluation protocol. On natural images with common object categories, SAM's zero-shot performance was competitive with strongly supervised models like RITM and FocalClick for interactive segmentation. However, on specialized domains such as medical imaging, satellite imagery, and industrial inspection, SAM's zero-shot performance lagged behind domain-specific models, highlighting the remaining domain gap. This performance differential motivated subsequent works like MedSAM and SAM-Adapter that adapt SAM to specific domains.

## Strengths

- **Unprecedented generalization**: SAM demonstrates remarkable zero-shot transfer across a wide variety of visual domains and segmentation tasks, establishing a new paradigm for general-purpose segmentation.
- **Flexible prompting interface**: The support for multiple prompt types (points, boxes, masks, text) and their combinations makes SAM highly versatile for diverse application scenarios.
- **Massive training scale**: The SA-1B dataset with 1.1 billion masks provides a training signal orders of magnitude larger than any previous segmentation dataset, contributing to robust feature learning.
- **Efficient architecture**: The separation of heavyweight image encoding from lightweight prompt-conditioned decoding enables real-time interactive use cases.
- **Ambiguity handling**: The multi-mask prediction with IoU scoring elegantly addresses the inherent ambiguity of sparse prompts.
- **Open release**: The model weights, code, and SA-1B dataset were released publicly, enabling widespread research and application.

## Limitations

- **Domain gap on specialized imagery**: SAM's zero-shot performance degrades substantially on medical images, satellite imagery, microscopy, and other domains with significantly different visual characteristics from natural images.
- **No semantic understanding**: SAM produces class-agnostic masks and does not assign semantic labels (category names) to segmented regions, limiting its utility for tasks requiring semantic understanding.
- **Boundary quality**: Fine-grained boundary precision can be lacking, particularly for objects with thin structures, holes, or complex boundaries that require high-resolution feature processing.
- **Text prompt limitations**: The text-based prompting capability was not fully developed or evaluated, limiting open-vocabulary segmentation applications.
- **Computational cost**: The ViT-H encoder is computationally expensive (requiring ~3.5 seconds per image on consumer GPUs), limiting deployment on edge devices.
- **Bias towards salient objects**: SAM's automatic mask generation tends to favor visually salient objects and may miss subtle or low-contrast structures.

## Connections

SAM serves as the foundational work for a family of subsequent models. SAM 2 extends the approach to video by introducing a streaming memory mechanism. MedSAM adapts SAM specifically for medical imaging through full fine-tuning on a large medical dataset. SAM-Adapter provides a parameter-efficient alternative to full fine-tuning by inserting adapter modules into the frozen image encoder. MedSAM-2 combines the ideas from SAM 2 and MedSAM by treating 3D medical volumes as video sequences. OMG-Seg takes a different approach to universal segmentation by using a CLIP backbone and unified decoder. The success of SAM has also influenced broader trends toward foundation models in computer vision, including applications in robotics, augmented reality, and content creation.

## References

- Dosovitskiy, A., et al. "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale." ICLR 2021.
- He, K., et al. "Masked Autoencoders Are Scalable Vision Learners." CVPR 2022.
- Radford, A., et al. "Learning Transferable Visual Models From Natural Language Supervision." ICML 2021 (CLIP).
- Sofiiuk, K., et al. "Reviving Iterative Training with Mask Guidance for Interactive Segmentation." ICIP 2022 (RITM).
- Lin, T.-Y., et al. "Focal Loss for Dense Object Detection." ICCV 2017.
- Milletari, F., et al. "V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation." 3DV 2016 (Dice loss).
- Brown, T., et al. "Language Models are Few-Shot Learners." NeurIPS 2020 (GPT-3, foundation model paradigm).
