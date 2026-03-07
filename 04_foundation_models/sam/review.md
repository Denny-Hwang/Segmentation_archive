---
title: "Segment Anything"
date: 2025-03-06
status: complete
tags: [foundation-model, promptable-segmentation, zero-shot, sa-1b]
difficulty: advanced
---

# SAM (Segment Anything Model)

## Paper Overview

**Title:** Segment Anything
**Authors:** Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C. Berg, Wan-Yen Lo, Piotr Dollar, Ross Girshick
**Venue:** ICCV 2023
**Institution:** Meta AI Research (FAIR)

SAM introduces a foundation model for image segmentation that is designed to be promptable, meaning it can produce valid segmentation masks given any combination of points, bounding boxes, rough masks, or text as input. The model is trained on a newly created dataset of over 1.1 billion masks, making it the largest segmentation dataset at the time of release.

## Motivation

Prior segmentation models were narrowly trained on specific datasets and task definitions (semantic, instance, panoptic). SAM aims to build a single model that generalizes across tasks and domains by drawing an analogy to large language models: train at massive scale on broad data, then prompt at inference time for the desired output.

## Architecture

SAM consists of three components:

### Image Encoder

- Based on a Vision Transformer (ViT-H) pretrained with MAE (Masked Autoencoder)
- Input images are resized to 1024x1024 and processed into a 64x64 feature embedding with 256 channels
- The image encoder runs once per image, producing reusable embeddings that can be paired with multiple prompts
- ViT-H has 632M parameters, making it the heaviest component but amortized over many prompt interactions

### Prompt Encoder

The prompt encoder handles two categories of prompts:

- **Sparse prompts** (points, boxes, text): Encoded using positional encodings combined with learned embeddings for each prompt type. Points use positional encoding plus a foreground/background indicator. Boxes are encoded as two points (top-left, bottom-right). Text prompts use CLIP text embeddings.
- **Dense prompts** (masks): Encoded using convolutional downsampling layers that map the input mask to a spatial embedding, then added element-wise to the image embedding.

### Mask Decoder

- A lightweight transformer-based decoder with only 2 layers
- Uses bidirectional cross-attention: prompt-to-image and image-to-prompt
- Produces 3 candidate masks at different granularity levels (whole, part, subpart) along with confidence scores (IoU predictions)
- The multi-mask output resolves ambiguity when a single point prompt could correspond to multiple valid segmentations
- Runs in approximately 50ms on CPU after the image embedding is precomputed

## Training

### Task Definition

The promptable segmentation task requires the model to return a valid segmentation mask for any given prompt. When a prompt is ambiguous, the model must produce at least one reasonable mask. This task is used both as a pretraining objective and as a downstream capability.

### Loss Function

The training loss combines focal loss and dice loss, computed over the predicted masks. During training with ambiguous prompts, only the minimum loss over the three output masks is backpropagated. An auxiliary IoU prediction head is supervised with MSE loss between predicted and actual IoU.

### Training Data Pipeline

Training leveraged the SA-1B data engine with three stages. The model was iteratively improved alongside the data collection, with each stage producing more masks to retrain the next version.

## Key Results

| Benchmark | Metric | SAM (ViT-H) |
|-----------|--------|-------------|
| COCO (zero-shot) | AR@1000 | 69.7 |
| LVIS (zero-shot) | AR@1000 | 75.4 |
| 23 diverse datasets | mIoU (1-point) | 60.6 |
| 23 diverse datasets | mIoU (oracle) | 73.0 |

### Zero-Shot Transfer Highlights

- Competitive or superior to fully supervised models on many benchmarks without any task-specific training
- Strong performance on edge detection (BSDS500), object proposal generation, and instance segmentation
- Demonstrated generalization to medical, aerial, underwater, and other out-of-distribution domains

## Model Variants

| Variant | Backbone | Parameters | Speed (img/s) |
|---------|----------|-----------|---------------|
| ViT-B | ViT-Base | ~91M | ~40 |
| ViT-L | ViT-Large | ~308M | ~20 |
| ViT-H | ViT-Huge | ~632M | ~8 |

## Strengths

- Unprecedented scale of training data (SA-1B) enables strong zero-shot generalization
- Promptable interface allows flexible interaction patterns for different use cases
- Efficient mask decoder enables real-time interactive segmentation once the image is encoded
- The multi-mask output strategy elegantly handles prompt ambiguity

## Limitations

- No built-in semantic understanding: SAM segments objects but does not classify them
- Struggles with fine-grained boundaries on thin structures (e.g., bicycle spokes, hair)
- Performance degrades on domain-specific data that is far from SA-1B distribution (e.g., medical imaging, remote sensing)
- Single-image model with no temporal or volumetric reasoning
- Text prompt capability was limited and not publicly released at launch

## Impact

SAM catalyzed a wave of research into foundation models for segmentation. It established the paradigm of large-scale pretraining followed by promptable inference, and the SA-1B dataset became a standard pretraining resource. Downstream works include MedSAM, SAM-Adapter, HQ-SAM, FastSAM, and many domain-specific adaptations.

## Citation

```
Kirillov, A., et al. "Segment Anything." ICCV 2023.
```
