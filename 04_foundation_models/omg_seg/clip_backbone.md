---
title: "CLIP Backbone in OMG-Seg"
date: 2025-03-06
status: planned
tags: [clip, vision-language, backbone, feature-extraction]
difficulty: advanced
---

# CLIP Backbone

## Overview

OMG-Seg uses CLIP's vision encoder as its backbone, a choice that distinguishes it from most segmentation models that rely on ImageNet-pretrained or MAE-pretrained backbones. CLIP (Contrastive Language-Image Pre-training) was trained on 400 million image-text pairs to align visual and textual representations in a shared embedding space. This vision-language alignment provides a unique advantage for segmentation: the backbone's features inherently encode semantic information about visual concepts, enabling open-vocabulary classification where new categories can be recognized at inference time simply by providing their text descriptions.

Using CLIP as a segmentation backbone requires careful adaptation because CLIP was designed for image-level classification (producing a single feature vector per image) rather than dense pixel-level prediction. OMG-Seg addresses this through multi-scale feature extraction from intermediate CLIP layers and feature pyramid construction, converting the classification-oriented backbone into a dense prediction engine.

## CLIP Architecture

CLIP's vision encoder (used in OMG-Seg) is a Vision Transformer (ViT-L/14) with 24 transformer layers, an embedding dimension of 1024, 16 attention heads, and an input patch size of 14x14 pixels. The model processes images by dividing them into non-overlapping patches, embedding each patch as a token, and applying the sequence of transformer layers with self-attention and MLP blocks. A class token ([CLS]) is prepended to the sequence and used for image-level classification in CLIP's original contrastive training.

The ViT-L/14 variant has approximately 304M parameters in the vision encoder alone. The model was trained using contrastive learning on 400M image-text pairs from the internet (the WIT dataset), where the objective was to maximize the cosine similarity between an image's [CLS] representation and its corresponding text's representation while minimizing similarity to non-matching pairs. This training produces features that encode rich semantic information about visual concepts, organized by their textual descriptions.

## Advantages of CLIP Features for Segmentation

CLIP features offer several advantages over traditional ImageNet-pretrained backbones for segmentation tasks:

First, **semantic richness**: CLIP features encode high-level semantic concepts aligned with natural language, meaning that features for visually similar but semantically different objects (e.g., a car versus a bus) are well-separated in feature space. This semantic organization provides a strong prior for classification that does not need to be learned from scratch during segmentation training.

Second, **open-vocabulary capability**: Because CLIP features are aligned with text embeddings, new categories can be introduced at inference time by computing text embeddings for their names. A segmentation model using CLIP features can classify a region as "fire hydrant" even if this category was never seen during segmentation training, simply by matching the region's visual features against the text embedding for "fire hydrant."

Third, **robustness and generalization**: CLIP was trained on a vastly more diverse dataset (400M internet image-text pairs) than ImageNet (1.2M curated images in 1000 categories). This diversity produces features that are more robust to distribution shift, domain change, and unusual visual appearances. Segmentation models using CLIP features have been shown to generalize better to out-of-distribution images than those using ImageNet features.

Fourth, **rich intermediate representations**: The deep transformer architecture produces informative features at every layer, with early layers encoding low-level visual patterns and later layers encoding high-level semantic concepts. This hierarchy is well-suited for multi-scale segmentation.

## Feature Extraction Strategy

Converting CLIP's single-scale ViT output into multi-scale feature maps suitable for segmentation requires extracting features from multiple transformer layers. OMG-Seg taps into layers at 1/4, 1/2, 3/4, and the full depth of the network (e.g., layers 6, 12, 18, and 24 for a 24-layer ViT-L). Each layer's output tokens are reshaped from the 1D sequence back to 2D spatial feature maps at the ViT's native spatial resolution (input_size / patch_size, e.g., 64x64 for 896x896 input with 14x14 patches).

These single-resolution features from different layers are then processed through a Feature Pyramid Network (FPN) that applies 1x1 convolutions to reduce all features to a common channel dimension (256) and progressive upsampling/downsampling to create a proper multi-scale pyramid at strides 4, 8, 16, and 32 relative to the input image. This multi-scale pyramid is essential for detecting objects at different scales, from small instances requiring high-resolution features to large regions benefiting from high-level semantic features.

An alternative approach used by some methods is to apply the ViT at multiple input resolutions and combine the outputs, but this is computationally expensive. OMG-Seg's layer-based extraction is more efficient because it requires only a single forward pass through the ViT.

## Text-Image Alignment

The vision-language alignment in CLIP is leveraged for classification in OMG-Seg's segmentation pipeline. After the decoder produces object queries, each query's class is determined by computing the cosine similarity between the query's feature vector and text embeddings for all candidate categories. Text embeddings are generated by passing category names through CLIP's text encoder (a transformer with 12 layers), optionally with prompt engineering templates such as "a photo of a {category}."

This alignment enables several powerful capabilities: (1) **open-vocabulary segmentation**, where the model can segment categories it was never trained to segment, as long as a text description is provided; (2) **zero-shot transfer across datasets**, where the model trained on COCO can be applied to ADE20K by simply changing the text category set; and (3) **natural language queries**, where a user can describe a target object in free text (e.g., "the red car on the left") and the model identifies the corresponding region.

The quality of text-image alignment depends on the specificity of the text description. Generic category names ("dog") work well for common objects, while more specific descriptions ("golden retriever puppy lying on a couch") can provide better discrimination for fine-grained categories. Prompt engineering (using templates like "a photo of a {category}" rather than just "{category}") has been shown to improve classification accuracy by 2-5 percentage points.

## Fine-Tuning CLIP for Dense Prediction

CLIP was designed for image-level tasks, and several adaptations are needed for dense (pixel-level) prediction. The primary challenge is that CLIP's training objective (contrastive matching of whole images with text) does not encourage spatially precise features. The [CLS] token aggregates global information but individual patch tokens may not capture fine-grained spatial details important for segmentation boundaries.

OMG-Seg addresses this through: (1) **fine-tuning with dense supervision**, where the CLIP backbone's parameters are updated during segmentation training with pixel-level loss signals that encourage spatially precise features; (2) **multi-scale feature extraction**, which provides high-resolution features from early layers even if later layers have coarser spatial information; (3) **a decoder with masked attention**, which forces the model to attend to specific spatial regions rather than global features. The fine-tuning is performed with a low learning rate (1e-5 for the CLIP backbone versus 1e-4 for the decoder) to preserve the pretrained semantic organization while improving spatial precision.

An alternative approach, explored in methods like MaskCLIP, avoids fine-tuning entirely by using CLIP features as-is and relying on the decoder to compensate for spatial imprecision. This preserves CLIP's open-vocabulary capability more fully but achieves lower segmentation quality than fine-tuned approaches. OMG-Seg's partial fine-tuning represents a compromise that retains most of the open-vocabulary capability while improving spatial precision.

## Comparison with Other Backbones

The choice of backbone significantly impacts segmentation performance across different dimensions:

| Backbone | Parameters | Semantic Richness | Spatial Precision | Open-Vocabulary | Training Data Scale |
|----------|-----------|-------------------|-------------------|-----------------|---------------------|
| CLIP ViT-L/14 | 304M | Excellent | Good (after FPN) | Yes | 400M image-text pairs |
| MAE ViT-H (SAM) | 632M | Good | Excellent | No | SA-1B (11M images) |
| Swin-L | 197M | Good | Excellent | No | ImageNet-22K (14M) |
| ResNet-101 | 44M | Moderate | Good | No | ImageNet-1K (1.2M) |
| Hiera-L (SAM 2) | 214M | Good | Excellent | No | SA-1B + SA-V |

CLIP features excel at classification and open-vocabulary tasks but are less spatially precise than MAE-pretrained ViT (used in SAM) or hierarchical architectures like Swin. SAM's MAE ViT-H produces superior boundary quality because MAE pretraining explicitly trains the model to reconstruct spatial details from masked patches. Swin Transformer's hierarchical design produces naturally multi-scale features, avoiding the need for layer-based extraction. However, none of these alternatives provide vision-language alignment, making CLIP uniquely suited for open-vocabulary segmentation.

## Implementation Notes

OMG-Seg uses the `open_clip` library to load pretrained CLIP ViT-L/14 weights. Multi-scale feature extraction is implemented by registering forward hooks on the specified transformer layers to capture intermediate activations. The FPN is implemented as a separate module that processes these intermediate features into a standard multi-scale pyramid. The CLIP text encoder is used at initialization to precompute text embeddings for all target categories, which are stored as a fixed tensor and used for classification during both training and inference.

For input preprocessing, images are resized to 896x896 pixels (a multiple of the 14-pixel patch size), producing 64x64 spatial tokens. The CLIP normalization (mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]) is applied to match CLIP's pretraining statistics. Inference speed is approximately 5 FPS on an A100 GPU for the full pipeline (CLIP encoding + FPN + decoder), with the CLIP backbone accounting for approximately 70% of the compute time.
