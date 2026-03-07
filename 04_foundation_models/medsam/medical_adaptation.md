---
title: "Medical Domain Adaptation in MedSAM"
date: 2025-03-06
status: planned
tags: [medical-adaptation, fine-tuning, domain-shift, transfer-learning]
difficulty: intermediate
---

# Medical Domain Adaptation

## Overview

Adapting SAM to the medical imaging domain requires bridging a substantial visual domain gap between natural photographs and clinical images. Medical images have fundamentally different characteristics: low contrast between adjacent structures, modality-specific noise patterns (e.g., speckle noise in ultrasound), vastly different spatial resolutions (from sub-micron pathology to millimeter-scale CT), and structures whose significance depends on domain knowledge rather than visual salience. MedSAM addresses this gap through end-to-end fine-tuning on a large-scale, multi-modality medical dataset, adapting all of SAM's learned representations to the medical domain while preserving the general feature extraction capabilities learned from natural images.

## Domain Gap Between Natural and Medical Images

The domain gap between natural and medical images manifests across several dimensions. First, natural images use RGB color channels optimized for human vision, while medical images use modality-specific intensity representations: CT images encode Hounsfield units reflecting tissue density, MRI signal intensities depend on tissue relaxation properties (T1, T2), and ultrasound images display acoustic reflectance patterns. Second, objects in natural images are typically well-defined with clear edges against contrasting backgrounds, while medical structures often have subtle, low-contrast boundaries (e.g., a tumor boundary in soft tissue may differ by only 5-10 Hounsfield units from surrounding tissue).

Third, the visual priors learned from natural images (that objects are roughly convex, have consistent texture, and are visually salient) do not hold in medical imaging, where structures can be highly irregular, have heterogeneous internal texture, and be visually subtle. Quantitative analysis shows that SAM's image encoder activations on medical images have significantly different statistical distributions compared to natural images: the mean activation magnitude is approximately 30% lower, and the spatial attention patterns are less focused, suggesting that the encoder's learned features are less discriminative for medical content.

## Fine-Tuning Approach

MedSAM fine-tunes all components of SAM (image encoder, prompt encoder, and mask decoder) end-to-end on medical data. The ViT-B backbone is used rather than ViT-H to balance performance and computational feasibility. All layers are unfrozen, allowing the model to adapt its learned representations at every level of the feature hierarchy. The learning rate is set to 1e-4 with cosine decay, and training proceeds for 100 epochs with a batch size of 16 on 4 A100 GPUs, requiring approximately 5 days of training.

The decision to fine-tune all parameters (rather than using parameter-efficient methods like LoRA or adapters) was motivated by the severity of the domain gap. Experiments showed that adapter-only or decoder-only fine-tuning recovered approximately 60-70% of the full fine-tuning improvement, suggesting that the image encoder's representations need substantial modification for medical images. However, full fine-tuning risks catastrophic forgetting of natural image features; MedSAM mitigates this by using a low learning rate and gradual warm-up, preserving useful low-level features while adapting high-level representations.

## Multi-Modality Training

Training a single model across 10 imaging modalities presents challenges in data balancing and representation learning. Each modality has different visual characteristics, so naive uniform sampling would lead to the model being dominated by whichever modality has the most training data (CT, with approximately 40% of samples). MedSAM addresses this through proportional sampling with a modality-mixing strategy: within each training batch, samples are drawn from multiple modalities to ensure the model sees diverse inputs throughout training.

The shared encoder must learn features useful for modalities as different as high-resolution pathology (where texture patterns at cellular scale are informative) and CT (where intensity values and anatomical context are key). Remarkably, the model learns to produce useful representations for all modalities simultaneously, suggesting that certain mid-level visual features (edges, boundaries, region homogeneity) are shared across modalities. However, performance on each modality is lower than what a modality-specific model could achieve: a CT-only fine-tuned SAM achieves approximately 2-3 dice points higher than MedSAM on CT benchmarks.

## Data Curation and Preprocessing

The training dataset was curated from over 30 publicly available medical imaging datasets, requiring extensive preprocessing to standardize formats. 3D volumes (CT, MRI) were sliced along the axial plane into individual 2D images, with empty slices (containing no annotated structures) removed. Images were resized to 1024x1024 pixels to match SAM's input resolution. Intensity normalization was applied per-modality: CT images were windowed to clinically relevant Hounsfield unit ranges (e.g., soft tissue window -175 to 250 HU), MRI images were normalized to zero mean and unit variance per volume, and other modalities were normalized to [0, 1] range.

Ground truth masks were converted from various annotation formats (NIfTI, DICOM-SEG, polygon annotations) into binary mask format. For multi-class annotations, each class was treated as a separate binary mask with its own bounding box prompt. This means a single image with liver and kidney annotations would generate two training samples, each with a different box prompt and binary mask target. Data augmentation included random horizontal and vertical flipping, rotation (up to 15 degrees), and elastic deformation.

## Impact of Adaptation on Performance

Fine-tuning improved SAM's medical segmentation performance dramatically. On internal validation, MedSAM achieved 87.2% mean dice compared to 62.4% for zero-shot SAM -- an improvement of approximately 25 dice points. The improvement varied by modality: CT and MRI saw the largest gains (28-32 dice points) because they are most dissimilar from natural images. Dermoscopy and endoscopy, which share more visual characteristics with natural photos, showed smaller but still significant gains (15-18 points).

Compared to training a ViT-B segmentation model from scratch on the same medical dataset, MedSAM achieved approximately 4 dice points higher performance, demonstrating that pre-training on SA-1B provides useful features even for the medical domain. The benefit of pre-training was most pronounced in low-data regimes: when only 10% of the medical training data was used, the pre-trained model outperformed the scratch model by 8 dice points, suggesting that SAM's pre-training provides strong regularization for medical segmentation.

## Generalization to Unseen Modalities

MedSAM's generalization was tested on 19 external datasets containing modalities, anatomies, or imaging protocols not present in the training set. On these unseen domains, MedSAM achieved 78.9% mean dice -- lower than the 87.2% on internal benchmarks but substantially higher than vanilla SAM's 55.1%. The generalization gap was smallest for modalities similar to those in training (e.g., a different MRI sequence or a different CT scanner) and largest for truly novel visual domains (e.g., intraoperative ultrasound or specialized microscopy).

These results suggest that MedSAM learns generalizable medical image features rather than simply memorizing the training distribution. However, a persistent 8-10 dice point gap between in-distribution and out-of-distribution performance indicates that further domain adaptation (either through additional fine-tuning on target-domain data or through domain-agnostic training strategies) is needed for clinical deployment on novel modalities.

## Comparison with Other Adaptation Methods

Full fine-tuning (MedSAM's approach) achieves the highest absolute performance but requires the most computational resources and training data. Adapter-based tuning (as in SAM-Adapter) trains only 2-5% of parameters, achieving approximately 80-85% of full fine-tuning performance with significantly less compute. LoRA adaptation of SAM (explored in concurrent works) trains approximately 1-2% of parameters and achieves similar performance to adapters. Head-only fine-tuning (freezing the encoder and training only the decoder) is the most efficient but recovers only about 50% of the full fine-tuning improvement.

The choice of adaptation strategy depends on the use case. For building a general-purpose medical segmentation tool (MedSAM's goal), full fine-tuning is justified because the one-time training cost is amortized across all future users. For adapting to a specific clinical task with limited data (e.g., a rare tumor type), parameter-efficient methods like adapters or LoRA are more appropriate because they reduce overfitting risk and training requirements.

## Implementation Notes

MedSAM is publicly available through GitHub and Hugging Face, with pre-trained weights for direct inference. The inference pipeline accepts a medical image (any modality) and a bounding box prompt, producing a binary segmentation mask. Input preprocessing (intensity normalization, resizing to 1024x1024) is handled by the provided code. Inference takes approximately 50ms per image on an A100 GPU. For 3D volumes, slices must be processed independently and then stacked, as MedSAM does not incorporate inter-slice context. The model can be further fine-tuned on task-specific data using the provided training scripts, which support custom datasets with minimal configuration.
