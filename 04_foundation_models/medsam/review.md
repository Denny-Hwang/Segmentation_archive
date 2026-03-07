---
title: "Segment Anything in Medical Images"
date: 2025-03-06
status: complete
tags: [foundation-model, medical-segmentation, domain-adaptation, sam]
difficulty: intermediate
---

# MedSAM

## Paper Overview

**Title:** Segment Anything in Medical Images
**Authors:** Jun Ma, Yuting He, Feifei Li, Lin Han, Chenyu You, Bo Wang
**Venue:** Nature Communications, 2024
**Institution:** University of Toronto / Vector Institute

MedSAM adapts the Segment Anything Model for medical image segmentation by fine-tuning SAM on a large-scale curated dataset of over 1.5 million medical image-mask pairs spanning 10 imaging modalities and more than 30 cancer types. The key design choice is to use bounding box prompts exclusively, making the model practical for clinical workflows where radiologists typically identify regions of interest.

## Motivation

SAM demonstrates impressive zero-shot segmentation on natural images, but its performance degrades significantly on medical images due to the large domain gap:

- Medical images have fundamentally different appearance characteristics (grayscale, low contrast, specific textures)
- Anatomical structures lack the clear boundaries common in natural images
- SA-1B contains virtually no medical imaging data
- Pathological findings (tumors, lesions) have highly variable appearance

MedSAM addresses this by fine-tuning SAM on a comprehensive medical dataset rather than training from scratch, leveraging the general visual understanding already captured in SAM's weights.

## Architecture

MedSAM retains SAM's three-component architecture:

### Image Encoder
- ViT-B backbone (chosen over ViT-H for efficiency in clinical settings)
- Pretrained weights from SAM, fine-tuned on medical data
- Input resolution: 1024x1024 (medical images resized accordingly)

### Prompt Encoder
- **Bounding box prompts only** during both training and inference
- The authors found that box prompts provide the best quality-effort tradeoff for medical applications
- Point prompts were excluded because they require more expertise to place effectively on medical structures
- Box prompts align naturally with how radiologists identify regions of interest

### Mask Decoder
- Lightweight transformer decoder identical to SAM's
- Fine-tuned alongside the encoder
- Single mask output (no multi-mask ambiguity resolution needed with box prompts)

## Training Dataset

### Scale and Composition

| Property | Value |
|----------|-------|
| Total image-mask pairs | 1,570,263 |
| Imaging modalities | 10+ |
| Anatomical structures | 30+ organ/tissue types |
| Cancer types | 30+ |
| Public dataset sources | 52 |

### Imaging Modalities Covered

1. **CT (Computed Tomography)** - largest portion of the dataset
2. **MRI (Magnetic Resonance Imaging)** - multiple sequences (T1, T2, FLAIR)
3. **X-ray** - chest, musculoskeletal
4. **Ultrasound** - abdominal, cardiac, obstetric
5. **Dermoscopy** - skin lesion imaging
6. **Endoscopy** - gastrointestinal
7. **Fundus photography** - retinal imaging
8. **Mammography** - breast imaging
9. **OCT (Optical Coherence Tomography)** - retinal layers
10. **Pathology** - histopathological slides

### Data Curation

- Sourced from 52 publicly available medical segmentation datasets
- Unified annotation format (converting all formats to binary masks with bounding boxes)
- Quality filtering to remove corrupted or poorly annotated samples
- 3D volumes (CT, MRI) were sliced into 2D images, with bounding boxes computed per slice

## Training Details

- Optimizer: AdamW with learning rate 1e-4
- Batch size: 160
- Training epochs: 100
- Loss: Combination of cross-entropy and dice loss
- Data augmentation: Random horizontal/vertical flips, random rotation
- All components (encoder + decoder) are fine-tuned end-to-end

## Key Results

### Comparison to SAM (Zero-Shot)

| Modality | SAM (zero-shot) DSC | MedSAM DSC |
|----------|---------------------|------------|
| CT (abdominal) | 0.62 | 0.87 |
| MRI (brain) | 0.55 | 0.84 |
| Dermoscopy | 0.70 | 0.90 |
| X-ray | 0.58 | 0.82 |
| Endoscopy | 0.60 | 0.88 |
| Ultrasound | 0.52 | 0.80 |

MedSAM improves over zero-shot SAM by 15-30 DSC points across modalities, demonstrating the necessity of domain-specific fine-tuning.

### Comparison to Specialized Models

MedSAM is competitive with modality-specific state-of-the-art models (nnU-Net, TransUNet, Swin-UNETR) on many benchmarks while being a single unified model. Specialized models still outperform MedSAM on their specific domains, but MedSAM offers the advantage of generality.

## Strengths

- Single model handles 10+ imaging modalities without modality-specific tuning
- Bounding box interface is intuitive and practical for clinical use
- Fine-tuning from SAM weights is more data-efficient than training from scratch
- Large and diverse training set reduces overfitting to any single modality
- Open-source release enables community adoption and further development

## Limitations

- Box-only prompts limit the model's ability to handle ambiguous cases where points or text would be more appropriate
- 2D processing means no native volumetric reasoning for 3D medical images
- Performance varies significantly across modalities (best on CT/dermoscopy, weaker on ultrasound)
- Cannot produce semantic labels (class-agnostic masks only)
- Requires a bounding box at inference time, which may need an upstream detection model for fully automatic pipelines

## Impact

MedSAM demonstrated that foundation model adaptation is viable for medical imaging and catalyzed a wave of follow-up work. It established the benchmark for medical foundation models and showed that a single model can reasonably cover the diversity of medical imaging modalities.

## Citation

```
Ma, J., et al. "Segment anything in medical images." Nature Communications 15, 654 (2024).
```
