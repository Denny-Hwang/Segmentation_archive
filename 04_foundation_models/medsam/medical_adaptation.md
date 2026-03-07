---
title: "Medical Domain Adaptation in MedSAM"
date: 2025-03-06
status: complete
tags: [medical-adaptation, fine-tuning, domain-shift, transfer-learning]
difficulty: intermediate
---

# Medical Domain Adaptation

## Overview

Adapting foundation segmentation models like SAM to medical imaging involves bridging a substantial domain gap. Natural images and medical images differ in fundamental ways, from pixel statistics to semantic content. This document covers the challenges of this domain shift, the strategies used in MedSAM and related work, and practical considerations for medical adaptation.

## The Domain Gap

### Visual Differences

| Property | Natural Images | Medical Images |
|----------|---------------|----------------|
| Color channels | RGB (3 channels) | Often grayscale (1 channel) or multi-modal |
| Dynamic range | 8-bit (0-255) | 12-16 bit (CT: -1000 to +3000 HU) |
| Contrast | High, varied | Often low, tissue-specific |
| Textures | Diverse, distinctive | Subtle, homogeneous |
| Boundaries | Sharp edges | Gradual transitions |
| Objects | Everyday items | Organs, lesions, anatomical structures |
| Background | Varied scenes | Black (CT/MRI) or uniform |

### Statistical Distribution Shift

SAM was trained on SA-1B, which contains virtually no medical images. The feature distributions learned by SAM's ViT encoder are therefore misaligned with medical image characteristics:

- Low-level features (edges, textures) are calibrated for natural image statistics
- Mid-level features (parts, shapes) encode object categories absent from medical imaging
- High-level features (object semantics) have no medical knowledge

Empirically, this results in SAM's zero-shot performance dropping by 15-35 DSC points on medical benchmarks compared to natural image benchmarks.

## Fine-Tuning Strategies

### Full Fine-Tuning (MedSAM Approach)

MedSAM fine-tunes all parameters of the SAM architecture (encoder + prompt encoder + decoder) on medical data.

**Advantages:**
- Maximum capacity for learning domain-specific features
- All layers can adapt to the new distribution
- Straightforward implementation

**Disadvantages:**
- Requires large amounts of medical training data to avoid overfitting
- Computationally expensive (full backpropagation through ViT)
- Risk of catastrophic forgetting of natural image capabilities
- Large storage footprint (one full model copy per adaptation)

### Head-Only Fine-Tuning

Only the mask decoder is fine-tuned while the image encoder remains frozen.

**Advantages:**
- Fast training (fewer parameters to update)
- Lower risk of overfitting on small datasets
- Preserves the encoder's general visual features

**Disadvantages:**
- Limited adaptation capacity since the encoder features remain fixed
- If the encoder features are poorly suited to medical data, the decoder cannot compensate
- Typically 5-10 DSC points below full fine-tuning

### Parameter-Efficient Fine-Tuning (PEFT)

Methods like LoRA and adapters modify only a small subset of parameters.

**Approaches:**
- **LoRA:** Low-rank updates to attention weight matrices, typically rank 4-16
- **Adapters:** Bottleneck layers inserted after each transformer block
- **Prompt tuning:** Learnable tokens prepended to the input sequence

These approaches train only 1-5% of the total parameters while achieving 80-95% of full fine-tuning performance. They are particularly valuable when:
- Medical data is limited (< 10K samples)
- Multiple domain-specific adaptations are needed simultaneously
- Deployment requires a single base model with swappable adapters

## Multi-Modality Training

### Challenge

Medical imaging spans fundamentally different modalities (CT, MRI, ultrasound, dermoscopy, etc.) with distinct physical imaging principles and visual characteristics. A single model must handle this diversity.

### MedSAM's Approach

MedSAM trains on all modalities simultaneously without modality-specific components:

1. All images are converted to 3-channel format (grayscale replicated or modality-specific normalization)
2. Intensity values are normalized to [0, 255] range per modality convention
3. The model learns shared representations across modalities
4. No modality indicator is provided to the model; it must infer the modality from visual content

### Alternative: Modality-Specific Heads

Some approaches use a shared encoder with modality-specific decoder heads:
- A routing mechanism selects the appropriate head based on modality
- Each head specializes in the characteristics of its modality
- The shared encoder still learns general features

This approach can outperform a single head but increases model complexity and requires modality labels at inference time.

## 3D Volume Handling

### The 2D Slice Approach

Most SAM adaptations (including MedSAM) process 3D medical volumes (CT, MRI) as stacks of 2D slices:

1. Each slice is treated as an independent image
2. Bounding boxes are computed per slice (using the 3D ground truth projected onto each slice)
3. No inter-slice consistency is enforced during inference

**Limitation:** This ignores valuable volumetric context. Adjacent slices contain highly correlated information that could improve segmentation accuracy and consistency.

### Volumetric Extensions

Subsequent works (including MedSAM-2) address this by:
- Treating volume slices as video frames and using temporal propagation
- Adding 3D convolutional layers to the encoder
- Post-processing with 3D connected component analysis

## Practical Considerations for Medical Adaptation

### Data Requirements

| Strategy | Minimum Samples | Recommended |
|----------|----------------|-------------|
| Zero-shot | 0 | N/A |
| Head-only fine-tuning | ~500 | ~2,000 |
| LoRA/Adapter | ~1,000 | ~5,000 |
| Full fine-tuning | ~10,000 | ~100,000+ |

### Preprocessing Pipeline

Medical images require careful preprocessing before being fed to SAM-based models:
1. **Window/level adjustment** (CT): Map relevant HU range to [0, 255]
2. **Intensity normalization** (MRI): Z-score or percentile-based normalization
3. **Resize:** Scale to 1024x1024 (SAM's expected input size)
4. **Channel replication:** Convert single-channel to 3-channel if needed

### Evaluation Metrics

Medical segmentation uses specific metrics:
- **Dice Similarity Coefficient (DSC):** Overlap between prediction and ground truth
- **Normalized Surface Distance (NSD):** Boundary accuracy within a tolerance
- **Hausdorff Distance (HD95):** Worst-case boundary error at 95th percentile
- These differ from COCO-style metrics (AP, AR) used in natural image evaluation

## Open Challenges

1. **Annotation scarcity:** Medical annotations require expert radiologists, making large datasets expensive
2. **Label noise:** Inter-annotator variability in medical imaging is high (10-20% disagreement)
3. **Rare conditions:** Uncommon pathologies have very few training samples
4. **Regulatory requirements:** Clinical deployment requires validation beyond standard benchmarks
5. **Privacy constraints:** Medical data cannot be freely shared, limiting dataset creation
