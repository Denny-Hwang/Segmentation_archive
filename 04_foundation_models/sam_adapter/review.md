---
title: "SAM-Adapter: Adapting SAM in Underperformed Scenes"
date: 2025-03-06
status: complete
tags: [adapter, parameter-efficient, domain-adaptation, sam]
difficulty: intermediate
---

# SAM-Adapter

## Paper Overview

**Title:** SAM-Adapter: Adapting Segment Anything Model in Underperformed Scenes
**Authors:** Tianrun Chen, Lanyun Zhu, Chaotao Ding, Runlong Cao, Yan Wang, Zejian Li, Lingyun Sun, Papa Mao, Ying Zang
**Venue:** ICCV 2023 Workshop
**Year:** 2023

SAM-Adapter proposes a parameter-efficient method for adapting SAM to domains where it underperforms, such as camouflaged object detection and shadow detection. The approach inserts lightweight adapter modules into SAM's ViT image encoder while keeping the original pretrained weights frozen, enabling domain-specific adaptation with minimal additional parameters.

## Motivation

While SAM achieves impressive zero-shot performance on common objects, it struggles on visually challenging scenes:

- **Camouflaged objects:** Objects that blend into their background (e.g., camouflaged animals, military concealment)
- **Shadow detection:** Identifying shadow regions that have subtle boundaries
- **Medical imaging:** Low-contrast structures in grayscale images
- **Remote sensing:** Small objects in aerial/satellite imagery

Full fine-tuning on these domains is expensive and risks catastrophic forgetting. SAM-Adapter offers an alternative: keep SAM's general knowledge intact and add small, trainable modules that inject domain-specific information.

## Architecture

### Adapter Module Design

Each adapter module is a bottleneck MLP inserted into the ViT encoder blocks:

```
Input features (dim D)
       |
   Linear (D -> d)     [down-projection, d << D]
       |
     ReLU
       |
   Linear (d -> D)     [up-projection]
       |
   Scale factor (s)
       |
  + Residual connection
       |
Output features (dim D)
```

Key parameters:
- **Bottleneck dimension d:** Typically 64 or 128 (vs. D=1280 for ViT-H)
- **Scale factor s:** Learned scalar that controls the adapter's influence, initialized to a small value (0.1)
- **Residual connection:** Ensures the adapter's contribution is additive and the model degrades gracefully

### Insertion Points

Adapters are inserted at two locations within each ViT transformer block:
1. **After the multi-head self-attention (MHSA) layer**
2. **After the feed-forward network (FFN) layer**

This follows the standard adapter placement from NLP (Houlsby et al., 2019), adapted for vision transformers.

### Task-Specific Information Injection

SAM-Adapter can optionally incorporate task-specific prior information:
- For shadow detection: depth maps or edge maps can be encoded and injected through the adapters
- For camouflaged detection: frequency domain features can supplement the spatial features
- The adapter's down-projection can accept concatenated features from auxiliary inputs

## Training

### What Is Trained

| Component | Trainable | Parameters |
|-----------|-----------|------------|
| ViT image encoder | Frozen | 0 |
| Adapter modules | Trained | ~4M (0.6% of ViT-H) |
| Prompt encoder | Frozen | 0 |
| Mask decoder | Trained | ~4M |
| **Total trainable** | | **~8M (1.2% of total)** |

### Training Configuration

- Optimizer: AdamW
- Learning rate: 1e-4 with cosine decay
- Batch size: 4-8
- Epochs: 20-50 (domain-dependent)
- Loss: Binary cross-entropy + dice loss
- Prompts during training: automatic bounding boxes derived from ground truth

## Key Results

### Camouflaged Object Detection

| Method | S-measure | E-measure | MAE |
|--------|-----------|-----------|-----|
| SAM (zero-shot) | 0.669 | 0.719 | 0.092 |
| SINet-V2 (specialized) | 0.820 | 0.882 | 0.048 |
| SAM-Adapter | 0.841 | 0.899 | 0.039 |

SAM-Adapter surpasses both zero-shot SAM and the specialized SINet-V2 model on camouflaged object detection.

### Shadow Detection

| Method | BER (lower is better) |
|--------|----------------------|
| SAM (zero-shot) | 12.8 |
| MTMT (specialized) | 5.1 |
| SAM-Adapter | 4.2 |

### Parameter Efficiency

| Adaptation Method | Trainable Params | COD S-measure |
|-------------------|-----------------|---------------|
| Full fine-tuning | 636M (100%) | 0.838 |
| Head-only | 4M (0.6%) | 0.762 |
| LoRA (rank 16) | 6M (0.9%) | 0.825 |
| **SAM-Adapter** | **8M (1.2%)** | **0.841** |

SAM-Adapter achieves the best performance while training only 1.2% of total parameters.

## Comparison to Other Adaptation Methods

### vs. Full Fine-Tuning
- SAM-Adapter matches or slightly exceeds full fine-tuning performance
- Requires 80x fewer trainable parameters
- Multiple domain adaptations can share the same frozen backbone

### vs. LoRA
- SAM-Adapter and LoRA achieve similar performance
- SAM-Adapter provides a more flexible architecture for injecting auxiliary information
- LoRA modifies existing weight matrices; adapters add new pathways

### vs. Prompt Tuning
- Visual prompt tuning (prepending learnable tokens) is even more parameter-efficient
- But typically achieves 3-5 points lower performance than SAM-Adapter
- SAM-Adapter modifies deeper feature representations, which is more effective for large domain shifts

## Strengths

- Minimal parameter overhead enables adaptation to many domains simultaneously
- Frozen backbone preserves SAM's general segmentation knowledge
- Modular design allows swapping adapters for different tasks at inference time
- Strong performance on challenging domains that SAM fails on out of the box
- Compatible with any SAM backbone variant (ViT-B, ViT-L, ViT-H)

## Limitations

- Still requires domain-specific labeled data for training
- Adapter design choices (bottleneck size, insertion points) require tuning per domain
- The mask decoder is also fine-tuned, which somewhat limits modularity
- Performance on in-distribution data (COCO, SA-1B-like) may slightly degrade
- Does not address the prompt engineering challenge (still needs good prompts at inference)

## Impact

SAM-Adapter established parameter-efficient adaptation as a viable strategy for extending foundation segmentation models to new domains. It influenced numerous follow-up works that apply adapters, LoRA, and other PEFT methods to SAM for specialized applications including medical imaging, remote sensing, and industrial inspection.

## Citation

```
Chen, T., et al. "SAM-Adapter: Adapting Segment Anything Model in Underperformed Scenes." ICCV Workshop 2023.
```
