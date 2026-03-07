---
title: "SAM-Adapter: Adapting SAM in Underperformed Scenes"
date: 2025-03-06
status: planned
tags: [adapter, parameter-efficient, domain-adaptation, sam]
difficulty: intermediate
---

# SAM-Adapter

## Meta Information

| Field | Value |
|-------|-------|
| **Paper Title** | SAM Fails to Segment Anything? -- SAM-Adapter: Adapting SAM in Underperformed Scenes |
| **Authors** | Chen, T., Zhu, L., Deng, C., Cao, R., Wang, Y., Zhang, S., Li, Z., Sun, L., Zang, Y., Mao, P. |
| **Year** | 2023 |
| **Venue** | arXiv |
| **arXiv** | [2304.09148](https://arxiv.org/abs/2304.09148) |
| **Difficulty** | Intermediate |

## One-Line Summary

SAM-Adapter inserts lightweight adapter modules into SAM's image encoder to enable parameter-efficient domain adaptation for challenging scenes where vanilla SAM underperforms.

## Motivation and Problem Statement

Despite SAM's impressive zero-shot generalization on natural images, systematic evaluation revealed significant failure modes in challenging visual domains. SAM struggles with camouflaged object detection (where objects blend into their background), shadow detection (where shadow boundaries are subtle), polyp segmentation in medical endoscopy, and other scenarios where target objects have low contrast, ambiguous boundaries, or atypical visual characteristics. On camouflaged object detection benchmarks like COD10K, SAM's zero-shot performance dropped to approximately 40-50% mean IoU, far below specialized models achieving 80%+.

The naive solution of full fine-tuning (as in MedSAM) requires substantial computational resources and risks catastrophic forgetting. SAM-Adapter proposes a parameter-efficient alternative: inserting small adapter modules into SAM's frozen image encoder. These adapters learn domain-specific features while preserving SAM's general visual representations, requiring training of only 2-5% of the total parameters. This approach is inspired by adapter tuning in NLP, where small bottleneck modules inserted into pre-trained transformers enable efficient task adaptation.

## Architecture Overview

SAM-Adapter modifies SAM by inserting adapter modules into each transformer block of the ViT image encoder. The adapters are placed after the multi-head self-attention (MHSA) and feed-forward network (FFN) sub-layers within each block, following the standard adapter placement strategy from NLP. The image encoder's original parameters are completely frozen, and only the adapter parameters (plus a task-specific prediction head) are trained. The prompt encoder and mask decoder are also kept frozen in the standard configuration, though the mask decoder can optionally be fine-tuned for additional performance.

### Key Components

- **Adapter Tuning**: See [adapter_tuning.md](adapter_tuning.md)

## Technical Details

### Adapter Module Design

Each adapter module follows a bottleneck architecture: a down-projection linear layer that reduces the feature dimension from the model's hidden size (e.g., 768 for ViT-B or 1280 for ViT-H) to a smaller bottleneck dimension (typically 64 or 128), a non-linear activation function (GELU), and an up-projection linear layer that restores the original dimension. A residual connection adds the adapter's output to the original feature, ensuring that the adapter can only modify the representation rather than replacing it entirely. Each adapter module has approximately 100K-200K parameters depending on the bottleneck dimension.

The adapter also includes a learnable scaling factor (initialized to a small value like 0.1) that controls the magnitude of the adapter's contribution relative to the frozen features. This initialization strategy ensures that at the start of training, the adapted model behaves nearly identically to the original SAM, with the adapters gradually learning to inject domain-specific information as training progresses.

### Insertion Points

Adapters are inserted at two locations within each ViT transformer block: after the multi-head self-attention layer and after the feed-forward network. For a ViT-B encoder with 12 transformer blocks, this results in 24 adapter modules totaling approximately 2.4M-4.8M trainable parameters (depending on bottleneck size). For ViT-H with 32 blocks, this increases to 64 adapter modules with approximately 6.4M-12.8M parameters. Ablation studies showed that inserting adapters at both locations outperforms inserting only after MHSA or only after FFN by approximately 1-2 IoU points.

### Frozen vs. Trainable Parameters

The training configuration freezes all of SAM's original parameters (image encoder, prompt encoder, mask decoder) and trains only the adapter modules and the task-specific output head. For ViT-B, this means approximately 4M trainable parameters out of 93M total -- roughly 4.3% of the model. For ViT-H, approximately 12M out of 636M are trainable -- roughly 1.9%. This parameter efficiency has two benefits: it drastically reduces GPU memory requirements for training (only adapter gradients need to be stored), and it prevents catastrophic forgetting of SAM's general visual capabilities.

### Task-Specific Heads

Since SAM's original mask decoder produces class-agnostic binary masks, SAM-Adapter adds task-specific prediction heads for structured prediction tasks. For camouflaged object detection and shadow detection, a lightweight decoder head (4-layer CNN with skip connections) replaces SAM's mask decoder to produce dense predictions at the input resolution. For medical segmentation tasks, the original mask decoder is retained but with an additional classification layer. The task-specific heads are always trained from scratch and typically contain 1-3M parameters.

### Training Strategy

Training uses standard supervised learning with task-specific loss functions: binary cross-entropy + dice loss for binary segmentation tasks, and structure-aware losses (weighted binary cross-entropy) for tasks like edge detection. The AdamW optimizer is used with learning rate 1e-4 for adapters and 1e-3 for the task-specific head, with cosine learning rate decay. Training typically converges in 30-50 epochs on a single GPU, requiring approximately 2-4 hours for datasets with 5,000-10,000 images. Data augmentation includes random flipping, rotation, and multi-scale resizing.

## Experiments and Results

### Target Domains

SAM-Adapter was evaluated on four challenging domains where vanilla SAM underperforms. Camouflaged object detection (COD10K, CAMO, NC4K): objects intentionally blend into their surroundings. Shadow detection (SBU, ISTD): detecting shadow regions in natural images. Polyp segmentation (Kvasir-SEG, CVC-ClinicDB): segmenting polyps in endoscopy images. Salient object detection (DUTS, DUT-OMRON): detecting visually prominent objects.

### Key Results

On camouflaged object detection (COD10K test set), SAM-Adapter achieved a mean IoU of 78.3%, compared to 42.1% for vanilla SAM zero-shot and 81.5% for the specialized SINet-v2 model. On shadow detection (SBU test), SAM-Adapter achieved 90.2% IoU versus 55.8% for SAM and 91.4% for specialized methods. On polyp segmentation (Kvasir-SEG), SAM-Adapter reached 87.1% dice compared to 61.3% for SAM and 89.2% for PraNet. These results demonstrate that adapters recover 85-95% of the gap between vanilla SAM and specialized models while training only 2-5% of parameters.

### Comparison with Full Fine-Tuning

Compared to full fine-tuning of SAM on the same target domains, SAM-Adapter achieved approximately 1-3 IoU points lower performance while using 20-50x fewer trainable parameters. Full fine-tuning of ViT-B on COD10K achieved 80.1% IoU versus 78.3% for SAM-Adapter. The small performance gap suggests that the adapter's bottleneck architecture captures the essential domain-specific features without requiring modification of the entire encoder. Moreover, SAM-Adapter preserves SAM's general capabilities: after adapter training, the model can still perform zero-shot segmentation on natural images by simply bypassing the adapters.

## Strengths

SAM-Adapter provides an efficient adaptation mechanism that achieves near-specialized-model performance with minimal computational cost. The frozen encoder preserves SAM's general visual features, enabling multi-task deployment (different adapter sets for different domains can share the same frozen encoder). Training requires only a single GPU and a few hours, making it accessible to researchers without large compute budgets. The adapter architecture is modular and composable: adapters trained for different tasks can be swapped without reloading the base model.

## Limitations

SAM-Adapter's performance consistently falls 1-3 points below full fine-tuning, which may matter in precision-critical applications. The approach requires task-specific training data, so it does not solve zero-shot domain adaptation. The adapter bottleneck limits the model's capacity to learn complex domain-specific features; for domains very different from natural images (e.g., electron microscopy, radar imagery), the bottleneck may be too restrictive. The task-specific heads must be designed and trained separately for each task, adding engineering complexity.

## Connections

SAM-Adapter is part of the broader ecosystem of SAM adaptation methods. MedSAM (Ma et al. 2024) takes the full fine-tuning approach for medical images, achieving higher peak performance at greater computational cost. The adapter architecture draws directly from the NLP literature, particularly Houlsby et al. 2019 ("Parameter-Efficient Transfer Learning for NLP"). LoRA (Hu et al. 2022) offers an alternative parameter-efficient adaptation strategy using low-rank weight updates. The comparison between SAM-Adapter and MedSAM informs the broader adaptation_strategies comparison in the _comparative directory.

## References

- Kirillov et al., "Segment Anything," ICCV 2023 (SAM foundation).
- Houlsby et al., "Parameter-Efficient Transfer Learning for NLP," ICML 2019 (adapter tuning).
- Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models," ICLR 2022 (alternative PEFT).
- Fan et al., "Camouflaged Object Detection," CVPR 2020 (COD10K dataset).
- Fan et al., "Pranet: Parallel Reverse Attention Network for Polyp Segmentation," MICCAI 2020 (polyp comparison).
- Ma et al., "Segment Anything in Medical Images," Nature Communications 2024 (MedSAM comparison).
