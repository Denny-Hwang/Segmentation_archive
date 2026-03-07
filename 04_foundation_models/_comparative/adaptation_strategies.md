---
title: "Adaptation Strategies for Foundation Segmentation Models"
date: 2025-03-06
status: planned
tags: [adaptation, fine-tuning, adapter, lora, parameter-efficient]
difficulty: intermediate
---

# Adaptation Strategies

## Overview

Foundation segmentation models like SAM are trained on massive, diverse datasets to learn general visual features, but their performance on specialized domains (medical imaging, remote sensing, industrial inspection, camouflaged objects) often falls short of domain-specific models. Adaptation strategies bridge this gap by modifying the model's behavior to suit a target domain while leveraging the pretrained representations. The strategies range from full fine-tuning (updating all parameters) to extremely lightweight approaches (tuning only bias terms), with each offering different trade-offs between adaptation capacity, computational cost, data requirements, and preservation of the original model's capabilities.

The choice of adaptation strategy is driven by practical constraints: available compute, training data volume, whether the original model's capabilities must be preserved, and the severity of the domain gap. This comparative analysis surveys the major strategies explored in the literature, with specific reference to how they have been applied to SAM and related foundation models.

## Full Fine-Tuning

Full fine-tuning updates all model parameters on domain-specific data, allowing the entire feature hierarchy to adapt to the target domain. This is the approach taken by MedSAM, which fine-tunes all of SAM ViT-B's 91M parameters on 1.57M medical image-mask pairs. Full fine-tuning provides the highest adaptation capacity and typically achieves the best absolute performance, particularly when the domain gap is large (e.g., natural images to CT/MRI).

The primary advantages are: maximum flexibility to restructure feature representations, no architectural constraints on what can be learned, and straightforward implementation. The disadvantages are significant: high computational cost (typically requiring multiple GPUs for several days), risk of catastrophic forgetting (the model loses its ability to handle the original domain), large storage requirements (a separate copy of the full model for each domain), and potential overfitting when target domain data is limited. Full fine-tuning is best suited for building a dedicated model for a specific domain when sufficient data and compute are available.

## Adapter-Based Tuning

Adapter tuning, as implemented in SAM-Adapter, inserts lightweight bottleneck modules into the frozen model's transformer blocks. Each adapter consists of a down-projection (d to r dimensions), nonlinear activation, and up-projection (r to d), with the output added residually to the original features. Only the adapter parameters and any task-specific heads are trained, while all original model parameters remain frozen.

Adapter tuning offers an excellent balance of efficiency and performance. With bottleneck dimension r=64 in a ViT-B (d=768), each adapter adds approximately 100K parameters, totaling approximately 1.2M across 12 layers (1.3% of ViT-B). SAM-Adapter achieves 95-98% of full fine-tuning performance on tasks like camouflaged object detection and shadow detection. The frozen backbone enables multi-task deployment with shared weights and prevents catastrophic forgetting. The main limitation is that adapters cannot fundamentally restructure the feature representations when the domain gap is very large.

## LoRA and QLoRA

Low-Rank Adaptation (LoRA) modifies existing weight matrices rather than inserting new modules. For a weight matrix W of dimensions d x d, LoRA adds a low-rank perturbation: W' = W + BA, where B is d x r and A is r x d (r << d). This effectively constrains the weight update to a low-rank subspace, dramatically reducing trainable parameters. LoRA is typically applied to the query and value projection matrices in self-attention, though it can be applied to any linear layer.

For SAM ViT-B with rank r=8, LoRA adds approximately 1.2M trainable parameters (comparable to adapters). Performance is generally within 1-2 points of full fine-tuning, similar to adapters. LoRA's advantage over adapters is that it modifies the attention mechanism directly, which can be beneficial for tasks requiring different attention patterns (e.g., attending to different spatial relationships in medical images). QLoRA extends this by quantizing the frozen weights to 4-bit precision, further reducing memory requirements during training.

LoRA has been applied to SAM for medical imaging (SAMed), remote sensing (RSPrompter), and other specialized domains. Results are generally comparable to adapter-based approaches, with LoRA slightly outperforming on tasks requiring attention modification and adapters slightly outperforming on tasks requiring feature transformation.

## Prompt Tuning

Visual Prompt Tuning (VPT) prepends a set of learnable tokens to the input sequence of each transformer layer. These "prompt" tokens interact with the image tokens through the existing self-attention mechanism, indirectly modifying the model's behavior without changing any weights. VPT trains only the prompt tokens (typically 10-50 tokens per layer), resulting in extremely few trainable parameters (0.1-0.5% of the model).

For SAM adaptation, prompt tuning has been explored but with limited success. The fundamental limitation is that prompt tokens can only influence the model through attention, which provides weaker adaptation than directly modifying features (adapters) or weights (LoRA). On domains with large domain gaps (medical, camouflaged objects), VPT typically underperforms adapters and LoRA by 3-8 IoU points. VPT is most effective for mild domain shifts where the model's existing features are mostly adequate and only need slight reweighting.

## Head-Only Fine-Tuning

The simplest adaptation strategy is to freeze the entire encoder and train only the decoder/head. For SAM, this means training only the mask decoder's 4M parameters while keeping the 91-632M encoder parameters frozen. This approach is extremely computationally efficient and cannot cause catastrophic forgetting in the encoder, but its adaptation capacity is limited because the encoder features remain unchanged.

Head-only fine-tuning recovers approximately 40-60% of the full fine-tuning improvement, depending on the domain gap. For small domain shifts (e.g., from COCO to a similar natural image dataset), head-only fine-tuning may be sufficient. For large domain shifts (natural images to medical imaging), it is clearly inadequate because the encoder's features are not discriminative for the target domain's structures. Head-only fine-tuning is primarily useful as a lower bound for benchmarking more sophisticated adaptation strategies.

## Domain-Specific Pretraining

An alternative to fine-tuning the supervised model is to perform additional self-supervised pretraining on unlabeled domain-specific data before supervised fine-tuning. For SAM, this would involve running additional MAE pretraining on medical images (or other domain images) before the supervised segmentation training. This approach can improve the encoder's feature representations for the target domain without requiring labeled data.

MedSAM's approach is a variant of this strategy: it performs supervised fine-tuning on a large medical dataset, which simultaneously adapts the encoder's features and trains the decoder. The distinction is that domain-specific pretraining uses self-supervised objectives (MAE, contrastive learning) on unlabeled data, while MedSAM uses supervised segmentation objectives on labeled data. Domain-specific pretraining is particularly valuable when large amounts of unlabeled domain data are available but labeled data is scarce.

## Comparison of Strategies

| Strategy | Trainable Params | Performance | Data Required | Compute Cost |
|----------|-----------------|-------------|---------------|--------------|
| Full Fine-Tuning | 100% (91-632M) | Best (100% baseline) | Large (10K-1M samples) | Very high (multi-GPU, days) |
| Adapter | 2-5% (2-5M) | 95-98% of full FT | Moderate (1K-10K) | Low (single GPU, hours) |
| LoRA | 1-3% (1-3M) | 93-97% of full FT | Moderate (1K-10K) | Low (single GPU, hours) |
| Prompt Tuning | 0.1-0.5% (0.1-0.5M) | 80-90% of full FT | Small (100-1K) | Very low (single GPU, 1-2h) |
| Head-Only | 4-5% (4M decoder) | 50-70% of full FT | Small-Moderate (100-10K) | Low (single GPU, hours) |

## Best Practices

Selecting an adaptation strategy should follow a decision tree based on practical constraints:

1. **If maximum performance is critical and compute/data are abundant**: Use full fine-tuning. This is the right choice for building a production medical segmentation system or a dedicated domain model.

2. **If the model must serve multiple domains or preserve original capabilities**: Use adapter-based tuning or LoRA. These methods allow sharing a frozen backbone across tasks with minimal per-task storage overhead.

3. **If training data is very limited (< 500 samples)**: Prefer adapter or LoRA over full fine-tuning, as the smaller parameter count reduces overfitting risk. Consider also using data augmentation and self-training.

4. **If the domain gap is small (target domain is visually similar to natural images)**: Head-only fine-tuning or prompt tuning may suffice, with adapters providing diminishing returns over simpler methods.

5. **If the domain gap is very large (target domain is visually dissimilar)**: Full fine-tuning is strongly preferred, as parameter-efficient methods may lack the capacity to restructure features sufficiently.

6. **If rapid experimentation is needed**: Start with adapters (fast training, good performance) to establish a baseline, then consider full fine-tuning only if the adapter performance is inadequate.

## Open Research Questions

Several important questions remain unresolved in foundation model adaptation for segmentation:

1. **Optimal adaptation granularity**: Should different encoder layers be adapted with different strategies (e.g., freeze early layers, use LoRA for middle layers, use adapters for later layers)? Layer-wise adaptation strategies are largely unexplored.

2. **Continual adaptation**: How can a model be progressively adapted to new domains without forgetting previous adaptations? Current methods require separate adapter sets per domain, but a unified continual learning approach would be more practical.

3. **Task-aware adaptation**: Should the adaptation strategy differ for different output tasks (e.g., boundary-focused adaptation for instance segmentation versus region-focused adaptation for semantic segmentation)?

4. **Combining adaptation methods**: Can adapters, LoRA, and prompt tuning be combined for superior performance? Initial explorations suggest marginal benefits, but the search space is large.

5. **Theoretical understanding**: Why do parameter-efficient methods work so well (95%+ of full fine-tuning performance with 2% of parameters)? Is there a principled way to determine the optimal number of adapter/LoRA parameters for a given domain gap?

6. **Self-supervised domain adaptation**: Can unlabeled target domain data be leveraged more effectively through domain-adaptive pretraining objectives? The interaction between self-supervised pretraining and supervised fine-tuning is not well understood.
