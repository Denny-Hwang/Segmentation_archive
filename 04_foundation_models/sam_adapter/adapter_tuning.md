---
title: "Adapter Tuning for SAM"
date: 2025-03-06
status: planned
tags: [adapter, parameter-efficient, fine-tuning, transfer-learning]
difficulty: intermediate
---

# Adapter Tuning

## Overview

Adapter tuning is a parameter-efficient fine-tuning (PEFT) strategy that inserts small, trainable bottleneck modules into a frozen pretrained model. In the context of SAM, adapter tuning addresses a practical challenge: SAM's image encoder contains hundreds of millions of parameters that encode general visual features, but these features are suboptimal for specialized domains. Rather than fine-tuning the entire encoder (expensive and risks catastrophic forgetting) or training only the decoder (insufficient for large domain gaps), adapter tuning provides a middle path that modifies the encoder's behavior through lightweight auxiliary modules while preserving the original weights.

The adapter approach originates from NLP, where Houlsby et al. (2019) demonstrated that inserting bottleneck layers into transformer blocks could match full fine-tuning performance on GLUE benchmarks while training less than 4% of parameters. SAM-Adapter applies this principle to vision transformers, showing that the same strategy is effective for bridging the domain gap between natural images and specialized visual domains such as shadow detection, camouflaged object segmentation, and medical imaging.

## Adapter Architecture

Each adapter module follows a bottleneck design with three components: a down-projection layer, a nonlinear activation, and an up-projection layer. The down-projection is a linear layer that maps from the transformer's hidden dimension d (e.g., 768 for ViT-B, 1024 for ViT-L, 1280 for ViT-H) to a bottleneck dimension r (typically 64 or 128). The nonlinear activation is GELU (Gaussian Error Linear Unit), chosen for consistency with the ViT's internal activation functions. The up-projection maps back from r to d. The adapter output is added residually to the transformer block's output, forming a skip connection that ensures the adapter's contribution is additive.

Mathematically, for input features h, the adapter computes: output = h + s * W_up(GELU(W_down(LN(h)))), where LN is optional layer normalization, W_down is the d-to-r projection, W_up is the r-to-d projection, and s is a learnable scaling factor (initialized to 0.1). The residual connection is critical: it ensures that at initialization (when adapter weights are near zero), the model behaves identically to the original SAM, and training gradually introduces domain-specific modifications.

## Parameter Efficiency

The parameter count per adapter is 2 * d * r + r (accounting for bias terms), which is approximately 100K for d=768, r=64. With one adapter per transformer block and 12 blocks in ViT-B, the total adapter parameters are approximately 1.2M. Including the task-specific head (typically 2-5M parameters), the total trainable parameters range from 3-7M, representing 3-7% of ViT-B's 91M parameters or less than 1% of ViT-H's 632M parameters.

This efficiency has practical implications beyond faster training. Storage is dramatically reduced: a task-specific adapter checkpoint is approximately 5-15MB compared to 350MB+ for a fully fine-tuned model. This enables a deployment pattern where a single frozen SAM backbone is loaded once, and different adapter modules are swapped in for different tasks with minimal overhead. GPU memory during training is also reduced by approximately 40-60% compared to full fine-tuning, because gradients are only computed for adapter and head parameters, not the frozen encoder.

## Adapter Placement Strategy

The placement of adapters within the transformer architecture affects adaptation performance. Three primary strategies have been explored: (1) after the multi-head self-attention (MHSA) block, (2) after the MLP block, and (3) after both. SAM-Adapter places adapters after the MLP block, based on the reasoning that the MLP is responsible for per-token feature transformation while attention handles token interaction. Domain adaptation primarily requires learning new feature transformations (e.g., what constitutes an edge in medical images versus natural images), making the post-MLP position more effective.

Ablation studies confirm this intuition: post-MLP adapters outperform post-attention adapters by approximately 1-2 IoU points. Placing adapters after both (dual adapters per block) provides marginal additional improvement (~0.5 IoU) at double the parameter cost, making it generally not worthwhile. Layer-wise analysis reveals that adapters in deeper layers (closer to the output) have a larger individual impact on performance than adapters in earlier layers, consistent with the observation that early ViT features are more universally transferable while later features are more task-specific.

## Training Procedure

Training adapters follows standard supervised learning with domain-specific data. The frozen SAM encoder performs a forward pass to produce intermediate features at each transformer block. At each block, the adapter module processes the features and adds its output residually. The adapted features are then passed to the task-specific head, which produces the final segmentation prediction. Loss is computed against ground-truth masks, and gradients flow only through the adapter modules and the head -- the encoder parameters receive no gradient updates.

Typical training hyperparameters include: AdamW optimizer with weight decay 0.01, learning rate 1e-3 for adapters (higher than full fine-tuning because adapters start near zero), learning rate 1e-4 for the task-specific head, cosine learning rate schedule with warm-up, and training for 50-100 epochs. Batch size depends on the input resolution but is typically 4-16. The training converges faster than full fine-tuning (approximately 2x fewer epochs to reach peak performance) because the search space is dramatically smaller.

Data augmentation follows standard practices for the target domain: random flipping, rotation, color jittering, and random resizing. For medical imaging, modality-specific augmentations (windowing perturbation for CT, intensity noise for MRI) are additionally applied. No special regularization beyond weight decay is typically needed, as the small parameter count of adapters inherently limits overfitting.

## Comparison with Other PEFT Methods

Several PEFT methods exist beyond adapters, each with distinct trade-offs:

**LoRA (Low-Rank Adaptation)**: Instead of inserting new modules, LoRA adds low-rank decomposition matrices (A and B, where A is d-by-r and B is r-by-d) to existing weight matrices, typically the query and value projections in attention layers. LoRA trains a similar number of parameters as adapters but modifies the attention mechanism directly rather than adding a post-hoc transformation. Performance is generally comparable to adapters (within 1 IoU point), but LoRA's integration into attention makes it more natural for tasks requiring modified attention patterns.

**Prompt Tuning / Visual Prompt Tuning (VPT)**: Prepends learnable tokens to the input sequence of each transformer layer. These "prompt" tokens interact with image tokens through attention, indirectly modifying the model's behavior. VPT trains fewer parameters than adapters (typically 0.1-0.5% of the model) but achieves lower performance on challenging domain adaptation tasks (2-5 IoU points below adapters), suggesting that input-level modification is less powerful than feature-level modification for significant domain shifts.

**Prefix Tuning**: Appends learnable key-value pairs to the attention mechanism, modifying attention outputs without changing the model weights. Similar in spirit to VPT but operates at the attention level rather than the input level. Performance falls between VPT and adapters.

**BitFit**: Fine-tunes only the bias terms of the pretrained model. Extremely parameter-efficient (< 0.1% of parameters) but substantially underperforms adapters on challenging domains (5-10 IoU points lower), indicating that bias-only modification is insufficient for bridging large domain gaps.

## Domain-Specific Results

Adapter tuning has been evaluated across multiple challenging domains where vanilla SAM underperforms:

- **Camouflaged object detection** (COD10K): Adapters improve SAM from 46% to 81% weighted F-measure, approaching the specialized SINet-V2 (81% F-measure) with only 3% trainable parameters.
- **Shadow detection** (SBU): Adapters improve from 58% to 92% balanced error rate, compared to 93% for the specialized MTMT-Net.
- **Polyp segmentation** (Kvasir-SEG): Adapters improve from 64% to 89% Dice, compared to 90% for the specialized PraNet.
- **Medical imaging** (various): Adapters applied to medical segmentation tasks achieve approximately 80-85% Dice, which is 3-5 points below MedSAM's full fine-tuning but substantially above vanilla SAM.

Across all domains, adapters recover 85-98% of the full fine-tuning improvement while training only 3-5% of parameters. The recovery rate is highest for domains closer to natural images (shadow detection, salient objects) and lowest for domains with the largest visual gap (medical imaging, remote sensing).

## When to Use Adapter Tuning

Adapter tuning is recommended in the following scenarios:

1. **Limited compute budget**: When full fine-tuning is too expensive (e.g., fine-tuning ViT-H requires 8+ A100 GPUs), adapters can be trained on a single consumer GPU.
2. **Multiple target domains**: When adapting SAM to several different domains, adapters allow sharing a single frozen backbone with per-domain adapter modules, dramatically reducing storage and deployment cost.
3. **Preserving original capabilities**: When the adapted model must also function on natural images (e.g., a general-purpose annotation tool with specialized modes), adapters preserve the original model's performance by keeping the encoder frozen.
4. **Limited training data**: Adapters have fewer parameters and are thus less prone to overfitting on small datasets (typically 100-1000 training images).
5. **Rapid experimentation**: Adapter training converges in 4-8 hours, enabling quick iteration on architectural choices and hyperparameters.

Adapter tuning is less suitable when: (1) the domain gap is extremely large (e.g., adapting to a completely novel imaging modality like terahertz imaging), where full encoder modification is necessary; (2) maximum absolute performance is required and compute is available; or (3) the promptable interface must be preserved (SAM-Adapter's task-specific heads remove the prompt-based interaction).

## Implementation Notes

Implementing adapter tuning for SAM involves modifying the ViT encoder's forward pass to insert adapter modules after each transformer block's MLP. This is typically done by subclassing the ViT block and adding the adapter as an additional module. The adapter weights are initialized with small random values (Kaiming initialization scaled by 0.1) to ensure near-zero initial contribution. The scaling factor is initialized to 0.1 and made learnable.

For practical deployment, adapter modules can be loaded and unloaded dynamically: the frozen SAM encoder is loaded once, and different adapter weight files are loaded depending on the task. This is achieved by maintaining separate `state_dict` entries for adapter parameters and loading them selectively. The overhead of adapter inference is minimal: adding adapters increases per-image inference time by less than 5% (approximately 2ms on A100), as the bottleneck computation is lightweight compared to the attention and MLP computations in the ViT blocks.
