---
title: "ViT Encoder for 3D Data in UNETR"
date: 2025-03-06
status: planned
tags: [vit, 3d-segmentation, volumetric, patch-embedding]
difficulty: intermediate
---

# ViT Encoder for 3D Data

## Overview

The ViT encoder for 3D data in UNETR extends the standard 2D Vision Transformer to handle volumetric inputs. The fundamental principle remains the same: an input is divided into non-overlapping patches, each patch is linearly projected to a fixed-dimensional embedding, positional information is added, and the resulting token sequence is processed through a stack of Transformer layers with multi-head self-attention and feed-forward networks. The key adaptation for 3D is that patches become cuboidal volumes ($P \times P \times P$ voxels), the positional encoding must represent 3D spatial locations, and the resulting token sequence encodes a 3D spatial grid rather than a 2D one.

This extension is conceptually straightforward but introduces important practical considerations around sequence length, computational cost, and the design of multi-scale feature extraction from a single-resolution encoder.

## 3D Patch Embedding

The 3D patch embedding layer partitions a volumetric input $\mathbf{x} \in \mathbb{R}^{H \times W \times D \times C_{in}}$ into a regular grid of non-overlapping 3D patches. Each patch has spatial dimensions $P \times P \times P$ (typically $P = 16$), resulting in $\frac{H}{P} \times \frac{W}{P} \times \frac{D}{P}$ patches. Each 3D patch is flattened into a 1D vector of dimension $P^3 \cdot C_{in}$ and mapped to the Transformer hidden dimension $d$ through a linear projection:

$$\mathbf{e}_i = \text{Flatten}(\mathbf{x}_i^{patch}) \cdot \mathbf{E}, \quad \mathbf{E} \in \mathbb{R}^{(P^3 \cdot C_{in}) \times d}$$

In practice, this linear projection is implemented as a 3D convolution with kernel size $P \times P \times P$ and stride $P$ in all three dimensions, which simultaneously extracts and projects the patches in a single operation:

$$\mathbf{E}_{3D} = \text{Conv3D}(\mathbf{x}, \text{kernel}=P, \text{stride}=P, \text{out\_channels}=d)$$

For the typical UNETR configuration with an input volume of $96 \times 96 \times 96$ and patch size $P = 16$, this produces $N = 6 \times 6 \times 6 = 216$ tokens. This relatively small sequence length is critical for making global self-attention computationally feasible in the 3D setting.

The patch size choice represents a trade-off: larger patches produce fewer tokens (lower computational cost) but coarser spatial representations, while smaller patches provide finer spatial detail but create longer sequences that are more expensive to process. With $P = 16$, each token represents a substantial volumetric region ($16^3 = 4096$ voxels), and fine spatial details within each patch are captured only through the linear projection.

## Positional Encoding for 3D

The ViT encoder in UNETR uses learned 1D positional encodings, following the standard ViT approach. Each of the $N$ token positions is assigned a learnable embedding vector $\mathbf{p}_i \in \mathbb{R}^d$, and these are added to the patch embeddings:

$$\mathbf{z}_0 = [\mathbf{e}_1 + \mathbf{p}_1; \mathbf{e}_2 + \mathbf{p}_2; \ldots; \mathbf{e}_N + \mathbf{p}_N]$$

Although the tokens are arranged in a 3D spatial grid, the positional encoding is applied to the flattened 1D sequence. The model must learn to recover the 3D spatial structure from the 1D ordering (which follows a raster scan pattern through the volume). Empirically, this works well because the flattening order is consistent and deterministic.

An alternative would be to use separate positional encodings for each spatial axis (height, width, depth) and combine them through addition or concatenation. For instance, with factorized encodings:

$$\mathbf{p}_{(h,w,d)} = \mathbf{p}_h^{(H)} + \mathbf{p}_w^{(W)} + \mathbf{p}_d^{(D)}$$

This factorized approach reduces the number of positional parameters from $N \cdot d$ to $(H' + W' + D') \cdot d$ and may generalize better to volumes of different sizes, but UNETR uses the simpler 1D learned encoding without reporting substantial performance differences.

## Multi-Scale Feature Extraction

A distinctive feature of UNETR's use of the ViT encoder is the extraction of intermediate representations from multiple Transformer layers to create multi-scale skip connections. While the ViT encoder itself operates at a single spatial resolution (all layers produce the same number of tokens with the same dimension), the representations at different depths encode features at different semantic levels:

- **Early layers (e.g., layer 3)**: Encode low-level features such as edges, textures, and local intensity patterns. These representations, when reshaped and upsampled, provide fine-grained spatial detail to the decoder.
- **Middle layers (e.g., layers 6 and 9)**: Encode mid-level features that capture local structure and regional context, bridging the gap between low-level and high-level representations.
- **Final layer (layer 12)**: Encodes high-level semantic features that capture global context and long-range dependencies across the entire volume.

To use these intermediate features as skip connections, each extracted representation $\mathbf{z}_l \in \mathbb{R}^{N \times d}$ is reshaped back to its 3D spatial arrangement $\frac{H}{P} \times \frac{W}{P} \times \frac{D}{P} \times d$ and then processed through deconvolution blocks to match the required spatial resolution for the corresponding decoder stage. For example, features from layer 3 are upsampled by $8\times$ to resolution $\frac{H}{2} \times \frac{W}{2} \times \frac{D}{2}$, while features from layer 12 remain at $\frac{H}{P}$ resolution.

This approach of using depth as a proxy for scale is different from hierarchical architectures (like Swin Transformer) that explicitly produce multi-resolution feature maps. It works because deeper Transformer layers progressively transform the representation from local to global features through the accumulation of self-attention operations.

## Computational Considerations

The global self-attention mechanism in UNETR's ViT encoder has computational cost $O(N^2 \cdot d)$ per layer, where $N$ is the number of tokens. The key computational properties:

**Self-attention cost per layer**: For $N = 216$ tokens and $d = 768$, the attention matrix computation requires $216^2 \times 768 \approx 36M$ operations per head. With 12 heads and 12 layers, the total self-attention cost is manageable but grows quadratically with volume size.

**Scaling limitations**: For larger volumes or finer patches, the token count increases cubically. A $192 \times 192 \times 192$ volume with $P = 16$ would produce $12^3 = 1728$ tokens, increasing the self-attention cost by $(1728/216)^2 = 64\times$. This cubic scaling of token count with volume size is the primary computational bottleneck.

**Memory requirements**: Training UNETR requires storing activations for all 12 Transformer layers, plus the decoder activations. For 3D inputs, this demands substantial GPU memory. Common strategies to manage memory include: using gradient checkpointing to trade computation for memory, processing sub-volumes (patches) during training and stitching during inference, and using mixed-precision training (FP16).

**Inference on full volumes**: At inference time, sliding window approaches are typically used, where the model processes overlapping sub-volumes and predictions are averaged in overlapping regions. This enables application to arbitrarily large volumes regardless of training volume size.

## Comparison with 3D CNN Encoders

| Property | ViT Encoder (UNETR) | 3D CNN Encoder (e.g., 3D U-Net) |
|----------|---------------------|----------------------------------|
| Receptive field | Global (full volume) from layer 1 | Local, grows with depth |
| Inductive biases | Minimal (only patch structure) | Strong (locality, translation equivariance) |
| Multi-scale features | Implicit (from different layers) | Explicit (from different stages) |
| Computational scaling | $O(N^2)$ in tokens | $O(N)$ in voxels |
| Data efficiency | Requires large pre-training data | Effective with smaller datasets |
| Parameter efficiency | Larger models (hidden dim applied uniformly) | Efficient channel progression |

3D CNN encoders have the advantage of strong inductive biases that enable data-efficient learning, linear computational scaling with input size, and explicit multi-resolution features through pooling operations. However, their local receptive fields limit the ability to model global spatial relationships, which is critical in volumetric medical imaging where structures may span large portions of the volume.

The ViT encoder in UNETR provides complementary strengths: global context from the first layer through self-attention, flexible learned representations without strong architectural constraints, and the ability to relate distant spatial locations directly. The trade-off is higher computational cost for larger inputs and greater dependence on pre-training to compensate for weaker inductive biases.

In practice, UNETR's ViT encoder achieves competitive or superior performance to 3D CNN encoders on standard benchmarks, particularly for structures that benefit from global context. However, the computational constraints limit the input resolution, and sliding-window inference is required for full-resolution volumetric predictions.

## Implementation Notes

Key implementation details for UNETR's 3D ViT encoder:

1. **Input resolution**: The standard training input size is $96 \times 96 \times 96$ voxels, extracted as random crops from the full volume during training. This is smaller than typical 3D CNN inputs due to memory constraints.

2. **Pre-training**: The encoder weights are initialized from a 2D ViT pre-trained on ImageNet-21K. The 2D patch embedding kernel is inflated to 3D by replicating the 2D kernel along the depth dimension and dividing by $P$ to maintain the expected activation scale. Positional embeddings are interpolated from 2D to 3D.

3. **Transformer configuration**: UNETR uses ViT-Base with $d = 768$, 12 heads, 12 layers, and MLP dimension $3072$. The total encoder parameter count is approximately 87M.

4. **Skip connection processing**: Each skip connection feature is processed through a sequence of $3 \times 3 \times 3$ convolutions, instance normalization, and ReLU activation after reshaping from the token sequence to 3D spatial format. Deconvolutional (transposed convolution) layers handle the upsampling to the target resolution.

5. **Training strategy**: UNETR is typically trained with the AdamW optimizer, cosine annealing learning rate schedule, and a warmup period. Data augmentation includes random cropping, flipping, rotation, and intensity scaling.
