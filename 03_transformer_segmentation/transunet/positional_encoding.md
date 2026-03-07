---
title: "Positional Encoding in TransUNet"
date: 2025-03-06
status: planned
tags: [positional-encoding, transformer, spatial-information]
difficulty: intermediate
---

# Positional Encoding in TransUNet

## Overview

Positional encoding is a critical component in the TransUNet architecture that provides spatial location information to the Transformer encoder. Since the self-attention mechanism in Transformers is inherently permutation-invariant -- it produces the same output regardless of the ordering of input tokens -- positional encodings are necessary to inject information about the spatial arrangement of tokens. In the context of medical image segmentation, preserving accurate spatial relationships is essential for producing precise segmentation maps, making the design of positional encodings particularly important.

In TransUNet, the positional encoding is added to the patch token embeddings before they are fed into the Transformer encoder layers. Each token corresponds to a spatial location in the 2D feature map produced by the CNN backbone, and the positional encoding ensures that the Transformer can distinguish between tokens from different spatial positions and learn spatial relationships during self-attention computation.

## Types of Positional Encoding

There are two main approaches to positional encoding in Vision Transformers:

**Fixed Sinusoidal Encodings** were originally proposed in the "Attention Is All You Need" paper for natural language processing. These use sine and cosine functions of different frequencies to encode position:

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d}}\right), \quad PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d}}\right)$$

where $pos$ is the position index, $i$ is the dimension index, and $d$ is the embedding dimension. These encodings have the theoretical advantage that the model can generalize to sequence lengths not seen during training, since the encoding function is defined for arbitrary positions.

**Learned Positional Encodings** are trainable parameter vectors that are optimized during training. Each position in the sequence is assigned a learnable embedding vector of the same dimension as the token embeddings. TransUNet, following the ViT convention, uses learned positional encodings. These are initialized randomly and updated through backpropagation. The learned approach allows the model to discover the most useful positional representations for the specific task and data distribution, though it fixes the maximum sequence length at initialization time.

## Spatial Position Representation

In TransUNet, the CNN backbone produces a 2D feature map of size $\frac{H}{16} \times \frac{W}{16}$ which is then flattened into a 1D sequence of $N = \frac{H}{16} \times \frac{W}{16}$ tokens. This flattening operation converts a 2D spatial grid into a 1D sequence, and the positional encoding must encode the original 2D spatial structure.

TransUNet uses a 1D learned positional encoding applied to the flattened sequence. While this may seem like a loss of 2D structural information, the model can implicitly recover 2D spatial relationships during training because the flattening order (row-major) is consistent. The positional embedding matrix $E_{pos} \in \mathbb{R}^{N \times D}$ is added element-wise to the token embedding matrix:

$$z_0 = [x_1 E; x_2 E; \ldots; x_N E] + E_{pos}$$

where $E \in \mathbb{R}^{d \times D}$ is the linear projection matrix that maps CNN feature vectors of dimension $d$ to the Transformer hidden dimension $D$.

An alternative approach used in some other architectures is to use separate encodings for the row and column dimensions (2D positional encoding). In this case, two encoding vectors $PE_{row} \in \mathbb{R}^{H' \times D/2}$ and $PE_{col} \in \mathbb{R}^{W' \times D/2}$ are concatenated to form the full positional encoding. While this provides a more explicit 2D structure, empirical studies have shown that 1D learned encodings perform comparably when sufficient training data is available, as the model learns to infer 2D relationships from the 1D ordering.

## Impact on Segmentation Performance

Positional encoding plays a measurably important role in TransUNet's segmentation accuracy. Without positional encoding, the Transformer encoder treats all tokens as an unordered set, losing the ability to reason about spatial relationships such as adjacency, relative position, and spatial extent of anatomical structures. Ablation experiments in the ViT literature show that removing positional encodings leads to substantial performance degradation, typically 2--5% in accuracy metrics.

For medical image segmentation specifically, positional information is crucial because anatomical structures have consistent spatial relationships. For example, in abdominal CT segmentation, the liver is typically located in the upper right quadrant, the spleen in the upper left, and the kidneys laterally. Positional encoding enables the Transformer to leverage these spatial priors, improving segmentation accuracy particularly for organs with consistent anatomical positions.

Furthermore, the positional encoding allows the self-attention mechanism to develop spatially structured attention patterns. Studies visualizing attention maps in TransUNet show that heads learn to attend to spatially coherent regions and that attention patterns exhibit structured spatial relationships that would be impossible without positional information.

## Comparison with Other Approaches

**Relative Position Bias** (used in Swin Transformer and Swin-Unet) encodes the relative positional offset between token pairs rather than absolute positions. The bias $B \in \mathbb{R}^{M^2 \times M^2}$ is added directly to the attention logits: $\text{Attention}(Q,K,V) = \text{Softmax}\left(\frac{QK^T}{\sqrt{d_k}} + B\right)V$. This approach has advantages for segmentation because spatial relationships between features (e.g., "this token is 3 positions to the right of that token") are often more informative than absolute positions. Relative position bias also naturally handles inputs of varying sizes without interpolation.

**Conditional Position Encoding (CPE)**, used in models like CPVT, generates positional encodings conditioned on the input features using depthwise convolutions. This provides a data-dependent positional signal that can adapt to the specific input, and naturally handles variable input resolutions without any modification.

**Rotary Position Embedding (RoPE)**, originally from NLP, encodes position through rotation of the query and key vectors, allowing relative position information to be naturally captured in the dot-product attention computation.

TransUNet's use of simple 1D learned positional encodings is effective but represents a relatively basic approach. Subsequent architectures in the segmentation domain have moved toward relative position biases (Swin-Unet, Swin UNETR) or conditional positional encodings, which tend to provide better generalization to different input sizes and stronger inductive biases about spatial relationships.

## Implementation Notes

In practice, TransUNet's positional encodings are implemented as an `nn.Parameter` tensor of shape $(N, D)$ where $N$ is the number of spatial tokens and $D$ is the hidden dimension. When using a standard input resolution of $224 \times 224$ with a patch size (after CNN backbone) effectively equivalent to $16 \times 16$, this yields $N = 14 \times 14 = 196$ position embeddings.

When applying TransUNet to input resolutions different from the pre-training resolution, the positional embeddings must be interpolated. The standard approach is to reshape the 1D positional embeddings back to a 2D grid, apply bicubic interpolation to the new resolution, and flatten again. This interpolation introduces a slight approximation but works well in practice for moderate resolution changes. It is worth noting that learned positional embeddings from ImageNet pre-trained ViT weights can be successfully transferred to medical imaging tasks despite the domain gap, as the basic spatial structure encoded by these embeddings is largely domain-agnostic.
