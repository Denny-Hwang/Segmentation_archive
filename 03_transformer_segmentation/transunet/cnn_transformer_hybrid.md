---
title: "CNN-Transformer Hybrid Encoding in TransUNet"
date: 2025-03-06
status: planned
tags: [transformer, cnn, hybrid-architecture, feature-extraction]
difficulty: intermediate
---

# CNN-Transformer Hybrid Encoding

## Overview

The CNN-Transformer hybrid encoder is the defining architectural innovation of TransUNet. Rather than using either a pure CNN or a pure Transformer for feature extraction, TransUNet sequentially combines both: a CNN backbone first extracts hierarchical local features from the input image, and then a Vision Transformer (ViT) encoder processes the CNN feature maps to model global dependencies through self-attention. This two-stage encoding strategy allows the model to benefit from the CNN's strong inductive biases for local feature extraction and translation equivariance, while also leveraging the Transformer's capacity for capturing long-range spatial relationships that are difficult for CNNs to model efficiently.

## CNN Feature Extractor

TransUNet uses a ResNet-50 backbone (pre-trained on ImageNet) as its CNN feature extractor. The ResNet processes the input image through an initial $7 \times 7$ convolution with stride 2, followed by batch normalization, ReLU activation, and max pooling. The image then passes through four residual stages, each composed of multiple bottleneck blocks. The stages progressively reduce spatial resolution while increasing channel depth:

- **Stage 1**: Output resolution $\frac{H}{4} \times \frac{W}{4}$, 256 channels
- **Stage 2**: Output resolution $\frac{H}{8} \times \frac{W}{8}$, 512 channels
- **Stage 3**: Output resolution $\frac{H}{16} \times \frac{W}{16}$, 1024 channels

Features from Stages 1, 2, and 3 are retained for use as skip connections in the decoder. The output of Stage 3 serves as the input to the Transformer encoder. In some configurations, only the first three stages of ResNet-50 are used (ResNet-50 V2 with 3 blocks), as the fourth stage's further downsampling would produce tokens at too low a resolution for the Transformer.

### Feature Map Resolution

The critical design decision is at which resolution to transition from CNN to Transformer processing. TransUNet uses $\frac{H}{16} \times \frac{W}{16}$, which for a $224 \times 224$ input yields a $14 \times 14$ grid of spatial locations. This is the same resolution used in standard ViT-Base, allowing direct use of pre-trained ViT weights. At this resolution, each spatial token captures a sufficiently large receptive field from the CNN while keeping the sequence length ($N = 196$) manageable for the quadratic self-attention computation. Using a higher resolution (e.g., $\frac{H}{8}$) would quadruple the number of tokens and increase self-attention cost by $16\times$, while a lower resolution would sacrifice too much spatial detail.

## Transformer Encoder

The ViT encoder operates on a 1D sequence of tokens derived from the CNN feature maps. It consists of $L = 12$ Transformer layers (for ViT-Base), each containing a multi-head self-attention (MSA) module and a position-wise feed-forward network (FFN).

### Tokenization of CNN Features

The CNN feature map of shape $\frac{H}{16} \times \frac{W}{16} \times d_{cnn}$ is first reshaped (flattened spatially) into a sequence of $N$ tokens, each of dimension $d_{cnn}$. A linear projection $E \in \mathbb{R}^{d_{cnn} \times D}$ maps each token to the Transformer hidden dimension $D = 768$. Learned positional encodings $E_{pos} \in \mathbb{R}^{N \times D}$ are then added:

$$z_0 = \text{Flatten}(F_{CNN}) \cdot E + E_{pos}$$

This tokenization process preserves the spatial correspondence between CNN feature locations and Transformer tokens, enabling meaningful skip connections and spatial reconstruction in the decoder.

### Self-Attention Mechanism

Each Transformer layer applies multi-head self-attention to model pairwise relationships between all tokens. For input $Z \in \mathbb{R}^{N \times D}$, queries, keys, and values are computed via learned projections:

$$Q = Z W_Q, \quad K = Z W_K, \quad V = Z W_V$$

where $W_Q, W_K, W_V \in \mathbb{R}^{D \times d_k}$. The attention output is:

$$\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

With $h = 12$ attention heads and $d_k = D/h = 64$ per head, the model can capture diverse types of spatial relationships in parallel. The key advantage over CNNs is that self-attention provides direct interaction between all spatial positions in a single layer, regardless of their distance in the input. A CNN would require many stacked layers to achieve a comparable effective receptive field.

The FFN that follows consists of two linear layers with a GELU activation:

$$\text{FFN}(x) = W_2 \cdot \text{GELU}(W_1 x + b_1) + b_2$$

where typically $W_1 \in \mathbb{R}^{D \times 4D}$ and $W_2 \in \mathbb{R}^{4D \times D}$, providing a 4x expansion in the intermediate dimension.

## Hybrid Design Rationale

The hybrid design is motivated by complementary strengths and weaknesses of CNNs and Transformers:

**CNNs excel at**: (1) Learning low-level features like edges, textures, and local patterns through hierarchical convolutions; (2) Operating with strong inductive biases (locality, translation equivariance) that enable data-efficient learning; (3) Processing high-resolution inputs efficiently due to the local nature of convolution operations.

**Transformers excel at**: (1) Modeling global dependencies through self-attention, where each position can directly attend to every other position; (2) Learning flexible, data-dependent attention patterns that adapt to each input; (3) Capturing complex spatial relationships that span the entire image.

By using a CNN for early feature extraction, TransUNet avoids the known weakness of pure ViT architectures: their poor performance on small to medium datasets due to a lack of inductive biases. The CNN provides well-structured local features that the Transformer can then reason about globally. Additionally, the CNN handles the highest-resolution processing stages where global self-attention would be prohibitively expensive.

## Comparison with Pure CNN and Pure Transformer

The TransUNet paper directly compares three paradigms, revealing clear trade-offs:

**Pure CNN (U-Net with ResNet backbone)** achieves solid baseline performance (DSC ~74.7% on Synapse) with efficient computation. However, it struggles on structures that require global context, such as the pancreas, where understanding the relationship between the organ and surrounding anatomy is important for accurate delineation.

**Pure Transformer (ViT + naive upsampling)** performs poorly when applied directly to medical segmentation (DSC ~61.5% on Synapse). The ViT alone loses too much fine-grained spatial information during tokenization, and without skip connections, the decoder cannot recover sufficient spatial detail for precise segmentation boundaries. The pure ViT also requires large-scale pre-training to compensate for its lack of inductive biases.

**Hybrid CNN-Transformer (TransUNet)** achieves the best performance (DSC ~77.5% on Synapse) by combining the benefits of both paradigms. The CNN provides rich local features and skip connections for the decoder, while the Transformer models global relationships in the bottleneck. This design has proven so effective that it has become a template for many subsequent architectures.

The ablation studies further show that the hybrid approach requires less training data to converge compared to pure ViT, while achieving better global context modeling compared to pure CNN, positioning it as a practical middle ground that is suitable for the relatively small medical imaging datasets commonly available.

## Implementation Notes

When implementing the hybrid encoder, several practical considerations are important:

1. **Weight initialization**: Both the ResNet and ViT components benefit from pre-training. The standard approach is to initialize ResNet with ImageNet-supervised weights and ViT with ImageNet-21K pre-trained weights. For ViT, the patch embedding layer weights from the standard ViT are replaced with the linear projection that maps CNN features to the Transformer dimension.

2. **Feature dimension matching**: The CNN output channel dimension ($d_{cnn} = 1024$ for ResNet-50 Stage 3) must be projected to the ViT hidden dimension ($D = 768$). This is handled by the linear embedding layer.

3. **Resolution consistency**: When using pre-trained ViT-Base weights (trained on $224 \times 224$ images with $16 \times 16$ patches, yielding 196 tokens), the CNN must also produce $14 \times 14$ feature maps. For different input resolutions, positional embedding interpolation is required.

4. **Training**: The entire model is trained end-to-end with both the CNN and Transformer components optimized jointly. Learning rate warmup and a polynomial decay schedule are typically used, with the SGD optimizer and an initial learning rate of 0.01.
