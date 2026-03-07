---
title: "UNETR: Transformers for 3D Medical Image Segmentation"
date: 2025-03-06
status: planned
tags: [vit, 3d-segmentation, medical-segmentation, volumetric]
difficulty: intermediate
---

# UNETR

## Meta Information

| Field | Value |
|-------|-------|
| **Paper Title** | UNETR: Transformers for 3D Medical Image Segmentation |
| **Authors** | Hatamizadeh, A., Tang, Y., Nath, V., Yang, D., Myronenko, A., Landman, B., Roth, H.R., Xu, D. |
| **Year** | 2022 |
| **Venue** | WACV |
| **arXiv** | [2103.10504](https://arxiv.org/abs/2103.10504) |
| **Difficulty** | Intermediate |

## One-Line Summary

UNETR uses a pure Vision Transformer as the encoder for 3D medical image segmentation, connecting intermediate transformer representations to a CNN-based decoder via skip connections.

## Motivation and Problem Statement

Prior to UNETR, transformer-based segmentation models such as TransUNet and Swin-Unet were designed exclusively for 2D images. However, medical imaging modalities like CT and MRI produce volumetric (3D) data where inter-slice spatial relationships carry important anatomical information. Processing 3D volumes slice-by-slice discards this contextual information and can lead to inconsistent segmentation across slices. While 3D CNNs (e.g., 3D U-Net, V-Net, nnU-Net) had been the standard approach for volumetric segmentation, they suffer from limited receptive fields and struggle to model long-range dependencies across the volume.

UNETR was motivated by the need to bring the long-range modeling capabilities of Transformers to 3D medical image segmentation. The key insight was that the Vision Transformer's global self-attention mechanism, which enables each spatial token to attend to all other tokens, is especially valuable in 3D where anatomical structures span large spatial extents and their segmentation benefits from global volumetric context. By treating 3D volumes as sequences of 3D patch tokens, UNETR directly extends the ViT paradigm to volumetric data.

A secondary motivation was to provide multi-scale feature representations from the Transformer encoder. Standard ViT produces features at a single scale, but segmentation requires multi-resolution features for the decoder. UNETR addresses this by extracting intermediate representations from multiple Transformer layers and connecting them to the decoder via skip connections, creating a U-Net-like architecture where the encoder is a pure ViT.

## Architecture Overview

UNETR follows a U-shaped encoder-decoder design. The encoder is a standard Vision Transformer operating on 3D patch tokens extracted from the input volume. The decoder is a CNN-based architecture that progressively upsamples features using transposed convolutions and residual blocks. The connection between the Transformer encoder and CNN decoder is achieved through skip connections that extract feature representations from intermediate Transformer layers, reshape them to 3D feature maps, and project them to appropriate dimensions before merging with the decoder.

The input volume of size $H \times W \times D$ is divided into non-overlapping 3D patches of size $P \times P \times P$ (typically $16 \times 16 \times 16$), producing a sequence of $N = \frac{H \times W \times D}{P^3}$ tokens. These tokens are linearly projected to the Transformer hidden dimension and processed through $L = 12$ Transformer layers. Feature maps are extracted from layers $\{3, 6, 9, 12\}$ and reshaped back to 3D spatial representations for skip connections.

### Key Components

- **ViT Encoder for 3D Data**: See [vit_encoder_3d.md](vit_encoder_3d.md)

## Technical Details

### 3D Input Tokenization

Given an input volume $\mathbf{x} \in \mathbb{R}^{H \times W \times D \times C_{in}}$ (where $C_{in}$ is typically 1 for CT or 4 for multi-modal MRI), the volume is partitioned into non-overlapping 3D patches of size $P \times P \times P$. Each patch is flattened to a 1D vector of dimension $P^3 \cdot C_{in}$ and linearly projected to the Transformer hidden dimension $d$:

$$\mathbf{z}_0 = [\mathbf{x}_p^1 \mathbf{E}; \mathbf{x}_p^2 \mathbf{E}; \ldots; \mathbf{x}_p^N \mathbf{E}] + \mathbf{E}_{pos}$$

where $\mathbf{E} \in \mathbb{R}^{P^3 C_{in} \times d}$ is the linear projection matrix and $\mathbf{E}_{pos} \in \mathbb{R}^{N \times d}$ provides learned positional encodings. For a typical input of $96 \times 96 \times 96$ with $P = 16$, this produces $N = 6 \times 6 \times 6 = 216$ tokens, each representing a $16^3$ voxel region.

### ViT Encoder

The ViT encoder consists of $L = 12$ standard Transformer layers, each applying multi-head self-attention (MSA) and a feed-forward network (FFN) with pre-norm layer normalization and residual connections:

$$\mathbf{z}_l' = \text{MSA}(\text{LN}(\mathbf{z}_{l-1})) + \mathbf{z}_{l-1}$$
$$\mathbf{z}_l = \text{FFN}(\text{LN}(\mathbf{z}_l')) + \mathbf{z}_l'$$

The encoder uses a hidden dimension of $d = 768$ and 12 attention heads. Unlike hierarchical encoders (e.g., Swin Transformer), UNETR's ViT maintains a constant spatial resolution and feature dimension throughout all 12 layers. This means all tokens interact at the same spatial scale, and multi-scale features are obtained by extracting representations at different depths rather than different resolutions.

### CNN Decoder with Skip Connections

The decoder takes skip connection features from four intermediate Transformer layers (layers 3, 6, 9, and 12) and the original input. Each skip connection involves: (1) extracting the token sequence from the specified Transformer layer, (2) reshaping it back to a 3D spatial grid, (3) projecting the feature dimension to the decoder's expected channels, and (4) applying deconvolutions to match the required resolution.

Features from earlier Transformer layers (layer 3) are upsampled more aggressively to match higher resolutions, while features from later layers require less upsampling. The decoder merges skip connection features with upsampled features from deeper decoder stages through concatenation, followed by residual convolutional blocks for feature refinement. The final layer uses a $1 \times 1 \times 1$ convolution to produce the per-voxel class predictions.

The decoder structure processes features at five resolution levels:
- $\frac{H}{32} \times \frac{W}{32} \times \frac{D}{32}$: From Transformer output (layer 12)
- $\frac{H}{16} \times \frac{W}{16} \times \frac{D}{16}$: Skip from layer 9
- $\frac{H}{8} \times \frac{W}{8} \times \frac{D}{8}$: Skip from layer 6
- $\frac{H}{4} \times \frac{W}{4} \times \frac{D}{4}$: Skip from layer 3
- $H \times W \times D$: Skip from original input

### Loss Function

UNETR uses a combination of soft Dice loss and cross-entropy loss:

$$\mathcal{L} = \mathcal{L}_{Dice} + \lambda \mathcal{L}_{CE}$$

The Dice loss directly optimizes the volumetric overlap between predicted and ground-truth segmentation, while cross-entropy provides pixel-wise classification gradients. Deep supervision is optionally applied at intermediate decoder stages to improve gradient flow during training.

## Experiments and Results

### Datasets

UNETR was evaluated on two major 3D medical image segmentation benchmarks:

1. **BTCV (Beyond The Cranial Vault)**: A multi-organ abdominal CT segmentation dataset with 30 cases and 13 organ annotations (spleen, right kidney, left kidney, gallbladder, esophagus, liver, stomach, aorta, inferior vena cava, portal and splenic veins, pancreas, right adrenal gland, left adrenal gland). The evaluation uses 5-fold cross-validation.

2. **MSD (Medical Segmentation Decathlon)**: Specifically the Brain Tumor and Spleen segmentation tasks. The brain tumor task uses multi-modal MRI (T1, T1ce, T2, FLAIR) for segmenting enhancing tumor, tumor core, and whole tumor regions.

### Key Results

On the BTCV dataset, UNETR achieved a mean Dice score of 78.4%, outperforming 3D CNN baselines including nnU-Net (78.1%) and Attention U-Net (71.5%). The improvement was most pronounced for organs that benefit from global context, such as the gallbladder (65.2% vs. 60.1% for Attention U-Net) and pancreas (63.2% vs. 58.8%).

On the MSD Brain Tumor task, UNETR achieved competitive results with mean Dice scores of 81.5% for the whole tumor, 78.2% for the tumor core, and 62.1% for the enhancing tumor. On the MSD Spleen task, UNETR achieved a Dice score of 96.4%.

### Ablation Studies

The ablation studies investigated: (1) the number of skip connections and which Transformer layers to use, finding that extracting from layers $\{3, 6, 9, 12\}$ (evenly spaced) was optimal; (2) the effect of pre-training, showing that ImageNet-21K pre-training improved performance by 3--5% Dice compared to training from scratch; (3) patch size, where $16 \times 16 \times 16$ provided the best trade-off between sequence length and spatial resolution; (4) the importance of the CNN decoder over simpler upsampling strategies, with residual convolutional blocks outperforming bilinear interpolation.

## Strengths

- **First Transformer encoder for 3D segmentation**: UNETR pioneered the application of Vision Transformers to 3D volumetric medical image segmentation, establishing a new architectural paradigm.
- **Global 3D context modeling**: The self-attention mechanism operates over the entire 3D volume, enabling the capture of long-range spatial dependencies that span the volumetric field of view.
- **Effective multi-scale skip connections**: By extracting features from multiple Transformer layers at different semantic levels, UNETR creates an effective multi-scale representation despite using a single-scale ViT encoder.
- **Strong empirical results**: UNETR matched or exceeded the performance of well-established 3D CNN architectures including nnU-Net.
- **Flexible framework**: The separation of Transformer encoder and CNN decoder allows independent improvements to either component.

## Limitations

- **Quadratic self-attention complexity**: Global self-attention over all 3D tokens is computationally expensive. For $96^3$ volumes with $16^3$ patches, the 216 tokens are manageable, but scaling to larger volumes or finer patches becomes prohibitive.
- **Single-resolution encoding**: Unlike hierarchical encoders (Swin Transformer), UNETR's ViT processes all tokens at the same resolution. The multi-scale skip connections approximate hierarchical features but at the cost of using the same Transformer dimension at all scales.
- **Large memory footprint**: Storing activations for 12 Transformer layers during training with 3D inputs requires substantial GPU memory, often necessitating smaller batch sizes or patch-based training.
- **Pre-training dependence**: UNETR benefits significantly from ImageNet pre-training despite the domain gap, and training from scratch yields notably inferior results.
- **Hybrid decoder**: While the encoder is a pure Transformer, the decoder uses CNN operations (transposed convolutions, residual blocks), making UNETR a partially hybrid architecture.

## Connections

UNETR extends the TransUNet concept to 3D volumetric data, replacing the CNN-Transformer hybrid encoder with a pure ViT encoder. Swin UNETR subsequently improved upon UNETR by replacing the ViT encoder with a Swin Transformer for more efficient hierarchical feature extraction and better scalability to higher resolutions. Both models share the same decoder design philosophy of connecting Transformer encoder features to a CNN decoder via skip connections. In the broader landscape, UNETR's approach of extracting intermediate Transformer representations for multi-scale features has been adopted by several subsequent 3D segmentation architectures.

## References

- Dosovitskiy, A., et al. "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale." ICLR 2021.
- Ronneberger, O., Fischer, P., and Brox, T. "U-Net: Convolutional Networks for Biomedical Image Segmentation." MICCAI 2015.
- Isensee, F., et al. "nnU-Net: A Self-Configuring Method for Deep Learning-Based Biomedical Image Segmentation." Nature Methods, 2021.
- Chen, J., et al. "TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation." arXiv 2021.
- Milletari, F., Navab, N., and Ahmadi, S. "V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation." 3DV 2016.
