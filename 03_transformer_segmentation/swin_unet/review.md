---
title: "Swin-Unet: Unet-like Pure Transformer for Medical Image Segmentation"
date: 2025-03-06
status: planned
tags: [swin-transformer, pure-transformer, medical-segmentation, u-net]
difficulty: intermediate
---

# Swin-Unet

## Meta Information

| Field | Value |
|-------|-------|
| **Paper Title** | Swin-Unet: Unet-like Pure Transformer for Medical Image Segmentation |
| **Authors** | Cao, H., Wang, Y., Chen, J., Jiang, D., Zhang, X., Tian, Q., Wang, M. |
| **Year** | 2021 |
| **Venue** | arXiv |
| **arXiv** | [2105.05537](https://arxiv.org/abs/2105.05537) |
| **Difficulty** | Intermediate |

## One-Line Summary

Swin-Unet is a pure transformer-based U-shaped architecture that uses Swin Transformer blocks with shifted window attention for both encoding and decoding in medical image segmentation.

## Motivation and Problem Statement

While TransUNet demonstrated the potential of combining CNNs with Transformers for medical image segmentation, it still relied on a CNN backbone for feature extraction, raising the question of whether a pure Transformer architecture could achieve comparable or superior results. CNN components, though effective, introduce specific inductive biases (locality, translation equivariance) that constrain the model's representational flexibility. A pure Transformer architecture could potentially learn more flexible feature representations directly from data.

However, directly applying the standard Vision Transformer (ViT) to segmentation tasks faces two fundamental challenges. First, ViT's global self-attention has quadratic computational complexity with respect to the number of tokens, making it impractical for the high-resolution feature maps needed in dense prediction. Second, ViT produces single-scale features without the hierarchical multi-resolution representations that are essential for segmentation decoders. The Swin Transformer, introduced by Liu et al., addressed both issues through windowed attention and hierarchical feature maps, making it a natural candidate for building pure Transformer segmentation architectures.

Swin-Unet was proposed to demonstrate that a carefully designed pure Transformer architecture, built entirely from Swin Transformer blocks, could match or exceed hybrid CNN-Transformer approaches for medical image segmentation while offering the benefits of a unified architectural paradigm.

## Architecture Overview

Swin-Unet adopts a symmetric U-shaped encoder-decoder structure, mirroring the classical U-Net design but replacing all convolutional operations with Swin Transformer blocks. The architecture consists of three main components: (1) a Swin Transformer encoder that hierarchically reduces spatial resolution while increasing feature dimensionality, (2) a bottleneck composed of Swin Transformer blocks at the lowest resolution, and (3) a symmetric Swin Transformer decoder that uses patch expanding layers to progressively upsample features back to the input resolution. Skip connections link corresponding encoder and decoder stages, enabling the transfer of fine-grained spatial information.

The input image of size $H \times W \times 3$ is first divided into non-overlapping $4 \times 4$ patches through a patch embedding layer, producing $\frac{H}{4} \times \frac{W}{4}$ tokens of dimension $C = 96$. The encoder then processes these tokens through three stages, each consisting of Swin Transformer blocks followed by a patch merging layer that halves the spatial dimensions and doubles the channel count. The decoder mirrors this structure with Swin Transformer blocks and patch expanding layers.

### Key Components

- **Shifted Window Mechanism**: See [shifted_window_mechanism.md](shifted_window_mechanism.md)
- **Patch Expanding Layer**: See [patch_expanding.md](patch_expanding.md)

## Technical Details

### Encoder Design

The encoder consists of four stages with progressively decreasing spatial resolution:

- **Stage 1**: $\frac{H}{4} \times \frac{W}{4}$ tokens, $C$ channels, 2 Swin Transformer blocks
- **Stage 2**: $\frac{H}{8} \times \frac{W}{8}$ tokens, $2C$ channels, 2 Swin Transformer blocks
- **Stage 3**: $\frac{H}{16} \times \frac{W}{16}$ tokens, $4C$ channels, 2 Swin Transformer blocks
- **Stage 4 (Bottleneck)**: $\frac{H}{32} \times \frac{W}{32}$ tokens, $8C$ channels, 2 Swin Transformer blocks

Between stages, patch merging layers concatenate features from groups of $2 \times 2$ neighboring tokens and apply a linear layer to reduce the concatenated dimension from $4C$ to $2C$, effectively performing a $2\times$ spatial downsampling while doubling the feature dimension. Each Swin Transformer block consists of a window-based multi-head self-attention (W-MSA) module followed by a shifted-window multi-head self-attention (SW-MSA) module, each accompanied by a 2-layer MLP with GELU activation and layer normalization.

### Bottleneck

The bottleneck at the deepest level of the U-shaped architecture consists of two consecutive Swin Transformer blocks operating at $\frac{H}{32} \times \frac{W}{32}$ resolution with $8C$ channels. At this resolution, the window-based attention can capture substantial spatial context within each window, and the shifted window mechanism ensures cross-window information flow. The bottleneck serves as the bridge between the contracting encoder and the expanding decoder, processing the most compressed representation of the input.

### Decoder Design

The decoder is designed to be architecturally symmetric to the encoder. At each stage, a patch expanding layer first upsamples the spatial resolution by $2\times$ while halving the channel dimension. The upsampled features are then concatenated with the corresponding skip connection features from the encoder and passed through a linear layer to adjust the channel dimension. Finally, two Swin Transformer blocks process the concatenated features.

The decoder stages produce features at resolutions $\frac{H}{16}$, $\frac{H}{8}$, and $\frac{H}{4}$, mirroring the encoder's hierarchy. A final $4\times$ patch expanding layer brings the features back to the original input resolution $H \times W$, followed by a linear projection to produce the segmentation map with the desired number of output classes.

### Skip Connections

Skip connections in Swin-Unet follow the same principle as in the original U-Net: features from each encoder stage are concatenated with the corresponding decoder stage features. Since both encoder and decoder operate entirely with Transformer blocks, the skip connections transfer Transformer-processed features rather than CNN feature maps. A key difference from TransUNet is that these features already contain global context information from the shifted window attention, providing richer representations to the decoder.

The concatenation doubles the channel dimension, which is then reduced by a linear projection layer. This linear fusion replaces the convolutional blocks used in U-Net for skip connection processing, maintaining the pure Transformer design philosophy.

### Loss Function

Swin-Unet uses a combination of cross-entropy loss and Dice loss, following the standard practice in medical image segmentation:

$$\mathcal{L} = \mathcal{L}_{CE} + \mathcal{L}_{Dice}$$

The Dice loss is computed per-class and averaged, providing direct optimization of the overlap metric and helping to address class imbalance. The cross-entropy loss provides stable gradient signals for pixel-wise classification.

## Experiments and Results

### Datasets

Swin-Unet was evaluated on two primary medical image segmentation benchmarks:

1. **Synapse Multi-Organ Segmentation**: 30 abdominal CT scans with 8 organ annotations, using the same split as TransUNet (18 training, 12 testing cases).

2. **ACDC (Automated Cardiac Diagnosis Challenge)**: 100 cardiac MRI patients with annotations for right ventricle, myocardium, and left ventricle.

### Key Results

On the Synapse dataset, Swin-Unet achieved a mean DSC of 79.13% and HD95 of 21.55 mm, outperforming TransUNet (77.48% DSC, 31.69 mm HD95) and establishing that a pure Transformer architecture could surpass hybrid designs. The improvement was particularly notable in the Hausdorff distance metric, indicating better boundary delineation. On ACDC, Swin-Unet achieved a mean DSC of 90.00%, with strong performance across all three cardiac structures.

Swin-Unet showed particular improvements on organs with complex shapes and unclear boundaries (e.g., pancreas, stomach), suggesting that the hierarchical shifted window attention captures multi-scale contextual information more effectively than the single-scale global attention in TransUNet's ViT encoder.

### Ablation Studies

Ablation studies demonstrated several important findings: (1) Pre-training on ImageNet-22K was essential for strong performance, with randomly initialized models showing significant degradation. (2) The symmetric decoder design with Swin Transformer blocks outperformed simpler upsampling decoders, validating the importance of Transformer-based decoding. (3) Skip connections remained important, with their removal causing a 2--3% drop in Dice score. (4) The shifted window mechanism was crucial, as using only regular window attention (without shifting) reduced cross-window communication and degraded segmentation quality at window boundaries.

## Strengths

- **Pure Transformer design** eliminates reliance on CNN components, enabling a unified architectural framework that can benefit uniformly from advances in Transformer research.
- **Computational efficiency** through shifted window attention achieves linear complexity relative to image size, unlike the quadratic complexity of global self-attention in ViT-based models.
- **Hierarchical feature extraction** naturally produces multi-scale representations through patch merging, aligning well with the requirements of dense prediction tasks.
- **Symmetric architecture** with the novel patch expanding decoder provides an elegant counterpart to the patch merging encoder, maintaining architectural consistency.
- **Strong empirical results** demonstrate that pure Transformer architectures can match or exceed hybrid CNN-Transformer approaches for medical image segmentation.

## Limitations

- **Heavy reliance on pre-training**: Without ImageNet-22K pre-training, Swin-Unet's performance degrades substantially, indicating that the pure Transformer architecture requires large-scale pre-training to compensate for its lack of CNN inductive biases.
- **Fixed window size**: The window-based attention restricts the receptive field within each attention layer to a fixed window size (typically $7 \times 7$), and while shifted windows enable cross-window communication, the effective receptive field growth is more gradual than with global attention.
- **2D-only architecture**: Like TransUNet, Swin-Unet operates on 2D slices and does not natively handle 3D volumetric data, limiting its applicability in settings where inter-slice context is important.
- **Patch expanding limitations**: The patch expanding layer uses a relatively simple linear operation for upsampling, which may not capture complex upsampling patterns as effectively as learned deconvolution operations.
- **Limited exploration of decoder design**: The symmetric decoder directly mirrors the encoder without exploring potentially more effective asymmetric designs or attention-based skip connection fusion.

## Connections

Swin-Unet builds directly upon the Swin Transformer backbone (Liu et al., 2021) and the U-Net architecture (Ronneberger et al., 2015). It demonstrates that the CNN components in TransUNet are not strictly necessary, though pre-training becomes more critical. Swin UNETR extends the Swin Transformer concept to 3D medical image segmentation by adapting the windowed attention mechanism for volumetric data. DS-TransUNet further develops the multi-scale idea by employing dual Swin Transformer encoders at different resolutions. In the natural image domain, Mask2Former and OneFormer use Swin Transformer as a backbone option, demonstrating the versatility of the Swin architecture across segmentation paradigms.

## References

- Liu, Z., et al. "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows." ICCV 2021.
- Ronneberger, O., Fischer, P., and Brox, T. "U-Net: Convolutional Networks for Biomedical Image Segmentation." MICCAI 2015.
- Chen, J., et al. "TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation." arXiv 2021.
- Dosovitskiy, A., et al. "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale." ICLR 2021.
- Vaswani, A., et al. "Attention Is All You Need." NeurIPS 2017.
