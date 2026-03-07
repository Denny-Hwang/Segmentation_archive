---
title: "TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation"
date: 2025-03-06
status: planned
tags: [transformer, cnn-hybrid, medical-segmentation, u-net, vit]
difficulty: intermediate
---

# TransUNet

## Meta Information

| Field | Value |
|-------|-------|
| **Paper Title** | TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation |
| **Authors** | Chen, J., Lu, Y., Yu, Q., Luo, X., Adeli, E., Wang, Y., Lu, L., Yuille, A.L., Zhou, Y. |
| **Year** | 2021 |
| **Venue** | arXiv |
| **arXiv** | [2102.04306](https://arxiv.org/abs/2102.04306) |
| **Difficulty** | Intermediate |

## One-Line Summary

TransUNet combines a CNN feature extractor with a Vision Transformer encoder to capture both local and global context for medical image segmentation, using a cascaded upsampler for dense prediction.

## Motivation and Problem Statement

Traditional CNN-based segmentation architectures such as U-Net and its variants rely on hierarchical feature extraction through convolutional layers. While effective at capturing local patterns, these models have inherently limited receptive fields that grow only linearly with network depth. This limitation makes it difficult to model long-range spatial dependencies that are often critical in medical image segmentation, where anatomical structures exhibit complex global relationships and context from distant regions can disambiguate local structures.

Prior to TransUNet, several works attempted to address the limited receptive field issue through dilated convolutions, attention gates, or non-local blocks appended to CNN architectures. However, these approaches either introduced substantial computational overhead or provided only partial improvements in global context modeling. The emergence of Vision Transformers (ViT) demonstrated that self-attention mechanisms could capture global dependencies effectively in image recognition tasks, but directly applying ViT to dense prediction tasks like segmentation posed challenges due to the loss of fine-grained spatial detail during tokenization.

TransUNet was designed to bridge this gap by leveraging the strengths of both CNNs and Transformers. The key insight is that a CNN backbone can first extract rich low-level features with strong spatial detail, and then a Transformer encoder can model global context over these features. By combining this hybrid encoder with a U-Net-style cascaded decoder with skip connections, TransUNet preserves fine-grained localization while benefiting from the global modeling capacity of self-attention.

## Architecture Overview

TransUNet follows a U-shaped encoder-decoder design. The encoder consists of two stages: a CNN feature extractor (ResNet-50) that processes the input image and produces feature maps at reduced resolution, followed by a Vision Transformer that operates on tokenized versions of these feature maps. The decoder uses a cascaded upsampler that progressively recovers spatial resolution, incorporating skip connections from intermediate CNN layers to restore fine-grained details.

The overall data flow proceeds as follows: an input image of size $H \times W$ is first processed by the CNN backbone to produce feature maps at $\frac{H}{16} \times \frac{W}{16}$ resolution. These feature maps are reshaped into a sequence of 1D tokens, augmented with positional encodings, and fed through 12 layers of Transformer blocks. The output tokens are then reshaped back to 2D and upsampled through the cascaded decoder with skip connections from three intermediate CNN stages.

### Key Components

- **CNN-Transformer Hybrid Encoder**: See [cnn_transformer_hybrid.md](cnn_transformer_hybrid.md)
- **Positional Encoding**: See [positional_encoding.md](positional_encoding.md)

## Technical Details

### Input Processing

The input medical image of size $H \times W \times C$ (where $C$ is typically 1 for grayscale or 3 for RGB) is first fed through the CNN backbone. For 2D segmentation, the input is typically a single slice or a multi-channel composite. The CNN backbone (ResNet-50) processes the image through its initial convolutional layer, batch normalization, and ReLU activation, followed by four residual stages. Features from the first three stages are saved for use as skip connections.

After the CNN produces feature maps at $\frac{H}{16} \times \frac{W}{16} \times d$ resolution (where $d$ is the channel dimension), these maps are reshaped into a sequence of $N = \frac{H}{16} \times \frac{W}{16}$ patch tokens, each of dimension $d$. A linear projection maps these tokens to the Transformer's hidden dimension $D$, and learned positional embeddings of the same dimension are added to encode spatial location information.

### Encoder Design

The hybrid encoder operates in two phases. In the CNN phase, a ResNet-50 (pre-trained on ImageNet) extracts hierarchical features at multiple resolutions: $\frac{H}{4}$, $\frac{H}{8}$, and $\frac{H}{16}$. The $\frac{H}{16}$ features serve as input to the Transformer phase.

The Transformer phase consists of $L = 12$ standard Transformer layers, each containing multi-head self-attention (MSA) and a feed-forward network (FFN) with GELU activation. Each layer applies layer normalization before both MSA and FFN, with residual connections:

$$z_l' = \text{MSA}(\text{LN}(z_{l-1})) + z_{l-1}$$
$$z_l = \text{FFN}(\text{LN}(z_l')) + z_l'$$

The self-attention mechanism allows each spatial token to attend to all other tokens, enabling the capture of long-range dependencies across the entire image. With hidden dimension $D = 768$ and 12 attention heads, each head operates on a 64-dimensional subspace.

### Decoder Design

The decoder employs a Cascaded Upsampler (CUP) that progressively increases the spatial resolution of the encoded features. The Transformer output tokens are first reshaped from a 1D sequence back to a 2D feature map of size $\frac{H}{16} \times \frac{W}{16} \times D$. This feature map is then projected to a lower dimension and upsampled through a series of stages.

Each upsampling stage consists of a $2\times$ bilinear upsampling operation followed by a $3 \times 3$ convolution. At each stage, the upsampled features are concatenated with the corresponding skip connection features from the CNN encoder, similar to the standard U-Net design. Three skip connections are used, from the CNN features at resolutions $\frac{H}{4}$, $\frac{H}{8}$, and $\frac{H}{16}$. The final segmentation map is produced by a $1 \times 1$ convolution that maps the decoded features to the number of target classes.

### Loss Function

TransUNet is trained using a combination of cross-entropy loss and Dice loss:

$$\mathcal{L} = \lambda_{ce} \mathcal{L}_{CE} + \lambda_{dice} \mathcal{L}_{Dice}$$

The cross-entropy loss provides pixel-wise classification supervision, while the Dice loss directly optimizes the overlap between predicted and ground-truth segmentation masks, helping to handle class imbalance that is common in medical image segmentation. Both losses are weighted equally in the default configuration ($\lambda_{ce} = \lambda_{dice} = 1$).

## Experiments and Results

### Datasets

TransUNet was primarily evaluated on two medical image segmentation benchmarks:

1. **Synapse Multi-Organ Segmentation**: A CT dataset containing 30 abdominal CT scans with annotations for 8 abdominal organs (aorta, gallbladder, left kidney, right kidney, liver, pancreas, spleen, stomach). The standard split uses 18 cases for training and 12 for testing, with evaluation on 2D axial slices.

2. **ACDC (Automated Cardiac Diagnosis Challenge)**: A cardiac MRI dataset with 100 patients annotated for right ventricle (RV), myocardium (Myo), and left ventricle (LV). The dataset is split into 70 training, 10 validation, and 20 test cases.

### Key Results

On the Synapse dataset, TransUNet achieved an average Dice Similarity Coefficient (DSC) of 77.48% and Hausdorff Distance (HD95) of 31.69 mm, outperforming prior CNN-based methods such as U-Net (DSC 74.68%), Att-UNet (DSC 75.57%), and the pure ViT baseline (DSC 61.50%). On ACDC, TransUNet achieved an average DSC of 89.71%, demonstrating competitive performance with improvements particularly on the more challenging right ventricle and myocardium classes.

Notably, TransUNet showed significant improvements on organs with complex boundaries and those requiring global context for accurate delineation, such as the pancreas and gallbladder, where long-range dependencies are particularly important for correct segmentation.

### Ablation Studies

The ablation studies revealed several key findings. First, the hybrid CNN-Transformer design substantially outperformed both pure CNN (U-Net) and pure Transformer (ViT with naive upsampling) approaches, validating the complementary benefits of local and global feature extraction. Second, the choice of CNN backbone resolution for tokenization mattered: using features at $\frac{H}{16}$ resolution provided the best trade-off between computational cost and representational capacity. Third, the cascaded upsampler with skip connections was shown to be essential for recovering fine-grained spatial details lost during encoding, with ablations showing significant degradation when skip connections were removed. Fourth, ImageNet pre-training for both the CNN backbone and the ViT encoder was important for achieving strong performance, reflecting the relatively small size of medical imaging datasets.

## Strengths

- **Principled hybrid design** that leverages the complementary strengths of CNNs (local features, translation equivariance) and Transformers (global context, long-range dependencies).
- **Strong empirical performance** on medical image segmentation benchmarks, particularly on structures requiring global context.
- **Flexible architecture** that can incorporate different CNN backbones and Transformer configurations.
- **Effective use of skip connections** in the U-Net style, enabling recovery of fine spatial details from the CNN encoder stages.
- **Pioneering work** that established the paradigm of combining CNNs and Vision Transformers for medical image segmentation, inspiring numerous follow-up works.

## Limitations

- **Computational cost**: The global self-attention in the ViT encoder has quadratic complexity $O(N^2)$ with respect to the number of tokens, which can be expensive for high-resolution inputs.
- **Reliance on pre-training**: Both the CNN backbone and the Transformer encoder benefit significantly from ImageNet pre-training, limiting the approach when pre-trained weights are unavailable (e.g., for non-standard input modalities).
- **2D-only design**: The original TransUNet operates on 2D slices, not fully exploiting 3D volumetric information available in CT and MRI datasets.
- **Fixed tokenization resolution**: The CNN-to-Transformer handoff occurs at a single fixed resolution ($\frac{H}{16}$), which may not be optimal for all anatomical structures and scales.
- **Limited decoder capacity**: The cascaded upsampler is relatively simple compared to more sophisticated decoder designs, potentially limiting the model's ability to capture complex decoder-side interactions.

## Connections

TransUNet is a foundational work in transformer-based medical image segmentation. It builds upon the Vision Transformer (ViT) architecture from Dosovitskiy et al. (2020) and the U-Net framework from Ronneberger et al. (2015). Several subsequent architectures directly extend or improve upon TransUNet's design: Swin-Unet replaces the ViT encoder with a Swin Transformer for more efficient hierarchical encoding; UNETR extends the concept to 3D volumetric segmentation; and DS-TransUNet introduces dual-scale encoding to capture features at multiple resolutions simultaneously. In the broader context, Mask2Former and OneFormer take a different approach by using transformer decoders for universal segmentation rather than focusing on the encoder side.

## References

- Dosovitskiy, A., et al. "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale." ICLR 2021.
- Ronneberger, O., Fischer, P., and Brox, T. "U-Net: Convolutional Networks for Biomedical Image Segmentation." MICCAI 2015.
- He, K., et al. "Deep Residual Learning for Image Recognition." CVPR 2016.
- Vaswani, A., et al. "Attention Is All You Need." NeurIPS 2017.
- Zhou, Z., et al. "UNet++: A Nested U-Net Architecture for Medical Image Segmentation." DLMIA 2018.
