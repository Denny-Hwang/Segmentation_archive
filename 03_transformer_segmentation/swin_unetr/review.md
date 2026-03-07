---
title: "Swin UNETR: Swin Transformers for Semantic Segmentation of Brain Tumors in MRI Images"
date: 2025-03-06
status: planned
tags: [swin-transformer, 3d-segmentation, brain-tumor, medical-segmentation]
difficulty: intermediate
---

# Swin UNETR

## Meta Information

| Field | Value |
|-------|-------|
| **Paper Title** | Swin UNETR: Swin Transformers for Semantic Segmentation of Brain Tumors in MRI Images |
| **Authors** | Hatamizadeh, A., Nath, V., Tang, Y., Yang, D., Roth, H.R., Xu, D. |
| **Year** | 2022 |
| **Venue** | CVPR |
| **arXiv** | [2201.01266](https://arxiv.org/abs/2201.01266) |
| **Difficulty** | Intermediate |

## One-Line Summary

Swin UNETR replaces the ViT encoder in UNETR with a Swin Transformer for efficient hierarchical 3D feature extraction, achieving state-of-the-art brain tumor segmentation on BraTS.

## Motivation and Problem Statement

While UNETR successfully demonstrated the application of Vision Transformers to 3D medical image segmentation, it inherited two significant limitations from the standard ViT architecture. First, global self-attention over all 3D tokens has quadratic computational complexity, limiting scalability to larger volumes or finer patch sizes. Second, the ViT encoder produces single-resolution features at every layer, and multi-scale representations must be approximated by extracting features from layers at different depths -- an indirect and suboptimal approach compared to architectures that explicitly produce hierarchical features.

Swin UNETR addresses both limitations by replacing the ViT encoder with a 3D Swin Transformer. The Swin Transformer's shifted window attention provides linear computational complexity with respect to image size, enabling processing of larger volumes or finer patches. Its hierarchical architecture with patch merging produces genuine multi-resolution feature maps at different stages, providing naturally multi-scale representations that are more suitable for dense prediction tasks.

Additionally, Swin UNETR introduces a self-supervised pre-training strategy tailored for 3D medical data, addressing the challenge that large-scale labeled datasets for 3D segmentation are scarce. By pre-training on unlabeled volumetric data, the model can learn useful 3D feature representations before fine-tuning on task-specific labeled data.

## Architecture Overview

Swin UNETR follows the same U-shaped encoder-decoder paradigm as UNETR, but with the ViT encoder replaced by a 3D Swin Transformer. The architecture consists of a hierarchical Swin Transformer encoder that produces multi-resolution feature maps at four stages, connected through skip connections to a CNN-based decoder that progressively upsamples and refines features for voxel-wise prediction.

The input volume is divided into 3D patches and processed through four Swin Transformer stages, each reducing spatial resolution by $2\times$ while increasing the channel dimension. The encoder outputs feature maps at resolutions $\frac{1}{2}$, $\frac{1}{4}$, $\frac{1}{8}$, and $\frac{1}{16}$ of the input (relative to the initial patch embedding resolution). These multi-scale features are connected to the CNN decoder via skip connections, and the decoder uses transposed convolutions and residual blocks to upsample back to the input resolution.

## Technical Details

### 3D Swin Transformer Encoder

The 3D Swin Transformer adapts the 2D Swin Transformer for volumetric data. The key modifications include:

**3D Window Partitioning**: Instead of 2D windows of size $M \times M$, the encoder uses 3D windows of size $M \times M \times M$ (typically $M = 7$). Self-attention is computed within each 3D window, and shifted 3D windows enable cross-window communication. The computational complexity is linear in the volume size: $O(V \cdot M^3)$ where $V$ is the number of voxels, compared to $O(V^2)$ for global attention.

**3D Patch Merging**: Between stages, 3D patch merging concatenates features from $2 \times 2 \times 2$ neighboring tokens and applies a linear layer, reducing spatial resolution by half in each dimension while doubling the channel count.

**3D Relative Position Bias**: The relative position bias is extended to 3D, with a bias table of shape $(2M-1) \times (2M-1) \times (2M-1)$ to encode relative positions along all three spatial axes.

The encoder stages are configured as:
- **Stage 1**: Resolution $\frac{H}{2} \times \frac{W}{2} \times \frac{D}{2}$, $C$ channels, 2 Swin Transformer blocks
- **Stage 2**: Resolution $\frac{H}{4} \times \frac{W}{4} \times \frac{D}{4}$, $2C$ channels, 2 Swin Transformer blocks
- **Stage 3**: Resolution $\frac{H}{8} \times \frac{W}{8} \times \frac{D}{8}$, $4C$ channels, 2 Swin Transformer blocks
- **Stage 4**: Resolution $\frac{H}{16} \times \frac{W}{16} \times \frac{D}{16}$, $8C$ channels, 2 Swin Transformer blocks

### Hierarchical Feature Maps

Unlike UNETR where multi-scale features are approximated by extracting from different depth layers of a single-resolution encoder, Swin UNETR produces genuine multi-resolution feature maps through its hierarchical architecture. Each encoder stage outputs features at a different spatial resolution with different channel dimensions, directly corresponding to different levels of the feature pyramid.

This hierarchical design provides several advantages for segmentation: features at higher resolutions (Stages 1--2) capture fine-grained spatial details important for boundary delineation, while features at lower resolutions (Stages 3--4) encode broader semantic context. The natural multi-scale representation eliminates the need for the upsampling deconvolutions that UNETR requires to project single-resolution Transformer features to different decoder resolutions.

### CNN Decoder

The decoder mirrors the standard UNETR decoder design with residual convolutional blocks and transposed convolutions. At each decoder level, features from the corresponding encoder stage (via skip connection) are concatenated with upsampled features from the previous decoder level:

$$\mathbf{d}_l = \text{ResBlock}([\text{Upsample}(\mathbf{d}_{l+1}); \mathbf{s}_l])$$

where $\mathbf{s}_l$ is the skip connection from encoder stage $l$, and $\text{ResBlock}$ consists of two $3 \times 3 \times 3$ convolutions with instance normalization and ReLU activation.

A key improvement over UNETR is that the skip connections directly use the encoder stage outputs at their native resolutions, without requiring additional deconvolution blocks to project from a uniform Transformer dimension. This simplifies the decoder and reduces the computational overhead.

### Self-Supervised Pre-Training

Swin UNETR introduces a self-supervised pre-training strategy that combines three pretext tasks to learn useful 3D representations from unlabeled data:

1. **Masked Volume Inpainting**: Random portions of the input volume are masked, and the model is trained to reconstruct the original voxel values, encouraging the encoder to learn contextual representations.

2. **3D Rotation Prediction**: The input volume is randomly rotated by one of four angles ($0^\circ$, $90^\circ$, $180^\circ$, $270^\circ$), and the model predicts the applied rotation, encouraging learning of orientation-aware features.

3. **Contrastive Learning**: Features from augmented views of the same volume are encouraged to be similar while features from different volumes are pushed apart, learning discriminative volumetric representations.

The pre-training is performed on 5,050 unlabeled CT volumes from publicly available datasets, providing the model with rich 3D prior knowledge before fine-tuning on task-specific labeled data.

### Loss Function

For fine-tuning, Swin UNETR uses a combination of Dice loss and cross-entropy loss:

$$\mathcal{L} = \mathcal{L}_{Dice} + \mathcal{L}_{CE}$$

The Dice loss is computed for each class separately and averaged, handling class imbalance. For brain tumor segmentation, the three target regions (whole tumor, tumor core, enhancing tumor) are evaluated independently with region-based Dice computation.

## Experiments and Results

### Datasets

Swin UNETR was primarily evaluated on:

1. **BraTS 2021**: The Brain Tumor Segmentation Challenge dataset containing 1,251 multi-modal MRI volumes (T1, T1ce, T2, FLAIR) with annotations for enhancing tumor (ET), tumor core (TC), and whole tumor (WT). The official validation set contains 219 cases and the test set 570 cases.

2. **BTCV**: The Beyond The Cranial Vault multi-organ CT segmentation dataset with 30 volumes and 13 organ labels, used for additional validation of multi-organ segmentation capabilities.

### Key Results

On BraTS 2021, Swin UNETR achieved state-of-the-art results:

| Region | Dice Score | HD95 (mm) |
|--------|-----------|-----------|
| Whole Tumor (WT) | 92.0% | 4.45 |
| Tumor Core (TC) | 88.3% | 6.73 |
| Enhancing Tumor (ET) | 82.5% | 9.14 |
| **Mean** | **87.6%** | **6.77** |

These results outperformed UNETR (mean Dice 84.2%) and other competing methods including nnU-Net (mean Dice 86.8%), demonstrating the benefits of hierarchical Swin features and self-supervised pre-training.

On BTCV, Swin UNETR achieved a mean Dice of 82.6%, outperforming UNETR (78.4%) and nnU-Net (81.3%), showing strong generalization across different anatomical regions and imaging modalities.

### Ablation Studies

Ablation studies demonstrated: (1) Self-supervised pre-training contributed approximately 2--3% Dice improvement compared to ImageNet initialization, validating the value of domain-specific 3D pre-training. (2) The hierarchical Swin Transformer encoder outperformed the flat ViT encoder by 2--4% Dice, confirming the importance of genuine multi-scale features. (3) The 3D shifted window mechanism was more effective than 3D global attention at comparable computational cost, particularly at higher resolutions. (4) Increasing the window size from $5^3$ to $7^3$ improved performance, with diminishing returns beyond $7^3$.

## Strengths

- **Efficient hierarchical encoding**: The 3D Swin Transformer provides genuine multi-scale features with linear computational complexity, overcoming both limitations of UNETR's ViT encoder.
- **Self-supervised pre-training**: The multi-task pre-training strategy enables effective use of unlabeled 3D medical data, reducing dependence on scarce labeled volumetric annotations.
- **State-of-the-art performance**: Swin UNETR achieved top results on the BraTS 2021 challenge and BTCV benchmark, demonstrating broad applicability.
- **Scalability**: The linear complexity of shifted window attention enables application to larger volumes and finer patch sizes compared to global attention approaches.
- **Natural multi-scale features**: Hierarchical patch merging produces feature maps at multiple resolutions without artificial depth-based approximation.

## Limitations

- **3D window constraints**: The 3D window attention restricts inter-window communication, and while shifted windows mitigate this, the effective receptive field growth per layer is limited compared to global attention.
- **Computational overhead**: Despite being more efficient than global attention, 3D Swin Transformers are still computationally demanding for high-resolution volumetric data, often requiring sliding-window inference.
- **Pre-training requirement**: Achieving the best results requires a substantial self-supervised pre-training phase on large unlabeled datasets, adding to the total training cost.
- **Fixed window size**: The window size must evenly divide the input dimensions, constraining input resolution choices.
- **Decoder asymmetry**: The decoder remains CNN-based, creating an asymmetry between Transformer encoding and convolutional decoding that may not optimally leverage the Transformer features.

## Connections

Swin UNETR is a direct evolution of UNETR, replacing the ViT encoder with a hierarchical Swin Transformer. It draws on the Swin Transformer (Liu et al., 2021) and its 3D extension for video understanding. The self-supervised pre-training strategy builds on advances in contrastive learning and masked image modeling. Swin UNETR can be viewed as the 3D counterpart of Swin-Unet, but with a CNN decoder instead of a Transformer decoder. In the broader landscape, it represents the convergence of efficient attention mechanisms (shifted windows), hierarchical feature extraction (multi-scale), and self-supervised learning for 3D medical image segmentation.

## References

- Hatamizadeh, A., et al. "UNETR: Transformers for 3D Medical Image Segmentation." WACV 2022.
- Liu, Z., et al. "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows." ICCV 2021.
- Liu, Z., et al. "Video Swin Transformer." CVPR 2022.
- Tang, Y., et al. "Self-Supervised Pre-Training of Swin Transformers for 3D Medical Image Analysis." CVPR 2022.
- Isensee, F., et al. "nnU-Net: A Self-Configuring Method for Deep Learning-Based Biomedical Image Segmentation." Nature Methods, 2021.
