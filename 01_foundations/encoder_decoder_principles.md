---
title: "Encoder-Decoder Architecture Principles"
date: 2025-03-06
status: in-progress
tags: [architecture, encoder-decoder, skip-connections, FPN, upsampling, bilinear, transposed-convolution, pixel-shuffle]
difficulty: intermediate
---

# Encoder-Decoder Architecture Principles

The encoder-decoder structure is the dominant architectural paradigm in image segmentation. This document explains the core principles: why encoding and decoding are necessary, how skip connections recover spatial detail, how Feature Pyramid Networks provide multi-scale representations, and how different upsampling strategies compare.

---

## 1. The Encoder-Decoder Framework

### 1.1 The Core Tension

Image segmentation requires two seemingly contradictory capabilities:

1. **Semantic understanding** (what): Recognizing high-level concepts like "person," "car," or "tumor" requires large receptive fields and abstract feature representations. This is achieved by progressive spatial downsampling.
2. **Spatial precision** (where): Producing per-pixel labels requires preserving fine spatial detail and sharp object boundaries. Downsampling destroys this information.

The encoder-decoder architecture resolves this tension by dividing the network into two complementary paths.

### 1.2 The Encoder (Contracting Path)

The encoder is typically a classification backbone (ResNet, VGG, EfficientNet, Swin Transformer, etc.) pretrained on ImageNet. It progressively reduces spatial resolution while increasing the number of feature channels and the semantic level of representation.

At each stage $s$, the encoder produces a feature map:

$$\mathbf{F}_s^{\text{enc}} \in \mathbb{R}^{B \times C_s \times \frac{H}{2^s} \times \frac{W}{2^s}}$$

Typical encoder stages (using ResNet as an example):

| Stage | Output Stride | Resolution (for 512x512 input) | Channels (ResNet-50) | Feature Type |
|-------|:---:|:---:|:---:|---|
| 0 (stem) | 2 | 256x256 | 64 | Low-level edges, textures |
| 1 | 4 | 128x128 | 256 | Local patterns, corners |
| 2 | 8 | 64x64 | 512 | Object parts |
| 3 | 16 | 32x32 | 1024 | Object-level semantics |
| 4 | 32 | 16x16 | 2048 | Scene-level semantics |

**Output stride** is the ratio of input spatial resolution to feature map resolution. Lower output stride means higher resolution features but more computation.

### 1.3 The Decoder (Expanding Path)

The decoder progressively upsamples the low-resolution, semantically-rich features back to the input resolution. At each stage, it refines the spatial detail by combining upsampled features with information from the encoder.

$$\mathbf{F}_s^{\text{dec}} = \text{Refine}\Big(\text{Upsample}(\mathbf{F}_{s+1}^{\text{dec}}),\ \mathbf{F}_s^{\text{enc}}\Big)$$

The final output is a dense prediction map:

$$\hat{Y} = \text{Conv}_{1 \times 1}(\mathbf{F}_0^{\text{dec}}) \in \mathbb{R}^{B \times K \times H \times W}$$

### 1.4 The Bottleneck

The bottleneck is the lowest-resolution feature map at the junction between encoder and decoder. It contains the most semantically abstract features with the largest receptive field. Some architectures add extra processing here (e.g., ASPP in DeepLab v3+, additional convolution blocks in U-Net).

---

## 2. Skip Connections

### 2.1 The Problem Without Skip Connections

If the decoder simply upsamples the bottleneck features to full resolution (as in FCN-32s), the result is semantically correct but spatially coarse. Object boundaries are blurry, small objects are lost, and fine structures are destroyed. This is because the encoder's downsampling operations (pooling, strided convolution) discard spatial information irreversibly.

### 2.2 How Skip Connections Help

Skip connections create direct pathways from encoder layers to corresponding decoder layers at the same spatial resolution. This allows the decoder to access fine-grained spatial features that were preserved in the encoder but lost through the bottleneck.

**Concatenation (U-Net style):**

$$\mathbf{F}_s^{\text{dec}} = \text{Conv}\Big(\text{Concat}\big[\text{Upsample}(\mathbf{F}_{s+1}^{\text{dec}}),\ \mathbf{F}_s^{\text{enc}}\big]\Big)$$

The encoder and decoder features are concatenated along the channel dimension and processed by convolution layers that learn to combine them.

**Addition (FCN / ResNet style):**

$$\mathbf{F}_s^{\text{dec}} = \text{Conv}\Big(\text{Upsample}(\mathbf{F}_{s+1}^{\text{dec}}) + \text{Conv}_{1\times1}(\mathbf{F}_s^{\text{enc}})\Big)$$

Encoder features are projected to match the channel dimension and element-wise added to the upsampled decoder features.

### 2.3 Comparison: Concatenation vs. Addition

| Aspect | Concatenation | Addition |
|--------|:---:|:---:|
| Information preservation | Higher (all channels retained) | Lower (channels merged) |
| Parameters | More (convolution over wider input) | Fewer |
| Representation capacity | Higher | Lower |
| Used in | U-Net, U-Net++, nnU-Net | FCN, FPN, DeepLab v3+ |
| Memory cost | Higher | Lower |

In practice, concatenation tends to produce slightly better results at the cost of more parameters. Addition is preferred when parameter efficiency matters or when the architecture already has sufficient capacity.

### 2.4 Skip Connection Variants

**Dense skip connections (U-Net++):**

Instead of connecting only corresponding encoder-decoder layers, U-Net++ introduces nested dense connections. Each decoder node receives features from all prior encoder and decoder nodes at the same or lower resolution:

```
X^{0,0} ------> X^{0,1} ------> X^{0,2} ------> X^{0,3}
  \                / \              / \              /
X^{1,0} ------> X^{1,1} ------> X^{1,2}
  \                / \              /
X^{2,0} ------> X^{2,1}
  \                /
X^{3,0}
```

Each $X^{i,j}$ with $j > 0$ receives inputs from $X^{i,0}, X^{i,1}, \dots, X^{i,j-1}$ (horizontal dense connections) and $\text{Upsample}(X^{i+1,j-1})$ (diagonal connection).

**Attention-gated skip connections (Attention U-Net):**

Not all encoder features are equally relevant. Attention gates learn to suppress irrelevant features and highlight salient ones before they are passed to the decoder:

$$\alpha_s = \sigma\Big(\psi^T \cdot \text{ReLU}\big(W_g \mathbf{F}_{s+1}^{\text{dec}} + W_x \mathbf{F}_s^{\text{enc}} + b\big)\Big)$$

$$\hat{\mathbf{F}}_s^{\text{enc}} = \alpha_s \odot \mathbf{F}_s^{\text{enc}}$$

where $\alpha_s$ is the spatial attention map and $\odot$ denotes element-wise multiplication.

### 2.5 Why Skip Connections Are Essential

Empirical evidence consistently shows large performance gains from skip connections:

| Architecture | Skip Connection Type | PASCAL VOC mIoU |
|-------------|---------------------|:---:|
| FCN-32s | None | 59.4 |
| FCN-16s | Addition (pool4) | 62.4 |
| FCN-8s | Addition (pool4 + pool3) | 62.7 |
| U-Net style | Concatenation (all levels) | ~65+ |

The ~3 mIoU point gap between FCN-32s and FCN-8s demonstrates that each additional skip connection recovers meaningful spatial information. Modern architectures use skip connections at every resolution level.

---

## 3. Feature Pyramid Networks (FPN)

### 3.1 Motivation

Objects in natural images span a wide range of scales. A small pedestrian at the horizon and a large truck in the foreground both need to be segmented accurately. Single-scale feature maps struggle with this: high-resolution features lack semantic depth, while low-resolution features lack spatial detail.

### 3.2 The FPN Architecture

Introduced by Lin et al. (2017) for object detection and later adopted for segmentation (Panoptic FPN, Kirillov et al., 2019).

FPN creates a multi-scale feature pyramid with strong semantics at all levels through a three-step process:

**Step 1: Bottom-up pathway** (the encoder). Standard forward pass through the backbone produces feature maps $\{C_2, C_3, C_4, C_5\}$ at strides $\{4, 8, 16, 32\}$.

**Step 2: Top-down pathway** with lateral connections. Starting from the coarsest level, upsample and merge with the corresponding bottom-up feature:

$$P_5 = \text{Conv}_{1\times1}(C_5)$$

$$P_4 = \text{Conv}_{1\times1}(C_4) + \text{Upsample}_{2\times}(P_5)$$

$$P_3 = \text{Conv}_{1\times1}(C_3) + \text{Upsample}_{2\times}(P_4)$$

$$P_2 = \text{Conv}_{1\times1}(C_2) + \text{Upsample}_{2\times}(P_3)$$

**Step 3:** Apply a $3 \times 3$ convolution to each $P_s$ to reduce aliasing from upsampling.

The result is a set of feature maps $\{P_2, P_3, P_4, P_5\}$ that all have 256 channels (a common choice) but at different spatial resolutions, each enriched with semantics from higher levels.

### 3.3 FPN for Segmentation

**Panoptic FPN** extends FPN for segmentation by adding a lightweight branch that merges all pyramid levels:

1. For each level $P_s$ (at stride $s$), apply a sequence of $3 \times 3$ convolutions with group normalization and ReLU, interleaved with $2\times$ bilinear upsampling, until the feature map reaches stride 4.
2. Element-wise sum all upsampled features into a single stride-4 feature map.
3. Apply a final $1 \times 1$ convolution to produce per-pixel class predictions.

### 3.4 FPN vs. Standard Encoder-Decoder

| Aspect | Standard Encoder-Decoder (U-Net) | FPN |
|--------|:---:|:---:|
| Multi-scale output | No (single output) | Yes (pyramid of features) |
| Semantic enrichment direction | Top-down only | Top-down with lateral connections |
| Connection type | Concatenation | Addition (lateral) |
| Output resolution | Full resolution | Multiple resolutions |
| Primary use | Single-task dense prediction | Multi-task (detection + segmentation) |

### 3.5 Advanced Multi-Scale Designs

**BiFPN (EfficientDet):** Adds bottom-up connections to FPN, creating bidirectional feature fusion. Uses learnable weights for each input:

$$P_s^{\text{out}} = \frac{w_1 P_s^{\text{td}} + w_2 P_s^{\text{in}}}{w_1 + w_2 + \epsilon}$$

**ASPP (DeepLab):** An alternative multi-scale approach. Instead of a feature pyramid, apply multiple parallel atrous convolutions at different dilation rates to the same feature map:

$$\text{ASPP}(\mathbf{F}) = \text{Conv}_{1\times1}(\text{Concat}[\text{Conv}_{r=1}(\mathbf{F}),\ \text{Conv}_{r=6}(\mathbf{F}),\ \text{Conv}_{r=12}(\mathbf{F}),\ \text{Conv}_{r=18}(\mathbf{F}),\ \text{GAP}(\mathbf{F})])$$

**HRNet (High-Resolution Network):** Maintains parallel multi-resolution streams throughout the network, with repeated multi-scale fusion. Unlike FPN (which creates multi-scale features at the end), HRNet preserves high-resolution features from the start.

---

## 4. Upsampling Strategies

Upsampling is needed in the decoder to restore spatial resolution. The three most common approaches are bilinear interpolation, transposed convolution, and PixelShuffle. Each has distinct trade-offs.

### 4.1 Bilinear Interpolation

**Method:** Compute each output pixel as a weighted average of the four nearest input pixels, with weights determined by the relative distance:

$$f(x, y) = \frac{1}{(x_2 - x_1)(y_2 - y_1)} \Big[ f(Q_{11})(x_2 - x)(y_2 - y) + f(Q_{21})(x - x_1)(y_2 - y) + f(Q_{12})(x_2 - x)(y - y_1) + f(Q_{22})(x - x_1)(y - y_1) \Big]$$

where $Q_{ij}$ are the four neighboring grid points.

**Implementation:** `F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)`

**Properties:**

| Property | Value |
|----------|-------|
| Learnable parameters | 0 |
| Computational cost | Very low |
| Artifacts | Smooth but can be blurry |
| Checkerboard artifacts | None |
| Memory | Minimal |

**When to use:** Default choice for general-purpose upsampling. Used in FPN, DeepLab, SegFormer decoders. When followed by convolution layers that can refine the result, bilinear interpolation is usually sufficient.

### 4.2 Transposed Convolution (Deconvolution)

**Method:** A learnable upsampling operation. Conceptually, it inserts zeros between input pixels, then applies a standard convolution. Equivalently, it can be viewed as a convolution that maps from lower to higher resolution with learned weights.

For an input feature map $\mathbf{F} \in \mathbb{R}^{C_{in} \times H \times W}$ and a transposed convolution with kernel size $k$, stride $s$, and padding $p$:

$$H_{out} = (H - 1) \times s - 2p + k$$

For $2\times$ upsampling: kernel $= 4$, stride $= 2$, padding $= 1$ gives $H_{out} = 2H$.

**Implementation:** `nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)`

**Properties:**

| Property | Value |
|----------|-------|
| Learnable parameters | $C_{in} \times C_{out} \times k \times k$ |
| Computational cost | Moderate (same as forward convolution) |
| Artifacts | **Checkerboard artifacts** (a well-known problem) |
| Memory | Moderate |

**Checkerboard artifacts:** When the kernel size is not divisible by the stride, the transposed convolution produces uneven overlap of contributions from neighboring input pixels. This creates a grid-like pattern in the output. Mitigation strategies:

1. Use kernel sizes divisible by the stride (e.g., kernel=4, stride=2 or kernel=2, stride=2).
2. Initialize with bilinear interpolation weights.
3. Replace with bilinear interpolation + standard convolution (see note below).

**When to use:** When the network needs to learn the upsampling operation (e.g., the original FCN, some GAN generators). In modern segmentation architectures, transposed convolutions have largely been replaced by bilinear interpolation followed by convolution, which avoids checkerboard artifacts.

### 4.3 PixelShuffle (Sub-Pixel Convolution)

**Method:** Introduced by Shi et al. (2016) for super-resolution. Rearranges elements from the channel dimension into spatial dimensions. For an upsampling factor $r$, the input has $r^2 \cdot C_{out}$ channels, and PixelShuffle reshapes this to $C_{out}$ channels at $r\times$ higher resolution.

$$\text{PixelShuffle}(\mathbf{F})_{c, rh+i, rw+j} = \mathbf{F}_{c \cdot r^2 + i \cdot r + j,\ h,\ w}$$

where $0 \leq i, j < r$.

In practice, PixelShuffle is preceded by a standard convolution that increases the channel count by $r^2$:

```python
nn.Sequential(
    nn.Conv2d(in_channels, out_channels * r * r, kernel_size=3, padding=1),
    nn.PixelShuffle(r)
)
```

**Properties:**

| Property | Value |
|----------|-------|
| Learnable parameters | Via preceding convolution |
| Computational cost | Moderate (convolution at low resolution + rearrange) |
| Artifacts | Minimal (but periodic patterns possible if not properly initialized) |
| Checkerboard artifacts | Rare (no overlapping contributions) |
| Memory | Moderate (temporary $r^2 \times$ channel expansion) |

**Advantage over transposed convolution:** The convolution operates at the lower resolution, which is computationally cheaper. The rearrangement itself is a zero-cost memory operation.

**When to use:** Popular in super-resolution. Less common in segmentation but used in some architectures (e.g., real-time segmentation networks that need efficient upsampling).

### 4.4 Comparison Table

| Method | Parameters | Speed | Quality | Artifacts | Typical Use in Segmentation |
|--------|:---:|:---:|:---:|:---:|---|
| Bilinear | 0 | Fast | Good (smooth) | None | FPN, DeepLab, SegFormer, most modern decoders |
| Transposed Conv | Many | Medium | Good (learnable) | Checkerboard risk | FCN, U-Net (original), some GANs |
| PixelShuffle | Many (in conv) | Medium | Good | Minimal | Real-time networks, super-resolution branches |
| Nearest Neighbor | 0 | Fastest | Lower (blocky) | Block artifacts | Occasionally in lightweight decoders |

### 4.5 The Modern Consensus

The dominant approach in modern segmentation architectures is:

$$\text{Upsample} = \text{Bilinear Interpolation}(2\times) + \text{Conv}_{3\times3} + \text{BN} + \text{ReLU}$$

This combination provides:
- No checkerboard artifacts (bilinear is smooth).
- Learnable refinement (the convolution corrects interpolation errors).
- Computational efficiency (the convolution operates at the target resolution, but bilinear upsampling is very cheap).

This "resize-conv" approach was popularized by Odena et al. (2016) as a drop-in replacement for transposed convolutions.

---

## 5. Putting It All Together: Architectural Recipes

### Recipe 1: Classic U-Net Style

```
Encoder: ResNet-50 (stages 0-4)
Skip: Concatenation at each resolution
Decoder: [Bilinear 2x -> Conv 3x3 -> BN -> ReLU] x4
Output: Conv 1x1 -> K classes
```

Best for: Medical imaging, small-to-medium datasets, tasks where boundary precision matters.

### Recipe 2: DeepLab v3+ Style

```
Encoder: ResNet-101 with output stride 16 (atrous convolutions in stage 4)
Bottleneck: ASPP module
Skip: One skip from low-level features (stride 4), 1x1 projection
Decoder: Bilinear 4x upsample + concat skip + Conv 3x3 + Bilinear 4x upsample
Output: Conv 1x1 -> K classes
```

Best for: Semantic segmentation on natural images, large datasets.

### Recipe 3: FPN / Mask2Former Style

```
Encoder: Swin Transformer (4 stages)
Multi-scale: FPN with top-down + lateral connections
Decoder: Transformer decoder with masked attention + multi-scale deformable attention
Output: Set of N mask predictions + N class predictions
```

Best for: Universal segmentation (semantic + instance + panoptic), complex scenes.

### Recipe 4: Lightweight / Real-Time

```
Encoder: MobileNetV3 or EfficientNet-B0
Skip: Addition (to save memory)
Decoder: [Bilinear 2x -> Depthwise Separable Conv 3x3] x3
Output: Conv 1x1 -> K classes
```

Best for: Edge deployment, mobile applications, video processing.

---

## 6. Design Principles Summary

1. **Use a pretrained encoder.** ImageNet pretraining provides a strong initialization. For domain-specific tasks (medical, satellite), consider domain-specific pretraining or self-supervised pretraining.

2. **Add skip connections at every resolution level.** The performance gain is consistent and well-documented. Use concatenation when you can afford the parameters; use addition when efficiency matters.

3. **Process multi-scale features.** Whether through FPN, ASPP, or parallel streams (HRNet), exploiting multi-scale information consistently improves segmentation of objects at different sizes.

4. **Prefer bilinear + conv over transposed convolution** for upsampling. It avoids artifacts and is equally expressive.

5. **Match decoder complexity to the task.** A lightweight MLP decoder (SegFormer) can suffice with a powerful transformer encoder. A heavy decoder is needed when the encoder is simple or when boundary precision is critical.

6. **Control output stride.** Lower output stride (8 vs. 32) preserves more spatial detail but increases computation. A common compromise: use output stride 16 in the encoder and upsample $4\times$ in the decoder, with one skip connection from stride-4 features.

---

## References

1. Long, J., Shelhamer, E., & Darrell, T. (2015). Fully Convolutional Networks for Semantic Segmentation. CVPR.
2. Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net. MICCAI.
3. Lin, T.-Y., Dollar, P., Girshick, R., He, K., Hariharan, B., & Belongie, S. (2017). Feature Pyramid Networks for Object Detection. CVPR.
4. Chen, L.-C., Zhu, Y., Papandreou, G., Schroff, F., & Adam, H. (2018). Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation. ECCV.
5. Shi, W., et al. (2016). Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network. CVPR.
6. Odena, A., Dumoulin, V., & Olah, C. (2016). Deconvolution and Checkerboard Artifacts. Distill.
7. Zhou, Z., Siddiquee, M. M. R., Tajbakhsh, N., & Liang, J. (2018). U-Net++: A Nested U-Net Architecture for Medical Image Segmentation. DLMIA.
8. Oktay, O., et al. (2018). Attention U-Net: Learning Where to Look for the Pancreas. MIDL.
9. Kirillov, A., Girshick, R., He, K., & Dollar, P. (2019). Panoptic Feature Pyramid Networks. CVPR.
10. Wang, J., et al. (2020). Deep High-Resolution Representation Learning for Visual Recognition. TPAMI.
