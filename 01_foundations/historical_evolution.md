---
title: "Historical Evolution of Image Segmentation"
date: 2025-03-06
status: in-progress
tags: [segmentation, history, FCN, U-Net, DeepLab, transformer, foundation-models, timeline]
difficulty: beginner
---

# Historical Evolution of Image Segmentation

This document traces the development of image segmentation methods from classical image processing techniques through to modern foundation models. The trajectory can be roughly divided into five eras, each defined by a dominant paradigm shift.

---

## Era 0: Classical Methods (Pre-2012)

Before deep learning, segmentation relied on hand-crafted features and energy minimization.

### Thresholding and Region-Based Methods (1960s--1980s)

- **Otsu's method (1979):** Automatic threshold selection by maximizing inter-class variance. Simple and effective for bimodal histograms but limited to grayscale images with clear foreground/background separation.
- **Region growing:** Start from seed pixels and iteratively merge neighboring pixels that satisfy a similarity criterion. Sensitive to seed placement and noise.
- **Watershed transform (1979, popularized 1990s):** Treat the gradient magnitude image as a topographic surface and flood from local minima. Produces an over-segmentation that can be merged in post-processing.

### Edge-Based and Contour Methods (1970s--2000s)

- **Canny edge detector (1986):** Produces thin, well-localized edges but does not directly produce closed contours or region labels.
- **Active contours / Snakes (Kass, Witkin, Terzopoulos, 1988):** An energy-minimizing spline that deforms to fit object boundaries. The energy has internal terms (smoothness) and external terms (image gradient). Limited by initialization sensitivity and difficulty with topological changes.
- **Level sets (Osher & Sethian, 1988; applied to segmentation by Caselles et al., 1993):** Represent contours implicitly as the zero level set of a higher-dimensional function. Naturally handle topological changes (splitting, merging). Computationally expensive.

### Graphical Models and Energy Minimization (2000s)

- **Markov Random Fields (MRFs) and Conditional Random Fields (CRFs):** Model spatial dependencies between pixel labels. The unary potential captures per-pixel evidence; the pairwise potential encourages label smoothness.
- **Graph cuts (Boykov & Jolly, 2001):** Formulate binary segmentation as a minimum cut / maximum flow problem on a graph. Extended to multi-label with alpha-expansion and alpha-beta swap moves. GrabCut (Rother et al., 2004) combined graph cuts with Gaussian Mixture Models for interactive foreground extraction.
- **Mean shift segmentation (Comaniciu & Meer, 2002):** Non-parametric clustering in the joint spatial-color space. Produces superpixels without requiring a predefined number of clusters.
- **SLIC superpixels (Achanta et al., 2012):** Simple Linear Iterative Clustering. Fast, compact superpixels that became a standard preprocessing step.

### Feature Engineering Era

- **Textons (Malik et al., 2001):** Cluster filter bank responses to create a texture vocabulary. Assign each pixel a texton label and use it as a feature for classification.
- **Deformable Part Models and HOG-based methods:** Object detection pipelines that could be extended to rough segmentation via figure-ground masks.
- **Random Forest classifiers on hand-crafted features:** Used extensively in medical image segmentation and remote sensing before deep learning.

**Limitations of classical methods:** Required task-specific feature engineering, struggled with semantic understanding, and generalized poorly across domains.

---

## Era 1: Fully Convolutional Networks (2014--2016)

### The FCN Revolution

**Paper:** Long, Shelhamer, & Darrell, *Fully Convolutional Networks for Semantic Segmentation* (CVPR 2015; arXiv 2014).

**Key insight:** Classification CNNs (AlexNet, VGGNet, GoogLeNet) can be repurposed for dense prediction by replacing fully connected layers with convolutional layers. The resulting network accepts arbitrary input sizes and produces spatial output maps.

**Architecture:**

1. Take a classification network (e.g., VGG-16) pretrained on ImageNet.
2. Replace the fully connected layers with $1 \times 1$ convolutions that output $K$ channels (one per class).
3. Upsample the coarse output to full resolution using learned deconvolution (transposed convolution) or bilinear interpolation.

**Skip connections:** FCN-32s (upsample 32x from the last layer) produced very coarse results. FCN-16s and FCN-8s added skip connections from earlier layers (pool4, pool3) to recover spatial detail:

$$\text{FCN-8s output} = \text{Upsample}_{2\times}(\text{Upsample}_{2\times}(\text{conv7}) + \text{pool4}) + \text{pool3}$$

**Impact:** Established the encoder-decoder paradigm that still dominates today. Showed that end-to-end training with per-pixel cross-entropy loss was viable. PASCAL VOC mIoU jumped from ~40% (pre-deep-learning) to ~62%.

### Concurrent and Follow-Up Work

- **SegNet (Badrinarayanan et al., 2015):** Encoder-decoder with unpooling layers that reuse max-pooling indices from the encoder, reducing parameters.
- **ParseNet (Liu et al., 2015):** Added global context features via global average pooling concatenated with local features.

---

## Era 2: U-Net and Specialized Architectures (2015--2017)

### U-Net

**Paper:** Ronneberger, Fischer, & Brox, *U-Net: Convolutional Networks for Biomedical Image Segmentation* (MICCAI 2015).

**Architecture:** A symmetric encoder-decoder with dense skip connections at every resolution level. The encoder contracts the spatial dimensions while increasing feature channels; the decoder expands back. Skip connections concatenate (not add) encoder features to decoder features at corresponding resolutions.

```
Encoder:           Decoder:
[572x572, 64]  --> copy & crop --> [388x388, 128]
[280x280, 128] --> copy & crop --> [196x196, 256]
[136x136, 256] --> copy & crop --> [104x104, 512]
[64x64, 512]   --> copy & crop --> [56x56, 1024]
                   [28x28, 1024] (bottleneck)
```

**Key contributions:**

- Demonstrated that segmentation networks could be trained effectively with very few annotated images (as few as 30) using extensive data augmentation (elastic deformations).
- The symmetric architecture with concatenation-based skip connections became the template for biomedical and general-purpose segmentation.
- Introduced overlap-tile strategy for seamless segmentation of large images.

**Legacy:** U-Net spawned a vast family of variants -- V-Net (3D medical), Attention U-Net, U-Net++, U-Net 3+, nnU-Net (self-configuring). As of 2025, U-Net variants remain dominant in medical image segmentation.

### Other Notable Architectures (2015--2017)

- **Dilated Convolutions / a trous convolutions (Yu & Koltun, 2016):** Increase receptive field without reducing spatial resolution by inserting gaps ("holes") in the convolution kernel. Became a core component of DeepLab.
- **PSPNet (Zhao et al., 2017):** Pyramid Pooling Module captures multi-scale context by pooling at multiple grid scales (1x1, 2x2, 3x3, 6x6) and concatenating upsampled features. Won the ImageNet Scene Parsing Challenge 2016.
- **RefineNet (Lin et al., 2017):** Multi-path refinement network that exploits features at all levels through long-range residual connections.

---

## Era 3: The DeepLab Family (2015--2018)

### DeepLab v1 (2015)

**Paper:** Chen, Papandreou, Kokkinos, Murphy, & Yuille, *Semantic Image Segmentation with Deep Convolutional Nets and Fully Connected CRFs* (ICLR 2015).

**Key ideas:**

- Used atrous (dilated) convolution to maintain spatial resolution in the deeper layers of VGG-16.
- Applied a dense CRF as a post-processing step to refine coarse CNN outputs. The CRF models long-range pairwise potentials based on pixel color and position:

$$E(\mathbf{x}) = \sum_i \psi_u(x_i) + \sum_{i < j} \psi_p(x_i, x_j)$$

where $\psi_p$ is a Gaussian kernel over color and position differences.

### DeepLab v2 (2017)

**Key addition: Atrous Spatial Pyramid Pooling (ASPP).** Apply multiple parallel atrous convolutions with different dilation rates (e.g., 6, 12, 18, 24) to capture multi-scale context. Concatenate the resulting feature maps.

$$\text{ASPP}(\mathbf{F}) = \text{Concat}\big[\text{Conv}_{r=6}(\mathbf{F}),\ \text{Conv}_{r=12}(\mathbf{F}),\ \text{Conv}_{r=18}(\mathbf{F}),\ \text{Conv}_{r=24}(\mathbf{F})\big]$$

Used ResNet-101 as the backbone. Achieved 79.7% mIoU on PASCAL VOC 2012.

### DeepLab v3 (2017)

- Improved ASPP with batch normalization and a global average pooling branch (image-level feature).
- Removed the CRF post-processing -- the CNN alone was sufficient.
- Employed output stride of 8 or 16 with multi-grid atrous convolutions in the last ResNet blocks.

### DeepLab v3+ (2018)

- Added a simple decoder module to recover object boundaries (rather than relying solely on bilinear upsampling).
- Used modified Xception as backbone with depthwise separable convolutions for efficiency.
- Achieved 89.0% mIoU on PASCAL VOC 2012 (with extra data and COCO pretraining).

**Legacy:** The DeepLab series established atrous convolution and ASPP as standard tools. The insight that multi-scale context is critical for segmentation influenced virtually all subsequent work.

---

## Era 4: Transformers Enter Segmentation (2020--2023)

### Vision Transformers (ViT) as Backbones

**ViT (Dosovitskiy et al., 2020)** demonstrated that pure transformer architectures could match or exceed CNNs on image classification. Key properties relevant to segmentation:

- Global self-attention captures long-range dependencies that CNNs can only achieve with very deep networks or large dilations.
- Patch-based tokenization: split the image into $16 \times 16$ patches, linearly embed each, and process with standard transformer layers.

### SETR -- Rethinking Semantic Segmentation (2021)

**Paper:** Zheng et al., *Rethinking Semantic Segmentation from a Sequence-to-Sequence Perspective with Transformers* (CVPR 2021).

First work to use a pure ViT encoder for semantic segmentation. Three decoder variants: naive upsampling (SETR-Naive), progressive upsampling (SETR-PUP), and multi-level aggregation (SETR-MLA).

### Segmenter (2021)

Pure transformer encoder-decoder for semantic segmentation. The decoder is a mask transformer that directly produces class-specific segmentation masks from encoded patch tokens.

### SegFormer (2021)

**Paper:** Xie et al., *SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers* (NeurIPS 2021).

**Key innovations:**

- Hierarchical transformer encoder (Mix Transformer, MiT) that produces multi-scale features, unlike ViT's single-scale output.
- Lightweight MLP decoder that aggregates multi-scale features -- no complex decoder needed.
- Efficient self-attention using spatial reduction (downsample keys and values).
- Achieved competitive results with significantly fewer FLOPs than prior transformer-based methods.

### Mask2Former -- Unified Segmentation (2022)

**Paper:** Cheng et al., *Masked-attention Mask Transformer for Universal Image Segmentation* (CVPR 2022).

**Key insight:** Semantic, instance, and panoptic segmentation can all be formulated as mask classification: predict a set of $N$ binary masks and assign each a class label (or "no object").

$$\text{Output} = \{(m_i, p_i)\}_{i=1}^{N}, \quad m_i \in [0,1]^{H \times W}, \quad p_i \in \Delta^{K+1}$$

**Architecture:**

- Backbone (Swin Transformer or ResNet) produces multi-scale features.
- Pixel decoder (with deformable attention) refines multi-scale features.
- Transformer decoder with masked cross-attention: each query only attends to the spatial region predicted by its mask from the previous layer.
- Trained with Hungarian matching loss (bipartite matching between predictions and ground truth).

**Impact:** Became the new de facto architecture for panoptic segmentation and achieved state-of-the-art across all three task types on COCO, ADE20K, and Cityscapes with a single architecture.

### Other Key Transformer Works

| Model | Year | Contribution |
|-------|------|-------------|
| Swin Transformer | 2021 | Hierarchical vision transformer with shifted windows; became dominant backbone |
| MaskFormer | 2021 | Precursor to Mask2Former; showed mask classification unifies semantic and panoptic |
| OneFormer | 2023 | Multi-task universal segmentation with task-conditioned queries |
| Mask DINO | 2023 | Unified detection and segmentation with DINO-style training |

---

## Era 5: Foundation Models (2023--Present)

### Segment Anything Model (SAM)

**Paper:** Kirillov et al., *Segment Anything* (ICCV 2023).

**Paradigm shift:** Instead of training task-specific models on task-specific datasets, train a single *promptable* model on an enormous dataset (SA-1B: 11M images, 1.1B masks) that can segment *anything* given a prompt.

**Architecture:**

- **Image encoder:** ViT-H (632M parameters), run once per image.
- **Prompt encoder:** Encodes points, boxes, masks, or text into prompt tokens.
- **Mask decoder:** Lightweight transformer decoder that combines image embeddings and prompt tokens to produce masks. Runs in ~50ms, enabling interactive use.

**Training:** Iterative data engine -- model-assisted annotation produced increasingly diverse and high-quality masks over three stages.

**Capabilities:** Zero-shot transfer to unseen tasks and domains. Strong out-of-the-box performance on diverse segmentation benchmarks without task-specific fine-tuning.

### SAM 2 (2024)

Extended SAM to video with a streaming architecture that processes frames sequentially, maintaining a memory bank of past predictions. Handles both image and video segmentation with a unified model. Trained on SA-V dataset (50.9K videos, 642.6K masklets).

### SEEM -- Segment Everything Everywhere All at Once (2023)

A unified model that accepts diverse prompts (point, box, text, scribble, referring expression, mask of another image) and can perform interactive, referring, and open-vocabulary segmentation.

### Grounded SAM and Variants (2023--2024)

Combine open-vocabulary object detection (Grounding DINO) with SAM to achieve open-vocabulary instance and panoptic segmentation. The detector proposes bounding boxes from text queries, and SAM segments within each box.

### EfficientSAM, FastSAM, MobileSAM (2023--2024)

Distilled or architecturally simplified versions of SAM for edge deployment. Trade some accuracy for significantly reduced latency and model size.

### Florence, UNINEXT, X-Decoder (2023)

Generalist models that handle segmentation alongside other vision tasks (detection, captioning, VQA) through shared architectures and multi-task training.

---

## Timeline Summary

```
1979  Otsu's Thresholding
1986  Canny Edge Detector
1988  Active Contours (Snakes)
2001  Graph Cuts (Boykov & Jolly)
2004  GrabCut
2012  SLIC Superpixels / AlexNet (ImageNet breakthrough)
2014  FCN (arXiv) -- deep learning enters segmentation
2015  U-Net, SegNet, DeepLab v1, ParseNet
2016  Dilated Convolutions, PSPNet
2017  DeepLab v2/v3, Mask R-CNN (instance seg.)
2018  DeepLab v3+, PANet
2019  Panoptic Segmentation defined, Panoptic FPN
2020  ViT, DETR (detection transformers)
2021  SETR, SegFormer, Swin Transformer, MaskFormer
2022  Mask2Former (universal segmentation)
2023  SAM, SEEM, Grounded SAM, OneFormer
2024  SAM 2 (video), EfficientSAM, SAM-HQ
2025  Open-vocabulary foundation models, unified architectures
```

---

## Key Takeaways

1. **From pixels to semantics:** Early methods operated on low-level features (edges, color). Deep learning introduced semantic understanding.
2. **From task-specific to universal:** The field has moved from specialized architectures per task toward unified models that handle semantic, instance, and panoptic segmentation.
3. **From closed-set to open-vocabulary:** Fixed class sets are giving way to models that can segment any concept described in natural language.
4. **From supervised to prompted:** Foundation models shift the paradigm from "train on your specific dataset" to "prompt a general model."
5. **Scale matters:** The most impactful recent advances (SAM, SAM 2) were enabled by massive datasets and large models, following the scaling paradigm established in NLP.

---

## References

1. Long, J., Shelhamer, E., & Darrell, T. (2015). Fully Convolutional Networks for Semantic Segmentation. CVPR.
2. Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net. MICCAI.
3. Chen, L.-C., et al. (2015--2018). DeepLab v1--v3+. ICLR / ECCV / arXiv.
4. Zhao, H., et al. (2017). Pyramid Scene Parsing Network. CVPR.
5. He, K., Gkioxari, G., Dollar, P., & Girshick, R. (2017). Mask R-CNN. ICCV.
6. Dosovitskiy, A., et al. (2020). An Image is Worth 16x16 Words. ICLR 2021.
7. Cheng, B., Misra, I., Schwing, A. G., Kirillov, A., & Girshick, R. (2022). Masked-attention Mask Transformer. CVPR.
8. Kirillov, A., et al. (2023). Segment Anything. ICCV.
9. Ravi, N., et al. (2024). SAM 2: Segment Anything in Images and Videos. arXiv.
