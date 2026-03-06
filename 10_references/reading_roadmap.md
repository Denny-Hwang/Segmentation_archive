# Paper Reading Roadmap

A structured guide for learning image segmentation, from beginner to cutting-edge research.

## Level 1: Foundations (Beginner)

Start here if you are new to deep learning and segmentation.

### Prerequisites
- Basic understanding of CNNs (convolution, pooling, activation functions)
- Familiarity with PyTorch or TensorFlow basics

### Essential Reads

1. **Fully Convolutional Networks for Semantic Segmentation** (Long et al., 2015)
   - *Why*: The seminal work that introduced end-to-end pixel-wise prediction with CNNs
   - arXiv: [1411.4038](https://arxiv.org/abs/1411.4038)

2. **U-Net: Convolutional Networks for Biomedical Image Segmentation** (Ronneberger et al., 2015)
   - *Why*: Introduced skip connections and encoder-decoder architecture; foundation for most medical segmentation
   - arXiv: [1505.04597](https://arxiv.org/abs/1505.04597)

3. **DeepLab: Semantic Image Segmentation with Deep Convolutional Nets and Fully Connected CRFs** (Chen et al., 2014)
   - *Why*: Introduced atrous (dilated) convolutions and CRF post-processing
   - arXiv: [1412.7062](https://arxiv.org/abs/1412.7062)

### Practice
- Implement U-Net from scratch in PyTorch
- Train on a small dataset (e.g., Kvasir-SEG or PASCAL VOC subset)
- Understand Dice coefficient and IoU metrics

---

## Level 2: Core Architectures (Intermediate)

Build a deeper understanding of the architectural design space.

### U-Net Family

4. **UNet++: A Nested U-Net Architecture** (Zhou et al., 2018)
   - *Why*: Dense skip connections and deep supervision
   - arXiv: [1807.10165](https://arxiv.org/abs/1807.10165)

5. **Attention U-Net: Learning Where to Look for the Pancreas** (Oktay et al., 2018)
   - *Why*: Attention gating for skip connections
   - arXiv: [1804.03999](https://arxiv.org/abs/1804.03999)

6. **3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation** (Cicek et al., 2016)
   - *Why*: Extension to 3D volumetric data
   - arXiv: [1606.06650](https://arxiv.org/abs/1606.06650)

### Scene Parsing

7. **Pyramid Scene Parsing Network (PSPNet)** (Zhao et al., 2017)
   - *Why*: Multi-scale context aggregation via pyramid pooling
   - arXiv: [1612.01105](https://arxiv.org/abs/1612.01105)

8. **Encoder-Decoder with Atrous Separable Convolution (DeepLab v3+)** (Chen et al., 2018)
   - *Why*: State-of-the-art atrous spatial pyramid pooling with decoder
   - arXiv: [1802.02611](https://arxiv.org/abs/1802.02611)

### Self-Configuring

9. **nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation** (Isensee et al., 2021)
   - *Why*: Demonstrates that engineering matters as much as architecture novelty
   - arXiv: [1809.10486](https://arxiv.org/abs/1809.10486)

### Practice
- Compare U-Net vs. UNet++ vs. Attention U-Net on the same dataset
- Experiment with different loss functions (CE, Dice, Focal)
- Train on Synapse or ACDC datasets

---

## Level 3: Transformers and Modern Methods (Advanced)

Understand how transformers revolutionized segmentation.

### Transformer Foundations

10. **An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale (ViT)** (Dosovitskiy et al., 2021)
    - *Why*: Foundation for all vision transformers
    - arXiv: [2010.11929](https://arxiv.org/abs/2010.11929)

11. **Swin Transformer: Hierarchical Vision Transformer using Shifted Windows** (Liu et al., 2021)
    - *Why*: Efficient hierarchical transformer serving as backbone for segmentation
    - arXiv: [2103.14030](https://arxiv.org/abs/2103.14030)

### Transformer Segmentation

12. **TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation** (Chen et al., 2021)
    - *Why*: First successful CNN-Transformer hybrid for medical segmentation
    - arXiv: [2102.04306](https://arxiv.org/abs/2102.04306)

13. **Swin-Unet: Unet-like Pure Transformer for Medical Image Segmentation** (Cao et al., 2021)
    - *Why*: Pure transformer encoder-decoder for medical segmentation
    - arXiv: [2105.05537](https://arxiv.org/abs/2105.05537)

14. **SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers** (Xie et al., 2021)
    - *Why*: Lightweight and effective transformer segmentation without positional encoding
    - arXiv: [2105.15203](https://arxiv.org/abs/2105.15203)

### Universal Segmentation

15. **Masked-attention Mask Transformer for Universal Image Segmentation (Mask2Former)** (Cheng et al., 2022)
    - *Why*: Unified architecture for semantic, instance, and panoptic segmentation
    - arXiv: [2112.01527](https://arxiv.org/abs/2112.01527)

16. **OneFormer: One Transformer to Rule Universal Image Segmentation** (Jain et al., 2023)
    - *Why*: Single multi-task model trained once for all segmentation tasks
    - arXiv: [2211.06220](https://arxiv.org/abs/2211.06220)

### Practice
- Fine-tune a pretrained SegFormer on a custom dataset
- Compare CNN-based vs. Transformer-based on Synapse benchmark
- Implement masked attention from Mask2Former

---

## Level 4: Cutting-Edge and Foundation Models

Explore the latest paradigm shifts in segmentation.

### Foundation Models

17. **Segment Anything (SAM)** (Kirillov et al., 2023)
    - *Why*: Paradigm shift to promptable, zero-shot segmentation with foundation models
    - arXiv: [2304.02643](https://arxiv.org/abs/2304.02643)

18. **SAM 2: Segment Anything in Images and Videos** (Ravi et al., 2024)
    - *Why*: Extension to video with memory-based tracking and streaming architecture
    - arXiv: [2408.00714](https://arxiv.org/abs/2408.00714)

### Adaptations and Efficiency

19. **EfficientSAM** (Xiong et al., 2024)
    - *Why*: Knowledge distillation for efficient SAM variants
    - arXiv: [2312.00863](https://arxiv.org/abs/2312.00863)

20. **Medical SAM Adapter** (Wu et al., 2023)
    - *Why*: Adapting foundation models to medical imaging with parameter-efficient fine-tuning
    - arXiv: [2304.12620](https://arxiv.org/abs/2304.12620)

### Practice
- Use SAM 2 for interactive video segmentation
- Adapt SAM to a medical domain using LoRA or adapter tuning
- Benchmark zero-shot SAM performance vs. supervised methods

---

## Suggested Reading Order

```
Level 1 (2-3 weeks)
  FCN -> U-Net -> DeepLab

Level 2 (3-4 weeks)
  UNet++ -> Attention U-Net -> 3D U-Net -> PSPNet -> DeepLab v3+ -> nnU-Net

Level 3 (4-6 weeks)
  ViT -> Swin Transformer -> TransUNet -> Swin-Unet -> SegFormer -> Mask2Former -> OneFormer

Level 4 (ongoing)
  SAM -> SAM 2 -> EfficientSAM -> Domain adaptations
```
