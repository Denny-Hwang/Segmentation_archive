---
title: "Experiment: Transformer vs CNN Segmentation"
date: 2025-01-15
status: planned
tags: [experiment, transformer, cnn, segformer, deeplabv3, comparison]
---

# Transformer vs CNN Segmentation Experiment

## Objective

Compare transformer-based and CNN-based segmentation architectures to understand their
relative strengths, weaknesses, and computational trade-offs. This experiment investigates
three core hypotheses about how architectural inductive biases (locality in CNNs vs. global
attention in transformers) affect segmentation quality across different object scales, boundary
precision, and data efficiency regimes.

---

## Models Compared

| Model | Type | Encoder | Decoder | Params (approx.) |
|-------|------|---------|---------|------------------|
| DeepLabV3+ | CNN | ResNet-50 | ASPP + lightweight decoder | ~26M |
| HRNet-W48 + OCR | CNN | HRNet-W48 | OCR (Object-Contextual Repr.) | ~71M |
| SegFormer-B2 | Transformer | Mix Transformer (MiT-B2) | Lightweight MLP decoder | ~25M |
| SegFormer-B4 | Transformer | Mix Transformer (MiT-B4) | Lightweight MLP decoder | ~64M |
| TransUNet | Hybrid | CNN (ResNet-50) + ViT-Base | CNN decoder with skip connections | ~105M |
| Swin-UNet | Transformer | Swin Transformer (Tiny) | Swin Transformer decoder | ~27M |

### Model Selection Rationale

**DeepLabV3+** represents the mature CNN paradigm: atrous spatial pyramid pooling captures
multi-scale context, while the encoder-decoder structure recovers spatial detail. It is a
strong, well-understood baseline.

**HRNet-W48 + OCR** maintains high-resolution representations throughout the network rather
than recovering them from low-resolution features. Combined with Object-Contextual
Representations, it represents the upper end of CNN performance.

**SegFormer** is a purpose-built transformer for segmentation. Its Mix Transformer encoder
uses overlapping patch embeddings and efficient self-attention, while the MLP decoder is
deliberately simple, relying on the encoder's multi-scale features. The B2 and B4 variants
test scaling behavior.

**TransUNet** is a hybrid that uses a CNN to extract local features, passes them through a
ViT to capture global context, and then decodes with CNN layers and skip connections. It tests
whether combining both paradigms yields the best of both worlds.

**Swin-UNet** is a pure transformer U-shaped architecture using shifted window attention for
computational efficiency. It tests whether transformers can fully replace CNNs in the
encoder-decoder paradigm.

---

## Datasets

### Primary: ADE20K (150 classes)

- **Task:** Multi-class semantic segmentation (150 classes)
- **Train/Val:** 20,210 / 2,000 images
- **Resolution:** Resize shorter side to 512, random crop to 512x512
- **Why:** Large-scale, diverse, challenging multi-class benchmark. Standard for comparing
  architectures.
- **Source:** [MIT CSAIL ADE20K](http://sceneparsing.csail.mit.edu/)

### Secondary: Synapse Multi-Organ CT (8 classes)

- **Task:** Multi-class medical segmentation (8 abdominal organs)
- **Train/Val:** 18 / 12 volumes (2D slices extracted)
- **Resolution:** 224x224 (following TransUNet protocol)
- **Why:** Medical domain with small, irregularly shaped objects. Tests generalization beyond
  natural images.
- **Source:** [Synapse Multi-Organ](https://www.synapse.org/#!Synapse:syn3193805)

### Data Augmentation

```python
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ADE20K augmentations
ade20k_train_transform = A.Compose([
    A.RandomResizedCrop(512, 512, scale=(0.5, 2.0)),
    A.HorizontalFlip(p=0.5),
    A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.5),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

# Synapse augmentations (more conservative for medical data)
synapse_train_transform = A.Compose([
    A.Resize(224, 224),
    A.HorizontalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=10, p=0.3),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])
```

---

## Hypotheses

1. **Transformers capture global context better**, improving segmentation of large, amorphous
   regions (e.g., sky, walls) where long-range dependencies matter.
2. **CNNs preserve local detail better**, resulting in crisper boundaries and better
   performance on small, thin objects (e.g., poles, fences).
3. **Hybrid models (TransUNet) offer a balanced trade-off**, combining local feature
   extraction with global reasoning.
4. **Transformers are more data-hungry**, performing worse than CNNs on the smaller Synapse
   dataset unless pretrained on ImageNet-21K.
5. **SegFormer is more parameter-efficient than traditional ViT-based models**, achieving
   comparable accuracy with fewer parameters and faster inference.

---

## Training Configuration

### Common Settings

| Parameter | Value |
|-----------|-------|
| Input resolution | 512x512 (ADE20K) / 224x224 (Synapse) |
| Batch size | 8 (ADE20K) / 16 (Synapse) |
| Optimizer | AdamW (weight decay = 0.01) |
| Learning rate | 6e-5 (transformers) / 1e-4 (CNNs) |
| LR schedule | Polynomial decay (power=1.0) with linear warmup (1500 iters) |
| Training iterations | 160K (ADE20K) / 20K (Synapse) |
| Loss function | Cross-entropy (ADE20K) / CE + Dice (Synapse) |
| Mixed precision | FP16 |
| Random seed | 42 |

### Per-Model Specifics

| Model | Pretrained Weights | Special Settings |
|-------|-------------------|------------------|
| DeepLabV3+ | ResNet-50 (ImageNet-1K) | Output stride = 16, ASPP rates = [6, 12, 18] |
| HRNet-W48 + OCR | HRNet-W48 (ImageNet-1K) | OCR mid-channels = 512 |
| SegFormer-B2 | MiT-B2 (ImageNet-1K) | Stochastic depth rate = 0.1 |
| SegFormer-B4 | MiT-B4 (ImageNet-1K) | Stochastic depth rate = 0.2 |
| TransUNet | ResNet-50 + ViT-B/16 (ImageNet-21K) | ViT patch size = 16, hidden dim = 768 |
| Swin-UNet | Swin-Tiny (ImageNet-1K) | Window size = 7, patch size = 4 |

---

## Evaluation Plan

### Accuracy Metrics

| Metric | Description |
|--------|-------------|
| mIoU | Mean Intersection-over-Union across all classes. Primary metric. |
| Per-class IoU | IoU for each class, to identify class-specific strengths. |
| Dice coefficient | Reported for Synapse dataset (following convention). |
| Pixel accuracy | Overall and per-class. |
| Boundary F1 | F1 score computed only on pixels within 2px of ground truth boundaries. |

### Scale-Dependent Analysis

Partition test images (or objects) by scale and compute mIoU per scale bucket:

| Scale Bucket | Object Area (% of image) | Purpose |
|-------------|-------------------------|---------|
| Small | < 1% | Tests fine-grained detection |
| Medium | 1% - 10% | Tests general segmentation |
| Large | > 10% | Tests global understanding |

This analysis directly tests Hypotheses 1 and 2.

### Efficiency Metrics

| Metric | Measurement Method |
|--------|-------------------|
| Parameters | `sum(p.numel() for p in model.parameters())` |
| FLOPs | `fvcore.nn.FlopCountAnalysis` at target resolution |
| GPU memory (peak) | `torch.cuda.max_memory_allocated()` during training |
| Inference speed (FPS) | Average over 500 forward passes on a single V100 |
| Throughput (images/sec) | Batch inference at optimal batch size |

### Robustness Tests

1. **Resolution sensitivity.** Evaluate each trained model at 0.5x, 0.75x, 1.0x, 1.25x, and
   1.5x the training resolution without retraining. Transformers should degrade more
   gracefully due to position embedding interpolation.
2. **Corruption robustness.** Apply Gaussian noise, blur, and contrast shifts to test images.
   Report mIoU degradation.
3. **Data efficiency.** Train on 10%, 25%, 50%, and 100% of training data. Plot mIoU vs.
   dataset size for each model.

---

## Expected Results

### ADE20K mIoU (approximate, based on published results)

| Model | mIoU | Params (M) | FLOPs (G) | FPS |
|-------|------|-----------|-----------|-----|
| DeepLabV3+ (R50) | ~44.0 | 26 | 177 | ~28 |
| HRNet-W48 + OCR | ~45.5 | 71 | 165 | ~14 |
| SegFormer-B2 | ~46.5 | 25 | 62 | ~30 |
| SegFormer-B4 | ~49.0 | 64 | 96 | ~18 |
| TransUNet | ~45.0 | 105 | 198 | ~10 |
| Swin-UNet | ~44.5 | 27 | 90 | ~22 |

### Synapse Multi-Organ Dice (approximate)

| Model | Average Dice | Aorta | Gallbladder | Kidney L | Kidney R |
|-------|-------------|-------|-------------|----------|----------|
| DeepLabV3+ | ~74.0 | ~85 | ~60 | ~80 | ~78 |
| SegFormer-B2 | ~76.0 | ~86 | ~63 | ~82 | ~80 |
| TransUNet | ~77.5 | ~87 | ~64 | ~82 | ~81 |
| Swin-UNet | ~75.0 | ~84 | ~61 | ~80 | ~79 |

---

## How to Run

```bash
# Train all models on ADE20K
for model in deeplabv3plus hrnet_ocr segformer_b2 segformer_b4 transunet swin_unet; do
    python train.py \
        --model $model \
        --dataset ade20k \
        --config configs/${model}_ade20k.yaml \
        --output results/ade20k/${model}/ \
        --seed 42
done

# Evaluate all models
python evaluate.py --results-dir results/ade20k/ --metrics mIoU boundary_f1 --output analysis/

# Scale-dependent analysis
python scale_analysis.py --results-dir results/ade20k/ --output analysis/scale/

# Resolution robustness test
python resolution_test.py --results-dir results/ade20k/ --scales 0.5 0.75 1.0 1.25 1.5

# Data efficiency curves
python data_efficiency.py --model segformer_b2 deeplabv3plus --fractions 0.1 0.25 0.5 1.0

# Generate comparison visualizations
python visualize_comparison.py --results-dir results/ade20k/ --num-samples 20
```

---

## Expected Timeline

| Phase | Duration | Notes |
|-------|----------|-------|
| Data preparation | 0.5 day | Download ADE20K, Synapse; preprocess |
| Training (ADE20K, 6 models) | 3-4 days | ~12-16 hrs per model on a single V100 |
| Training (Synapse, 6 models) | 1 day | Much smaller dataset |
| Evaluation + analysis | 1 day | Metrics, visualizations, robustness tests |
| Report writing | 0.5 day | Compile results and conclusions |
| **Total** | **~6-7 days** | With a single V100 GPU |

---

## Files

| File | Description |
|------|-------------|
| `configs/` | Per-model YAML configuration files |
| `train.py` | Unified training script for all models |
| `evaluate.py` | Compute metrics on test set |
| `scale_analysis.py` | Scale-dependent mIoU analysis |
| `resolution_test.py` | Multi-resolution robustness evaluation |
| `data_efficiency.py` | Data efficiency curve generation |
| `visualize_comparison.py` | Qualitative prediction comparison grids |

---

## References

1. Chen, L.-C., et al. (2018). Encoder-Decoder with Atrous Separable Convolution for
   Semantic Image Segmentation (DeepLabV3+). ECCV 2018.
2. Wang, J., et al. (2020). Deep High-Resolution Representation Learning for Visual
   Recognition (HRNet). TPAMI 2020.
3. Xie, E., et al. (2021). SegFormer: Simple and Efficient Design for Semantic Segmentation
   with Transformers. NeurIPS 2021.
4. Chen, J., et al. (2021). TransUNet: Transformers Make Strong Encoders for Medical Image
   Segmentation. arXiv:2102.04306.
5. Cao, H., et al. (2022). Swin-UNet: Unet-like Pure Transformer for Medical Image
   Segmentation. ECCV 2022 Workshops.
6. Yuan, Y., et al. (2020). Object-Contextual Representations for Semantic Segmentation
   (OCR). ECCV 2020.
7. Zhou, B., et al. (2019). Semantic Understanding of Scenes through the ADE20K Dataset.
   IJCV 2019.
