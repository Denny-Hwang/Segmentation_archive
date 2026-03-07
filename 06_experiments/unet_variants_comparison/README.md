---
title: "Experiment: U-Net Variants Comparison"
date: 2025-01-15
status: planned
tags: [experiment, unet, unet++, attention-unet, comparison]
---

# U-Net Variants Comparison Experiment

## Objective

Compare multiple U-Net variants under identical training conditions to measure the impact of
architectural changes (nested skip connections, attention gates, pretrained encoders) on
segmentation quality, computational cost, and convergence behavior.

This experiment is designed to answer the question: **do the added architectural complexities
of U-Net successors translate into meaningful accuracy gains, and at what computational cost?**

---

## Models Compared

| Model | Key Modification | Reference | Param Count (approx.) |
|-------|-----------------|-----------|----------------------|
| U-Net | Baseline encoder-decoder with skip connections | Ronneberger 2015 | ~31M |
| U-Net++ | Nested, dense skip connections with deep supervision | Zhou 2018 | ~36M |
| Attention U-Net | Attention gates on skip connections | Oktay 2018 | ~34M |
| U-Net + ResNet34 encoder | Pretrained ImageNet encoder backbone | segmentation_models_pytorch | ~24M |
| U-Net + EfficientNet-B3 encoder | Pretrained efficient backbone | segmentation_models_pytorch | ~13M |

### Model Descriptions

**U-Net (Baseline).** The original symmetric encoder-decoder with four downsampling stages,
crop-and-concatenate skip connections, and two 3x3 convolutions per block. We use the modern
variant with padded convolutions and batch normalization.

**U-Net++.** Redesigns skip connections as dense blocks. Instead of a single connection from
encoder level `i` to decoder level `i`, U-Net++ introduces intermediate convolutional nodes
at every level, forming a nested structure. Deep supervision is applied to multiple decoder
outputs, allowing pruning at inference time.

**Attention U-Net.** Adds attention gate modules at each skip connection. The attention gates
learn to suppress irrelevant regions in the encoder feature maps while highlighting salient
features, using a gating signal from the decoder. This is especially useful when the target
structure is small relative to the image.

**U-Net + ResNet34.** Replaces the custom encoder with a ResNet34 backbone pretrained on
ImageNet. The decoder remains a standard U-Net decoder. This tests whether transfer learning
from natural images benefits medical segmentation.

**U-Net + EfficientNet-B3.** Uses a more parameter-efficient pretrained encoder to test
whether a smaller model with compound scaling can match or exceed larger variants.

---

## Datasets

We recommend running this comparison on two datasets to test generalizability:

### Primary: ISIC 2018 Skin Lesion Segmentation

- **Task:** Binary segmentation (lesion vs. background)
- **Images:** 2,594 dermoscopy images with ground truth masks
- **Resolution:** Resize to 256x256 for this experiment
- **Split:** 70% train / 15% validation / 15% test (stratified by diagnosis)
- **Source:** [ISIC Archive](https://challenge.isic-archive.com/)

### Secondary: Montgomery County Chest X-Ray (Lung Segmentation)

- **Task:** Binary segmentation (lung fields vs. background)
- **Images:** 138 posterior-anterior chest X-rays
- **Resolution:** Resize to 512x512
- **Split:** 5-fold cross-validation (given the small dataset size)
- **Source:** [NIH/LHCH](https://lhncbc.nlm.nih.gov/LHC-downloads/downloads.html)

### Data Preprocessing Pipeline

```python
import albumentations as A
from albumentations.pytorch import ToTensorV2

train_transform = A.Compose([
    A.Resize(256, 256),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
    A.ElasticTransform(alpha=120, sigma=120 * 0.05, p=0.3),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.3),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

val_transform = A.Compose([
    A.Resize(256, 256),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])
```

---

## Controlled Variables

All models are trained under identical conditions to ensure a fair comparison:

| Variable | Value | Rationale |
|----------|-------|-----------|
| Optimizer | Adam | Widely used, stable convergence |
| Learning rate | 1e-4 | Standard starting point for Adam |
| LR scheduler | CosineAnnealingLR (T_max=100) | Smooth decay, avoids sharp drops |
| Batch size | 8 | Fits all models in ~8 GB GPU memory |
| Loss function | 0.5 * BCE + 0.5 * Dice Loss | Combines pixel-level and region-level supervision |
| Epochs | 100 | Sufficient for convergence on these datasets |
| Early stopping | Patience=15 on validation Dice | Prevents overfitting |
| Random seed | 42 | Reproducibility |
| Input size | 256x256 | Uniform across all models |
| Weight init | He normal (scratch models) / ImageNet (pretrained) | Standard practice |
| Mixed precision | FP16 via torch.cuda.amp | Faster training, lower memory |

---

## Training Protocol

### Step 1: Data Preparation

```bash
python prepare_data.py --dataset isic2018 --output data/isic2018/ --split 70-15-15
```

### Step 2: Training Each Model

```bash
# Train all variants sequentially with identical configs
for model in unet unetplusplus attention_unet unet_resnet34 unet_efficientb3; do
    python train.py \
        --model $model \
        --dataset data/isic2018/ \
        --epochs 100 \
        --batch-size 8 \
        --lr 1e-4 \
        --loss bce_dice \
        --seed 42 \
        --output results/$model/ \
        --wandb-project unet-variants
done
```

### Step 3: Evaluation

```bash
python evaluate.py --results-dir results/ --output analysis/
```

### Step 4: Statistical Testing

```bash
python statistical_tests.py --results-dir results/ --test paired-ttest
```

---

## Evaluation

### Quantitative Metrics

| Metric | Description | Implementation |
|--------|-------------|----------------|
| **Dice Coefficient** | 2*TP / (2*TP + FP + FN). Primary metric for region overlap. | `torchmetrics.Dice` |
| **IoU (Jaccard Index)** | TP / (TP + FP + FN). Stricter than Dice. | `torchmetrics.JaccardIndex` |
| **Pixel Accuracy** | (TP + TN) / Total. Can be misleading with class imbalance. | Manual computation |
| **Hausdorff Distance (95th)** | Maximum boundary error at the 95th percentile. | `surface_distance` library |
| **Sensitivity (Recall)** | TP / (TP + FN). Measures false negative rate. | Manual computation |
| **Specificity** | TN / (TN + FP). Measures false positive rate. | Manual computation |

### Efficiency Metrics

| Metric | How to Measure |
|--------|---------------|
| Parameter count | `sum(p.numel() for p in model.parameters())` |
| FLOPs | `fvcore.nn.FlopCountAnalysis` or `ptflops` |
| GPU memory (peak) | `torch.cuda.max_memory_allocated()` |
| Inference time | Average over 100 forward passes (after warmup) |
| Training time | Wall-clock time for full training run |

### Qualitative Analysis

- Select 10 challenging test images (small objects, ambiguous boundaries, low contrast).
- Generate side-by-side prediction visualizations for all models.
- Create error maps highlighting false positives (red) and false negatives (blue).
- Examine failure modes: does each architecture fail differently?

### Statistical Significance Testing

Run a paired t-test (or Wilcoxon signed-rank test for non-normal distributions) on per-image
Dice scores across test images. Report p-values for each pair of models. A difference is
considered significant at p < 0.05.

---

## Expected Results Table

Based on published literature and typical performance on similar datasets:

| Model | Dice (ISIC 2018) | IoU (ISIC 2018) | Params (M) | Inference (ms) |
|-------|------------------|-----------------|------------|----------------|
| U-Net (scratch) | ~0.87 | ~0.78 | 31.0 | ~15 |
| U-Net++ | ~0.89 | ~0.80 | 36.6 | ~22 |
| Attention U-Net | ~0.88 | ~0.79 | 34.8 | ~19 |
| U-Net + ResNet34 | ~0.90 | ~0.82 | 24.4 | ~12 |
| U-Net + EfficientNet-B3 | ~0.90 | ~0.82 | 13.2 | ~14 |

**Expected conclusions:**

1. Pretrained encoder variants (ResNet34, EfficientNet-B3) should outperform from-scratch
   models, confirming the value of transfer learning even for medical images.
2. U-Net++ and Attention U-Net should show modest improvements over vanilla U-Net, with
   Attention U-Net particularly benefiting on images with small target structures.
3. EfficientNet-B3 encoder should match ResNet34 performance with roughly half the parameters.
4. All models should converge within 60-80 epochs with the cosine annealing schedule.

---

## How to Run

```bash
# Full pipeline: prepare data, train all models, evaluate, and generate report
python run_comparison.py --config config.yaml

# Or run individual steps
python run_comparison.py --config config.yaml --stage prepare
python run_comparison.py --config config.yaml --stage train --model unet
python run_comparison.py --config config.yaml --stage evaluate
python run_comparison.py --config config.yaml --stage report
```

---

## Configuration (config.yaml)

```yaml
experiment:
  name: unet_variants_comparison
  seed: 42
  device: cuda

dataset:
  name: isic2018
  root: data/isic2018/
  image_size: [256, 256]
  train_split: 0.7
  val_split: 0.15
  test_split: 0.15
  num_workers: 4

models:
  - name: unet
    encoder: vanilla
    pretrained: false
  - name: unetplusplus
    encoder: vanilla
    pretrained: false
    deep_supervision: true
  - name: attention_unet
    encoder: vanilla
    pretrained: false
  - name: unet_resnet34
    encoder: resnet34
    pretrained: true
  - name: unet_efficientb3
    encoder: efficientnet-b3
    pretrained: true

training:
  epochs: 100
  batch_size: 8
  optimizer: adam
  lr: 1e-4
  scheduler: cosine
  loss: bce_dice
  early_stopping_patience: 15
  mixed_precision: true

evaluation:
  metrics: [dice, iou, pixel_accuracy, hausdorff95, sensitivity, specificity]
  num_qualitative_samples: 10
  statistical_test: paired_ttest
```

---

## Files

| File | Description |
|------|-------------|
| `config.yaml` | Comparison experiment configuration |
| `run_comparison.py` | Script to train all variants sequentially |
| `train.py` | Single-model training script |
| `evaluate.py` | Compute metrics on test set for all trained models |
| `analyze_results.py` | Results analysis, tables, and visualizations |
| `statistical_tests.py` | Paired statistical significance tests |
| `prepare_data.py` | Download and preprocess dataset |

---

## Troubleshooting

- **Out of memory:** Reduce batch size to 4, or enable gradient accumulation over 2 steps.
- **Slow training:** Ensure mixed precision is enabled and `num_workers >= 4`.
- **Diverging loss:** Try reducing learning rate to 3e-5 or warming up for 5 epochs.
- **Poor U-Net++ convergence:** Deep supervision can be unstable; try weighting auxiliary
  losses at 0.3 instead of 1.0.

---

## References

1. Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional Networks for
   Biomedical Image Segmentation. MICCAI 2015.
2. Zhou, Z., et al. (2018). UNet++: A Nested U-Net Architecture for Medical Image
   Segmentation. DLMIA 2018.
3. Oktay, O., et al. (2018). Attention U-Net: Learning Where to Look for the Pancreas.
   MIDL 2018.
4. Iakubovskii, P. (2019). Segmentation Models PyTorch.
   https://github.com/qubvel/segmentation_models.pytorch
5. Codella, N., et al. (2019). Skin Lesion Analysis Toward Melanoma Detection 2018:
   A Challenge Hosted by the ISIC. arXiv:1902.03368.
