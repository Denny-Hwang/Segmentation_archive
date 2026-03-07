---
title: "Cross-Repository Tricks and Gotchas"
date: 2025-01-15
status: planned
tags: [cross-repo, tricks, gotchas, debugging, performance]
---

# Tricks and Gotchas

## Purpose

This document collects subtle implementation details, common pitfalls, and non-obvious techniques discovered during code analysis. These are things not typically mentioned in papers or documentation.

## Numerical Stability

### Loss Function Stability

**Dice loss epsilon**: Every Dice loss implementation needs a small epsilon in the denominator to avoid division by zero when a class has no pixels in the batch:

```python
# Correct:
dice = (2 * intersection + eps) / (sum_pred + sum_target + eps)  # eps=1e-6 or 1e-7

# Wrong (division by zero when class absent):
dice = (2 * intersection) / (sum_pred + sum_target)
```

nnU-Net uses `smooth=1e-5` in the numerator and denominator. Some implementations add epsilon only to the denominator, while others add it to both (the "Laplace smoothing" variant). The difference is negligible for non-empty classes but matters when a class has zero pixels.

**Log-softmax vs softmax + log**: Always use `F.log_softmax()` or `nn.CrossEntropyLoss` (which combines log_softmax + NLL internally) rather than computing `torch.log(F.softmax(x))`. The latter is numerically unstable when softmax outputs are near zero:

```python
# Numerically stable (used by all repos):
loss = F.cross_entropy(logits, targets)  # log-sum-exp trick internally

# Numerically unstable:
probs = F.softmax(logits, dim=1)
loss = F.nll_loss(torch.log(probs + 1e-8), targets)  # Needs epsilon hack
```

### Gradient Issues

**Gradient clipping** is used by Pytorch-UNet (`clip_grad_norm_(params, 1.0)`) and nnU-Net (`clip_grad_norm_(params, 12.0)`). Without clipping, deep networks with Dice loss can produce exploding gradients, especially early in training when predictions are poor and the loss landscape is steep.

**Deep supervision** (nnU-Net) helps gradient flow by injecting loss signals at intermediate decoder levels, preventing the vanishing gradient problem in very deep networks (6+ encoder stages).

**LeakyReLU** (nnU-Net) instead of ReLU prevents "dead neurons" that can accumulate during training with aggressive learning rates.

## Data Pipeline Gotchas

### Mask Encoding

**Integer vs one-hot encoding**: Loss functions expect different formats:
- `nn.CrossEntropyLoss`: Integer-encoded targets of shape `(B, H, W)` with values in `[0, n_classes-1]`
- `nn.BCEWithLogitsLoss`: Float targets of shape `(B, n_classes, H, W)` (one-hot)
- Dice loss implementations vary -- some expect one-hot, others expect integer

**Off-by-one in class indices**: The `ignore_index=255` convention (MMSegmentation, nnU-Net) means class 0 is a valid class (usually background). Mixing up "0 = ignore" and "0 = background" is a common source of training bugs.

**Mask value normalization**: Some datasets store masks with pixel values 0 and 255 (for visualization), but the model expects 0 and 1. Pytorch-UNet's `BasicDataset` handles this by detecting unique values and normalizing, but custom datasets often trip over this.

### Augmentation Pitfalls

**Spatial transforms must apply identically to image and mask**:

```python
# CORRECT: Apply same random transform to both
import albumentations as A
transform = A.Compose([
    A.RandomCrop(256, 256),
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=30),
], additional_targets={'mask': 'mask'})
result = transform(image=image, mask=mask)  # Same crop/flip/rotation applied to both

# WRONG: Independent random transforms
image = random_crop(image)  # Different crop region!
mask = random_crop(mask)    # Misaligned with image
```

**Interpolation for masks**: Resizing masks must use nearest-neighbor interpolation. Bilinear/bicubic interpolation creates non-integer values that corrupt class labels. This is a silent bug -- training may appear to work but produce poor results.

**Color augmentations**: Brightness, contrast, hue, saturation changes must be applied to the image ONLY, never to the mask. This seems obvious but is easy to get wrong when using generic transform pipelines.

### DataLoader Workers

**`num_workers > 0` with `fork`**: On Linux, forked workers share file descriptors and can cause issues with HDF5 files, LMDB databases, or custom lazy-loading datasets. Solutions:
- Use `mp_start_method='spawn'` (slower but safer)
- Open file handles in `__getitem__`, not `__init__`
- Use `persistent_workers=True` (PyTorch 1.8+) to avoid repeated fork overhead

**Shared memory limits**: Large datasets with many workers can exhaust `/dev/shm` (default 64MB in Docker). Symptoms: `RuntimeError: unable to open shared memory object`. Fix: increase `--shm-size` in Docker or reduce `num_workers`.

## Architecture Gotchas

### Spatial Dimension Mismatch

When input dimensions are not divisible by `2^depth`, downsampling produces odd-sized feature maps that cannot be exactly upsampled back:

```
Input: (1, 3, 257, 257)
After 4x MaxPool2d(2): (1, 512, 16, 16)  -- floor division: 257/16 = 16.0625 -> 16
After 4x Upsample(2):  (1, 64, 256, 256)  -- 16 * 16 = 256, not 257!
```

Solutions across repos:
- **Pytorch-UNet**: `F.pad()` in the `Up` block to match skip connection dimensions
- **nnU-Net**: Patch-based training with patch sizes chosen to be divisible by the stride product
- **MMSegmentation**: `align_corners` parameter in interpolation; padding to divisible sizes
- **SAM 2**: Pad input to 1024x1024 (always square, always divisible)

### Channel Count After Concatenation

After concatenating skip connections, the channel count doubles. The subsequent convolution must account for this:

```python
# Bug: forgot that concatenation doubles channels
self.conv = Conv2d(256, 256, 3)  # Wrong: input is 512 after concat
# Fix:
self.conv = Conv2d(512, 256, 3)  # Correct: 256 (decoder) + 256 (skip) = 512
```

In bilinear mode in Pytorch-UNet, the channel handling is more subtle: `DoubleConv` uses `mid_channels = in_channels // 2` to compensate for the doubled input.

### Bilinear vs Transposed Convolution

**Checkerboard artifacts**: `ConvTranspose2d` with `kernel_size=2, stride=2` can produce checkerboard patterns in the output because the kernel's receptive fields overlap in a grid pattern. Solutions:
- Use `kernel_size=4, stride=2, padding=1` (overlapping, smoother)
- Use bilinear upsampling + regular convolution (SMP default approach)
- Use pixel shuffle / sub-pixel convolution

**Alignment issues**: `align_corners=True` vs `False` in bilinear interpolation affects pixel alignment at boundaries. Inconsistent settings between training and inference cause subtle accuracy drops. MMSegmentation standardizes on `align_corners=False`.

## Training Gotchas

### Learning Rate for Pretrained vs New Layers

Pretrained encoders need lower learning rates than randomly initialized decoders:

```python
# SMP / MMSegmentation pattern:
optimizer = torch.optim.SGD([
    {'params': model.encoder.parameters(), 'lr': 1e-4},   # 10x lower for pretrained
    {'params': model.decoder.parameters(), 'lr': 1e-3},   # Full LR for new layers
], momentum=0.9)

# MMSegmentation config approach:
optim_wrapper = dict(
    paramwise_cfg=dict(custom_keys={'backbone': dict(lr_mult=0.1)})
)
```

Without differential LRs, the pretrained encoder can "forget" its learned representations (catastrophic forgetting), while the decoder does not converge fast enough.

### Batch Normalization with Small Batches

BatchNorm statistics become noisy with small batches (batch_size < 8):

| Normalization | When to Use | Repos Using It |
|--------------|-------------|----------------|
| BatchNorm | batch_size >= 16 | Pytorch-UNet, SMP (default), keras-unet |
| SyncBatchNorm | Multi-GPU, effective batch >= 16 | MMSegmentation |
| InstanceNorm | batch_size = 1-4 (medical 3D) | nnU-Net |
| GroupNorm | batch_size = 1-8, any setting | Alternative in SMP, MMSeg |
| LayerNorm | Transformer architectures | SAM 2 |

nnU-Net's use of InstanceNorm is critical: medical 3D volumes are so large that batch sizes of 2 are common, making BatchNorm unusable.

### Mixed Precision Pitfalls

Operations that can fail under FP16:
- **Large reductions** (sum over many elements): Can overflow FP16 range (max ~65504). Dice loss over large images is vulnerable.
- **Softmax**: The exp() operation can overflow. PyTorch's `F.cross_entropy` handles this internally but manual softmax may not.
- **Small learning rates**: With `GradScaler`, very small LRs can be rounded to zero in FP16.

Pytorch-UNet and nnU-Net both use `GradScaler` to handle FP16 gradient underflow. SAM 2 trains in BF16 (bfloat16) which has the same exponent range as FP32 and avoids most overflow issues.

## Evaluation Gotchas

### IoU vs Dice Metric Differences

Dice and IoU are monotonically related but produce different numeric values:
```
Dice = 2 * IoU / (1 + IoU)
IoU  = Dice / (2 - Dice)
```

**Micro vs macro averaging**:
- Micro: Compute metric globally across all pixels/classes (weights by class frequency)
- Macro: Compute metric per class, then average (treats all classes equally)
- nnU-Net reports per-class Dice (then averages) -- essentially macro averaging
- MMSegmentation supports both via config

**Boundary effects**: Small objects have disproportionately large boundary-to-area ratios, so boundary prediction errors affect their IoU/Dice much more than large objects.

### Evaluation on Original vs Resized Images

**Always evaluate on the original resolution** for fair comparison:
- nnU-Net: Predictions are resampled back to original spacing before evaluation
- MMSegmentation: Supports `mode='whole'` (original size) or `mode='slide'` (tiled)
- Pytorch-UNet: Predictions at the training scale; must upsample before comparison

Evaluating on resized images inflates metrics because small errors are absorbed by the lower resolution.

## Repository-Specific Gotchas

### Pytorch-UNet
- Default 5 epochs is grossly insufficient for convergence. Use 50+ epochs.
- `wandb` is a hard dependency for training; set `WANDB_MODE=disabled` if not using it.
- The `scale` parameter defaults to 0.5, halving input resolution. This is easily missed and dramatically affects quality.
- No validation metric tracking for best model -- all checkpoints are saved equally.

### SMP
- The `encoder_weights="imagenet"` parameter requires internet on first use to download weights. Offline environments need pre-downloaded weights.
- Decoder output channels must match what `SegmentationHead` expects. Changing `decoder_channels` without adjusting the head causes shape mismatches.
- `activation=None` (default) returns raw logits. Users expecting probabilities often forget to apply sigmoid/softmax.

### nnU-Net
- The three environment variables (`nnUNet_raw`, `nnUNet_preprocessed`, `nnUNet_results`) MUST be set before any operation. Missing them produces cryptic errors.
- Plans files are dataset-specific. You cannot use plans from one dataset on another.
- The default 1000-epoch training takes 1-7 days depending on dataset size and GPU.
- Cross-validation fold results are not automatically aggregated; you must run `nnUNetv2_find_best_configuration` separately.

### SAM 2
- Input images MUST be resized to 1024x1024 (longest side) for the model to work correctly. The model was trained at this resolution and performance degrades at other sizes.
- Point prompts use (x, y) coordinates, not (row, col). This is image convention, not matrix convention.
- Video mode requires frames to be provided in temporal order. Random access is not supported.
- GPU memory usage scales with `num_maskmem` (memory bank size). Reduce this parameter for long videos on memory-constrained GPUs.

### MMSegmentation
- Config inheritance can create deeply nested dependencies that are hard to trace. Use `python tools/print_config.py config_file.py` to see the fully resolved config.
- `SyncBatchNorm` requires distributed training (`torch.distributed.launch`). Using it with a single GPU causes errors.
- The iteration-based training loop means there are no "epochs" -- monitor iterations instead.
- `data_root` paths in configs are relative to the working directory, not the config file location. This is a frequent source of "file not found" errors.
