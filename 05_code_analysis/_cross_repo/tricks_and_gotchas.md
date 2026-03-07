---
title: "Cross-Repository Tricks and Gotchas"
date: 2025-01-15
status: complete
tags: [cross-repo, tricks, gotchas, debugging, performance]
---

# Tricks and Gotchas

## Purpose

This document collects subtle implementation details, common pitfalls, and non-obvious techniques discovered during code analysis. These are things not typically mentioned in papers or documentation.

## Numerical Stability

### Loss Function Stability

Dice loss requires careful handling to avoid numerical issues. The denominator can be zero when both prediction and target are empty (no foreground pixels), leading to `0/0 = NaN`. Every implementation adds an epsilon value, but the choice of epsilon matters:

```python
# Standard Dice loss with epsilon (used by Pytorch-UNet, SMP, nnU-Net)
def dice_loss(pred, target, epsilon=1e-6):
    intersection = (pred * target).sum(dim=(-1, -2))
    union = pred.sum(dim=(-1, -2)) + target.sum(dim=(-1, -2))
    dice = (2 * intersection + epsilon) / (union + epsilon)
    return 1 - dice.mean()
```

A common mistake is adding epsilon only to the denominator (`intersection / (union + eps)`), which produces a non-zero loss even when both prediction and target are perfectly empty. The correct approach adds epsilon to both numerator and denominator (`(intersection + eps) / (union + eps)`), which gives a Dice score close to 1.0 when both are empty, correctly indicating "no error."

`CrossEntropyLoss` and `BCEWithLogitsLoss` in PyTorch are inherently stable because they use the log-sum-exp trick internally. Never compute `softmax` followed by `log` manually -- use `log_softmax` or the combined loss functions. SMP's loss implementations follow this pattern correctly; custom loss implementations often do not.

### Gradient Issues

Gradient clipping is used by Pytorch-UNet (`clip_grad_norm_(model.parameters(), 1.0)`) and nnU-Net to prevent exploding gradients. This is especially important when combining multiple loss functions (CE + Dice), as the Dice loss gradient can be very large when predictions are poor (early training) because small changes in the denominator cause large changes in the loss value.

nnU-Net uses a gradient clipping norm of 12.0 (much more permissive than Pytorch-UNet's 1.0), which prevents catastrophic gradient explosions while allowing larger gradient steps during normal training. Overly aggressive clipping (e.g., norm=0.1) can severely slow convergence.

## Data Pipeline Gotchas

### Mask Encoding

The most common data pipeline bug in segmentation is mismatched mask encoding. Different loss functions expect different formats:

```python
# CrossEntropyLoss: expects integer class indices [B, H, W] with values in [0, C-1]
loss = nn.CrossEntropyLoss()(logits, target_indices)  # logits: [B, C, H, W]

# BCEWithLogitsLoss: expects float targets [B, 1, H, W] in [0, 1]
loss = nn.BCEWithLogitsLoss()(logits, target_float)

# Common bug: passing one-hot masks to CrossEntropyLoss
# CrossEntropyLoss does NOT accept one-hot targets in standard PyTorch
# (it was added in PyTorch 2.0+ with label_smoothing, but the default expects indices)
```

nnU-Net handles this by always converting masks to integer format during preprocessing. Pytorch-UNet's `BasicDataset` converts binary masks (0/255) to 0/1 but can produce unexpected results with multi-class RGB masks if the user does not manually convert them to integer class indices.

### Augmentation Pitfalls

Geometric augmentations (rotation, flipping, elastic deformation, scaling) must be applied identically to both image and mask. This is straightforward with libraries like `albumentations` (which applies transforms to image and mask simultaneously) but error-prone with `torchvision.transforms` (which processes image and mask independently unless you use functional transforms):

```python
# WRONG: different random transforms for image and mask
transform = transforms.RandomHorizontalFlip()
image = transform(image)  # May or may not flip
mask = transform(mask)    # Independent random decision!

# CORRECT: use functional API with shared random state
if random.random() > 0.5:
    image = TF.hflip(image)
    mask = TF.hflip(mask)
```

Intensity augmentations (brightness, contrast, color jitter) should only be applied to the image, never to the mask. Applying color jitter to a mask would corrupt the class labels.

nnU-Net uses `batchgenerators` library which handles image-mask consistency automatically. It applies augmentations to a combined data-seg array, ensuring geometric transforms are identical.

### DataLoader Workers

Setting `num_workers > 0` in PyTorch's `DataLoader` spawns separate processes for data loading, which can cause several issues:

- **Memory duplication**: Each worker creates its own copy of the dataset object, including any cached data. With `num_workers=8` and a large in-memory dataset, memory usage can increase 8x.
- **Random seed issues**: Workers inherit the parent's random state, leading to identical augmentations across workers unless seeds are properly set. PyTorch provides `worker_init_fn` to set unique seeds per worker.
- **Shared memory limits**: On Linux, too many workers can exhaust `/dev/shm` (shared memory), causing cryptic `RuntimeError: DataLoader worker is killed`. The fix is to increase shared memory size or reduce `num_workers`.
- **Zombie processes**: If training crashes without proper cleanup, worker processes may persist as zombies. Use `try/finally` blocks or signal handlers for clean shutdown.

Pytorch-UNet sets `num_workers=os.cpu_count()` which can be too aggressive on shared machines. A safer default is `num_workers=4` or `num_workers=min(4, os.cpu_count())`.

## Architecture Gotchas

### Spatial Dimension Mismatch

When the input spatial dimensions are not divisible by `2^depth` (where depth is the number of downsampling stages), skip connections will have mismatched dimensions. Different repos handle this differently:

- **Pytorch-UNet**: Pads the upsampled tensor with `F.pad()` to match the skip connection. Works for any input size but adds zero-value pixels at boundaries.
- **nnU-Net**: Enforces that patch sizes are divisible by the total stride during planning. Mismatches never occur by construction.
- **SMP**: Pads input to the nearest valid size before encoding, then crops the output. The `SegmentationModel.forward()` handles this transparently.
- **MMSegmentation**: Relies on the user to choose compatible input sizes. Mismatches cause runtime errors.

### Channel Count After Concatenation

When implementing U-Net skip connections, the channel count after concatenation must exactly match the expected input channels of the subsequent convolution. A common bug:

```python
# If encoder outputs 256 channels and decoder upsamples to 256 channels,
# concatenation produces 512 channels, NOT 256.
x = torch.cat([x_skip, x_up], dim=1)  # 256 + 256 = 512 channels
conv = nn.Conv2d(512, 256, 3, padding=1)  # Input must be 512, not 256!
```

In Pytorch-UNet's bilinear mode, this is further complicated by the `mid_channels` parameter in `DoubleConv`, which reduces intermediate computation. The channel arithmetic is: upsample (keep channels) + concat (double channels) + DoubleConv(in=double, mid=in//2, out=target).

### Bilinear vs Transposed Convolution

Transposed convolutions (`ConvTranspose2d`) can produce **checkerboard artifacts** when the kernel size is not divisible by the stride. This is because overlapping output regions receive contributions from different numbers of input pixels. The standard remedy is to use `kernel_size=2, stride=2` (no overlap) or to use bilinear upsampling followed by a regular convolution:

```python
# Artifact-free upsampling alternatives:
# Option 1: bilinear + conv (used by Pytorch-UNet bilinear mode)
x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
x = conv(x)

# Option 2: pixel shuffle (sub-pixel convolution)
x = nn.PixelShuffle(upscale_factor=2)(conv_expand(x))

# Option 3: ConvTranspose2d with kernel=2, stride=2 (no overlap)
x = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)(x)
```

## Training Gotchas

### Learning Rate for Pretrained vs New Layers

When using a pretrained encoder with a randomly initialized decoder, applying the same learning rate to both components often leads to poor results. The pretrained encoder needs a lower learning rate to avoid destroying learned features, while the decoder needs a higher learning rate to converge quickly:

```python
# Differential learning rates (common in SMP and MMSegmentation)
optimizer = torch.optim.Adam([
    {'params': model.encoder.parameters(), 'lr': 1e-5},   # Lower LR
    {'params': model.decoder.parameters(), 'lr': 1e-4},   # Higher LR
])
```

MMSegmentation handles this via `paramwise_cfg` in the optimizer config, which applies learning rate multipliers to different parameter groups based on their names or layer types. SMP leaves this to the user.

### Batch Normalization with Small Batches

BatchNorm performs poorly with very small batch sizes (batch_size < 4) because the batch statistics become noisy estimates of the true population statistics. This is common in medical imaging where large 3D volumes force batch_size=1-2:

- **nnU-Net**: Uses InstanceNorm (`nn.InstanceNorm3d`) for 3D configurations and batch_size=2, which normalizes each sample independently.
- **Pytorch-UNet**: Uses BatchNorm2d regardless of batch size, which can cause training instability with batch_size=1.
- **MMSegmentation**: Uses SyncBN (`nn.SyncBatchNorm`) for multi-GPU training, which aggregates statistics across GPUs to increase the effective batch size.
- **GroupNorm** (`nn.GroupNorm(num_groups=32)`) is a batch-size-independent alternative that works well for any batch size, used in some SAM 2 components.

### Mixed Precision Pitfalls

AMP (Automatic Mixed Precision) can cause issues in specific scenarios:

- **Dice loss with FP16**: The Dice loss computation involves division of two sums. In FP16, large sums can overflow (>65504) or small sums can underflow to zero. The epsilon value must be large enough to matter in FP16 (1e-6 may round to 0 in FP16; use 1e-4 or compute Dice in FP32).
- **Softmax in FP16**: Can produce numerically unstable results for large logit values. PyTorch's `F.cross_entropy` handles this correctly (log-sum-exp trick), but manual `F.softmax(x) + torch.log(x)` can fail.
- **GradScaler with multi-loss**: When using multiple losses with different magnitudes, the GradScaler's dynamic scaling can oscillate. nnU-Net handles this by computing all losses within the same `autocast` context and summing them before calling `backward()`.

## Evaluation Gotchas

### IoU vs Dice Metric Differences

IoU (Intersection over Union, also called Jaccard Index) and Dice coefficient are monotonically related (`Dice = 2*IoU / (1+IoU)`) but produce different numerical values. Dice is always >= IoU for the same prediction. A Dice score of 0.90 corresponds to an IoU of 0.818. Papers and competitions may report either metric, so always check which is being used when comparing results.

**Micro vs macro averaging**: Micro averaging computes the metric globally (sum all intersections / sum all unions), while macro averaging computes per-class metrics and takes the mean. Macro averaging gives equal weight to rare classes, while micro averaging is dominated by frequent classes. nnU-Net reports both; Pytorch-UNet reports macro Dice by default.

### Evaluation on Original vs Resized Images

When training at reduced resolution (e.g., Pytorch-UNet's `--scale 0.5`), evaluation should ideally be done on the original resolution by upsampling predictions. However, Pytorch-UNet evaluates on the reduced resolution by default, which inflates Dice scores because small boundary errors are less penalized at lower resolution. nnU-Net always evaluates on the original resolution by resampling predictions back to the original spacing.

## Repository-Specific Gotchas

### Pytorch-UNet

The default training configuration (5 epochs, lr=1e-5, batch_size=1, RMSprop) is only suitable for quick demos. For real training, use at least 50 epochs, consider switching to Adam, and increase batch size if memory allows. The `wandb` dependency will cause training to fail if not installed or authenticated -- set `WANDB_MODE=disabled` to bypass. The validation `drop_last=True` setting discards the last incomplete batch, which can skew validation metrics on small datasets.

### SMP

When switching encoder architectures, the decoder's input channel expectations may not match the new encoder's output channels. SMP handles this automatically, but users who modify the encoder or decoder independently must ensure `decoder_channels` matches `encoder.out_channels`. The `auxiliary_params` option for adding a classification head can interfere with segmentation loss if not properly configured.

### nnU-Net

The three environment variables (`nnUNet_raw`, `nnUNet_preprocessed`, `nnUNet_results`) must be set before any operation. Missing variables produce cryptic import-time errors. The dataset naming convention (`DatasetXXX_Name`) with a three-digit numeric ID is strictly enforced. Resuming training from a checkpoint requires the exact same preprocessed data path; moving data to a different directory will fail. The planner targets 8GB VRAM by default; on larger GPUs, pass `--gpu_memory_target` to utilize available memory.

### SAM 2

SAM 2 requires specific checkpoint files that must be downloaded separately (they are not included in the pip package). The model expects 1024x1024 input resolution; other resolutions will work but produce suboptimal results because the positional encodings were trained for 1024x1024. Video inference with `SAM2VideoPredictor` requires frames to be provided as a directory of images or a video file -- it does not accept raw tensors directly. The memory bank size is a critical tuning parameter for long videos: too small misses re-appearing objects, too large wastes memory and may introduce stale information.

### MMSegmentation

The `_delete_=True` pattern in configs is essential when switching between fundamentally different backbones (e.g., CNN to Transformer), but forgetting it causes silent config merging bugs where stale fields from the parent config produce incorrect model architectures. The `SyncBN` default requires multi-GPU training; for single-GPU, switch to `BN` in the norm_cfg. The iteration-based training loop counts individual iterations, not epochs -- confusing these will result in training for far too long or too little. Config validation with `python tools/print_config.py` should always be run before training to catch inheritance errors.
