---
title: "Pytorch-UNet - Reverse Engineering Notes"
date: 2025-01-15
status: planned
parent: "unet_pytorch/repo_overview.md"
tags: [unet, reverse-engineering, implementation-details]
---

# Pytorch-UNet Reverse Engineering Notes

## Purpose

This document captures implementation details that are **not obvious** from the paper or README -- things only discoverable by reading the source code carefully.

## Hidden Implementation Details

### Padding Strategy

The original U-Net paper (Ronneberger et al., 2015) uses **valid (unpadded) convolutions**, meaning spatial dimensions shrink after each 3x3 convolution. This required cropping skip connections to match decoder feature map sizes. The output was smaller than the input (388x388 output from 572x572 input).

This implementation uses **same-padding** (`padding=1` on all 3x3 Conv2d layers), so spatial dimensions are preserved through convolutions. This is a major departure:
- Output size equals input size (no need for tiling/overlap inference)
- Skip connections have matching spatial dimensions (with minor +-1 pixel corrections via `F.pad`)
- Simpler to use with arbitrary input sizes

The tradeoff: padding introduces zero-valued border pixels that can subtly affect features near image edges, but this is negligible in practice and universally preferred in modern implementations.

### Weight Initialization

This implementation does **not** define any custom weight initialization. It relies entirely on PyTorch's default initialization:
- `nn.Conv2d`: Kaiming uniform initialization (He et al., 2015) -- `fan_in` mode with `a=sqrt(5)`
- `nn.BatchNorm2d`: weight=1, bias=0 (identity transform initially)
- `nn.ConvTranspose2d`: Same Kaiming uniform as Conv2d

The original U-Net paper recommended drawing initial weights from a Gaussian distribution with `std = sqrt(2/N)` where N is the number of incoming nodes, which is essentially the same as Kaiming initialization. So the default behavior is a close match.

### Bilinear Mode Channel Handling

A subtle but critical difference between the two upsampling modes:

```python
# In UNet.__init__:
factor = 2 if bilinear else 1
self.down4 = Down(512, 1024 // factor)  # 512 (bilinear) vs 1024 (transposed)
self.up1 = Up(1024, 512 // factor, bilinear)
self.up2 = Up(512, 256 // factor, bilinear)
self.up3 = Up(256, 128 // factor, bilinear)
self.up4 = Up(128, 64, bilinear)
```

When `bilinear=True`:
- Bottleneck outputs 512 channels (not 1024)
- Each decoder stage outputs half the channels compared to transposed conv mode
- `DoubleConv` in `Up` uses `mid_channels = in_channels // 2` to further reduce parameters
- Overall: ~17.3M parameters vs ~31M for transposed convolution

This is because bilinear upsampling has no learnable parameters, so the model compensates by keeping channel counts lower to avoid excessive computation in the subsequent convolutions.

## Performance-Critical Choices

### Memory Optimization

- **`inplace=True` on ReLU**: Modifies activations in-place, saving one tensor allocation per ReLU. This can reduce peak memory by ~15-20% in deep networks
- **`set_to_none=True` in `optimizer.zero_grad()`**: Sets gradients to `None` instead of zero tensors, saving memory allocations
- **AMP (mixed precision)**: FP16 forward/backward pass halves activation memory
- **`GradScaler`**: Prevents gradient underflow under FP16
- **No explicit `torch.cuda.empty_cache()`**: The implementation relies on PyTorch's memory allocator

Not implemented (but could help):
- Gradient checkpointing (`torch.utils.checkpoint`) for trading compute for memory
- Activation offloading to CPU

### Numerical Stability

- **`BCEWithLogitsLoss` over `BCELoss`**: For binary segmentation, logits are passed directly (no manual sigmoid), which is numerically more stable due to the log-sum-exp trick
- **`CrossEntropyLoss`**: Similarly combines log-softmax and NLL in a numerically stable way
- **Dice loss epsilon**: The `dice_loss` function uses `+ 1e-6` (or similar) in the denominator to avoid division by zero when a class has no pixels
- **Gradient clipping**: `clip_grad_norm_(model.parameters(), 1.0)` prevents exploding gradients

## Paper vs Code Discrepancies

| Aspect | Paper (Ronneberger 2015) | This Implementation |
|--------|-------------------------|---------------------|
| Padding | Valid convolutions (output < input) | Same-padding (`padding=1`), output == input |
| Normalization | None mentioned | BatchNorm2d after every Conv2d |
| Upsampling | "Up-convolution" (learned 2x2) | Bilinear (default) or ConvTranspose2d (optional) |
| Input Size | Fixed 572x572 | Arbitrary (ideally divisible by 16) |
| Depth | 4 downsampling stages | 4 downsampling stages (same) |
| Base Filters | 64 | 64 (same) |
| Loss | Weighted cross-entropy + morphological weight map | CE + Dice (no weight maps) |
| Data Augmentation | Elastic deformations, shifts, rotations | None built-in |
| Dropout | Dropout at bottleneck | No dropout |
| Output Activation | Pixel-wise softmax | Raw logits (softmax in loss) |

## Gotchas and Pitfalls

1. **Input size not divisible by 16**: The 4 maxpool stages each halve dimensions. Odd dimensions cause off-by-one issues that `F.pad` in the `Up` block corrects, but this adds unnecessary computation. Prefer inputs where H and W are multiples of 16.

2. **Default hyperparameters are for demo purposes**: 5 epochs at LR=1e-5 with batch_size=1 will not produce a converged model. These are minimal defaults for quick testing.

3. **No data augmentation**: The lack of built-in augmentation means the model will overfit quickly on small datasets. You must add augmentation externally.

4. **Bilinear mode default**: The `bilinear=True` default produces a smaller model but with no learnable upsampling. For some tasks, `bilinear=False` (transposed convolution) performs better.

5. **Checkpoint format**: Only `state_dict` is saved, not optimizer/scheduler state. You cannot resume training from a checkpoint without re-implementing save/load logic.

6. **Wandb dependency**: Training will fail if `wandb` is not installed or not logged in. Disable with `WANDB_MODE=disabled`.

7. **Multi-class mask format**: Masks must contain integer class indices (0, 1, 2, ...) as pixel values. RGB masks will not work without conversion.

## Lessons Learned

1. **Same-padding is universally preferred**: No modern U-Net implementation uses valid convolutions. The complexity of crop-based skip connections is not worth the theoretical purity.

2. **BatchNorm is practically mandatory**: The original paper had no normalization, but training is significantly more stable and faster with BatchNorm. For batch_size=1 scenarios (common in medical imaging), use InstanceNorm or GroupNorm instead.

3. **Bilinear vs transposed convolution is a minor decision**: Both work well; bilinear is simpler and cheaper, transposed conv is more expressive. The performance difference is typically <1% on benchmarks.

4. **Dice loss stabilizes training for imbalanced classes**: Cross-entropy alone struggles when foreground is small. The Dice component provides a region-based objective that is invariant to class frequency.

5. **Keep the architecture simple, invest in data**: This clean U-Net implementation shows that a simple architecture with good data handling can be competitive. Over-engineering the architecture often yields diminishing returns.
