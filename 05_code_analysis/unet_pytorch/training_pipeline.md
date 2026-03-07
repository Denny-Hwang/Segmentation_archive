---
title: "Pytorch-UNet - Training Pipeline"
date: 2025-01-15
status: planned
parent: "unet_pytorch/repo_overview.md"
tags: [unet, training, pytorch, optimization]
---

# Pytorch-UNet Training Pipeline

## Training Script Overview

Source file: `train.py`

## Optimizer Configuration

From `train.py`, the optimizer is configured as:

```python
optimizer = optim.RMSprop(model.parameters(),
                          lr=learning_rate,
                          weight_decay=weight_decay,
                          momentum=momentum,
                          foreach=True)
```

- **Optimizer**: RMSprop (not Adam or SGD, which is an unusual choice for segmentation)
- **Learning rate**: 1e-5 (very conservative default)
- **Weight decay**: 1e-8 (minimal regularization)
- **Momentum**: 0.999
- **`foreach=True`**: Uses the faster fused implementation available in recent PyTorch versions

## Learning Rate Scheduler

```python
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)
```

- **Type**: `ReduceLROnPlateau` -- reduces LR when the validation Dice score stops improving
- **Mode**: `'max'` because the monitored metric is Dice score (higher is better)
- **Patience**: 5 epochs of no improvement before reducing LR
- **Factor**: 0.1 (default) -- divides LR by 10 on plateau

## Loss Function

The loss combines cross-entropy and Dice loss:

```python
criterion = nn.CrossEntropyLoss() if model.n_classes > 1 \
            else nn.BCEWithLogitsLoss()
# ...
loss = criterion(masks_pred, true_masks) \
     + dice_loss(
         F.softmax(masks_pred, dim=1).float(),
         F.one_hot(true_masks, model.n_classes).permute(0, 3, 1, 2).float(),
         multiclass=True
     )
```

### Implementation Details

- **Binary segmentation** (`n_classes=1`): Uses `BCEWithLogitsLoss` (sigmoid applied internally) + binary Dice loss
- **Multi-class** (`n_classes > 1`): Uses `CrossEntropyLoss` (softmax + NLL combined) + multi-class Dice loss
- **Dice loss** implementation (in `utils/dice_score.py`) computes 1 - Dice coefficient, averaged over all classes
- The two losses are summed with **equal weight** (no balancing coefficients)
- `F.softmax` is applied to predictions before Dice calculation, while `CrossEntropyLoss` applies its own internal softmax
- True masks are one-hot encoded for the Dice calculation: `F.one_hot(true_masks, n_classes).permute(0, 3, 1, 2)`

## Training Loop

### Per-Epoch Flow

```python
for epoch in range(1, epochs + 1):
    model.train()
    epoch_loss = 0
    with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}') as pbar:
        for batch in train_loader:
            images, true_masks = batch['image'], batch['mask']
            images = images.to(device)
            true_masks = true_masks.to(device)

            with torch.autocast(device.type if device.type != 'mps' else 'cpu',
                                enabled=amp):
                masks_pred = model(images)
                loss = criterion(masks_pred, true_masks) + dice_loss(...)

            optimizer.zero_grad(set_to_none=True)
            grad_scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
            grad_scaler.step(optimizer)
            grad_scaler.update()

            pbar.update(images.shape[0])
            epoch_loss += loss.item()

    # Validation + scheduler step
    val_score = evaluate(model, val_loader, device, amp)
    scheduler.step(val_score)
```

Key flow: forward pass -> loss computation -> backward pass -> gradient clipping -> optimizer step -> validation -> LR scheduling.

### Gradient Accumulation

Gradient accumulation is **not** implemented. Each batch triggers an independent `optimizer.step()`. For large images with batch_size=1, this means updates happen every single sample.

Gradient clipping **is** used: `torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)` with a default max norm of 1.0.

### Mixed Precision

**Yes**, AMP (Automatic Mixed Precision) is used via `torch.autocast` and `GradScaler`:

```python
grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
with torch.autocast(device.type, enabled=amp):
    masks_pred = model(images)
    loss = ...
grad_scaler.scale(loss).backward()
grad_scaler.step(optimizer)
grad_scaler.update()
```

- Enabled by default via the `--amp` flag
- Reduces memory usage and speeds up training on GPUs with Tensor Cores (V100, A100, etc.)
- The `GradScaler` prevents underflow in FP16 gradients by dynamically scaling the loss

## Validation

Source file: `evaluate.py`

The `evaluate()` function computes the Dice score on the validation set:

```python
@torch.inference_mode()
def evaluate(net, dataloader, device, amp):
    net.eval()
    dice_score = 0
    for batch in dataloader:
        image, mask_true = batch['image'], batch['mask']
        mask_pred = net(image.to(device))
        # Convert to one-hot / binary predictions
        # Compute Dice coefficient
        dice_score += multiclass_dice_coeff(...)
    return dice_score / max(num_val_batches, 1)
```

- Uses `@torch.inference_mode()` (more efficient than `torch.no_grad()`)
- Computes **Dice coefficient** (not Dice loss) as the validation metric
- Multi-class Dice is averaged across all classes (macro averaging)
- The score is used by `ReduceLROnPlateau` scheduler

## Checkpointing

```python
Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
state_dict = model.state_dict()
torch.save(state_dict, str(dir_checkpoint / f'checkpoint_epoch{epoch}.pth'))
```

- Saves a checkpoint **every epoch** to `checkpoints/` directory
- Saves only `state_dict` (not the full model), which is the recommended practice
- No "best model" tracking -- all epoch checkpoints are saved
- Checkpoints include only model weights, not optimizer state or scheduler state (so training cannot be resumed mid-run without modification)

## Logging

- **Weights & Biases (wandb)**: Primary logging platform, initialized in `train.py` with `wandb.init()`
- Logs: learning rate, training loss, validation Dice score, step count, epoch, and sample predictions as images
- **tqdm**: Progress bars for per-epoch batch iteration
- **Python logging**: Standard library logging for informational messages (model config, data split sizes)

```python
experiment = wandb.init(project='U-Net', resume='allow')
experiment.config.update(dict(epochs=epochs, batch_size=batch_size, ...))
experiment.log({'learning rate': optimizer.param_groups[0]['lr'],
                'validation Dice': val_score,
                'images': wandb.Image(image), ...})
```

## Hyperparameter Defaults

| Hyperparameter | Default Value | Notes |
|---------------|---------------|-------|
| Learning Rate | 1e-5 | Very low; may need tuning per dataset |
| Batch Size | 1 | Conservative; increase with available VRAM |
| Epochs | 5 | Short; production training needs 50-200+ |
| Optimizer | RMSprop | Uncommon choice; Adam often works better |
| Weight Decay | 1e-8 | Minimal regularization |
| Momentum | 0.999 | High momentum for RMSprop |
| AMP | Enabled | Mixed precision by default |
| Gradient Clipping | 1.0 | Max gradient norm |
| Image Scale | 0.5 | Halves input resolution |
| Val % | 10% | 10% of data held out for validation |

## Reproduction Notes

1. **Prepare data**: Place images in `data/imgs/` and masks in `data/masks/` with matching filenames
2. **Install dependencies**: `pip install -r requirements.txt` (torch, torchvision, pillow, tqdm, wandb)
3. **Run training**: `python train.py --epochs 50 --batch-size 4 --scale 1.0 --classes 2`
4. **Monitor**: Check wandb dashboard for loss curves and predictions
5. **Predict**: `python predict.py -i input.png -o output.png --model checkpoints/checkpoint_epoch50.pth`
6. **Key tip**: The default 5 epochs is insufficient for convergence; use at least 50 epochs for real training
7. **Memory**: With `--scale 1.0` and large images, batch_size=1 may be necessary even on 16GB GPUs
