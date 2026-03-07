---
title: "Pytorch-UNet - Data Pipeline"
date: 2025-01-15
status: planned
parent: "unet_pytorch/repo_overview.md"
tags: [unet, data, pytorch, preprocessing, augmentation]
---

# Pytorch-UNet Data Pipeline

## Dataset Class

Source file: `utils/data_loading.py`

### Supported Formats

The `BasicDataset` class (and its subclass `CarvanaDataset`) in `utils/data_loading.py` supports:

- **Image formats**: Any format readable by PIL/Pillow -- PNG, JPEG, BMP, TIFF, GIF, etc.
- **Mask formats**: Same as images; masks are loaded as PIL Images and converted to numpy arrays
- **Mask encoding**: Unique pixel values in the mask define class IDs. Binary masks use 0/1 (or 0/255 which gets normalized). Multi-class masks use integer values 0..N-1.

File discovery uses `pathlib.Path.glob('*')` and filters by known image suffixes, then matches image filenames to mask filenames (with configurable `mask_suffix`).

### Directory Structure Expected

```
data/
├── imgs/          # Input images
│   ├── image1.png
│   ├── image2.jpg
│   └── ...
└── masks/         # Corresponding masks
    ├── image1_mask.png    # mask_suffix="_mask" by default
    ├── image2_mask.png
    └── ...
```

The `CarvanaDataset` subclass sets `mask_suffix='_mask'` automatically for the Carvana car segmentation dataset.

## Preprocessing

### Image Preprocessing

In `BasicDataset.preprocess()`:

```python
@staticmethod
def preprocess(mask_values, pil_img, scale, is_mask):
    w, h = pil_img.size
    newW, newH = int(scale * w), int(scale * h)
    pil_img = pil_img.resize((newW, newH),
                              resample=Image.NEAREST if is_mask else Image.BICUBIC)
    img = np.asarray(pil_img)
```

- **Resizing**: Images are resized by a `scale` factor (default 0.5). BICUBIC interpolation is used for images.
- **Normalization**: Image pixel values are divided by 255.0 to produce [0, 1] range
- **Dtype**: Converted to `np.float32` for images
- **Transposition**: HWC to CHW format (`img.transpose((2, 0, 1))`) for PyTorch convention

### Mask Preprocessing

- **Resizing**: Masks use NEAREST interpolation to avoid creating invalid class values
- **Binary case**: If `mask_values == [0, 1]`, the mask is kept as-is (single-channel float)
- **Multi-class case**: Masks are one-hot encoded using `np.all()` comparison against each unique mask value, producing shape `(n_classes, H, W)`
- **Dtype**: Masks are converted to `np.int64` (long) for `CrossEntropyLoss` compatibility

## Augmentation

The base Pytorch-UNet implementation does **not** include built-in augmentation. The `BasicDataset.__getitem__` only applies:
1. Loading from disk
2. Resize by scale factor
3. Normalization

Users are expected to add augmentation externally via `torchvision.transforms` or `albumentations`. The training script `train.py` applies only horizontal flips via a random transform in some versions. The simplicity is intentional -- the repo focuses on architecture clarity over training pipelines.

## DataLoader Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| Batch Size | 1 (default) | Configurable via `--batch-size` CLI arg |
| Num Workers | `os.cpu_count()` | Uses all available CPU cores |
| Pin Memory | `True` | Enables faster CPU-to-GPU transfer |
| Shuffle | `True` (train) / `False` (val) | Standard practice |

From `train.py`:

```python
loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)
train_loader = DataLoader(train_set, shuffle=True, **loader_args)
val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)
```

Note: `drop_last=True` is used for the validation loader to avoid issues with batch normalization on incomplete batches.

## Data Flow Diagram

```
Raw Image (PNG/JPG)  +  Mask (PNG)
    │                       │
    ├── PIL.Image.open()    ├── PIL.Image.open()
    │                       │
    ├── Resize (BICUBIC)    ├── Resize (NEAREST)
    │   scale * (H, W)     │   scale * (H, W)
    │                       │
    ├── np.asarray()        ├── np.asarray()
    ├── / 255.0             ├── One-hot encode (multi-class)
    ├── .float32            ├── .int64
    ├── HWC -> CHW          │
    │                       │
    ├── torch.as_tensor()   ├── torch.as_tensor()
    │                       │
    └─── DataLoader (batched, shuffled, pinned) ───┘
                    │
                    └── Training Loop
```

## Compatibility Notes

- **Input size**: The model works with arbitrary input sizes, but dimensions should ideally be divisible by 16 (2^4 for the 4 downsampling stages) to avoid the need for padding in skip connections
- **Scale factor**: Default scale=0.5 halves the image; for full-resolution training, set `--scale 1.0` but expect higher memory usage
- **Grayscale images**: Set `n_channels=1` on the model; the dataset handles grayscale by keeping the single channel
- **Large datasets**: No lazy loading or memory-mapped files -- all preprocessing happens on-the-fly per `__getitem__` call, which is standard but means I/O can bottleneck training on slow storage
- **Carvana-specific**: The `CarvanaDataset` subclass expects `.gif` masks with `_mask` suffix and 1000x1000 pixel car images
