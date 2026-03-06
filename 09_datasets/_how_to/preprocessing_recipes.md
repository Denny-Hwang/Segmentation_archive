# Preprocessing Recipes

Common preprocessing pipelines for segmentation datasets.

## Medical Image Preprocessing

### CT Volume Preprocessing (Synapse, LiTS)

```python
import numpy as np
import nibabel as nib
from scipy.ndimage import zoom


def preprocess_ct_volume(nifti_path, target_spacing=(1.0, 1.0, 1.0)):
    """Standard CT preprocessing pipeline.

    Steps:
    1. Load NIfTI volume
    2. Resample to isotropic spacing
    3. Clip HU values to soft-tissue window
    4. Normalize to [0, 1]
    """
    img = nib.load(nifti_path)
    volume = img.get_fdata()
    spacing = img.header.get_zooms()

    # Resample to target spacing
    scale = [s / t for s, t in zip(spacing, target_spacing)]
    volume = zoom(volume, scale, order=3)

    # Clip to soft-tissue CT window
    volume = np.clip(volume, -125, 275)

    # Normalize to [0, 1]
    volume = (volume - volume.min()) / (volume.max() - volume.min() + 1e-8)

    return volume.astype(np.float32)
```

### Cardiac MRI Preprocessing (ACDC)

```python
def preprocess_cardiac_mri(nifti_path):
    """ACDC cardiac MRI preprocessing.

    Steps:
    1. Load NIfTI
    2. Z-score normalization per volume
    3. Resize to standard resolution
    """
    img = nib.load(nifti_path)
    volume = img.get_fdata().astype(np.float32)

    # Z-score normalization
    mean = volume.mean()
    std = volume.std()
    volume = (volume - mean) / (std + 1e-8)

    return volume
```

## Natural Image Preprocessing

### Standard Segmentation Augmentation (COCO, ADE20K, VOC)

```python
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transforms(img_size=512):
    """Standard training augmentation pipeline."""
    return A.Compose([
        A.RandomResizedCrop(height=img_size, width=img_size, scale=(0.5, 2.0)),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        A.GaussianBlur(blur_limit=(3, 7), p=0.3),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def get_val_transforms(img_size=512):
    """Validation transform (resize + normalize only)."""
    return A.Compose([
        A.Resize(height=img_size, width=img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
```

### Cityscapes Preprocessing

```python
def get_cityscapes_transforms(crop_size=768):
    """Cityscapes-specific preprocessing."""
    return A.Compose([
        A.RandomCrop(height=crop_size, width=crop_size),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
```

## Common Mask Processing

### Label Encoding

```python
def encode_segmentation_mask(mask, class_mapping):
    """Convert RGB mask to class index mask.

    Args:
        mask: HxWx3 RGB mask array.
        class_mapping: Dict mapping RGB tuples to class indices.

    Returns:
        HxW integer array of class indices.
    """
    encoded = np.zeros(mask.shape[:2], dtype=np.int64)
    for rgb, class_idx in class_mapping.items():
        match = np.all(mask == rgb, axis=-1)
        encoded[match] = class_idx
    return encoded
```

### Ignore Index Handling

```python
def apply_ignore_index(mask, ignore_classes, ignore_value=255):
    """Set certain classes to ignore index for loss computation."""
    for cls in ignore_classes:
        mask[mask == cls] = ignore_value
    return mask
```

## nnU-Net-style Automatic Preprocessing

For fully automatic preprocessing, consider using nnU-Net's built-in pipeline:

```bash
nnUNetv2_plan_and_preprocess -d DATASET_ID --verify_dataset_integrity
```

This automatically determines optimal spacing, normalization, and patch size based on dataset statistics.
