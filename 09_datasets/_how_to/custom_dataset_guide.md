# Custom Dataset Guide

How to create and organize your own segmentation dataset for training.

## Step 1: Data Collection

- Collect images relevant to your segmentation task
- Ensure consistent image quality and resolution where possible
- Aim for diversity in lighting, viewpoints, and object appearances
- Minimum recommended: 100-500 images for fine-tuning, 1000+ for training from scratch

## Step 2: Annotation

### Recommended Tools

| Tool | Type | Best For |
|---|---|---|
| [Label Studio](https://labelstud.io/) | Web-based | Polygon and brush annotation |
| [CVAT](https://www.cvat.ai/) | Web-based | Team annotation workflows |
| [Labelme](https://github.com/wkentaro/labelme) | Desktop | Quick polygon annotation |
| [SAM + Grounding DINO](https://github.com/IDEA-Research/Grounded-Segment-Anything) | Automatic | Semi-automatic with prompts |

### Annotation Formats

```
# COCO format (JSON with RLE or polygon)
{
  "images": [...],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "segmentation": [[x1, y1, x2, y2, ...]],
      "area": 1234,
      "bbox": [x, y, w, h]
    }
  ],
  "categories": [...]
}

# Simple mask format (PNG files)
# Pixel values correspond to class indices
# 0 = background, 1 = class_1, 2 = class_2, ...
```

## Step 3: Directory Structure

### Recommended Layout

```
my_dataset/
├── images/
│   ├── train/
│   │   ├── img_001.jpg
│   │   └── ...
│   └── val/
│       ├── img_100.jpg
│       └── ...
├── masks/
│   ├── train/
│   │   ├── img_001.png
│   │   └── ...
│   └── val/
│       ├── img_100.png
│       └── ...
├── class_mapping.json
└── dataset_info.yaml
```

### class_mapping.json

```json
{
  "0": "background",
  "1": "object_class_1",
  "2": "object_class_2",
  "255": "ignore"
}
```

## Step 4: PyTorch Dataset Class

```python
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np


class CustomSegDataset(Dataset):
    """Custom segmentation dataset.

    Args:
        image_dir: Path to image directory.
        mask_dir: Path to mask directory.
        transform: Albumentations transform pipeline.
    """

    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = sorted(image_dir.glob("*.jpg")) + sorted(image_dir.glob("*.png"))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        mask_path = self.mask_dir / img_path.with_suffix(".png").name

        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"))

        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]

        return image, mask.long()
```

## Step 5: Quality Checks

```python
def validate_dataset(image_dir, mask_dir):
    """Run basic quality checks on a custom dataset."""
    images = sorted(image_dir.glob("*"))
    masks = sorted(mask_dir.glob("*"))

    print(f"Images: {len(images)}, Masks: {len(masks)}")
    assert len(images) == len(masks), "Image/mask count mismatch!"

    for img_path, mask_path in zip(images, masks):
        img = Image.open(img_path)
        mask = Image.open(mask_path)
        assert img.size == mask.size, f"Size mismatch: {img_path.name}"

    # Check class distribution
    all_classes = set()
    for mask_path in masks:
        mask = np.array(Image.open(mask_path))
        all_classes.update(np.unique(mask))

    print(f"Unique classes found: {sorted(all_classes)}")
```

## Step 6: Integration with Frameworks

### nnU-Net Format

```bash
# Convert to nnU-Net format:
# nnUNet_raw/DatasetXXX_MyDataset/
#   imagesTr/     # Training images (case_XXXX_0000.nii.gz)
#   labelsTr/     # Training labels (case_XXXX.nii.gz)
#   imagesTs/     # Test images
#   dataset.json  # Dataset description
```

### MMSegmentation Format

```python
# Add custom dataset config:
# configs/_base_/datasets/my_dataset.py
dataset_type = 'CustomDataset'
data_root = 'data/my_dataset'
img_suffix = '.jpg'
seg_map_suffix = '.png'
```
