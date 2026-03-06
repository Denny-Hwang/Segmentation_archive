"""
Common augmentation pipelines for image segmentation using albumentations.

Provides pre-configured augmentation pipelines for training, validation,
and test-time augmentation scenarios.

Usage:
    from _common.augmentation import get_training_augmentation, get_validation_augmentation

    train_transform = get_training_augmentation(image_size=(256, 256))
    val_transform = get_validation_augmentation(image_size=(256, 256))

    # Apply to image and mask
    result = train_transform(image=image, mask=mask)
    aug_image, aug_mask = result["image"], result["mask"]
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Tuple, Optional


def get_training_augmentation(
    image_size: Tuple[int, int] = (256, 256),
    horizontal_flip: bool = True,
    vertical_flip: bool = False,
    rotation_limit: int = 15,
    brightness_limit: float = 0.2,
    contrast_limit: float = 0.2,
    elastic_transform: bool = False,
    normalize: bool = True,
    mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
    std: Tuple[float, ...] = (0.229, 0.224, 0.225),
) -> A.Compose:
    """Create a training augmentation pipeline.

    All spatial transforms are applied consistently to both image and mask.
    Color transforms are applied only to the image.

    Args:
        image_size: Target (height, width) for resizing.
        horizontal_flip: Enable random horizontal flipping.
        vertical_flip: Enable random vertical flipping.
        rotation_limit: Maximum rotation angle in degrees.
        brightness_limit: Maximum brightness adjustment factor.
        contrast_limit: Maximum contrast adjustment factor.
        elastic_transform: Enable elastic deformation (useful for medical images).
        normalize: Whether to apply ImageNet normalization.
        mean: Normalization mean per channel.
        std: Normalization std per channel.

    Returns:
        albumentations Compose pipeline.
    """
    transforms = [
        A.Resize(height=image_size[0], width=image_size[1]),
    ]

    # Spatial augmentations (applied to both image and mask)
    if horizontal_flip:
        transforms.append(A.HorizontalFlip(p=0.5))
    if vertical_flip:
        transforms.append(A.VerticalFlip(p=0.5))
    if rotation_limit > 0:
        transforms.append(
            A.ShiftScaleRotate(
                shift_limit=0.05,
                scale_limit=0.1,
                rotate_limit=rotation_limit,
                border_mode=0,  # cv2.BORDER_CONSTANT
                p=0.5,
            )
        )
    if elastic_transform:
        transforms.append(A.ElasticTransform(alpha=120, sigma=6, p=0.3))

    # Color augmentations (applied only to image)
    if brightness_limit > 0 or contrast_limit > 0:
        transforms.append(
            A.RandomBrightnessContrast(
                brightness_limit=brightness_limit,
                contrast_limit=contrast_limit,
                p=0.5,
            )
        )

    transforms.extend([
        A.GaussNoise(p=0.2),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
    ])

    # Normalization and tensor conversion
    if normalize:
        transforms.append(A.Normalize(mean=mean, std=std))
    transforms.append(ToTensorV2())

    return A.Compose(transforms)


def get_validation_augmentation(
    image_size: Tuple[int, int] = (256, 256),
    normalize: bool = True,
    mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
    std: Tuple[float, ...] = (0.229, 0.224, 0.225),
) -> A.Compose:
    """Create a validation/test augmentation pipeline (resize + normalize only).

    Args:
        image_size: Target (height, width) for resizing.
        normalize: Whether to apply ImageNet normalization.
        mean: Normalization mean per channel.
        std: Normalization std per channel.

    Returns:
        albumentations Compose pipeline.
    """
    transforms = [
        A.Resize(height=image_size[0], width=image_size[1]),
    ]

    if normalize:
        transforms.append(A.Normalize(mean=mean, std=std))
    transforms.append(ToTensorV2())

    return A.Compose(transforms)


def get_tta_augmentations(
    image_size: Tuple[int, int] = (256, 256),
    mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
    std: Tuple[float, ...] = (0.229, 0.224, 0.225),
) -> list:
    """Create a list of test-time augmentation (TTA) transforms.

    Returns multiple augmentation pipelines. During inference, run the model
    on each augmented version and average the predictions.

    Args:
        image_size: Target (height, width) for resizing.
        mean: Normalization mean per channel.
        std: Normalization std per channel.

    Returns:
        List of albumentations Compose pipelines.
    """
    base = [
        A.Resize(height=image_size[0], width=image_size[1]),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ]

    tta_transforms = [
        # Original
        A.Compose(base),
        # Horizontal flip
        A.Compose([A.HorizontalFlip(p=1.0)] + base),
        # Vertical flip
        A.Compose([A.VerticalFlip(p=1.0)] + base),
        # Both flips
        A.Compose([A.HorizontalFlip(p=1.0), A.VerticalFlip(p=1.0)] + base),
    ]

    return tta_transforms
