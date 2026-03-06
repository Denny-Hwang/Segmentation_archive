"""
Common visualization utilities for image segmentation experiments.

Provides functions to overlay segmentation masks on images, plot training
curves, and compare predictions across models.

Usage:
    from _common.visualization import overlay_mask, plot_training_curves

    # Overlay a mask on an image
    fig = overlay_mask(image, mask, alpha=0.5)
    fig.savefig("overlay.png")

    # Plot training curves from a log file
    plot_training_curves(log_path="training_log.csv")
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
from typing import Optional, Sequence, Union

# Default color palette for segmentation classes
DEFAULT_PALETTE = [
    [0, 0, 0],        # background (black)
    [255, 0, 0],      # class 1 (red)
    [0, 255, 0],      # class 2 (green)
    [0, 0, 255],      # class 3 (blue)
    [255, 255, 0],    # class 4 (yellow)
    [255, 0, 255],    # class 5 (magenta)
    [0, 255, 255],    # class 6 (cyan)
    [128, 0, 0],      # class 7 (dark red)
    [0, 128, 0],      # class 8 (dark green)
    [0, 0, 128],      # class 9 (dark blue)
]


def overlay_mask(
    image: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.5,
    palette: Optional[Sequence[Sequence[int]]] = None,
    title: Optional[str] = None,
    figsize: tuple = (8, 8),
) -> plt.Figure:
    """Overlay a segmentation mask on an image.

    Args:
        image: Input image as numpy array of shape (H, W, 3) with values in [0, 255]
               or [0.0, 1.0].
        mask: Segmentation mask of shape (H, W) with integer class labels.
        alpha: Transparency of the overlay (0 = fully transparent, 1 = fully opaque).
        palette: List of RGB colors for each class. Defaults to DEFAULT_PALETTE.
        title: Optional title for the figure.
        figsize: Figure size in inches.

    Returns:
        matplotlib Figure object.
    """
    if palette is None:
        palette = DEFAULT_PALETTE

    # Normalize image to [0, 1] if needed
    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 255.0
    image = np.clip(image, 0.0, 1.0)

    # Create colored mask
    h, w = mask.shape[:2]
    colored_mask = np.zeros((h, w, 3), dtype=np.float32)
    for cls_id in range(int(mask.max()) + 1):
        if cls_id < len(palette):
            color = np.array(palette[cls_id], dtype=np.float32) / 255.0
            colored_mask[mask == cls_id] = color

    # Blend image and mask
    blended = image * (1 - alpha) + colored_mask * alpha

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.imshow(np.clip(blended, 0, 1))
    ax.axis("off")
    if title:
        ax.set_title(title)
    plt.tight_layout()

    return fig


def show_prediction_comparison(
    image: np.ndarray,
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    palette: Optional[Sequence[Sequence[int]]] = None,
    title: str = "Prediction Comparison",
    figsize: tuple = (18, 6),
) -> plt.Figure:
    """Show image, ground truth, and prediction side by side.

    Args:
        image: Input image of shape (H, W, 3).
        ground_truth: Ground truth mask of shape (H, W).
        prediction: Predicted mask of shape (H, W).
        palette: Color palette for mask visualization.
        title: Figure title.
        figsize: Figure size.

    Returns:
        matplotlib Figure object.
    """
    if palette is None:
        palette = DEFAULT_PALETTE

    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 255.0

    num_classes = max(int(ground_truth.max()), int(prediction.max())) + 1

    def colorize_mask(mask: np.ndarray) -> np.ndarray:
        h, w = mask.shape[:2]
        colored = np.zeros((h, w, 3), dtype=np.float32)
        for cls_id in range(num_classes):
            if cls_id < len(palette):
                color = np.array(palette[cls_id], dtype=np.float32) / 255.0
                colored[mask == cls_id] = color
        return colored

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    axes[0].imshow(np.clip(image, 0, 1))
    axes[0].set_title("Input Image")
    axes[0].axis("off")

    axes[1].imshow(colorize_mask(ground_truth))
    axes[1].set_title("Ground Truth")
    axes[1].axis("off")

    axes[2].imshow(colorize_mask(prediction))
    axes[2].set_title("Prediction")
    axes[2].axis("off")

    fig.suptitle(title)
    plt.tight_layout()

    return fig


def plot_training_curves(
    train_losses: Sequence[float],
    val_losses: Optional[Sequence[float]] = None,
    train_metrics: Optional[dict] = None,
    val_metrics: Optional[dict] = None,
    title: str = "Training Curves",
    figsize: tuple = (14, 5),
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Plot training and validation loss/metric curves.

    Args:
        train_losses: List of training losses per epoch.
        val_losses: Optional list of validation losses per epoch.
        train_metrics: Optional dict of {metric_name: [values_per_epoch]} for training.
        val_metrics: Optional dict of {metric_name: [values_per_epoch]} for validation.
        title: Figure title.
        figsize: Figure size.
        save_path: Optional path to save the figure.

    Returns:
        matplotlib Figure object.
    """
    has_metrics = train_metrics is not None or val_metrics is not None
    ncols = 2 if has_metrics else 1
    fig, axes = plt.subplots(1, ncols, figsize=figsize)

    if ncols == 1:
        axes = [axes]

    # Plot losses
    epochs = range(1, len(train_losses) + 1)
    axes[0].plot(epochs, train_losses, label="Train Loss", color="blue")
    if val_losses is not None:
        axes[0].plot(epochs, val_losses, label="Val Loss", color="red")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot metrics
    if has_metrics:
        if train_metrics:
            for name, values in train_metrics.items():
                axes[1].plot(range(1, len(values) + 1), values, label=f"Train {name}")
        if val_metrics:
            for name, values in val_metrics.items():
                axes[1].plot(
                    range(1, len(values) + 1), values, label=f"Val {name}", linestyle="--"
                )
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Score")
        axes[1].set_title("Metrics")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    fig.suptitle(title)
    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: Optional[Sequence[str]] = None,
    normalize: bool = True,
    title: str = "Confusion Matrix",
    figsize: tuple = (8, 8),
    cmap: str = "Blues",
) -> plt.Figure:
    """Plot a confusion matrix as a heatmap.

    Args:
        cm: Confusion matrix of shape (num_classes, num_classes).
        class_names: Optional list of class names.
        normalize: Whether to normalize rows to sum to 1.
        title: Figure title.
        figsize: Figure size.
        cmap: Colormap name.

    Returns:
        matplotlib Figure object.
    """
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)
        cm_display = cm.astype(np.float64) / row_sums
    else:
        cm_display = cm

    num_classes = cm.shape[0]
    if class_names is None:
        class_names = [str(i) for i in range(num_classes)]

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    im = ax.imshow(cm_display, interpolation="nearest", cmap=cmap)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(num_classes),
        yticks=np.arange(num_classes),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True Label",
        xlabel="Predicted Label",
        title=title,
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    fmt = ".2f" if normalize else "d"
    thresh = cm_display.max() / 2.0
    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(
                j, i, format(cm_display[i, j], fmt),
                ha="center", va="center",
                color="white" if cm_display[i, j] > thresh else "black",
            )

    plt.tight_layout()
    return fig
