"""Example image loading and label-map visualization helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ARCHIVE_ROOT = Path(__file__).resolve().parent.parent.parent
EXAMPLES_DIR = ARCHIVE_ROOT / "assets" / "examples"
MASKS_DIR = EXAMPLES_DIR / "masks"

# Fixed qualitative palette so GT and predictions share class colors.
# Index -1 (unmatched segments) renders as dark gray.
PALETTE = np.array(
    [
        [77, 175, 74],  # green
        [55, 126, 184],  # blue
        [228, 26, 28],  # red
        [255, 127, 0],  # orange
        [152, 78, 163],  # purple
        [255, 217, 47],  # yellow
        [166, 86, 40],  # brown
        [247, 129, 191],  # pink
        [0, 206, 209],  # teal
        [128, 128, 0],  # olive
    ],
    dtype=np.uint8,
)
UNMATCHED_COLOR = np.array([64, 64, 64], dtype=np.uint8)


@dataclass
class Example:
    key: str
    title: str
    classes: list[str]
    difficulty: str
    teaches: str
    image_path: Path
    mask_path: Path
    source: str = ""
    gt_note: str = ""


def load_manifest() -> dict[str, Example]:
    """Load example metadata from assets/examples/examples.json."""
    manifest_path = EXAMPLES_DIR / "examples.json"
    if not manifest_path.exists():
        return {}
    with open(manifest_path) as f:
        raw = json.load(f)
    examples = {}
    for key, meta in raw.items():
        img = EXAMPLES_DIR / f"{key}.png"
        mask = MASKS_DIR / f"{key}.png"
        if img.exists() and mask.exists():
            examples[key] = Example(
                key=key,
                title=meta["title"],
                classes=meta["classes"],
                difficulty=meta["difficulty"],
                teaches=meta["teaches"],
                image_path=img,
                mask_path=mask,
                source=meta.get("source", ""),
                gt_note=meta.get("gt_note", ""),
            )
    return examples


def load_image(path: Path) -> np.ndarray:
    from PIL import Image

    return np.asarray(Image.open(path).convert("RGB"))


def load_mask(path: Path) -> np.ndarray:
    from PIL import Image

    return np.asarray(Image.open(path)).astype(np.int32)


def colorize_labels(labels: np.ndarray, n_colors: int | None = None) -> np.ndarray:
    """Map an integer label map to RGB using the shared palette.

    Label -1 (unmatched) becomes dark gray. When there are more labels
    than palette entries (e.g. raw superpixels), colors cycle.
    """
    out = np.empty((*labels.shape, 3), dtype=np.uint8)
    out[labels < 0] = UNMATCHED_COLOR
    pos = labels >= 0
    out[pos] = PALETTE[labels[pos] % len(PALETTE)]
    return out


def overlay_labels(
    image: np.ndarray, labels: np.ndarray, alpha: float = 0.55
) -> np.ndarray:
    """Blend a colorized label map over the image, with boundary lines."""
    color = colorize_labels(labels)
    blend = image.astype(np.float32) * (1 - alpha) + color.astype(np.float32) * alpha
    blend = blend.astype(np.uint8)
    # Draw segment boundaries in white for crispness
    edges = np.zeros(labels.shape, dtype=bool)
    edges[:-1, :] |= labels[:-1, :] != labels[1:, :]
    edges[:, :-1] |= labels[:, :-1] != labels[:, 1:]
    blend[edges] = [255, 255, 255]
    return blend


def resize_pair(
    image: np.ndarray, mask: np.ndarray | None, max_size: int
) -> tuple[np.ndarray, np.ndarray | None]:
    """Downscale image (bilinear) and mask (nearest) to max_size on the long side."""
    from PIL import Image

    h, w = image.shape[:2]
    if max(h, w) <= max_size:
        return image, mask
    scale = max_size / max(h, w)
    new_size = (int(w * scale), int(h * scale))
    image_r = np.asarray(Image.fromarray(image).resize(new_size, Image.BILINEAR))
    mask_r = None
    if mask is not None:
        mask_r = np.asarray(
            Image.fromarray(mask.astype(np.uint8)).resize(new_size, Image.NEAREST)
        ).astype(np.int32)
    return image_r, mask_r


def class_legend_html(class_names: list[str]) -> str:
    """HTML color chips mapping palette colors to class names."""
    chips = []
    for i, name in enumerate(class_names):
        r, g, b = PALETTE[i % len(PALETTE)]
        chips.append(
            f'<span style="background:rgb({r},{g},{b});color:#fff;'
            f"padding:2px 10px;border-radius:10px;font-size:0.85em;"
            f'margin-right:4px;white-space:nowrap;">{i}: {name}</span>'
        )
    return " ".join(chips)
