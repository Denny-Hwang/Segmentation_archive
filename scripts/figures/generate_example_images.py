#!/usr/bin/env python3
"""Generate synthetic example images + ground-truth masks for the Segmentation Lab.

Every example is drawn simultaneously onto an RGB image and an integer
label mask, so the explorer can compute real evaluation metrics (IoU,
Dice, boundary F1, ...) against a known ground truth.

Outputs (all copyright-free, fully deterministic):
    assets/examples/<name>.png            RGB image (512x512)
    assets/examples/masks/<name>.png      Label mask, mode "L", pixel = class id
    assets/examples/examples.json         Class names + teaching notes per example

Each example is designed to stress a different family of segmentation
algorithms (global thresholding, clustering, region growing, graph-based,
edge/texture handling), which is what makes the Lab pedagogically useful.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = REPO_ROOT / "assets" / "examples"
MASK_DIR = OUT_DIR / "masks"

W, H = 512, 512


class Canvas:
    """Draw shapes into an RGB image and a class-id label mask at once."""

    def __init__(self, bg_color: str | tuple, bg_class: int = 0):
        from PIL import Image, ImageDraw

        self.img = Image.new("RGB", (W, H), bg_color)
        self.mask = Image.new("L", (W, H), bg_class)
        self.draw = ImageDraw.Draw(self.img)
        self.mdraw = ImageDraw.Draw(self.mask)

    def rectangle(self, xy, fill, cls: int, **kw):
        self.draw.rectangle(xy, fill=fill, **kw)
        self.mdraw.rectangle(xy, fill=cls)

    def ellipse(self, xy, fill, cls: int, **kw):
        self.draw.ellipse(xy, fill=fill, **kw)
        self.mdraw.ellipse(xy, fill=cls)

    def polygon(self, xy, fill, cls: int, **kw):
        self.draw.polygon(xy, fill=fill, **kw)
        self.mdraw.polygon(xy, fill=cls)

    def line(self, xy, fill, cls: int, width: int = 1):
        self.draw.line(xy, fill=fill, width=width)
        self.mdraw.line(xy, fill=cls, width=width)

    def add_noise(self, sigma: float, seed: int = 0):
        """Add Gaussian noise to the image only (ground truth unchanged)."""
        import numpy as np
        from PIL import Image

        rng = np.random.default_rng(seed)
        arr = np.asarray(self.img, dtype=np.float32)
        arr = arr + rng.normal(0.0, sigma, arr.shape)
        self.img = Image.fromarray(np.clip(arr, 0, 255).astype("uint8"))

    def save(self, name: str):
        self.img.save(OUT_DIR / f"{name}.png")
        self.mask.save(MASK_DIR / f"{name}.png")


def make_shapes_basic() -> dict:
    c = Canvas("white")
    c.rectangle([50, 50, 200, 200], "#4A90D9", 1)
    c.ellipse([250, 80, 450, 280], "#FF6B6B", 2)
    c.polygon([(300, 350), (450, 480), (150, 480)], "#50C878", 3)
    c.rectangle([30, 300, 130, 480], "#FFD700", 4)
    c.save("shapes_basic")
    return {
        "title": "Basic Shapes",
        "classes": ["background", "square", "circle", "triangle", "bar"],
        "difficulty": "easy",
        "teaches": (
            "High-contrast, noise-free regions: nearly every algorithm "
            "succeeds here. Use it as a sanity check and to learn each "
            "method's parameters before moving to harder cases."
        ),
    }


def make_shapes_noisy() -> dict:
    c = Canvas("#d8d8d8")
    c.rectangle([60, 60, 210, 210], "#7a7a7a", 1)
    c.ellipse([260, 90, 460, 290], "#9a9a9a", 2)
    c.polygon([(310, 350), (460, 490), (160, 490)], "#5f5f5f", 3)
    c.add_noise(sigma=28.0, seed=7)
    c.save("shapes_noisy")
    return {
        "title": "Noisy Low-Contrast Shapes",
        "classes": ["background", "square", "circle", "triangle"],
        "difficulty": "hard",
        "teaches": (
            "Heavy Gaussian noise + low gray-level contrast. Global "
            "thresholding (Otsu) becomes unstable, per-pixel clustering "
            "produces speckle, while region-based methods (Felzenszwalb, "
            "Chan-Vese) that integrate evidence over areas degrade more "
            "gracefully. Shows why denoising/regularization matters."
        ),
    }


def make_gradient_road() -> dict:
    c = Canvas("black")
    # Sky-to-ground illumination gradient — breaks global thresholds.
    for y in range(H):
        r = int(50 + 150 * (y / H))
        g = int(120 + 100 * (1 - y / H))
        c.line([(0, y), (W, y)], (r, g, 80), 0)
    c.rectangle([180, 0, 330, H], "#555555", 1)
    c.line([(255, 0), (255, H)], "yellow", 2, width=4)
    for y in range(0, H, 40):
        c.rectangle([250, y, 260, y + 20], "white", 2)
    c.save("road_synthetic")
    return {
        "title": "Road with Illumination Gradient",
        "classes": ["terrain", "road", "lane markings"],
        "difficulty": "medium",
        "teaches": (
            "A smooth illumination gradient across the scene: a single "
            "global threshold cannot separate terrain from road everywhere. "
            "Color clustering and graph-based methods cope better. The thin "
            "lane markings also expose class-imbalance problems — overall "
            "pixel accuracy stays high even when markings are missed."
        ),
    }


def make_cells() -> dict:
    c = Canvas("#1a1a2e")
    rnd = random.Random(42)
    for _ in range(25):
        x, y = rnd.randint(30, W - 30), rnd.randint(30, H - 30)
        r = rnd.randint(20, 60)
        color = rnd.choice(["#e94560", "#c73e57", "#d64566", "#b53a52", "#e94560"])
        c.ellipse([x - r, y - r, x + r, y + r], color, 1, outline="white", width=1)
    c.save("cells_synthetic")
    return {
        "title": "Touching Cells",
        "classes": ["background", "cell"],
        "difficulty": "medium",
        "teaches": (
            "Overlapping blob-like objects, as in microscopy. Binary "
            "foreground extraction is easy, but separating touching "
            "instances is not — this is where watershed with distance-"
            "transform markers shines. Also shows that a perfect semantic "
            "score can hide a poor instance separation."
        ),
    }


def make_city_blocks() -> dict:
    c = Canvas("#2d3436")
    rnd = random.Random(42)
    colors = ["#fdcb6e", "#6c5ce7", "#00b894", "#e17055", "#0984e3", "#d63031"]
    idx = 0
    for row in range(0, H, 80):
        for col in range(0, W, 80):
            margin = rnd.randint(4, 15)
            color = colors[idx % len(colors)]
            c.rectangle(
                [col + margin, row + margin, col + 80 - margin, row + 80 - margin],
                color,
                (idx % len(colors)) + 1,
            )
            idx += 1
    c.save("city_blocks")
    return {
        "title": "City Blocks",
        "classes": [
            "street",
            "block yellow",
            "block purple",
            "block green",
            "block orange",
            "block blue",
            "block red",
        ],
        "difficulty": "easy",
        "teaches": (
            "Many small regions with distinct colors. Clustering with the "
            "right K recovers the palette; superpixel methods (SLIC) "
            "over-segment neatly along the grid. Try setting K wrong and "
            "watch classes merge — a lesson in model selection."
        ),
    }


def make_nature() -> dict:
    c = Canvas("#87CEEB")
    c.rectangle([0, 300, W, H], "#228B22", 1)
    for tx in [80, 200, 350, 450]:
        c.rectangle([tx - 8, 220, tx + 8, 300], "#8B4513", 2)
        c.ellipse([tx - 40, 160, tx + 40, 260], "#006400", 3)
    c.ellipse([400, 30, 480, 110], "#FFD700", 4)
    c.save("nature_synthetic")
    return {
        "title": "Nature Scene",
        "classes": ["sky", "grass", "trunk", "foliage", "sun"],
        "difficulty": "medium",
        "teaches": (
            "Semantically distinct but small classes (trunks, sun) versus "
            "large ones (sky, grass). Mean IoU punishes missing the small "
            "classes; pixel accuracy barely notices. Grass vs. foliage are "
            "both green — close in color space, so clustering may merge "
            "them while graph methods keep them apart spatially."
        ),
    }


def make_indoor() -> dict:
    c = Canvas("#D2B48C")
    c.rectangle([0, 200, W, H], "#8B7355", 1)
    c.rectangle([50, 250, 200, 450], "#A0522D", 2)
    c.rectangle([300, 100, 480, 350], "#4682B4", 3)
    c.ellipse([350, 60, 430, 100], "#C0C0C0", 4)
    c.save("indoor_synthetic")
    return {
        "title": "Indoor Scene",
        "classes": ["wall", "floor", "table", "cabinet", "lamp"],
        "difficulty": "medium",
        "teaches": (
            "Several beige/brown surfaces with similar hues (wall, floor, "
            "table): color-only methods blur their boundaries, while "
            "edge/graph-aware methods keep straight architectural edges. "
            "A good case for comparing boundary F1 across algorithms."
        ),
    }


def make_animal() -> dict:
    c = Canvas("#87CEEB")
    c.rectangle([0, 350, W, H], "#90EE90", 1)
    c.ellipse([180, 250, 330, 380], "#333333", 2)
    c.ellipse([220, 200, 290, 270], "#333333", 2)
    c.polygon([(225, 205), (240, 170), (255, 205)], "#333333", 2)
    c.polygon([(260, 205), (275, 170), (290, 205)], "#333333", 2)
    c.save("animal_synthetic")
    return {
        "title": "Animal Silhouette",
        "classes": ["sky", "grass", "cat"],
        "difficulty": "easy",
        "teaches": (
            "A dark object on bright background — the classic figure/ground "
            "problem where Otsu thresholding and Chan-Vese active contours "
            "excel. Compare their boundaries around the ears: curvature is "
            "where energy-based methods smooth away detail."
        ),
    }


def make_tissue() -> dict:
    c = Canvas("#FFE4E1")
    rnd = random.Random(123)
    for _ in range(40):
        x, y = rnd.randint(10, W - 10), rnd.randint(10, H - 10)
        r = rnd.randint(10, 35)
        shade = rnd.randint(180, 240)
        c.ellipse(
            [x - r, y - r, x + r, y + r],
            (shade, shade - 40, shade - 60),
            1,
            outline=(150, 80, 80),
            width=1,
        )
    c.add_noise(sigma=10.0, seed=3)
    c.save("tissue_synthetic")
    return {
        "title": "Tissue (Low Contrast)",
        "classes": ["background", "nuclei"],
        "difficulty": "hard",
        "teaches": (
            "Low foreground/background contrast plus noise, as in "
            "histopathology. Threshold choice becomes ambiguous and small "
            "IoU differences between algorithms matter. Dice — the standard "
            "medical metric — is more forgiving of small errors on small "
            "structures than IoU; compare both columns here."
        ),
    }


def make_texture() -> dict:
    from PIL import ImageDraw

    c = Canvas("white")
    # Left half: horizontal stripes. Right half: dots. Same colors!
    d: ImageDraw.ImageDraw = c.draw
    for y in range(0, H, 16):
        d.rectangle([0, y, W // 2, y + 7], fill="#333333")
    for y in range(8, H, 20):
        for x in range(W // 2 + 8, W, 20):
            d.ellipse([x - 4, y - 4, x + 4, y + 4], fill="#333333")
    c.mdraw.rectangle([0, 0, W // 2, H], fill=0)
    c.mdraw.rectangle([W // 2 + 1, 0, W, H], fill=1)
    c.save("texture_halves")
    return {
        "title": "Texture: Stripes vs Dots",
        "classes": ["stripes", "dots"],
        "difficulty": "hard",
        "teaches": (
            "Both halves use the exact same two colors — only the texture "
            "differs. Every color/intensity-based algorithm fails by "
            "design, revealing a shared blind spot of classical methods "
            "and motivating learned features (CNNs/Transformers), which "
            "capture texture through local receptive fields."
        ),
    }


EXAMPLES = {
    "shapes_basic": make_shapes_basic,
    "shapes_noisy": make_shapes_noisy,
    "cells_synthetic": make_cells,
    "city_blocks": make_city_blocks,
    "road_synthetic": make_gradient_road,
    "nature_synthetic": make_nature,
    "indoor_synthetic": make_indoor,
    "animal_synthetic": make_animal,
    "tissue_synthetic": make_tissue,
    "texture_halves": make_texture,
}


def generate_images() -> None:
    try:
        import PIL  # noqa: F401
    except ImportError:
        print("Pillow not installed - cannot generate example images")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MASK_DIR.mkdir(parents=True, exist_ok=True)

    manifest = {}
    for name, fn in EXAMPLES.items():
        manifest[name] = fn()
        print(f"  generated {name}")

    # Merge with any existing entries (e.g. real photos from
    # prepare_real_examples.py) instead of overwriting them.
    manifest_path = OUT_DIR / "examples.json"
    merged = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    merged.update(manifest)
    manifest_path.write_text(json.dumps(merged, indent=2) + "\n")

    print(f"Generated {len(manifest)} examples (image + mask) in {OUT_DIR}")


if __name__ == "__main__":
    generate_images()
