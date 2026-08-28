#!/usr/bin/env python3
"""Prepare real-photograph examples (with reference masks) for the Segmentation Lab.

Uses photographs bundled with scikit-image, all public domain / CC0 /
no-known-restrictions, so they can be committed to the repository:

- coins ........ Ancient Greek coins (Brooklyn Museum, no known restrictions)
- coffee ....... Espresso cup (Rachel Michetti, CC0, courtesy Pikolo Espresso Bar)
- ihc .......... Immunohistochemistry tissue (no known restrictions)
- hubble ....... Hubble eXtreme Deep Field (NASA, public domain)

Unlike the synthetic examples, real photographs have no exact ground
truth. Each mask here is a *reference annotation* produced by the
deterministic pipeline in this script (stated per example in `gt_note`)
and visually verified by a human. Expect a small honest error band near
boundaries - which is itself worth teaching: real benchmarks (Cityscapes,
COCO, BraTS) also carry annotation noise.

Everything is deterministic: re-running reproduces identical files.
Entries are merged into assets/examples/examples.json alongside the
synthetic ones from generate_example_images.py.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = REPO_ROOT / "assets" / "examples"
MASK_DIR = OUT_DIR / "masks"


def make_coins():
    """Binary coin masks via marker watershed on the Sobel gradient."""
    import numpy as np
    from scipy import ndimage as ndi
    from skimage import data
    from skimage.filters import sobel
    from skimage.segmentation import watershed

    gray = data.coins()
    markers = np.zeros_like(gray, dtype=np.int32)
    markers[gray < 30] = 1  # certain background
    markers[gray > 150] = 2  # certain coin
    seg = watershed(sobel(gray), markers) == 2
    seg = ndi.binary_fill_holes(seg)
    lab, nlab = ndi.label(seg)
    sizes = ndi.sum(seg, lab, range(1, nlab + 1))
    seg = np.isin(lab, np.nonzero(sizes >= 300)[0] + 1)
    image = np.stack([gray] * 3, axis=-1)
    meta = {
        "title": "Photo: Ancient Coins",
        "classes": ["background", "coin"],
        "difficulty": "medium",
        "teaches": (
            "A real photograph with a textured background and uneven "
            "illumination: plain global thresholding leaks or eats coins, "
            "which is why the classic recipe is edges + watershed. Also a "
            "natural instance case - watch watershed separate all 24 coins."
        ),
        "source": "scikit-image `coins` - Brooklyn Museum, no known copyright restrictions",
        "gt_note": (
            "Reference mask: marker-controlled watershed on the Sobel "
            "gradient (markers: gray<30 background, gray>150 coin), holes "
            "filled, regions <300px removed; human-verified (24/24 coins)."
        ),
    }
    return image, seg.astype(np.uint8), meta


def make_coffee():
    """Multi-class mask drawn manually as verified geometric primitives."""
    import numpy as np
    from PIL import Image, ImageDraw
    from skimage import data

    image = data.coffee()
    m = Image.new("L", (600, 400), 0)
    d = ImageDraw.Draw(m)
    d.ellipse([78, 127, 530, 383], fill=1)  # saucer
    d.ellipse([158, 28, 398, 140], fill=2)  # cup rim
    d.ellipse([178, 75, 390, 250], fill=2)  # cup upper body
    d.ellipse([200, 150, 378, 287], fill=2)  # cup lower body
    d.ellipse([192, 220, 272, 312], fill=2)  # handle outer
    d.ellipse([202, 257, 236, 290], fill=1)  # handle hole (saucer visible)
    d.ellipse([218, 106, 356, 203], fill=3)  # coffee liquid
    d.ellipse([354, 246, 446, 318], fill=4)  # spoon bowl
    d.polygon([(398, 60), (415, 57), (423, 248), (396, 258)], fill=4)  # spoon handle
    mask = np.array(m, dtype=np.uint8)
    meta = {
        "title": "Photo: Espresso Cup",
        "classes": ["table", "saucer", "cup", "coffee", "spoon"],
        "difficulty": "hard",
        "teaches": (
            "An everyday photo: specular highlights on ceramic, shadows on "
            "the saucer, a metallic spoon reflecting its surroundings, and "
            "a wooden table sharing hues with the saucer. Color clustering "
            "splits the white cup into highlight/shadow clusters; graph "
            "methods leak through the soft shadow edges. This is why "
            "real-world segmentation needed learned features."
        ),
        "source": (
            "scikit-image `coffee` - photo by Rachel Michetti, CC0, "
            "courtesy Pikolo Espresso Bar"
        ),
        "gt_note": (
            "Reference mask: hand-annotated geometric primitives (ellipses/"
            "polygons) verified against the photo; boundaries are "
            "approximate within a few pixels - real datasets carry similar "
            "annotation noise."
        ),
    }
    return image, mask, meta


def make_ihc():
    """DAB-positive epithelium via color deconvolution (smoothed reference)."""
    import numpy as np
    from scipy import ndimage as ndi
    from skimage import data
    from skimage.color import rgb2hed

    image = data.immunohistochemistry()
    dab = ndi.gaussian_filter(rgb2hed(image)[:, :, 2], 4)
    m = dab > np.percentile(dab, 45)
    m = ndi.binary_closing(m, iterations=2)
    lab, nlab = ndi.label(m)
    sizes = ndi.sum(m, lab, range(1, nlab + 1))
    m = np.isin(lab, np.nonzero(sizes >= 1500)[0] + 1)
    m = ndi.binary_fill_holes(m)
    meta = {
        "title": "Photo: IHC Tissue",
        "classes": ["stroma/background", "DAB-positive epithelium"],
        "difficulty": "hard",
        "teaches": (
            "Real histopathology: the brown DAB stain marks epithelium, "
            "blue hematoxylin the stroma. Intensity alone cannot separate "
            "the stains (both are dark) - color is essential, and even in "
            "color the boundary is genuinely fuzzy. Compare Dice vs "
            "Boundary F1 here: overlap can look fine while contours are "
            "ragged, the everyday reality of medical segmentation."
        ),
        "source": "scikit-image `immunohistochemistry` - no known copyright restrictions",
        "gt_note": (
            "Reference mask: DAB channel of HED color deconvolution, "
            "Gaussian σ=4, threshold at its 45th percentile, closed, "
            "regions <1500px removed, holes filled; human-verified. "
            "Stain-derived, so treat boundaries as approximate."
        ),
    }
    return image, m.astype(np.uint8), meta


def make_hubble():
    """Bright astronomical sources by luminance threshold + minimum area."""
    import numpy as np
    from PIL import Image
    from scipy import ndimage as ndi
    from skimage import data
    from skimage.color import rgb2gray

    image = data.hubble_deep_field()
    m = rgb2gray(image) > 0.16
    m = ndi.binary_opening(m, iterations=1)
    lab, nlab = ndi.label(m)
    sizes = ndi.sum(m, lab, range(1, nlab + 1))
    m = np.isin(lab, np.nonzero(sizes >= 12)[0] + 1)

    # Downscale to keep the repository small (mask with NEAREST)
    target_w = 640
    h, w = image.shape[:2]
    scale = target_w / w
    new_size = (target_w, int(h * scale))
    image = np.asarray(Image.fromarray(image).resize(new_size, Image.BILINEAR))
    m = (
        np.asarray(
            Image.fromarray(m.astype(np.uint8) * 255).resize(new_size, Image.NEAREST)
        )
        > 127
    )
    meta = {
        "title": "Photo: Hubble Deep Field",
        "classes": ["space", "bright source"],
        "difficulty": "medium",
        "teaches": (
            "Astronomical source extraction: thousands of tiny objects on a "
            "near-black background with sensor noise. Pixel accuracy is "
            "meaningless (predicting 'all space' scores ~95%) - this is the "
            "class-imbalance lesson at its most extreme. Region size "
            "filtering and threshold choice dominate the result."
        ),
        "source": "scikit-image `hubble_deep_field` - NASA, public domain",
        "gt_note": (
            "Reference mask: sources brighter than 0.16 luminance after "
            "opening, regions <12px removed (a stated brightness/area "
            "criterion, as astronomical catalogs use); human-verified."
        ),
    }
    return image, m.astype(np.uint8), meta


REAL_EXAMPLES = {
    "photo_coins": make_coins,
    "photo_coffee": make_coffee,
    "photo_ihc": make_ihc,
    "photo_hubble": make_hubble,
}


def prepare() -> None:
    try:
        import skimage  # noqa: F401
        from PIL import Image
    except ImportError as exc:
        print(f"Missing dependency ({exc.name}) - cannot prepare real examples")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MASK_DIR.mkdir(parents=True, exist_ok=True)

    manifest_path = OUT_DIR / "examples.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}

    for key, fn in REAL_EXAMPLES.items():
        image, mask, meta = fn()
        Image.fromarray(image).save(OUT_DIR / f"{key}.png")
        Image.fromarray(mask).save(MASK_DIR / f"{key}.png")
        manifest[key] = meta
        print(f"  prepared {key} {image.shape[:2]}")

    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Prepared {len(REAL_EXAMPLES)} real-photo examples in {OUT_DIR}")


if __name__ == "__main__":
    prepare()
