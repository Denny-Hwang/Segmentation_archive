#!/usr/bin/env python3
"""Generate synthetic example images for the Segmentation Playground.

Creates simple geometric / pattern images that are copyright-free,
saved to assets/examples/. These serve as demo inputs when users
do not upload their own image.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = REPO_ROOT / "assets" / "examples"


def generate_images() -> None:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        print("Pillow not installed – cannot generate example images")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    W, H = 512, 512

    # 1 - Colored shapes on white
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    d.rectangle([50, 50, 200, 200], fill="#4A90D9")
    d.ellipse([250, 80, 450, 280], fill="#FF6B6B")
    d.polygon([(300, 350), (450, 480), (150, 480)], fill="#50C878")
    d.rectangle([30, 300, 130, 480], fill="#FFD700")
    img.save(OUT_DIR / "shapes_basic.png")

    # 2 - Overlapping circles (cell-like)
    img = Image.new("RGB", (W, H), "#1a1a2e")
    d = ImageDraw.Draw(img)
    import random
    random.seed(42)
    for _ in range(25):
        x, y = random.randint(30, W - 30), random.randint(30, H - 30)
        r = random.randint(20, 60)
        color = random.choice(["#e94560", "#0f3460", "#16213e", "#533483", "#e94560"])
        d.ellipse([x - r, y - r, x + r, y + r], fill=color, outline="white", width=1)
    img.save(OUT_DIR / "cells_synthetic.png")

    # 3 - Grid / city-like blocks
    img = Image.new("RGB", (W, H), "#2d3436")
    d = ImageDraw.Draw(img)
    colors = ["#fdcb6e", "#6c5ce7", "#00b894", "#e17055", "#0984e3", "#d63031"]
    idx = 0
    for row in range(0, H, 80):
        for col in range(0, W, 80):
            margin = random.randint(4, 15)
            c = colors[idx % len(colors)]
            d.rectangle([col + margin, row + margin, col + 80 - margin, row + 80 - margin], fill=c)
            idx += 1
    img.save(OUT_DIR / "city_blocks.png")

    # 4 - Gradient with stripes (road-like)
    img = Image.new("RGB", (W, H), "black")
    d = ImageDraw.Draw(img)
    for y in range(H):
        r = int(50 + 150 * (y / H))
        g = int(120 + 100 * (1 - y / H))
        b = 80
        d.line([(0, y), (W, y)], fill=(r, g, b))
    # Road
    d.rectangle([180, 0, 330, H], fill="#555555")
    d.line([(255, 0), (255, H)], fill="yellow", width=3)
    for y in range(0, H, 40):
        d.rectangle([250, y, 260, y + 20], fill="white")
    img.save(OUT_DIR / "road_synthetic.png")

    # 5 - Nature-like (sky + ground + tree shapes)
    img = Image.new("RGB", (W, H), "#87CEEB")
    d = ImageDraw.Draw(img)
    d.rectangle([0, 300, W, H], fill="#228B22")
    # Trees
    for tx in [80, 200, 350, 450]:
        d.rectangle([tx - 8, 220, tx + 8, 300], fill="#8B4513")
        d.ellipse([tx - 40, 160, tx + 40, 260], fill="#006400")
    # Sun
    d.ellipse([400, 30, 480, 110], fill="#FFD700")
    img.save(OUT_DIR / "nature_synthetic.png")

    # 6 - Indoor-like (floor + walls + furniture)
    img = Image.new("RGB", (W, H), "#F5F5DC")
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, W, 200], fill="#D2B48C")  # wall
    d.rectangle([0, 200, W, H], fill="#8B7355")  # floor
    d.rectangle([50, 250, 200, 450], fill="#A0522D")  # table
    d.rectangle([300, 100, 480, 350], fill="#4682B4")  # cabinet
    d.ellipse([350, 60, 430, 100], fill="#C0C0C0")  # lamp
    img.save(OUT_DIR / "indoor_synthetic.png")

    # 7 - Animal-like silhouettes
    img = Image.new("RGB", (W, H), "#87CEEB")
    d = ImageDraw.Draw(img)
    d.rectangle([0, 350, W, H], fill="#90EE90")
    # Simple cat silhouette
    d.ellipse([180, 250, 330, 380], fill="#333333")  # body
    d.ellipse([220, 200, 290, 270], fill="#333333")  # head
    d.polygon([(225, 205), (240, 170), (255, 205)], fill="#333333")  # ear L
    d.polygon([(260, 205), (275, 170), (290, 205)], fill="#333333")  # ear R
    img.save(OUT_DIR / "animal_synthetic.png")

    # 8 - Medical-like (tissue pattern)
    img = Image.new("RGB", (W, H), "#FFE4E1")
    d = ImageDraw.Draw(img)
    random.seed(123)
    for _ in range(40):
        x, y = random.randint(10, W - 10), random.randint(10, H - 10)
        r = random.randint(10, 35)
        shade = random.randint(180, 240)
        d.ellipse([x - r, y - r, x + r, y + r], fill=(shade, shade - 40, shade - 60),
                  outline=(150, 80, 80), width=1)
    img.save(OUT_DIR / "tissue_synthetic.png")

    print(f"Generated {len(list(OUT_DIR.glob('*.png')))} example images in {OUT_DIR}")


if __name__ == "__main__":
    generate_images()
