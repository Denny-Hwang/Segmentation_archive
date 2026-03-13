#!/usr/bin/env python3
"""Generate publication-ready figures for the Segmentation Archive.

Produces:
  - Taxonomy diagram (from Mermaid source)
  - Timeline / evolution chart (Matplotlib)
  - Model comparison table & bar chart (Matplotlib)
  - Pipeline diagram (from Mermaid source)

Usage:
    python scripts/figures/generate_figures.py

Outputs are saved to docs/figures/*.png.
Mermaid diagrams require `mmdc` (mermaid-cli) for PNG rendering; if unavailable
the script prints instructions and skips those outputs.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
FIGURES_DIR = REPO_ROOT / "docs" / "figures"
MERMAID_DIR = FIGURES_DIR  # .mermaid sources live alongside .png outputs


def _has_mmdc() -> bool:
    return shutil.which("mmdc") is not None


def render_mermaid(src: Path, out: Path) -> bool:
    """Render a .mermaid file to PNG using mermaid-cli."""
    if not _has_mmdc():
        print(f"  [SKIP] mmdc not found – cannot render {src.name}")
        return False
    try:
        subprocess.run(
            ["mmdc", "-i", str(src), "-o", str(out), "-b", "transparent", "-w", "2048"],
            check=True,
            capture_output=True,
        )
        print(f"  [OK] {out.name}")
        return True
    except subprocess.CalledProcessError as exc:
        print(f"  [FAIL] {src.name}: {exc.stderr.decode()[:200]}")
        return False


# ---------------------------------------------------------------------------
# Matplotlib figures
# ---------------------------------------------------------------------------

def generate_comparison_chart() -> None:
    """Generate a horizontal bar chart comparing models across datasets."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("  [SKIP] matplotlib not installed – skipping comparison chart")
        return

    benchmarks = {
        "Synapse Multi-Organ\n(mDSC %)": {
            "U-Net": 76.85, "Att. U-Net": 77.77, "UNet++": 78.30,
            "TransUNet": 77.48, "Swin-Unet": 79.13, "nnU-Net": 82.50,
        },
        "ADE20K\n(mIoU %)": {
            "DeepLab v3+": 45.47, "SegFormer-B5": 51.80,
            "Mask2Former": 56.01, "OneFormer": 57.40,
        },
        "Cityscapes val\n(mIoU %)": {
            "DeepLab v3+": 80.90, "SegFormer-B5": 84.00,
            "Mask2Former": 83.30, "OneFormer": 84.40,
        },
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    colors = ["#4A90D9", "#50C878", "#FF6B6B", "#FFD700", "#DA70D6", "#20B2AA"]

    for ax, (title, data) in zip(axes, benchmarks.items()):
        models = list(data.keys())
        scores = list(data.values())
        bars = ax.barh(models, scores, color=colors[:len(models)], edgecolor="white")
        ax.set_xlabel("Score")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.invert_yaxis()
        for bar, score in zip(bars, scores):
            ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                    f"{score:.1f}", va="center", fontsize=9)

    fig.suptitle("Model Performance Comparison (representative benchmarks)",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    out = FIGURES_DIR / "model_comparison_chart.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {out.name}")


def generate_timeline_chart() -> None:
    """Generate a visual timeline figure using matplotlib."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [SKIP] matplotlib not installed – skipping timeline chart")
        return

    events = [
        (2014, "FCN", "semantic", -1),
        (2015, "U-Net", "medical", 1),
        (2015, "DeepLab v1", "semantic", -1.5),
        (2016, "3D U-Net", "medical", 1.5),
        (2017, "PSPNet", "semantic", -1),
        (2017, "Mask R-CNN", "instance", -2),
        (2018, "UNet++", "medical", 1),
        (2018, "DeepLab v3+", "semantic", -1.5),
        (2020, "ViT", "foundation", 2),
        (2021, "TransUNet", "medical", 1.5),
        (2021, "Swin-Unet", "medical", 1),
        (2021, "SegFormer", "semantic", -1),
        (2021, "nnU-Net", "medical", 2),
        (2022, "Mask2Former", "universal", -2),
        (2023, "OneFormer", "universal", -1.5),
        (2023, "SAM", "foundation", 2),
        (2024, "SAM 2", "foundation", 2.5),
    ]

    cat_colors = {
        "semantic": "#4A90D9",
        "medical": "#FF6B6B",
        "instance": "#FFD700",
        "foundation": "#50C878",
        "universal": "#DA70D6",
    }

    fig, ax = plt.subplots(figsize=(16, 6))
    ax.axhline(0, color="gray", linewidth=0.8, zorder=0)

    for year, name, cat, yoff in events:
        color = cat_colors.get(cat, "#888")
        ax.scatter(year, 0, s=80, color=color, zorder=5)
        ax.annotate(
            name, (year, 0), xytext=(0, yoff * 25),
            textcoords="offset points", ha="center", fontsize=8,
            fontweight="bold",
            arrowprops=dict(arrowstyle="-", color=color, lw=0.8),
            color=color,
        )

    # Legend
    for cat, color in cat_colors.items():
        ax.scatter([], [], color=color, label=cat.title(), s=60)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)

    ax.set_xlim(2013, 2025)
    ax.set_yticks([])
    ax.set_xlabel("Year")
    ax.set_title("Evolution of Image Segmentation Methods", fontweight="bold")
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    fig.tight_layout()

    out = FIGURES_DIR / "timeline_evolution_chart.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {out.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    print("Generating figures …")

    # Mermaid diagrams
    mermaid_files = {
        "taxonomy_diagram.mermaid": "taxonomy_diagram.png",
        "pipeline_diagram.mermaid": "pipeline_diagram.png",
        "timeline_evolution.mermaid": "timeline_evolution_mermaid.png",
    }
    for src_name, out_name in mermaid_files.items():
        src = MERMAID_DIR / src_name
        if src.exists():
            render_mermaid(src, FIGURES_DIR / out_name)
        else:
            print(f"  [SKIP] {src_name} not found")

    # Matplotlib figures
    generate_comparison_chart()
    generate_timeline_chart()

    print("\nDone. Figures saved to docs/figures/")
    if not _has_mmdc():
        print(
            "\nNote: Install mermaid-cli for Mermaid diagram rendering:\n"
            "  npm install -g @mermaid-js/mermaid-cli"
        )


if __name__ == "__main__":
    main()
