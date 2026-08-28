"""Metrics Guide - study segmentation evaluation metrics interactively.

Companion page to the Segmentation Lab: an interactive playground where a
synthetic prediction can be perturbed (shift, shrink, ragged edges,
spurious blobs) while every metric reacts live, followed by full study
cards for each metric and guidance on choosing metrics per task.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from seg_lab.metric_info import METRIC_INFO  # noqa: E402
from seg_lab.metrics import (  # noqa: E402
    adjusted_rand_index,
    boundary_f1_per_class,
    hausdorff95_per_class,
    iou_dice_per_class,
    pixel_accuracy,
)

st.set_page_config(page_title="Metrics Guide - Segmentation Archive", layout="wide")

SIZE = 256


# ---------------------------------------------------------------------------
# Interactive demo helpers
# ---------------------------------------------------------------------------


def disk_mask(cx: float, cy: float, r: float, ragged: float = 0.0) -> np.ndarray:
    """Binary disk; `ragged` adds sinusoidal perturbation to the radius."""
    yy, xx = np.mgrid[0:SIZE, 0:SIZE]
    dy, dx = yy - cy, xx - cx
    dist = np.hypot(dx, dy)
    if ragged > 0:
        theta = np.arctan2(dy, dx)
        r_theta = r * (
            1
            + ragged * 0.12 * np.sin(9 * theta)
            + ragged * 0.08 * np.sin(17 * theta + 1.3)
        )
        return dist <= r_theta
    return dist <= r


def compute_demo_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    p, g = pred.astype(np.int32), gt.astype(np.int32)
    ious, dices = iou_dice_per_class(p, g, 2)
    bf1 = boundary_f1_per_class(p, g, 2, tolerance=2)
    hd = hausdorff95_per_class(p, g, 2)
    return {
        "Pixel Acc": pixel_accuracy(p, g),
        "IoU (fg)": ious[1],
        "Dice (fg)": dices[1],
        "Boundary F1": float(np.nanmean(bf1)),
        "HD95 (px)": float(np.nanmean(hd)),
        "ARI": adjusted_rand_index(p, g),
    }


def demo_overlay(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Visualize GT vs prediction: TP green, FN blue, FP red."""
    img = np.full((SIZE, SIZE, 3), 245, dtype=np.uint8)
    img[gt & pred] = [80, 200, 120]  # true positive
    img[gt & ~pred] = [70, 130, 220]  # false negative (missed)
    img[~gt & pred] = [230, 80, 80]  # false positive (hallucinated)
    return img


def render_interactive_demo() -> None:
    st.markdown("## 🎛️ Feel the metrics")
    st.markdown(
        "A ground-truth disk (radius 60px) versus a prediction you distort. "
        "Move the sliders and watch **which metrics react and which stay "
        "silent** - that difference is the entire art of choosing metrics."
    )

    ctrl, viz, nums = st.columns([1.1, 1.2, 1.4])
    with ctrl:
        shift = st.slider("Shift prediction (px)", -60, 60, 0, 2)
        scale = st.slider("Scale prediction (%)", 40, 160, 100, 5)
        ragged = st.slider("Ragged boundary", 0.0, 1.0, 0.0, 0.1)
        blob = st.slider(
            "Spurious blob distance (px)",
            0,
            90,
            0,
            10,
            help="Adds a small false-positive blob this far from the object. "
            "0 = no blob.",
        )

    gt = disk_mask(SIZE / 2, SIZE / 2, 60)
    pred = disk_mask(SIZE / 2 + shift, SIZE / 2, 60 * scale / 100, ragged=ragged)
    if blob > 0:
        pred |= disk_mask(SIZE / 2 + 60 + blob, SIZE / 2 - 60 - blob * 0.3, 9)

    metrics = compute_demo_metrics(pred, gt)

    with viz:
        st.image(
            demo_overlay(pred, gt),
            caption="🟩 correct (TP) · 🟦 missed (FN) · 🟥 hallucinated (FP)",
            use_container_width=True,
        )
    with nums:
        cols = st.columns(3)
        for i, (name, val) in enumerate(metrics.items()):
            fmt = f"{val:.1f}" if "HD95" in name else f"{val:.3f}"
            cols[i % 3].metric(name, fmt)
        st.caption(
            "Things to try — **Shift by ~10px**: IoU drops ~2× faster than "
            "pixel accuracy; Boundary F1 collapses first. **IoU vs Dice**: "
            "Dice is always ≥ IoU, related by Dice = 2·IoU/(1+IoU). "
            "**Spurious blob**: Dice barely moves (~0.01) while HD95 "
            "explodes — that is why medical challenges report both. "
            "**Ragged boundary**: overlap metrics stay high while Boundary "
            "F1 plummets."
        )


# ---------------------------------------------------------------------------
# Study cards
# ---------------------------------------------------------------------------


def render_metric_cards() -> None:
    st.markdown("## 📖 Metric study cards")
    tabs = st.tabs([info["name"] for info in METRIC_INFO.values()])
    for tab, (key, info) in zip(tabs, METRIC_INFO.items()):
        with tab:
            direction = (
                "⬆️ higher is better" if info["higher_better"] else "⬇️ lower is better"
            )
            st.markdown(
                f"**Also known as:** {info['aka']} · **Range:** {info['range']} · "
                f"{direction}"
            )
            st.latex(info["formula"])
            st.markdown(info["intuition"])
            col_p, col_c = st.columns(2)
            with col_p:
                st.markdown("#### ✅ Strengths")
                for p in info["pros"]:
                    st.markdown(f"- {p}")
            with col_c:
                st.markdown("#### ⚠️ Weaknesses")
                for c in info["cons"]:
                    st.markdown(f"- {c}")
            st.markdown("#### 🧨 Limitations & pitfalls")
            st.markdown(info["limitations"])
            st.success(f"**Use when:** {info['use_when']}")


def render_choosing_guide() -> None:
    st.markdown("## 🧭 Which metric for which task?")
    st.markdown(
        """
| Task | Primary metrics | Why |
|---|---|---|
| Semantic segmentation benchmarks (Cityscapes, ADE20K) | **mIoU** (+ per-class IoU) | Class-balanced; universal comparison currency |
| Medical image segmentation | **Dice + HD95** | Dice for overlap, HD95 for worst-case contour error - both required by BraTS/Decathlon |
| Boundary-critical tasks (matting, lesion contours) | **Boundary F1** + an overlap metric | Overlap metrics cannot see contour quality |
| Unsupervised / classical segmentation | **ARI** first, then mapped per-class metrics | No label mapping bias; then diagnose per class |
| Instance segmentation (COCO) | **AP over IoU thresholds** | Detection-style: per-object matching, not per-pixel *(not computed in the Lab)* |
| Panoptic segmentation | **PQ = SQ × RQ** | Unifies semantic + instance quality *(not computed in the Lab)* |
| Class-imbalanced screening (tiny lesions) | **Recall / FNR per class** + Dice | Missing the small class is the costly error |
        """
    )
    st.markdown(
        """
### Reporting checklist (the part reviewers check)

1. **Never compare IoU against Dice** - same information, different scales
   (Dice = 2·IoU/(1+IoU)); state which one you report.
2. **State the empty-class convention** - skipped, scored 1, or scored 0
   changes means by whole points on sparse datasets.
3. **State averaging** - macro (per-class then mean, standard) vs
   frequency-weighted (rewards big classes) differ hugely under imbalance.
4. **Scale distance tolerances with resolution** - Boundary-F1 θ and HD95
   in pixels are not comparable across image sizes; report physical units
   when available (mm in medical).
5. **Report per-class tables, not just means** - means hide exactly the
   failures that matter (rare classes, thin structures).
6. **Pair complementary metrics** - one overlap metric (mIoU or Dice) +
   one boundary metric (BF1 or HD95) covers most blind spots.
        """
    )


def main() -> None:
    st.title("📏 Segmentation Metrics Guide")
    st.markdown(
        "*No single number can summarize a segmentation.* Every metric "
        "answers a different question - and stays blind to a different "
        "failure. This guide lets you **feel** each metric interactively, "
        "study its formula and limitations, and pick the right combination "
        "for your task. Apply them to real algorithms in the "
        "**Segmentation Lab**."
    )
    st.markdown("---")
    render_interactive_demo()
    st.markdown("---")
    render_metric_cards()
    st.markdown("---")
    render_choosing_guide()


if __name__ == "__main__":
    main()
