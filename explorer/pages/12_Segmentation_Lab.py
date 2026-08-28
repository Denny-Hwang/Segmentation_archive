"""Segmentation Lab - experiment with classical algorithms on real examples.

The heart of the archive's "learn by doing" experience: pick an example
image with known ground truth (or upload your own), run several classical
segmentation algorithms side-by-side with tunable parameters, and compare
them with a full evaluation-metric suite - each metric explained inline
with its strengths, weaknesses, and limitations.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from seg_lab.algorithms import ALGORITHMS, run_algorithm  # noqa: E402
from seg_lab.examples import (  # noqa: E402
    class_legend_html,
    load_image,
    load_manifest,
    load_mask,
    overlay_labels,
    resize_pair,
)
from seg_lab.metric_info import METRIC_COLUMNS, METRIC_INFO  # noqa: E402
from seg_lab.metrics import compute_all  # noqa: E402

st.set_page_config(page_title="Segmentation Lab - Segmentation Archive", layout="wide")

DIFFICULTY_BADGE = {"easy": "🟢 easy", "medium": "🟡 medium", "hard": "🔴 hard"}
DEFAULT_ALGOS = ["otsu", "kmeans", "felzenszwalb", "watershed"]


@st.cache_data(show_spinner=False)
def cached_run(alg_key: str, image: np.ndarray, params_items: tuple) -> tuple:
    """Run one algorithm, cached on (algorithm, image content, parameters)."""
    t0 = time.perf_counter()
    labels = run_algorithm(alg_key, image, dict(params_items))
    return labels, time.perf_counter() - t0


@st.cache_data(show_spinner=False)
def cached_metrics(
    labels: np.ndarray, gt: np.ndarray, n_classes: int, mapping: str, tol: int
) -> dict:
    return compute_all(labels, gt, n_classes, mapping=mapping, boundary_tol=tol)


def render_param_widgets(alg_key: str) -> dict:
    """Render sidebar widgets from an algorithm's parameter specs."""
    alg = ALGORITHMS[alg_key]
    values = {}
    for p in alg.params:
        wkey = f"lab_{alg_key}_{p.name}"
        if p.ptype == "choice":
            values[p.name] = st.selectbox(
                p.label,
                p.choices,
                index=p.choices.index(p.default),
                key=wkey,
                help=p.help or None,
            )
        elif p.ptype == "int":
            values[p.name] = st.slider(
                p.label,
                int(p.min),
                int(p.max),
                int(p.default),
                int(p.step or 1),
                key=wkey,
                help=p.help or None,
            )
        else:
            values[p.name] = st.slider(
                p.label,
                float(p.min),
                float(p.max),
                float(p.default),
                float(p.step or 0.1),
                key=wkey,
                help=p.help or None,
            )
    return values


def render_algorithm_notes(alg_key: str) -> None:
    alg = ALGORITHMS[alg_key]
    with st.expander(f"How {alg.name} works ({alg.category}, {alg.year})"):
        st.markdown(alg.how_it_works)
        col_s, col_w = st.columns(2)
        with col_s:
            st.markdown("**Strengths**")
            for s in alg.strengths:
                st.markdown(f"- {s}")
        with col_w:
            st.markdown("**Weaknesses**")
            for w in alg.weaknesses:
                st.markdown(f"- {w}")


def render_metric_explainers(keys: list[str]) -> None:
    """Inline metric study cards: intuition, formula, pros/cons, limitations."""
    st.markdown("#### What do these metrics mean?")
    st.caption(
        "Every metric tells a different story - and hides a different failure. "
        "Full details, interactive demos, and a metric-selection guide live in "
        "the **Metrics Guide** page."
    )
    for key in keys:
        info = METRIC_INFO[key]
        direction = "higher is better" if info["higher_better"] else "lower is better"
        with st.expander(f"{info['name']} · {info['aka']} · {direction}"):
            st.markdown(info["intuition"])
            st.latex(info["formula"])
            col_p, col_c = st.columns(2)
            with col_p:
                st.markdown("**Strengths**")
                for p in info["pros"]:
                    st.markdown(f"- {p}")
            with col_c:
                st.markdown("**Weaknesses**")
                for c in info["cons"]:
                    st.markdown(f"- {c}")
            st.markdown(f"**Limitations & pitfalls:** {info['limitations']}")
            st.markdown(f"**Use when:** {info['use_when']}")


def render_metric_charts(rows: list[dict]) -> None:
    """Grouped bar chart for 0-1 metrics plus a separate HD95 chart."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        return

    names = [r["Algorithm"] for r in rows]
    ratio_metrics = [
        ("pixel_accuracy", "Pixel Acc"),
        ("mean_iou", "mIoU"),
        ("mean_dice", "mDice"),
        ("boundary_f1", "Boundary F1"),
        ("ari", "ARI"),
    ]
    col1, col2 = st.columns([2, 1])
    with col1:
        fig = go.Figure()
        for mkey, mname in ratio_metrics:
            fig.add_bar(name=mname, x=names, y=[r[mkey] for r in rows])
        fig.update_layout(
            barmode="group",
            title="Score metrics (higher is better)",
            yaxis_range=[0, 1],
            height=380,
            margin=dict(t=40, b=10),
            legend=dict(orientation="h", y=-0.25),
        )
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig2 = go.Figure()
        fig2.add_bar(x=names, y=[r["hd95"] for r in rows], marker_color="#e17055")
        fig2.update_layout(
            title="HD95 in pixels (lower is better)",
            height=380,
            margin=dict(t=40, b=10),
        )
        st.plotly_chart(fig2, use_container_width=True)


def main() -> None:
    st.title("🧪 Segmentation Lab")
    st.markdown(
        "Run **classical segmentation algorithms** side-by-side on examples "
        "with known ground truth, tune their parameters live, and see how "
        "every evaluation metric responds. No GPU, no downloads - everything "
        "computes in seconds on CPU. For deep models, visit the "
        "**Playground** page; for metric theory, the **Metrics Guide**."
    )

    examples = load_manifest()

    # ----- Sidebar: algorithms & options -----
    st.sidebar.subheader("Algorithms")
    alg_keys = st.sidebar.multiselect(
        "Compare",
        options=list(ALGORITHMS.keys()),
        default=[k for k in DEFAULT_ALGOS if k in ALGORITHMS],
        format_func=lambda k: f"{ALGORITHMS[k].name} ({ALGORITHMS[k].category})",
    )
    alg_params: dict[str, dict] = {}
    for k in alg_keys:
        with st.sidebar.expander(f"⚙️ {ALGORITHMS[k].name}"):
            st.caption(ALGORITHMS[k].summary)
            alg_params[k] = render_param_widgets(k)

    st.sidebar.markdown("---")
    st.sidebar.subheader("Evaluation settings")
    mapping = st.sidebar.radio(
        "Segment → class mapping",
        ["majority", "hungarian"],
        help=(
            "Classical algorithms output anonymous segments, so segments must "
            "be mapped to ground-truth classes before scoring. **majority** "
            "(many-to-one) gives each segment its best-overlap class - "
            "measures achievable accuracy but flatters over-segmentation. "
            "**hungarian** (one-to-one) allows each class only one segment - "
            "harsher, but exposes over-segmentation. Compare both!"
        ),
    )
    boundary_tol = st.sidebar.slider(
        "Boundary F1 tolerance θ (px)",
        1,
        5,
        2,
        help="A predicted boundary pixel within θ px of a true boundary counts as correct.",
    )
    max_size = st.sidebar.select_slider(
        "Processing size (px)",
        options=[256, 320, 384, 512],
        value=384,
        help="Images are downscaled to this size before segmentation. Smaller = faster.",
    )
    overlay_alpha = st.sidebar.slider("Overlay opacity", 0.2, 0.9, 0.55, 0.05)

    # ----- Input selection -----
    st.markdown("---")
    example_tab, upload_tab = st.tabs(["📚 Example Gallery", "📤 Upload Your Own"])

    image = gt = None
    class_names: list[str] = []
    teaches = ""

    with example_tab:
        if not examples:
            st.warning(
                "No example images found. Run "
                "`python scripts/figures/generate_example_images.py` first."
            )
        else:
            # Real photographs first, then synthetic teaching images
            keys = sorted(examples, key=lambda k: not k.startswith("photo_"))
            cols = st.columns(5)
            for i, key in enumerate(keys):
                ex = examples[key]
                with cols[i % 5]:
                    st.image(str(ex.image_path), use_container_width=True)
                    picked = st.button(
                        f"{ex.title}",
                        key=f"pick_{key}",
                        help=f"{DIFFICULTY_BADGE[ex.difficulty]} — {ex.teaches}",
                        use_container_width=True,
                    )
                    if picked:
                        st.session_state["lab_example"] = key
            chosen = st.session_state.get("lab_example", keys[0])
            ex = examples[chosen]
            image = load_image(ex.image_path)
            gt = load_mask(ex.mask_path)
            class_names = ex.classes
            teaches = ex.teaches
            st.info(
                f"**{ex.title}** · {DIFFICULTY_BADGE[ex.difficulty]} · "
                f"{len(class_names)} classes\n\n💡 **What this example teaches:** {teaches}"
            )
            if ex.source:
                st.caption(f"📷 **Source:** {ex.source}")
            if ex.gt_note:
                st.caption(f"🏷️ **Ground truth:** {ex.gt_note}")

    with upload_tab:
        uploaded = st.file_uploader(
            "Upload an image (JPG / PNG). Uploads have no ground truth, so "
            "supervised metrics are unavailable - you still get side-by-side "
            "visual comparison, timing, and segment counts.",
            type=["jpg", "jpeg", "png", "bmp", "webp"],
        )
        if uploaded is not None:
            from PIL import Image

            image = np.asarray(Image.open(uploaded).convert("RGB"))
            gt = None
            class_names = []

    if image is None:
        st.info("Pick an example above to begin.")
        return
    if not alg_keys:
        st.warning("Select at least one algorithm in the sidebar.")
        return

    image, gt = resize_pair(image, gt, max_size)
    n_classes = len(class_names) if gt is not None else 0

    # ----- Ground truth row -----
    st.markdown("---")
    gt_cols = st.columns(4)
    with gt_cols[0]:
        st.image(image, caption="Input image", use_container_width=True)
    if gt is not None:
        with gt_cols[1]:
            st.image(
                overlay_labels(image, gt, alpha=overlay_alpha),
                caption="Ground truth overlay",
                use_container_width=True,
            )
        with gt_cols[2]:
            st.markdown("**Classes**")
            st.markdown(class_legend_html(class_names), unsafe_allow_html=True)
            st.caption(
                "Predicted segments are recolored to these class colors after "
                "mapping; unmatched segments (Hungarian mode) appear dark gray."
            )

    # ----- Run algorithms -----
    st.markdown("### Results")
    result_rows: list[dict] = []
    per_class_store: dict[str, dict] = {}
    n_cols = min(len(alg_keys), 4)
    grid = st.columns(n_cols)

    for i, alg_key in enumerate(alg_keys):
        alg = ALGORITHMS[alg_key]
        with grid[i % n_cols]:
            st.markdown(f"**{alg.name}**")
            try:
                with st.spinner(f"Running {alg.name}…"):
                    labels, elapsed = cached_run(
                        alg_key, image, tuple(sorted(alg_params[alg_key].items()))
                    )
            except Exception as exc:  # pragma: no cover - defensive UI path
                st.error(f"{alg.name} failed: {exc}")
                continue

            if gt is not None:
                m = cached_metrics(labels, gt, n_classes, mapping, boundary_tol)
                shown = m["mapped"]
                caption = (
                    f"{elapsed:.2f}s · {m['n_segments']} raw segments → "
                    f"{len(np.unique(shown[shown >= 0]))} classes"
                )
                result_rows.append(
                    {
                        "Algorithm": alg.name,
                        "Category": alg.category,
                        "Segments": m["n_segments"],
                        "Time (s)": round(elapsed, 2),
                        "pixel_accuracy": m["pixel_accuracy"],
                        "mean_iou": m["mean_iou"],
                        "mean_dice": m["mean_dice"],
                        "boundary_f1": m["boundary_f1"],
                        "hd95": m["hd95"],
                        "ari": m["ari"],
                    }
                )
                per_class_store[alg.name] = m["per_class"]
            else:
                shown = labels
                caption = f"{elapsed:.2f}s · {len(np.unique(labels))} segments"

            st.image(
                overlay_labels(image, shown, alpha=overlay_alpha),
                caption=caption,
                use_container_width=True,
            )
            render_algorithm_notes(alg_key)

    # ----- Metrics -----
    if result_rows:
        st.markdown("---")
        st.markdown("### 📊 Evaluation metrics on this example")
        st.caption(
            f"Mapping: **{mapping}** · Boundary tolerance θ = {boundary_tol}px · "
            "hover column headers for metric definitions; full explanations below."
        )

        import pandas as pd

        df = pd.DataFrame(result_rows).set_index("Algorithm")
        rename = {k: label for k, label, _ in METRIC_COLUMNS}
        df = df.rename(columns=rename)
        score_cols = [label for k, label, _ in METRIC_COLUMNS if k != "hd95"]
        styled = (
            df.style.format({label: fmt for _, label, fmt in METRIC_COLUMNS})
            .highlight_max(subset=score_cols, color="rgba(80, 200, 120, 0.35)")
            .highlight_min(subset=["HD95 (px)"], color="rgba(80, 200, 120, 0.35)")
        )
        st.dataframe(styled, use_container_width=True)
        st.caption(
            "Green = best score per column. Watch for disagreements between "
            "columns - e.g. high pixel accuracy but low mIoU means small "
            "classes were missed; high mDice but poor Boundary F1 means "
            "ragged contours; a big gap between majority and Hungarian "
            "mapping (toggle in the sidebar) means over-segmentation."
        )

        render_metric_charts(result_rows)

        # Per-class breakdown
        with st.expander("🔍 Per-class breakdown (find *which* class failed)"):
            metric_choice = st.radio(
                "Metric",
                ["iou", "dice", "boundary_f1", "hd95"],
                horizontal=True,
                format_func=lambda k: {
                    "iou": "IoU",
                    "dice": "Dice",
                    "boundary_f1": "Boundary F1",
                    "hd95": "HD95 (px)",
                }[k],
            )
            per_class_df = pd.DataFrame(
                {name: store[metric_choice] for name, store in per_class_store.items()},
                index=[f"{i}: {c}" for i, c in enumerate(class_names)],
            )
            fmt = "{:.1f}" if metric_choice == "hd95" else "{:.3f}"
            st.dataframe(
                per_class_df.style.format(fmt, na_rep="—"),
                use_container_width=True,
            )
            st.caption(
                "Mean metrics hide per-class failures: a thin or rare class "
                "(lane markings, tree trunks) can be entirely missed while "
                "the mean still looks respectable."
            )

        st.markdown("---")
        render_metric_explainers([k for k, _, _ in METRIC_COLUMNS])
    elif gt is None:
        st.markdown("---")
        st.info(
            "Uploaded images have no ground truth, so supervised metrics are "
            "not computed. Switch to the Example Gallery for the full metric "
            "experience, or study each metric in the **Metrics Guide** page."
        )

    st.markdown("---")
    st.caption(
        "All algorithms are classical (1967–2012), pre-deep-learning methods "
        "running on CPU via scikit-image/scipy. Their failure modes - noise, "
        "illumination gradients, texture, touching objects - are precisely "
        "what motivated learned segmentation models. Continue the story in "
        "**Paper Reviews** (FCN → U-Net → SegFormer → SAM) and try deep "
        "models in the **Playground**."
    )


if __name__ == "__main__":
    main()
