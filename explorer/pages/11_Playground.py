"""Playground - Try deep segmentation models on your own images.

Upload an image or select an example, pick one or more models, and
compare segmentation results side-by-side. Picking an example with a
ground-truth mask also scores every model with the full evaluation
metric suite (pixel accuracy, mIoU, Dice, boundary F1, HD95, ARI),
exactly like the classical algorithms in the Segmentation Lab - so deep
and classical methods can be compared on the same examples with the
same numbers.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from seg_lab import deep_models  # noqa: E402
from seg_lab.examples import (  # noqa: E402
    class_legend_html,
    load_image,
    load_manifest,
    load_mask,
    overlay_labels,
    resize_pair,
)
from seg_lab.metric_info import METRIC_COLUMNS  # noqa: E402
from seg_lab.metrics import compute_all  # noqa: E402

st.set_page_config(page_title="Playground - Segmentation Archive", layout="wide")

ARCHIVE_ROOT = Path(__file__).resolve().parent.parent.parent
EXAMPLES_DIR = ARCHIVE_ROOT / "assets" / "examples"

# ---------------------------------------------------------------------------
# Model registry & backends (implemented in seg_lab/deep_models.py)
# ---------------------------------------------------------------------------

MODEL_INFO = deep_models.MODEL_INFO


def resolve_token() -> str | None:
    """Hugging Face token from Streamlit secrets, falling back to the env."""
    try:
        for key in deep_models.TOKEN_ENV_VARS:
            if key in st.secrets:
                value = str(st.secrets[key]).strip()
                if value:
                    return value
    except Exception:  # noqa: BLE001 - no secrets.toml on this host
        pass
    return deep_models.token_from_env()


@st.cache_resource(show_spinner="Loading model…")
def load_pipeline(model_key: str):
    """Load a local HuggingFace pipeline once per model, cached across reruns."""
    return deep_models.load_local_pipeline(model_key)


@st.cache_resource(show_spinner=False)
def api_client(token_fingerprint: str, _token: str):
    """Reuse one Inference API client per token (hashed, never cached raw)."""
    return deep_models.build_api_client(_token)


@st.cache_data(show_spinner=False, max_entries=32)
def cached_inference(
    backend: str, model_key: str, image_bytes: bytes, _token: str | None = None
) -> tuple:
    """Infer once per (backend, model, image); overlay tweaks reuse the cache."""
    if backend == deep_models.HF_API:
        fingerprint = hashlib.sha256((_token or "").encode()).hexdigest()[:16]
        client = api_client(fingerprint, _token)
        return deep_models.run_hf_api(client, model_key, image_bytes)
    return deep_models.run_local(load_pipeline(model_key), image_bytes)


def segment_colors(n: int):
    """Deterministic distinct colors, shared by overlay and legend."""
    import numpy as np

    rng = np.random.default_rng(42)
    return rng.integers(60, 230, size=(max(n, 1), 3), dtype=np.uint8)


def blend_masks(image, results, alpha: float = 0.5):
    """Create a color overlay of segmentation masks on the original image."""
    import numpy as np
    from PIL import Image

    img_array = np.array(image.convert("RGB"))
    overlay = img_array.copy()
    colors = segment_colors(len(results))

    for i, seg in enumerate(results):
        m = seg["mask"]
        if m.size != image.size:
            m = m.resize(image.size, Image.NEAREST)
        mask = np.array(m.convert("L")) > 127
        overlay[mask] = (
            overlay[mask] * (1 - alpha) + colors[i % len(colors)] * alpha
        ).astype(np.uint8)

    return Image.fromarray(overlay)


def build_class_legend(results) -> str:
    """Return an HTML legend mapping colours to class labels."""
    colors = segment_colors(len(results))
    lines = []
    for i, seg in enumerate(results):
        c = colors[i % len(colors)]
        hex_color = f"#{c[0]:02x}{c[1]:02x}{c[2]:02x}"
        label = seg.get("label", f"segment_{i}")
        raw_score = seg.get("score")
        try:
            score_pct = f" ({float(raw_score):.0%})" if raw_score is not None else ""
        except (TypeError, ValueError):
            score_pct = ""
        lines.append(
            f'<span style="background:{hex_color};padding:2px 8px;'
            f'border-radius:3px;color:#fff;font-size:0.85em;">'
            f"{label}{score_pct}</span>"
        )
    return " ".join(lines)


def results_to_labelmap(results, shape: tuple[int, int]):
    """Convert HF pipeline output (list of per-class PIL masks) to a label map.

    Pixels covered by no mask stay 0; each segment paints i+1. Masks that
    come back at a different resolution are nearest-resized to `shape`.
    """
    import numpy as np
    from PIL import Image

    labels = np.zeros(shape, dtype=np.int32)
    for i, seg in enumerate(results):
        m = seg["mask"]
        if m.size != (shape[1], shape[0]):
            m = m.resize((shape[1], shape[0]), Image.NEAREST)
        labels[np.array(m.convert("L")) > 127] = i + 1
    return labels


def render_metric_table(rows: list[dict], mapping: str) -> None:
    """Comparison table across models, styled like the Segmentation Lab."""
    import pandas as pd

    df = pd.DataFrame(rows).set_index("Model")
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
        f"Mapping: **{mapping}** · green = best per column. The same metric "
        "suite the Segmentation Lab applies to classical algorithms - run "
        "them on this example there and compare the two tables directly."
    )


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------


def main():
    st.title("Segmentation Playground")
    st.markdown(
        "Upload your own image or pick an example, select one or more deep "
        "models, and compare results side-by-side. Picking a **ground-truth "
        "example** also scores every model with the same metric suite as the "
        "**Segmentation Lab**, so deep and classical methods are directly "
        "comparable."
    )

    examples = load_manifest()

    # ----- Sidebar: where the models run -----
    st.sidebar.subheader("Inference Backend")
    backends = [deep_models.LOCAL, deep_models.HF_API]
    local_missing = deep_models.missing_local_packages()
    api_missing = deep_models.missing_api_packages()
    backend = st.sidebar.radio(
        "Run models",
        backends,
        index=backends.index(
            deep_models.HF_API if local_missing else deep_models.LOCAL
        ),
        format_func=lambda b: deep_models.BACKEND_LABELS[b],
        help=(
            "**Local** downloads the weights and runs them on this machine's "
            "CPU - it needs `torch` *and* `torchvision`, because from "
            "transformers 5.x every SegFormer image processor is built on "
            "torchvision. **Hugging Face API** posts the image to the Hub and "
            "gets the masks back: no torch, no weights, no memory ceiling, "
            "but it needs a free access token."
        ),
    )

    token = resolve_token()
    blocker: str | None = None

    if backend == deep_models.LOCAL:
        if local_missing:
            blocker = (
                "**Local backend unavailable — missing:** "
                f"`{'`, `'.join(local_missing)}`\n\n"
                "```bash\n"
                f"{deep_models.install_hint(local_missing)}\n"
                "```\n"
                "…or switch the sidebar backend to **Hugging Face API "
                "(remote)**, which needs none of these packages."
            )
        else:
            st.sidebar.caption("Weights are downloaded once and run on CPU here.")
    else:
        if api_missing:
            blocker = (
                "**Hugging Face API backend unavailable — missing:** "
                f"`{'`, `'.join(api_missing)}`\n\n"
                "```bash\n"
                f"{deep_models.install_hint(api_missing)}\n"
                "```"
            )
        entered = st.sidebar.text_input(
            "Hugging Face access token",
            type="password",
            placeholder="using stored token" if token else "hf_...",
            help=(
                "A free **read** token from "
                "https://huggingface.co/settings/tokens with 'Make calls to "
                "Inference Providers' enabled. It is also picked up from "
                "`HF_TOKEN` in the environment or in "
                "`.streamlit/secrets.toml`, so you never have to paste it here."
            ),
        ).strip()
        token = entered or token
        if token:
            st.sidebar.caption(
                f"Token loaded from {'this field' if entered else 'secrets / environment'}."
            )
        elif not blocker:
            blocker = (
                "**No Hugging Face token.** Create a free read token at "
                "[huggingface.co/settings/tokens]"
                "(https://huggingface.co/settings/tokens), then paste it in "
                "the sidebar, export `HF_TOKEN=...`, or add "
                '`HF_TOKEN = "hf_..."` to `.streamlit/secrets.toml`.'
            )

    if blocker:
        st.warning(blocker)

    # ----- Sidebar: model selection & options -----
    st.sidebar.subheader("Model Selection")
    selected_models = st.sidebar.multiselect(
        "Models to compare",
        list(MODEL_INFO.keys()),
        default=[list(MODEL_INFO.keys())[0]],
    )

    for m in selected_models:
        st.sidebar.caption(f"**{m}**: {MODEL_INFO[m]['description']}")
        if backend == deep_models.LOCAL and MODEL_INFO[m]["weight"] == "heavy":
            st.sidebar.caption(
                "⚠️ ~340 MB of weights and well over 1 GB of RAM on CPU. The "
                "API backend runs it on the Hub instead."
            )

    st.sidebar.markdown("---")
    st.sidebar.subheader("Options")
    overlay_alpha = st.sidebar.slider("Overlay opacity", 0.1, 0.9, 0.5, 0.05)
    max_size = st.sidebar.select_slider(
        "Max input size (px)",
        options=[256, 384, 512, 640, 768],
        value=512,
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("Evaluation")
    mapping = st.sidebar.radio(
        "Segment → class mapping",
        ["majority", "hungarian"],
        help=(
            "Used when a ground-truth example is selected. The model's "
            "predicted segments are mapped onto the example's classes before "
            "scoring: **majority** assigns each segment its best-overlap "
            "class; **hungarian** allows one segment per class and counts "
            "the rest as errors. See the Metrics Guide for details."
        ),
    )

    # ----- Image input (an upload always wins over a previously picked example)
    st.markdown("---")
    input_tab, example_tab = st.tabs(["Upload Image", "Example Images"])

    uploaded_image = None
    example_image = None
    gt = None
    class_names: list[str] = []

    with input_tab:
        uploaded = st.file_uploader(
            "Upload an image (JPG / PNG). Uploads have no ground truth, so "
            "metrics are shown only for the examples below.",
            type=["jpg", "jpeg", "png", "bmp", "webp"],
        )
        if uploaded is not None:
            from PIL import Image

            uploaded_image = Image.open(uploaded).convert("RGB")

    with example_tab:
        if examples:
            # Real photographs first, then synthetic teaching images
            keys = sorted(examples, key=lambda k: not k.startswith("photo_"))
            cols = st.columns(5)
            for i, key in enumerate(keys):
                ex = examples[key]
                with cols[i % 5]:
                    st.image(str(ex.image_path), use_container_width=True)
                    if st.button(
                        ex.title,
                        key=f"ex_{key}",
                        help=ex.teaches,
                        use_container_width=True,
                    ):
                        st.session_state["_playground_example"] = key

            chosen = st.session_state.get("_playground_example")
            if chosen and chosen in examples:
                ex = examples[chosen]
                from PIL import Image

                example_image = Image.fromarray(load_image(ex.image_path))
                gt = load_mask(ex.mask_path)
                class_names = ex.classes
                col_a, col_b = st.columns([3, 1])
                col_a.success(
                    f"Using example: **{ex.title}** · {len(class_names)} "
                    "ground-truth classes → metrics enabled"
                )
                if col_b.button("Clear example"):
                    del st.session_state["_playground_example"]
                    st.rerun()
        else:
            st.info(
                "No example images found. Run "
                "`python scripts/figures/generate_example_images.py` and "
                "`python scripts/figures/prepare_real_examples.py` first."
            )

    if uploaded_image is not None:
        pil_image, gt, class_names = uploaded_image, None, []
    else:
        pil_image = example_image

    # ----- Run inference -----
    if pil_image is None:
        st.info("Select or upload an image above to begin.")
        return

    if blocker:
        st.info("Resolve the note above to run inference.")
        return

    if not selected_models:
        st.warning("Select at least one model in the sidebar.")
        return

    import io

    import numpy as np
    from PIL import Image as PILImage

    image_np, gt = resize_pair(np.asarray(pil_image), gt, max_size)
    pil_image = PILImage.fromarray(image_np)
    n_classes = len(class_names) if gt is not None else 0

    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    image_bytes = buf.getvalue()

    st.markdown("---")
    st.subheader("Results")
    in_col, gt_col, legend_col = st.columns([1, 1, 1])
    with in_col:
        st.image(pil_image, caption="Input image", use_container_width=True)
    if gt is not None:
        with gt_col:
            st.image(
                overlay_labels(image_np, gt, alpha=overlay_alpha),
                caption="Ground truth overlay",
                use_container_width=True,
            )
        with legend_col:
            st.markdown("**Ground-truth classes**")
            st.markdown(class_legend_html(class_names), unsafe_allow_html=True)
            st.caption(
                "These models predict ADE20K scene classes (wall, sky, "
                "floor…), not this example's labels - so predicted segments "
                "are mapped to the ground-truth classes by best overlap "
                "before scoring. Metrics measure how well the model's "
                "*partition* matches the ground truth, not whether the "
                "class names are right."
            )

    metric_rows: list[dict] = []
    result_cols = st.columns(max(len(selected_models), 1))

    for idx, model_key in enumerate(selected_models):
        with result_cols[idx % len(result_cols)]:
            st.markdown(f"#### {model_key}")
            with st.spinner(f"Running {model_key}…"):
                try:
                    results, elapsed = cached_inference(
                        backend, model_key, image_bytes, token
                    )
                except deep_models.BackendError as exc:
                    st.error(str(exc))
                    continue
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Inference failed: {exc}")
                    continue

            st.caption(
                f"Inference time: **{elapsed:.2f}s** | Segments: **{len(results)}**"
            )

            overlay = blend_masks(pil_image, results, alpha=overlay_alpha)
            st.image(overlay, caption="Segmentation overlay", use_container_width=True)

            legend_html = build_class_legend(results)
            st.markdown(legend_html, unsafe_allow_html=True)

            if gt is not None:
                labels = results_to_labelmap(results, gt.shape)
                m = compute_all(labels, gt, n_classes, mapping=mapping)
                st.image(
                    overlay_labels(image_np, m["mapped"], alpha=overlay_alpha),
                    caption="Mapped to ground-truth classes",
                    use_container_width=True,
                )
                metric_rows.append(
                    {
                        "Model": model_key,
                        "Time (s)": round(elapsed, 2),
                        "pixel_accuracy": m["pixel_accuracy"],
                        "mean_iou": m["mean_iou"],
                        "mean_dice": m["mean_dice"],
                        "boundary_f1": m["boundary_f1"],
                        "hd95": m["hd95"],
                        "ari": m["ari"],
                    }
                )

    if metric_rows:
        st.markdown("---")
        st.markdown("### 📊 Evaluation metrics on this example")
        render_metric_table(metric_rows, mapping)
        st.caption(
            "Metric definitions, strengths, and limitations are covered in "
            "the **Metrics Guide** page."
        )

    st.markdown("---")
    st.caption(
        f"Backend: **{deep_models.BACKEND_LABELS[backend]}**. Local runs "
        "download the weights on first use and infer on CPU; the API backend "
        "keeps nothing locally and runs the model on Hugging Face. Either way "
        "results are cached per (backend, model, image), so overlay tweaks "
        "are instant."
    )


if __name__ == "__main__":
    main()
