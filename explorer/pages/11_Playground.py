"""Playground - Try segmentation models on your own images.

Upload an image or select an example, pick one or more models,
and compare segmentation results side-by-side.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

st.set_page_config(page_title="Playground - Segmentation Archive", layout="wide")

ARCHIVE_ROOT = Path(__file__).resolve().parent.parent.parent
EXAMPLES_DIR = ARCHIVE_ROOT / "assets" / "examples"

# ---------------------------------------------------------------------------
# Model registry — lightweight wrappers around HF pipelines
# ---------------------------------------------------------------------------

MODEL_INFO: dict[str, dict] = {
    "SegFormer-B0 (ADE20K)": {
        "hf_model": "nvidia/segformer-b0-finetuned-ade-512-512",
        "task": "image-segmentation",
        "description": "Lightweight SegFormer (3.7M params). Fast on CPU.",
        "weight": "light",
    },
    "SegFormer-B1 (ADE20K)": {
        "hf_model": "nvidia/segformer-b1-finetuned-ade-512-512",
        "task": "image-segmentation",
        "description": "Mid-size SegFormer (13.7M params). Good accuracy/speed trade-off.",
        "weight": "mid",
    },
    "SegFormer-B5 (ADE20K)": {
        "hf_model": "nvidia/segformer-b5-finetuned-ade-640-640",
        "task": "image-segmentation",
        "description": "Large SegFormer (84M params). High accuracy, slower.",
        "weight": "heavy",
    },
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner="Loading model…")
def load_pipeline(model_key: str):
    """Load a HuggingFace pipeline, cached across reruns."""
    info = MODEL_INFO[model_key]
    try:
        from transformers import pipeline
    except ImportError:
        st.error(
            "`transformers` is not installed. "
            "Run `pip install transformers torch` to enable model inference."
        )
        return None

    try:
        pipe = pipeline(
            task=info["task"],
            model=info["hf_model"],
            device=-1,  # CPU; change to 0 for GPU
        )
        return pipe
    except Exception as exc:
        st.error(f"Failed to load **{model_key}**: {exc}")
        return None


def run_inference(pipe, image):
    """Run the segmentation pipeline and return results + elapsed time."""
    t0 = time.perf_counter()
    results = pipe(image)
    elapsed = time.perf_counter() - t0
    return results, elapsed


def blend_masks(image, results, alpha: float = 0.5):
    """Create a color overlay of segmentation masks on the original image."""
    from PIL import Image
    import numpy as np

    img_array = np.array(image.convert("RGB"))
    overlay = img_array.copy()

    # Generate distinct colours for each segment
    np.random.seed(42)
    n = len(results)
    colors = np.random.randint(60, 230, size=(max(n, 1), 3), dtype=np.uint8)

    for i, seg in enumerate(results):
        mask = np.array(seg["mask"].convert("L")) > 127
        overlay[mask] = (
            overlay[mask] * (1 - alpha) + colors[i % n] * alpha
        ).astype(np.uint8)

    return Image.fromarray(overlay)


def build_class_legend(results) -> str:
    """Return a Markdown legend mapping colours to class labels."""
    import numpy as np

    np.random.seed(42)
    n = len(results)
    colors = np.random.randint(60, 230, size=(max(n, 1), 3), dtype=np.uint8)

    lines = []
    for i, seg in enumerate(results):
        c = colors[i % n]
        hex_color = f"#{c[0]:02x}{c[1]:02x}{c[2]:02x}"
        label = seg.get("label", f"segment_{i}")
        score = seg.get("score", 0)
        lines.append(
            f'<span style="background:{hex_color};padding:2px 8px;'
            f'border-radius:3px;color:#fff;font-size:0.85em;">'
            f"{label} ({score:.0%})</span>"
        )
    return " ".join(lines)


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

def main():
    st.title("Segmentation Playground")
    st.markdown(
        "Upload your own image or pick an example, select one or more models, "
        "and compare segmentation results side-by-side."
    )

    # Check dependencies
    _deps_ok = True
    try:
        import transformers  # noqa: F401
        import torch  # noqa: F401
        from PIL import Image  # noqa: F401
        import numpy  # noqa: F401
    except ImportError as exc:
        st.warning(
            f"**Missing dependency:** `{exc.name}`. "
            "Install required packages:\n\n"
            "```bash\n"
            "pip install transformers torch Pillow numpy\n"
            "```"
        )
        _deps_ok = False

    # ----- Sidebar: model selection & options -----
    st.sidebar.subheader("Model Selection")
    selected_models = st.sidebar.multiselect(
        "Models to compare",
        list(MODEL_INFO.keys()),
        default=[list(MODEL_INFO.keys())[0]],
    )

    for m in selected_models:
        st.sidebar.caption(f"**{m}**: {MODEL_INFO[m]['description']}")

    st.sidebar.markdown("---")
    st.sidebar.subheader("Options")
    overlay_alpha = st.sidebar.slider("Overlay opacity", 0.1, 0.9, 0.5, 0.05)
    max_size = st.sidebar.select_slider(
        "Max input size (px)",
        options=[256, 384, 512, 640, 768],
        value=512,
    )

    # ----- Image input -----
    st.markdown("---")
    input_tab, example_tab = st.tabs(["Upload Image", "Example Images"])

    pil_image = None

    with input_tab:
        uploaded = st.file_uploader(
            "Upload an image (JPG / PNG)",
            type=["jpg", "jpeg", "png", "bmp", "webp"],
        )
        if uploaded is not None:
            from PIL import Image
            pil_image = Image.open(uploaded).convert("RGB")

    with example_tab:
        example_files = sorted(EXAMPLES_DIR.glob("*.png")) if EXAMPLES_DIR.exists() else []
        if example_files:
            cols = st.columns(min(len(example_files), 4))
            for i, ef in enumerate(example_files):
                with cols[i % 4]:
                    st.image(str(ef), caption=ef.stem.replace("_", " ").title(), use_container_width=True)
                    if st.button("Use", key=f"ex_{ef.stem}"):
                        st.session_state["_playground_example"] = str(ef)

            chosen = st.session_state.get("_playground_example")
            if chosen:
                from PIL import Image
                pil_image = Image.open(chosen).convert("RGB")
                st.success(f"Using example: {Path(chosen).stem}")
        else:
            st.info(
                "No example images found. Run "
                "`python scripts/figures/generate_example_images.py` "
                "to create synthetic examples."
            )

    # ----- Run inference -----
    if pil_image is None:
        st.info("Select or upload an image above to begin.")
        return

    if not _deps_ok:
        st.warning("Install missing dependencies to run inference.")
        return

    if not selected_models:
        st.warning("Select at least one model in the sidebar.")
        return

    # Resize
    from PIL import Image as PILImage
    w, h = pil_image.size
    if max(w, h) > max_size:
        ratio = max_size / max(w, h)
        pil_image = pil_image.resize(
            (int(w * ratio), int(h * ratio)), PILImage.LANCZOS
        )

    st.markdown("---")
    st.subheader("Results")

    result_cols = st.columns(max(len(selected_models), 1))

    for idx, model_key in enumerate(selected_models):
        with result_cols[idx % len(result_cols)]:
            st.markdown(f"#### {model_key}")
            pipe = load_pipeline(model_key)
            if pipe is None:
                st.error("Model not available.")
                continue

            with st.spinner(f"Running {model_key}…"):
                try:
                    results, elapsed = run_inference(pipe, pil_image)
                except Exception as exc:
                    st.error(f"Inference failed: {exc}")
                    continue

            st.caption(f"Inference time: **{elapsed:.2f}s** | Segments: **{len(results)}**")

            # Original
            st.image(pil_image, caption="Original", use_container_width=True)

            # Overlay
            overlay = blend_masks(pil_image, results, alpha=overlay_alpha)
            st.image(overlay, caption="Segmentation overlay", use_container_width=True)

            # Legend
            legend_html = build_class_legend(results)
            st.markdown(legend_html, unsafe_allow_html=True)

    st.markdown("---")
    st.caption(
        "Models are loaded from Hugging Face Hub on first use and cached locally. "
        "All inference runs on CPU by default."
    )


if __name__ == "__main__":
    main()
