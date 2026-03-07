"""Architecture Gallery - Visual architecture comparison and exploration."""

import streamlit as st
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None

st.set_page_config(page_title="Architecture Gallery - Segmentation Archive", layout="wide")

ARCHIVE_ROOT = Path(__file__).resolve().parent.parent.parent
ARCH_DIRS = [
    ARCHIVE_ROOT / "02_unet_family",
    ARCHIVE_ROOT / "03_transformer_segmentation",
    ARCHIVE_ROOT / "04_foundation_models",
    ARCHIVE_ROOT / "01_foundations",
]


def load_architectures() -> list[dict]:
    """Load architecture documentation files."""
    architectures = []
    for arch_dir in ARCH_DIRS:
        if not arch_dir.exists():
            continue

        for md_file in sorted(arch_dir.rglob("*.md")):
            if md_file.name.startswith("_") or md_file.name == "README.md":
                continue
            try:
                content = md_file.read_text(encoding="utf-8")
                architectures.append({
                    "name": md_file.stem.replace("_", " ").title(),
                    "content": content,
                    "file_path": str(md_file.relative_to(ARCHIVE_ROOT)),
                    "category": arch_dir.name,
                })
            except Exception:
                continue

    return architectures


def main():
    st.title("Architecture Gallery")
    st.markdown(
        "Visual exploration of segmentation model architectures. "
        "Browse architecture documentation, diagrams, and component descriptions."
    )

    architectures = load_architectures()

    if not architectures:
        st.info(
            "No architecture documents found. Add Markdown files to "
            "`02_unet_family/`, `03_transformer_segmentation/`, or "
            "`04_foundation_models/` to populate this gallery."
        )

        # Show placeholder gallery
        st.markdown("---")
        st.subheader("Architecture Categories")

        categories = {
            "CNN-Based": ["U-Net", "UNet++", "Attention U-Net", "DeepLab v3+", "PSPNet"],
            "Transformer-Based": ["TransUNet", "Swin-Unet", "SegFormer"],
            "Universal / Panoptic": ["Mask2Former", "OneFormer"],
            "Foundation Models": ["SAM", "SAM 2"],
            "3D / Volumetric": ["3D U-Net", "V-Net"],
        }

        for category, models in categories.items():
            with st.expander(category, expanded=True):
                cols = st.columns(len(models))
                for i, model in enumerate(models):
                    with cols[i]:
                        st.markdown(f"**{model}**")
                        st.caption("Documentation pending")
        return

    # Category filter
    categories = sorted({a["category"] for a in architectures})
    selected = st.sidebar.selectbox("Category", ["All"] + categories)

    filtered = architectures
    if selected != "All":
        filtered = [a for a in filtered if a["category"] == selected]

    st.markdown(f"Showing **{len(filtered)}** architectures")
    st.markdown("---")

    for arch in filtered:
        with st.expander(f"**{arch['name']}**", expanded=False):
            st.markdown(arch["content"][:2000])
            st.caption(f"Source: `{arch['file_path']}`")


if __name__ == "__main__":
    main()
