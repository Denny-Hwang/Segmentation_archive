"""Home - Dashboard with overview tabs, key diagrams, and archive stats."""

import sys
import streamlit as st
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from components.frontmatter import strip_frontmatter
from components.mermaid_render import render_mermaid_file

st.set_page_config(page_title="Home - Segmentation Archive", layout="wide")

ARCHIVE_ROOT = Path(__file__).resolve().parent.parent.parent
FIGURES_DIR = ARCHIVE_ROOT / "docs" / "figures"


def count_files(directory: Path, pattern: str = "*.md") -> int:
    if not directory.exists():
        return 0
    return len(list(directory.rglob(pattern)))


def main():
    st.title("Dashboard")
    st.markdown("Overview of the Segmentation Archive — research, models, and tools.")

    # ---------- Tabs ----------
    tab_overview, tab_stats, tab_quick = st.tabs([
        "Overview", "Archive Statistics", "Quick Start"
    ])

    # ===== Tab 1: Overview =====
    with tab_overview:
        st.markdown(
            "### Image Segmentation Research at a Glance\n\n"
            "This archive covers the full landscape of image segmentation: "
            "from classical CNN approaches through transformer hybrids to "
            "modern foundation models like SAM."
        )

        taxonomy_mermaid = FIGURES_DIR / "taxonomy_diagram.mermaid"
        if taxonomy_mermaid.exists():
            st.markdown("#### Segmentation Taxonomy")
            render_mermaid_file(taxonomy_mermaid, height=520)

        pipeline_mermaid = FIGURES_DIR / "pipeline_diagram.mermaid"
        if pipeline_mermaid.exists():
            st.markdown("#### Typical Segmentation Pipeline")
            render_mermaid_file(pipeline_mermaid, height=380)

        timeline_png = FIGURES_DIR / "timeline_evolution_chart.png"
        if timeline_png.exists():
            st.markdown("#### Historical Evolution")
            st.image(str(timeline_png), use_container_width=True)

        comparison_png = FIGURES_DIR / "model_comparison_chart.png"
        if comparison_png.exists():
            st.markdown("#### Model Comparison (Representative Benchmarks)")
            st.image(str(comparison_png), use_container_width=True)

    # ===== Tab 2: Archive Statistics =====
    with tab_stats:
        st.subheader("Archive Statistics")
        col1, col2, col3, col4 = st.columns(4)

        paper_dirs = ["02_unet_family", "03_transformer_segmentation", "04_foundation_models"]
        paper_count = sum(count_files(ARCHIVE_ROOT / d) for d in paper_dirs)

        with col1:
            st.metric("Paper Reviews", paper_count)
        with col2:
            st.metric("Foundation Docs", count_files(ARCHIVE_ROOT / "01_foundations"))
        with col3:
            st.metric("Dataset Cards", count_files(ARCHIVE_ROOT / "09_datasets"))
        with col4:
            st.metric("Reference Docs", count_files(ARCHIVE_ROOT / "10_references"))

        st.markdown("---")
        st.subheader("Archive Sections")

        sections = [
            ("01_foundations", "Foundations", "Core concepts, taxonomy, metrics, and loss functions"),
            ("02_unet_family", "U-Net Family", "U-Net variants: original, ++, Attention, 3D, R2, etc."),
            ("03_transformer_segmentation", "Transformer Segmentation", "TransUNet, Swin-Unet, SegFormer, Mask2Former, OneFormer"),
            ("04_foundation_models", "Foundation Models", "SAM, SAM2, MedSAM, and adaptation strategies"),
            ("05_code_analysis", "Code Analysis", "Implementation deep-dives and code walkthroughs"),
            ("06_experiments", "Experiments", "Training utilities, metrics, visualization, and augmentation"),
            ("07_visualizations", "Visualizations", "Evolution trees, timelines, and architecture diagrams"),
            ("08_repo_tracker", "Repo Tracker", "GitHub repository and paper tracking"),
            ("09_datasets", "Datasets", "Dataset documentation and preprocessing guides"),
            ("10_references", "References", "Glossary, surveys, and reading roadmap"),
        ]

        for section_dir, title, description in sections:
            section_path = ARCHIVE_ROOT / section_dir
            exists = section_path.exists()
            file_count = count_files(section_path) if exists else 0

            with st.expander(f"{title} ({file_count} files)", expanded=False):
                st.markdown(f"**Status**: {'Active' if exists else 'Pending'}")
                st.markdown(f"**Description**: {description}")
                if exists:
                    files = sorted(section_path.rglob("*.md"))[:10]
                    if files:
                        st.markdown("**Recent files:**")
                        for f in files:
                            st.markdown(f"- `{f.relative_to(ARCHIVE_ROOT)}`")

    # ===== Tab 3: Quick Start =====
    with tab_quick:
        st.subheader("Quick Start")
        st.markdown(
            """
**Navigate the archive** using the sidebar pages:

| Page | What you'll find |
|------|-----------------|
| **Paper Reviews** | Detailed analyses with filters, search, and metadata |
| **Architecture Gallery** | Model architecture documentation |
| **Benchmark Compare** | Side-by-side performance comparison |
| **Timeline** | Historical evolution of segmentation methods |
| **Figures Gallery** | All diagrams & charts (download/view) |
| **Playground** | Run segmentation models on your images |

**Run locally:**

```bash
cd explorer
pip install -r requirements.txt
streamlit run app.py
```

**Upstream / original research:**
This repository is an independent review & learning resource.
No single upstream research repository; individual papers are cited
within each review document (see `10_references/bibliography.bib`).
"""
        )

    st.markdown("---")
    st.caption("Segmentation Archive Explorer")


if __name__ == "__main__":
    main()
