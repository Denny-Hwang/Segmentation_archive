"""Figures Gallery - Browse all research diagrams, charts, and visual assets."""

import sys
import streamlit as st
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from components.figure_gallery import render_figure_gallery
from components.mermaid_render import render_mermaid, mermaid_png_download_button

st.set_page_config(page_title="Figures Gallery - Segmentation Archive", layout="wide")

ARCHIVE_ROOT = Path(__file__).resolve().parent.parent.parent
FIGURES_DIR = ARCHIVE_ROOT / "docs" / "figures"
VIZ_DIR = ARCHIVE_ROOT / "07_visualizations"


def main():
    st.title("Figures Gallery")
    st.markdown(
        "Publication-ready figures, diagrams, and charts from the archive. "
        "Download images or view interactive Mermaid diagrams."
    )

    tab_main, tab_arch, tab_gen = st.tabs([
        "Research Figures", "Architecture Diagrams", "Generate Figures"
    ])

    with tab_main:
        render_figure_gallery(FIGURES_DIR)

    with tab_arch:
        st.markdown("### Architecture Evolution Diagrams")
        arch_dir = VIZ_DIR / "architecture_diagrams"
        if arch_dir.exists():
            mermaid_files = sorted(arch_dir.glob("*.mermaid"))
            if mermaid_files:
                for mmd in mermaid_files:
                    caption = mmd.stem.replace("_", " ").title()
                    with st.expander(f"**{caption}**", expanded=True):
                        code = mmd.read_text(encoding="utf-8")
                        render_mermaid(code, height=450)
                        dl_col1, dl_col2 = st.columns(2)
                        with dl_col1:
                            st.download_button(
                                "Download .mermaid source",
                                code,
                                file_name=mmd.name,
                                mime="text/plain",
                                key=f"dl_arch_src_{mmd.name}",
                            )
                        with dl_col2:
                            mermaid_png_download_button(
                                code,
                                filename=mmd.stem + ".png",
                                label="Download PNG (via mermaid.ink)",
                                key=f"dl_arch_png_{mmd.name}",
                            )
            else:
                st.info("No `.mermaid` files found in architecture diagrams.")
        else:
            st.info("No architecture diagrams found in `07_visualizations/architecture_diagrams/`.")

    with tab_gen:
        st.markdown("### Regenerate Figures")
        st.markdown(
            "Run the figure generation script to create/update all figures:\n\n"
            "```bash\n"
            "python scripts/figures/generate_figures.py\n"
            "```\n\n"
            "**Requirements:** `matplotlib`, `Pillow`. "
            "For Mermaid PNG rendering: `npm install -g @mermaid-js/mermaid-cli`.\n\n"
            "**Sources:**\n"
            "- `docs/figures/*.mermaid` — Mermaid diagram sources\n"
            "- `scripts/figures/generate_figures.py` — Python figure generator\n"
        )

    st.markdown("---")
    st.caption("Segmentation Archive Explorer — Figures Gallery")


if __name__ == "__main__":
    main()
