"""Figure gallery component for displaying research diagrams and charts."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

try:
    from streamlit_mermaid import st_mermaid
    HAS_MERMAID = True
except ImportError:
    HAS_MERMAID = False


def render_figure_gallery(figures_dir: Path) -> None:
    """Display all figures in a directory with captions and download buttons.

    Args:
        figures_dir: Directory containing figure files (.png, .mermaid).
    """
    if not figures_dir.exists():
        st.info("No figures directory found.")
        return

    png_files = sorted(figures_dir.glob("*.png"))
    mermaid_files = sorted(figures_dir.glob("*.mermaid"))

    if not png_files and not mermaid_files:
        st.info("No figures found in the gallery.")
        return

    # PNG figures
    if png_files:
        st.markdown("### Rendered Figures")
        cols = st.columns(min(len(png_files), 2))
        for i, fig_path in enumerate(png_files):
            with cols[i % 2]:
                caption = fig_path.stem.replace("_", " ").title()
                st.image(str(fig_path), caption=caption, use_container_width=True)
                with open(fig_path, "rb") as f:
                    st.download_button(
                        f"Download {fig_path.name}",
                        f.read(),
                        file_name=fig_path.name,
                        mime="image/png",
                        key=f"dl_{fig_path.name}",
                    )

    # Mermaid diagrams (interactive)
    if mermaid_files:
        st.markdown("### Interactive Diagrams (Mermaid)")
        for mmd_path in mermaid_files:
            caption = mmd_path.stem.replace("_", " ").title()
            with st.expander(f"**{caption}**", expanded=False):
                code = mmd_path.read_text(encoding="utf-8")
                if HAS_MERMAID:
                    try:
                        st_mermaid(code, height=450)
                    except Exception:
                        st.code(code, language="mermaid")
                else:
                    st.code(code, language="mermaid")
                st.download_button(
                    f"Download {mmd_path.name}",
                    code,
                    file_name=mmd_path.name,
                    mime="text/plain",
                    key=f"dl_{mmd_path.name}",
                )
