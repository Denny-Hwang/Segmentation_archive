"""Segmentation Archive Explorer - Main Streamlit Application.

A visual explorer for the Segmentation Archive, providing interactive
access to paper reviews, architecture diagrams, experiments, benchmarks,
and learning resources.

Run with:
    streamlit run app.py
"""

import streamlit as st
from pathlib import Path

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Segmentation Archive Explorer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ARCHIVE_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def render_sidebar():
    """Render the sidebar navigation and info panel."""
    st.sidebar.title("Segmentation Archive")
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
        **Navigation**

        Use the pages in the sidebar to explore:

        1. Home - Dashboard overview
        2. Paper Reviews - Browse paper analyses
        3. Architecture Gallery - Visual architecture guide
        4. Code Analysis - Implementation deep-dives
        5. Experiments - Results and benchmarks
        6. Benchmark Compare - Side-by-side comparison
        7. Repo Tracker - GitHub repo monitoring
        8. Timeline - Historical evolution
        9. Reading Roadmap - Learning path
        10. Figures Gallery - Diagrams & charts
        11. Playground - Try models live

        ---
        *Segmentation Archive Explorer*
        """
    )


# ---------------------------------------------------------------------------
# Main page
# ---------------------------------------------------------------------------
def main():
    """Render the main landing page."""
    render_sidebar()

    st.title("Segmentation Archive Explorer")
    st.markdown(
        """
        Welcome to the **Segmentation Archive Explorer**, an interactive
        dashboard for navigating the comprehensive image segmentation
        knowledge base.

        ### Getting Started

        Select a page from the sidebar to begin exploring. Each page provides
        a different lens into the archive:

        - **Paper Reviews**: Read detailed analyses of key segmentation papers
        - **Architecture Gallery**: Visualize and compare model architectures
        - **Code Analysis**: Explore implementation details and code patterns
        - **Experiments**: View training results, metrics, and comparisons
        - **Benchmark Compare**: Side-by-side model performance comparison
        - **Repo Tracker**: Monitor GitHub repositories and paper citations
        - **Timeline**: Trace the historical evolution of segmentation methods
        - **Reading Roadmap**: Follow a structured learning path
        - **Figures Gallery**: Browse and download all research diagrams
        - **Playground**: Run segmentation models on your own images

        ### Archive Structure

        This explorer reads from the archive's YAML registries and Markdown
        files to provide a unified, searchable interface for all collected
        knowledge.
        """
    )

    # Quick search on landing page
    st.markdown("---")
    st.subheader("Quick Search")
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from components.search_bar import render_search_bar, display_search_results
    results = render_search_bar(ARCHIVE_ROOT, placeholder="Search papers, concepts, architectures...")
    if results:
        display_search_results(results)

    # Quick stats
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Archive Sections", "10+")
    with col2:
        st.metric("Tracked Papers", "15+")
    with col3:
        st.metric("Tracked Repos", "9")
    with col4:
        st.metric("Datasets Documented", "14")


if __name__ == "__main__":
    main()
