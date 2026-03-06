"""Home - Dashboard page showing archive stats and recent updates."""

import streamlit as st
from pathlib import Path

st.set_page_config(page_title="Home - Segmentation Archive", layout="wide")

ARCHIVE_ROOT = Path(__file__).resolve().parent.parent.parent


def count_files(directory: Path, pattern: str = "*.md") -> int:
    """Count files matching a pattern in a directory tree."""
    if not directory.exists():
        return 0
    return len(list(directory.rglob(pattern)))


def main():
    st.title("Dashboard")
    st.markdown("Overview of the Segmentation Archive contents and recent activity.")

    st.markdown("---")

    # Archive statistics
    st.subheader("Archive Statistics")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        paper_count = count_files(ARCHIVE_ROOT / "01_paper_reviews")
        st.metric("Paper Reviews", paper_count)

    with col2:
        arch_count = count_files(ARCHIVE_ROOT / "02_architecture")
        st.metric("Architecture Docs", arch_count)

    with col3:
        dataset_count = count_files(ARCHIVE_ROOT / "09_datasets")
        st.metric("Dataset Cards", dataset_count)

    with col4:
        ref_count = count_files(ARCHIVE_ROOT / "10_references")
        st.metric("Reference Docs", ref_count)

    st.markdown("---")

    # Archive sections
    st.subheader("Archive Sections")

    sections = [
        ("01_paper_reviews", "Paper Reviews", "Detailed analyses of key segmentation papers"),
        ("02_architecture", "Architecture", "Model architecture documentation and diagrams"),
        ("03_code_analysis", "Code Analysis", "Implementation deep-dives and code walkthroughs"),
        ("04_experiments", "Experiments", "Training experiments, results, and configurations"),
        ("05_benchmarks", "Benchmarks", "Performance comparisons across models and datasets"),
        ("06_knowledge_graph", "Knowledge Graph", "Paper relationships and citation networks"),
        ("07_timeline", "Timeline", "Historical evolution of segmentation methods"),
        ("08_repo_tracker", "Repo Tracker", "GitHub repository and paper tracking"),
        ("09_datasets", "Datasets", "Dataset documentation and preprocessing guides"),
        ("10_references", "References", "Glossary, surveys, and reading roadmap"),
    ]

    for section_dir, title, description in sections:
        section_path = ARCHIVE_ROOT / section_dir
        exists = section_path.exists()
        status = "Active" if exists else "Pending"
        file_count = count_files(section_path) if exists else 0

        with st.expander(f"{title} ({file_count} files)", expanded=False):
            st.markdown(f"**Status**: {status}")
            st.markdown(f"**Description**: {description}")
            if exists:
                files = sorted(section_path.rglob("*.md"))[:10]
                if files:
                    st.markdown("**Recent files:**")
                    for f in files:
                        st.markdown(f"- `{f.relative_to(ARCHIVE_ROOT)}`")

    st.markdown("---")
    st.caption("Segmentation Archive Explorer")


if __name__ == "__main__":
    main()
