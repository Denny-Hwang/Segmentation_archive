"""Reading Roadmap Navigator - Structured learning path for segmentation."""

import streamlit as st
from pathlib import Path
import sys

# Add explorer to path for component imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from components.paper_figures import render_paper_figures_inline

st.set_page_config(page_title="Reading Roadmap - Segmentation Archive", layout="wide")

ARCHIVE_ROOT = Path(__file__).resolve().parent.parent.parent

ROADMAP = {
    "Level 1: Foundations (Beginner)": {
        "description": "Start here if you are new to deep learning and segmentation.",
        "papers": [
            {
                "title": "Fully Convolutional Networks for Semantic Segmentation",
                "authors": "Long et al.",
                "year": 2015,
                "arxiv": "1411.4038",
                "reason": "Seminal work introducing end-to-end pixel-wise prediction with CNNs",
            },
            {
                "title": "U-Net: Convolutional Networks for Biomedical Image Segmentation",
                "authors": "Ronneberger et al.",
                "year": 2015,
                "arxiv": "1505.04597",
                "reason": "Introduced skip connections and encoder-decoder architecture",
            },
            {
                "title": "DeepLab: Semantic Image Segmentation with Deep Convolutional Nets and Fully Connected CRFs",
                "authors": "Chen et al.",
                "year": 2014,
                "arxiv": "1412.7062",
                "reason": "Introduced atrous convolutions and CRF post-processing",
            },
        ],
    },
    "Level 2: Core Architectures (Intermediate)": {
        "description": "Build deeper understanding of the architectural design space.",
        "papers": [
            {
                "title": "UNet++: A Nested U-Net Architecture",
                "authors": "Zhou et al.",
                "year": 2018,
                "arxiv": "1807.10165",
                "reason": "Dense skip connections and deep supervision",
            },
            {
                "title": "Attention U-Net: Learning Where to Look for the Pancreas",
                "authors": "Oktay et al.",
                "year": 2018,
                "arxiv": "1804.03999",
                "reason": "Attention gating for skip connections",
            },
            {
                "title": "PSPNet: Pyramid Scene Parsing Network",
                "authors": "Zhao et al.",
                "year": 2017,
                "arxiv": "1612.01105",
                "reason": "Multi-scale context via pyramid pooling",
            },
            {
                "title": "DeepLab v3+",
                "authors": "Chen et al.",
                "year": 2018,
                "arxiv": "1802.02611",
                "reason": "State-of-the-art atrous spatial pyramid pooling with decoder",
            },
            {
                "title": "nnU-Net",
                "authors": "Isensee et al.",
                "year": 2021,
                "arxiv": "1809.10486",
                "reason": "Self-configuring method demonstrating engineering importance",
            },
        ],
    },
    "Level 3: Transformers and Modern Methods (Advanced)": {
        "description": "Understand how transformers revolutionized segmentation.",
        "papers": [
            {
                "title": "An Image is Worth 16x16 Words (ViT)",
                "authors": "Dosovitskiy et al.",
                "year": 2021,
                "arxiv": "2010.11929",
                "reason": "Foundation for all vision transformers",
            },
            {
                "title": "TransUNet",
                "authors": "Chen et al.",
                "year": 2021,
                "arxiv": "2102.04306",
                "reason": "First successful CNN-Transformer hybrid for medical segmentation",
            },
            {
                "title": "SegFormer",
                "authors": "Xie et al.",
                "year": 2021,
                "arxiv": "2105.15203",
                "reason": "Lightweight and effective transformer segmentation",
            },
            {
                "title": "Mask2Former",
                "authors": "Cheng et al.",
                "year": 2022,
                "arxiv": "2112.01527",
                "reason": "Unified architecture for all segmentation tasks",
            },
            {
                "title": "OneFormer",
                "authors": "Jain et al.",
                "year": 2023,
                "arxiv": "2211.06220",
                "reason": "Single multi-task model for all segmentation",
            },
        ],
    },
    "Level 4: Cutting-Edge and Foundation Models": {
        "description": "Explore the latest paradigm shifts.",
        "papers": [
            {
                "title": "Segment Anything (SAM)",
                "authors": "Kirillov et al.",
                "year": 2023,
                "arxiv": "2304.02643",
                "reason": "Paradigm shift to promptable zero-shot segmentation",
            },
            {
                "title": "SAM 2: Segment Anything in Images and Videos",
                "authors": "Ravi et al.",
                "year": 2024,
                "arxiv": "2408.00714",
                "reason": "Extension to video with memory-based architecture",
            },
            {
                "title": "EfficientSAM",
                "authors": "Xiong et al.",
                "year": 2024,
                "arxiv": "2312.00863",
                "reason": "Knowledge distillation for efficient SAM variants",
            },
        ],
    },
}


def _on_checkbox_change(title: str, key: str):
    """Callback to sync checkbox state with completed_papers immediately."""
    if st.session_state[key]:
        st.session_state.completed_papers.add(title)
    else:
        st.session_state.completed_papers.discard(title)


def main():
    st.title("📚 Reading Roadmap")
    st.markdown(
        "A structured learning path from beginner to cutting-edge "
        "in image segmentation research."
    )

    # Progress tracking
    if "completed_papers" not in st.session_state:
        st.session_state.completed_papers = set()

    total_papers = sum(len(level["papers"]) for level in ROADMAP.values())
    completed = len(st.session_state.completed_papers)

    # Progress bar with percentage
    progress_pct = completed / total_papers if total_papers > 0 else 0
    st.progress(progress_pct)
    st.caption(f"Progress: {completed}/{total_papers} papers read ({progress_pct:.0%})")

    # Reset button
    if completed > 0:
        if st.button("Reset progress", type="secondary"):
            st.session_state.completed_papers = set()
            # Clear all checkbox keys
            for level_data in ROADMAP.values():
                for paper in level_data["papers"]:
                    key = f"paper_{paper['arxiv']}"
                    if key in st.session_state:
                        st.session_state[key] = False
            st.rerun()

    st.markdown("---")

    for level_name, level_data in ROADMAP.items():
        level_completed = sum(
            1 for p in level_data["papers"]
            if p["title"] in st.session_state.completed_papers
        )
        total_in_level = len(level_data["papers"])
        all_done = level_completed == total_in_level

        level_icon = "✅" if all_done else "📖"
        st.subheader(f"{level_icon} {level_name} ({level_completed}/{total_in_level})")
        st.markdown(f"*{level_data['description']}*")

        for paper in level_data["papers"]:
            key = f"paper_{paper['arxiv']}"
            is_done = paper["title"] in st.session_state.completed_papers

            col1, col2 = st.columns([4, 1])

            with col2:
                st.checkbox(
                    "Read",
                    value=is_done,
                    key=key,
                    on_change=_on_checkbox_change,
                    args=(paper["title"], key),
                )

            with col1:
                if is_done:
                    st.markdown(
                        f"~~**{paper['title']}**~~ ✅ "
                        f"({paper['authors']}, {paper['year']})"
                    )
                else:
                    st.markdown(
                        f"**{paper['title']}** "
                        f"({paper['authors']}, {paper['year']})"
                    )
                st.caption(
                    f"Why read this: {paper['reason']} | "
                    f"[arXiv:{paper['arxiv']}](https://arxiv.org/abs/{paper['arxiv']}) | "
                    f"[PDF](https://arxiv.org/pdf/{paper['arxiv']})"
                )

                # Personal notes per paper
                notes_key = f"notes_{paper['arxiv']}"
                if st.session_state.get(f"show_notes_{paper['arxiv']}", False):
                    st.text_area(
                        "My notes",
                        key=notes_key,
                        height=80,
                        placeholder="Write your notes about this paper...",
                    )
                st.button(
                    "📝 Notes" if not st.session_state.get(f"show_notes_{paper['arxiv']}", False) else "Hide notes",
                    key=f"toggle_notes_{paper['arxiv']}",
                    on_click=lambda k=f"show_notes_{paper['arxiv']}": st.session_state.update({k: not st.session_state.get(k, False)}),
                )
                # Show key architecture figure if available
                render_paper_figures_inline(paper["arxiv"])

        st.markdown("---")


if __name__ == "__main__":
    main()
