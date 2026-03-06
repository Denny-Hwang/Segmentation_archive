"""Reading Roadmap Navigator - Structured learning path for segmentation."""

import streamlit as st
from pathlib import Path

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


def main():
    st.title("Reading Roadmap")
    st.markdown(
        "A structured learning path from beginner to cutting-edge "
        "in image segmentation research."
    )

    # Progress tracking
    if "completed_papers" not in st.session_state:
        st.session_state.completed_papers = set()

    total_papers = sum(len(level["papers"]) for level in ROADMAP.values())
    completed = len(st.session_state.completed_papers)

    st.progress(completed / total_papers if total_papers > 0 else 0)
    st.caption(f"Progress: {completed}/{total_papers} papers read")

    st.markdown("---")

    for level_name, level_data in ROADMAP.items():
        level_completed = sum(
            1 for p in level_data["papers"]
            if p["title"] in st.session_state.completed_papers
        )

        st.subheader(f"{level_name} ({level_completed}/{len(level_data['papers'])})")
        st.markdown(f"*{level_data['description']}*")

        for paper in level_data["papers"]:
            col1, col2 = st.columns([4, 1])

            with col1:
                is_done = paper["title"] in st.session_state.completed_papers
                prefix = "[Done]" if is_done else ""
                st.markdown(
                    f"**{prefix} {paper['title']}** "
                    f"({paper['authors']}, {paper['year']})"
                )
                st.caption(
                    f"Why read this: {paper['reason']} | "
                    f"[arXiv:{paper['arxiv']}](https://arxiv.org/abs/{paper['arxiv']})"
                )

            with col2:
                if st.checkbox(
                    "Read",
                    value=paper["title"] in st.session_state.completed_papers,
                    key=f"paper_{paper['arxiv']}",
                ):
                    st.session_state.completed_papers.add(paper["title"])
                else:
                    st.session_state.completed_papers.discard(paper["title"])

        st.markdown("---")


if __name__ == "__main__":
    main()
