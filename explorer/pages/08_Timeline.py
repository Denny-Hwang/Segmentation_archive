"""Historical Timeline - Trace the evolution of segmentation methods."""

import streamlit as st
from pathlib import Path

try:
    import plotly.graph_objects as go
except ImportError:
    go = None

st.set_page_config(page_title="Timeline - Segmentation Archive", layout="wide")

ARCHIVE_ROOT = Path(__file__).resolve().parent.parent.parent

# Key milestones in segmentation history
TIMELINE_DATA = [
    {"year": 2012, "name": "AlexNet", "category": "foundation",
     "desc": "Deep CNN revolution begins with ImageNet breakthrough"},
    {"year": 2014, "name": "FCN", "category": "semantic",
     "desc": "First end-to-end CNN for pixel-wise prediction"},
    {"year": 2015, "name": "U-Net", "category": "medical",
     "desc": "Encoder-decoder with skip connections for biomedical segmentation"},
    {"year": 2015, "name": "DeepLab v1", "category": "semantic",
     "desc": "Atrous convolution and CRF for semantic segmentation"},
    {"year": 2016, "name": "3D U-Net", "category": "medical",
     "desc": "Extension of U-Net to volumetric segmentation"},
    {"year": 2016, "name": "V-Net", "category": "medical",
     "desc": "Volumetric CNN with Dice loss for medical images"},
    {"year": 2017, "name": "PSPNet", "category": "semantic",
     "desc": "Pyramid pooling for multi-scale context aggregation"},
    {"year": 2017, "name": "Mask R-CNN", "category": "instance",
     "desc": "Instance segmentation by extending Faster R-CNN"},
    {"year": 2018, "name": "UNet++", "category": "medical",
     "desc": "Nested U-Net with dense skip connections"},
    {"year": 2018, "name": "Attention U-Net", "category": "medical",
     "desc": "Attention gating for skip connections"},
    {"year": 2018, "name": "DeepLab v3+", "category": "semantic",
     "desc": "ASPP with encoder-decoder refinement"},
    {"year": 2020, "name": "ViT", "category": "foundation",
     "desc": "Vision Transformer proves transformers work for images"},
    {"year": 2021, "name": "TransUNet", "category": "medical",
     "desc": "CNN-Transformer hybrid for medical segmentation"},
    {"year": 2021, "name": "Swin Transformer", "category": "foundation",
     "desc": "Hierarchical vision transformer with shifted windows"},
    {"year": 2021, "name": "Swin-Unet", "category": "medical",
     "desc": "Pure Swin Transformer encoder-decoder"},
    {"year": 2021, "name": "SegFormer", "category": "semantic",
     "desc": "Efficient transformer design for segmentation"},
    {"year": 2021, "name": "nnU-Net", "category": "medical",
     "desc": "Self-configuring framework published in Nature Methods"},
    {"year": 2022, "name": "Mask2Former", "category": "universal",
     "desc": "Universal segmentation with masked attention"},
    {"year": 2023, "name": "OneFormer", "category": "universal",
     "desc": "One transformer for all segmentation tasks"},
    {"year": 2023, "name": "SAM", "category": "foundation",
     "desc": "Segment Anything - promptable foundation model"},
    {"year": 2024, "name": "SAM 2", "category": "foundation",
     "desc": "Segment Anything in images and videos"},
]

CATEGORY_COLORS = {
    "foundation": "#4A90D9",
    "semantic": "#50C878",
    "medical": "#FF6B6B",
    "instance": "#FFD700",
    "universal": "#DA70D6",
}


def main():
    st.title("Segmentation Timeline")
    st.markdown(
        "The historical evolution of image segmentation methods, "
        "from early CNNs to modern foundation models."
    )

    # Category filter
    categories = sorted({d["category"] for d in TIMELINE_DATA})
    selected_cats = st.multiselect(
        "Filter by category",
        categories,
        default=categories,
    )

    filtered = [d for d in TIMELINE_DATA if d["category"] in selected_cats]

    st.markdown("---")

    # Timeline visualization
    if go is not None and filtered:
        fig = go.Figure()

        for item in filtered:
            color = CATEGORY_COLORS.get(item["category"], "#888888")
            fig.add_trace(go.Scatter(
                x=[item["year"]],
                y=[item["category"]],
                mode="markers+text",
                marker=dict(size=16, color=color),
                text=[item["name"]],
                textposition="top center",
                hovertemplate=(
                    f"<b>{item['name']}</b> ({item['year']})<br>"
                    f"{item['desc']}<extra></extra>"
                ),
                showlegend=False,
            ))

        fig.update_layout(
            title="Segmentation Methods Timeline",
            xaxis_title="Year",
            yaxis_title="Category",
            height=500,
            xaxis=dict(dtick=1),
        )
        st.plotly_chart(fig, use_container_width=True)

    # Detailed timeline list
    st.subheader("Detailed Timeline")

    current_year = None
    for item in sorted(filtered, key=lambda x: x["year"]):
        if item["year"] != current_year:
            current_year = item["year"]
            st.markdown(f"### {current_year}")

        color = CATEGORY_COLORS.get(item["category"], "#888888")
        st.markdown(
            f"- **{item['name']}** [{item['category']}] - {item['desc']}"
        )


if __name__ == "__main__":
    main()
