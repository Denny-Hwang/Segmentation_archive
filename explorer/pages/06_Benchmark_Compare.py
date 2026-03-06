"""Benchmark Comparison Tool - Side-by-side model performance comparison."""

import streamlit as st
from pathlib import Path

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import plotly.express as px
    import plotly.graph_objects as go
except ImportError:
    px = None
    go = None

try:
    import yaml
except ImportError:
    yaml = None

st.set_page_config(page_title="Benchmark Compare - Segmentation Archive", layout="wide")

ARCHIVE_ROOT = Path(__file__).resolve().parent.parent.parent
BENCHMARKS_DIR = ARCHIVE_ROOT / "05_benchmarks"

# Placeholder benchmark data for demonstration
SAMPLE_BENCHMARKS = {
    "Synapse Multi-Organ (mDSC %)": {
        "U-Net": 76.85,
        "Attention U-Net": 77.77,
        "UNet++": 78.30,
        "TransUNet": 77.48,
        "Swin-Unet": 79.13,
        "nnU-Net": 82.50,
        "HiFormer": 80.39,
    },
    "Synapse Multi-Organ (mHD95 mm)": {
        "U-Net": 31.10,
        "Attention U-Net": 29.20,
        "UNet++": 28.50,
        "TransUNet": 31.69,
        "Swin-Unet": 21.55,
        "nnU-Net": 15.20,
        "HiFormer": 20.77,
    },
    "ADE20K (mIoU %)": {
        "PSPNet (R101)": 44.39,
        "DeepLab v3+ (R101)": 45.47,
        "SegFormer-B5": 51.80,
        "Mask2Former (Swin-L)": 56.01,
        "OneFormer (Swin-L)": 57.40,
    },
    "Cityscapes val (mIoU %)": {
        "PSPNet (R101)": 79.70,
        "DeepLab v3+ (R101)": 80.90,
        "SegFormer-B5": 84.00,
        "Mask2Former (Swin-L)": 83.30,
        "OneFormer (Swin-L)": 84.40,
    },
}


def load_benchmark_data() -> dict:
    """Load benchmark data from YAML files or return sample data."""
    if BENCHMARKS_DIR.exists() and yaml is not None:
        for yaml_file in BENCHMARKS_DIR.rglob("*.yaml"):
            try:
                with open(yaml_file, "r") as f:
                    data = yaml.safe_load(f)
                if data and "benchmarks" in data:
                    return data["benchmarks"]
            except Exception:
                continue

    return SAMPLE_BENCHMARKS


def main():
    st.title("Benchmark Comparison")
    st.markdown(
        "Compare model performance across datasets and metrics. "
        "Select a benchmark to view detailed comparisons."
    )

    benchmarks = load_benchmark_data()

    if pd is None:
        st.error("pandas is required for this page. Install with: pip install pandas")
        return

    # Benchmark selector
    benchmark_names = list(benchmarks.keys())
    selected = st.selectbox("Select Benchmark", benchmark_names)

    if selected:
        data = benchmarks[selected]
        df = pd.DataFrame(
            list(data.items()),
            columns=["Model", "Score"],
        ).sort_values("Score", ascending="HD" in selected or "mm" in selected)

        st.markdown("---")

        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("Results Table")
            st.dataframe(df, use_container_width=True, hide_index=True)

        with col2:
            st.subheader("Visualization")
            if px is not None:
                ascending = "HD" in selected or "mm" in selected
                df_sorted = df.sort_values("Score", ascending=not ascending)
                fig = px.bar(
                    df_sorted,
                    x="Model",
                    y="Score",
                    title=selected,
                    color="Score",
                    color_continuous_scale="viridis",
                )
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.bar_chart(df.set_index("Model"))

    # Multi-benchmark comparison
    st.markdown("---")
    st.subheader("Multi-Benchmark Comparison")

    selected_benchmarks = st.multiselect(
        "Select benchmarks to compare",
        benchmark_names,
        default=benchmark_names[:2] if len(benchmark_names) >= 2 else benchmark_names,
    )

    if selected_benchmarks and len(selected_benchmarks) > 1:
        all_models = set()
        for bm in selected_benchmarks:
            all_models.update(benchmarks[bm].keys())

        comparison_data = {"Model": sorted(all_models)}
        for bm in selected_benchmarks:
            comparison_data[bm] = [
                benchmarks[bm].get(m, None) for m in sorted(all_models)
            ]

        comp_df = pd.DataFrame(comparison_data)
        st.dataframe(comp_df, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
