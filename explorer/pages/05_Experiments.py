"""Experiments Dashboard - View training results and experiment logs."""

import streamlit as st
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import plotly.express as px
except ImportError:
    px = None

st.set_page_config(page_title="Experiments - Segmentation Archive", layout="wide")

ARCHIVE_ROOT = Path(__file__).resolve().parent.parent.parent
EXPERIMENTS_DIR = ARCHIVE_ROOT / "04_experiments"


def load_experiment_configs() -> list[dict]:
    """Load experiment configuration YAML files."""
    configs = []
    if not EXPERIMENTS_DIR.exists() or yaml is None:
        return configs

    for yaml_file in sorted(EXPERIMENTS_DIR.rglob("*.yaml")):
        if yaml_file.name.startswith("_"):
            continue
        try:
            with open(yaml_file, "r") as f:
                data = yaml.safe_load(f)
            if data:
                data["_file"] = str(yaml_file.relative_to(ARCHIVE_ROOT))
                configs.append(data)
        except Exception:
            continue

    return configs


def main():
    st.title("Experiments Dashboard")
    st.markdown(
        "View and compare training experiments, results, and configurations."
    )

    configs = load_experiment_configs()

    if not configs:
        st.info(
            "No experiment configurations found. Add YAML files to "
            "`04_experiments/` to populate this dashboard."
        )

        # Placeholder data
        st.markdown("---")
        st.subheader("Example Experiment Summary")

        if pd is not None:
            sample_data = {
                "Model": ["U-Net", "UNet++", "TransUNet", "Swin-Unet", "nnU-Net"],
                "Dataset": ["Synapse"] * 5,
                "mDSC (%)": [76.85, 78.30, 77.48, 79.13, 82.50],
                "mHD95 (mm)": [31.10, 28.50, 31.69, 21.55, 15.20],
                "Params (M)": [31.0, 36.6, 105.3, 27.2, 31.2],
            }
            df = pd.DataFrame(sample_data)
            st.dataframe(df, use_container_width=True)

            if px is not None:
                fig = px.bar(
                    df,
                    x="Model",
                    y="mDSC (%)",
                    title="Mean Dice Score Comparison (Synapse Dataset)",
                    color="Model",
                )
                st.plotly_chart(fig, use_container_width=True)
        return

    # Display experiment configs
    st.markdown("---")
    for config in configs:
        name = config.get("experiment_name", config.get("_file", "Unknown"))
        with st.expander(f"**{name}**"):
            st.json(config)


if __name__ == "__main__":
    main()
