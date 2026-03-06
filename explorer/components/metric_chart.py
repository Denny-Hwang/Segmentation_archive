"""Metric chart component for visualizing experiment results."""

from typing import Any

import streamlit as st

try:
    import plotly.express as px
    import plotly.graph_objects as go

    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

try:
    import pandas as pd

    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


def render_bar_chart(
    data: dict[str, float],
    title: str = "",
    x_label: str = "Model",
    y_label: str = "Score",
    color_scale: str = "viridis",
    sort_ascending: bool = False,
) -> None:
    """Render a bar chart comparing metric values across models.

    Args:
        data: Mapping of model names to metric values.
        title: Chart title.
        x_label: X-axis label.
        y_label: Y-axis label.
        color_scale: Plotly color scale name.
        sort_ascending: Whether to sort bars in ascending order.
    """
    if not HAS_PLOTLY or not HAS_PANDAS:
        st.warning("Install plotly and pandas for chart rendering.")
        st.json(data)
        return

    df = pd.DataFrame(list(data.items()), columns=[x_label, y_label])
    df = df.sort_values(y_label, ascending=sort_ascending)

    fig = px.bar(
        df,
        x=x_label,
        y=y_label,
        title=title,
        color=y_label,
        color_continuous_scale=color_scale,
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)


def render_radar_chart(
    models: dict[str, dict[str, float]],
    title: str = "Multi-Metric Comparison",
) -> None:
    """Render a radar chart comparing multiple metrics across models.

    Args:
        models: Mapping of model names to metric dicts.
            Example: {"U-Net": {"DSC": 0.76, "HD95": 31.1, "Params": 31.0}}
        title: Chart title.
    """
    if not HAS_PLOTLY:
        st.warning("Install plotly for radar chart rendering.")
        return

    fig = go.Figure()

    for model_name, metrics in models.items():
        categories = list(metrics.keys())
        values = list(metrics.values())
        # Close the polygon
        values.append(values[0])
        categories.append(categories[0])

        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill="toself",
            name=model_name,
        ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True)),
        showlegend=True,
        title=title,
    )
    st.plotly_chart(fig, use_container_width=True)


def render_metric_table(
    data: list[dict[str, Any]],
    title: str = "",
    highlight_best: bool = True,
) -> None:
    """Render a metric comparison table.

    Args:
        data: List of row dictionaries.
        title: Optional table title.
        highlight_best: Whether to highlight best values (not yet implemented).
    """
    if not HAS_PANDAS:
        st.warning("Install pandas for table rendering.")
        return

    if title:
        st.subheader(title)

    df = pd.DataFrame(data)
    st.dataframe(df, use_container_width=True, hide_index=True)
