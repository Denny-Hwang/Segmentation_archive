"""Knowledge graph component for visualizing paper relationships."""

from typing import Any

import streamlit as st

try:
    from streamlit_agraph import agraph, Node, Edge, Config

    HAS_AGRAPH = True
except ImportError:
    HAS_AGRAPH = False


# Default node colors by category
CATEGORY_COLORS = {
    "architecture": "#4A90D9",
    "medical": "#FF6B6B",
    "semantic": "#50C878",
    "foundation": "#FFD700",
    "loss-function": "#DA70D6",
    "framework": "#FF8C00",
    "panoptic": "#00CED1",
    "default": "#888888",
}


def render_knowledge_graph(
    nodes_data: list[dict[str, Any]],
    edges_data: list[dict[str, Any]],
    title: str = "Paper Relationship Graph",
    height: int = 600,
    physics: bool = True,
) -> None:
    """Render an interactive knowledge graph of paper relationships.

    Args:
        nodes_data: List of node dicts with keys: id, label, category (optional).
        edges_data: List of edge dicts with keys: source, target, label (optional).
        title: Graph title.
        height: Graph height in pixels.
        physics: Whether to enable physics simulation.
    """
    st.subheader(title)

    if not HAS_AGRAPH:
        st.info(
            "Install `streamlit-agraph` for interactive graph visualization: "
            "`pip install streamlit-agraph`"
        )
        # Fallback text representation
        st.markdown("**Nodes:**")
        for node in nodes_data:
            st.markdown(f"- {node.get('label', node.get('id'))}")
        st.markdown("**Edges:**")
        for edge in edges_data:
            label = edge.get("label", "relates to")
            st.markdown(f"- {edge['source']} --[{label}]--> {edge['target']}")
        return

    # Graph controls toolbar
    ctrl_cols = st.columns([1, 1, 1, 1, 4])
    with ctrl_cols[0]:
        if st.button("🔄 Reset View", key=f"reset_graph_{title}", help="Reset zoom and position to default"):
            # Increment a counter to force re-render of the graph
            reset_key = f"_graph_reset_{title}"
            st.session_state[reset_key] = st.session_state.get(reset_key, 0) + 1
    with ctrl_cols[1]:
        physics = st.checkbox("Physics", value=physics, key=f"physics_{title}", help="Toggle physics simulation")
    with ctrl_cols[2]:
        hierarchical = st.checkbox("Hierarchical", value=False, key=f"hier_{title}", help="Toggle hierarchical layout")
    with ctrl_cols[3]:
        height = st.select_slider(
            "Height",
            options=[400, 600, 800, 1000],
            value=height,
            key=f"height_{title}",
        )

    # Legend
    used_categories = {node.get("category", "default") for node in nodes_data}
    legend_items = []
    for cat, color in CATEGORY_COLORS.items():
        if cat in used_categories:
            legend_items.append(f'<span style="color:{color}">●</span> {cat}')
    if legend_items:
        st.markdown(
            " &nbsp;&nbsp; ".join(legend_items),
            unsafe_allow_html=True,
        )

    # Build graph version key to force re-render on reset
    reset_key = f"_graph_reset_{title}"
    version = st.session_state.get(reset_key, 0)

    nodes = []
    for node in nodes_data:
        category = node.get("category", "default")
        color = CATEGORY_COLORS.get(category, CATEGORY_COLORS["default"])
        nodes.append(Node(
            id=node["id"],
            label=node.get("label", node["id"]),
            size=node.get("size", 20),
            color=color,
            title=node.get("title", node.get("label", "")),
        ))

    edges = []
    for edge in edges_data:
        edges.append(Edge(
            source=edge["source"],
            target=edge["target"],
            label=edge.get("label", ""),
            color=edge.get("color", "#666666"),
        ))

    config = Config(
        width="100%",
        height=height,
        directed=True,
        physics=physics,
        hierarchical=hierarchical,
    )

    agraph(nodes=nodes, edges=edges, config=config)


def build_citation_graph(papers: list[dict[str, Any]]) -> tuple[list, list]:
    """Build a citation graph from paper metadata.

    This constructs nodes and edges based on a 'cites' or 'builds_on'
    field in each paper entry.

    Args:
        papers: List of paper metadata dicts. Each may have a 'cites' list
            of paper IDs that it references.

    Returns:
        Tuple of (nodes_data, edges_data) for render_knowledge_graph.
    """
    nodes = []
    edges = []
    paper_ids = set()

    for paper in papers:
        paper_id = paper.get("id", paper.get("title", ""))
        if paper_id in paper_ids:
            continue
        paper_ids.add(paper_id)

        nodes.append({
            "id": paper_id,
            "label": paper.get("title", paper_id),
            "category": paper.get("category", "default"),
            "title": f"{paper.get('title', '')} ({paper.get('year', '')})",
        })

        for cited_id in paper.get("cites", []):
            edges.append({
                "source": paper_id,
                "target": cited_id,
                "label": "cites",
            })

        for parent_id in paper.get("builds_on", []):
            edges.append({
                "source": paper_id,
                "target": parent_id,
                "label": "builds on",
                "color": "#FF6B6B",
            })

    return nodes, edges
