"""Architecture diagram viewer component for the explorer."""

import streamlit as st


def render_mermaid_diagram(mermaid_code: str, height: int = 400) -> None:
    """Render a Mermaid.js diagram in Streamlit.

    Attempts to use streamlit-mermaid if available, falls back to
    displaying the raw Mermaid code.

    Args:
        mermaid_code: Mermaid diagram definition string.
        height: Height of the rendered diagram in pixels.
    """
    try:
        from streamlit_mermaid import st_mermaid

        st_mermaid(mermaid_code, height=height)
    except ImportError:
        st.info(
            "Install `streamlit-mermaid` for interactive diagrams: "
            "`pip install streamlit-mermaid`"
        )
        st.code(mermaid_code, language="mermaid")


def render_architecture_diagram(
    name: str,
    diagram_code: str | None = None,
    description: str = "",
    components: list[dict] | None = None,
) -> None:
    """Render an architecture visualization with optional component details.

    Args:
        name: Architecture name (e.g., 'U-Net').
        diagram_code: Mermaid diagram definition.
        description: Architecture description text.
        components: List of component dicts with 'name' and 'description' keys.
    """
    st.subheader(name)

    if description:
        st.markdown(description)

    if diagram_code:
        render_mermaid_diagram(diagram_code)

    if components:
        st.markdown("**Components:**")
        for comp in components:
            with st.expander(comp.get("name", "Component")):
                st.markdown(comp.get("description", ""))


def extract_mermaid_from_markdown(markdown_text: str) -> list[str]:
    """Extract Mermaid code blocks from Markdown text.

    Args:
        markdown_text: Raw Markdown content.

    Returns:
        List of Mermaid diagram code strings.
    """
    diagrams = []
    in_mermaid = False
    current_block = []

    for line in markdown_text.split("\n"):
        if line.strip().startswith("```mermaid"):
            in_mermaid = True
            current_block = []
        elif line.strip() == "```" and in_mermaid:
            in_mermaid = False
            diagrams.append("\n".join(current_block))
        elif in_mermaid:
            current_block.append(line)

    return diagrams
