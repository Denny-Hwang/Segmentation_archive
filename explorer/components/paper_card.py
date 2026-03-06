"""Paper card component for displaying paper summaries in the explorer."""

from typing import Any

import streamlit as st


def render_paper_card(
    title: str,
    authors: str | list[str] = "",
    year: int | str = "",
    venue: str = "",
    arxiv: str = "",
    category: str = "",
    tags: list[str] | None = None,
    summary: str = "",
    expanded: bool = False,
) -> None:
    """Render a paper summary card in Streamlit.

    Args:
        title: Paper title.
        authors: Author name(s), either a string or list of strings.
        year: Publication year.
        venue: Publication venue (conference/journal).
        arxiv: arXiv paper ID.
        category: Paper category (e.g., architecture, loss-function).
        tags: List of topic tags.
        summary: Brief summary or abstract.
        expanded: Whether the card starts expanded.
    """
    if isinstance(authors, list):
        authors_str = ", ".join(authors)
    else:
        authors_str = authors

    with st.expander(f"**{title}** ({year})", expanded=expanded):
        col1, col2 = st.columns([3, 1])

        with col1:
            if authors_str:
                st.markdown(f"**Authors**: {authors_str}")
            if summary:
                st.markdown(summary)

        with col2:
            if venue:
                st.markdown(f"**Venue**: {venue}")
            if category:
                st.markdown(f"**Category**: {category}")
            if arxiv:
                st.markdown(
                    f"**arXiv**: [{arxiv}](https://arxiv.org/abs/{arxiv})"
                )
            if tags:
                st.markdown(f"**Tags**: {', '.join(tags)}")


def render_paper_card_from_dict(paper: dict[str, Any], **kwargs) -> None:
    """Render a paper card from a dictionary of metadata.

    Args:
        paper: Dictionary with paper metadata fields.
        **kwargs: Additional keyword arguments passed to render_paper_card.
    """
    render_paper_card(
        title=paper.get("title", "Untitled"),
        authors=paper.get("authors", ""),
        year=paper.get("year", ""),
        venue=paper.get("venue", ""),
        arxiv=paper.get("arxiv", ""),
        category=paper.get("category", ""),
        tags=paper.get("tags"),
        summary=paper.get("summary", paper.get("description", "")),
        **kwargs,
    )
