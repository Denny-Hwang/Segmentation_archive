"""Full-text search component for the archive explorer."""

from pathlib import Path
from typing import Any

import streamlit as st

try:
    from whoosh.index import open_dir, exists_in
    from whoosh.qparser import MultifieldParser

    HAS_WHOOSH = True
except ImportError:
    HAS_WHOOSH = False


def render_search_bar(
    archive_root: Path,
    index_dir: str = "explorer/data/.search_index",
    placeholder: str = "Search the archive...",
) -> list[dict[str, Any]]:
    """Render a search bar and return matching results.

    Uses Whoosh full-text search index if available, otherwise falls
    back to basic string matching across Markdown files.

    Args:
        archive_root: Root path of the segmentation archive.
        index_dir: Relative path to the Whoosh index directory.
        placeholder: Placeholder text for the search input.

    Returns:
        List of result dicts with 'title', 'path', and 'snippet' keys.
    """
    query = st.text_input("Search", placeholder=placeholder, key="archive_search")

    if not query:
        return []

    index_path = archive_root / index_dir

    if HAS_WHOOSH and index_path.exists() and exists_in(str(index_path)):
        return _search_whoosh(index_path, query)
    else:
        return _search_fallback(archive_root, query)


def _search_whoosh(index_path: Path, query: str) -> list[dict[str, Any]]:
    """Search using a Whoosh full-text index.

    Args:
        index_path: Path to the Whoosh index directory.
        query: Search query string.

    Returns:
        List of search result dicts.
    """
    ix = open_dir(str(index_path))
    parser = MultifieldParser(["title", "content"], schema=ix.schema)
    parsed_query = parser.parse(query)

    results = []
    with ix.searcher() as searcher:
        hits = searcher.search(parsed_query, limit=20)
        for hit in hits:
            results.append({
                "title": hit.get("title", "Untitled"),
                "path": hit.get("path", ""),
                "snippet": hit.highlights("content", top=3) or "",
                "score": hit.score,
            })

    return results


def _search_fallback(
    archive_root: Path, query: str
) -> list[dict[str, Any]]:
    """Basic fallback search using string matching on Markdown files.

    Args:
        archive_root: Root path of the archive.
        query: Search query string.

    Returns:
        List of search result dicts.
    """
    results = []
    query_lower = query.lower()

    for md_file in sorted(archive_root.rglob("*.md")):
        # Skip explorer and hidden directories
        rel = str(md_file.relative_to(archive_root))
        if rel.startswith("explorer/") or "/." in rel:
            continue

        try:
            content = md_file.read_text(encoding="utf-8")
        except Exception:
            continue

        if query_lower in content.lower():
            # Extract a snippet around the first match
            idx = content.lower().find(query_lower)
            start = max(0, idx - 100)
            end = min(len(content), idx + len(query) + 100)
            snippet = content[start:end].replace("\n", " ").strip()

            results.append({
                "title": md_file.stem.replace("_", " ").title(),
                "path": rel,
                "snippet": f"...{snippet}...",
                "score": 1.0,
            })

    return results[:20]


def display_search_results(results: list[dict[str, Any]]) -> None:
    """Display search results in the Streamlit UI.

    Args:
        results: List of result dicts from render_search_bar.
    """
    if not results:
        st.info("No results found.")
        return

    st.markdown(f"**{len(results)} result(s) found**")

    for result in results:
        with st.expander(f"{result['title']} - `{result['path']}`"):
            if result.get("snippet"):
                st.markdown(result["snippet"])
