"""Paper Review Explorer - Browse and filter paper analyses."""

import re
import sys

import streamlit as st
from pathlib import Path

# Add explorer to path for component imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from components.paper_figures import render_paper_figures

try:
    import frontmatter
except ImportError:
    frontmatter = None

st.set_page_config(page_title="Paper Reviews - Segmentation Archive", layout="wide")

ARCHIVE_ROOT = Path(__file__).resolve().parent.parent.parent
REVIEW_DIRS = [
    ARCHIVE_ROOT / "02_unet_family",
    ARCHIVE_ROOT / "03_transformer_segmentation",
    ARCHIVE_ROOT / "04_foundation_models",
]

# Regex to strip YAML frontmatter (---\n...\n---) from raw markdown
_FRONTMATTER_RE = re.compile(r"\A---\s*\n.*?\n---\s*\n?", re.DOTALL)


def _strip_frontmatter(text: str) -> str:
    """Remove YAML frontmatter block from markdown text."""
    return _FRONTMATTER_RE.sub("", text).lstrip("\n")


def _parse_yaml_frontmatter(text: str) -> dict:
    """Minimal YAML frontmatter parser (fallback when python-frontmatter unavailable)."""
    meta: dict = {}
    match = re.match(r"\A---\s*\n(.*?)\n---\s*\n?", text, re.DOTALL)
    if not match:
        return {"content": text}
    yaml_block = match.group(1)
    content = text[match.end():]
    for line in yaml_block.splitlines():
        if ":" not in line:
            continue
        key, _, val = line.partition(":")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if val.startswith("[") and val.endswith("]"):
            val = [v.strip().strip('"').strip("'") for v in val[1:-1].split(",")]
        meta[key] = val
    meta["content"] = content.lstrip("\n")
    return meta


def load_paper_reviews() -> list[dict]:
    """Load all paper review files with their frontmatter metadata."""
    reviews = []
    for reviews_dir in REVIEW_DIRS:
        if not reviews_dir.exists():
            continue

        for md_file in sorted(reviews_dir.rglob("*.md")):
            if md_file.name.startswith("_") or md_file.name == "README.md":
                continue
            try:
                if frontmatter:
                    post = frontmatter.load(md_file)
                    meta = dict(post.metadata)
                    meta["content"] = post.content
                else:
                    raw = md_file.read_text(encoding="utf-8")
                    meta = _parse_yaml_frontmatter(raw)
                meta["file_path"] = str(md_file.relative_to(ARCHIVE_ROOT))
                meta.setdefault("title", md_file.stem.replace("_", " ").title())
                meta.setdefault("category", reviews_dir.name)
                meta.setdefault("year", "unknown")
                meta.setdefault("tags", [])
                reviews.append(meta)
            except Exception:
                continue

    return reviews


def main():
    st.title("Paper Reviews")
    st.markdown("Browse and filter detailed analyses of segmentation papers.")

    reviews = load_paper_reviews()

    if not reviews:
        st.info(
            "No paper reviews found. Add Markdown files with YAML frontmatter "
            "to `02_unet_family/`, `03_transformer_segmentation/`, or "
            "`04_foundation_models/` to populate this page."
        )
        return

    # Filters
    st.sidebar.subheader("Filters")

    categories = sorted({r.get("category", "uncategorized") for r in reviews})
    selected_category = st.sidebar.selectbox(
        "Category", ["All"] + categories
    )

    all_tags = sorted({t for r in reviews for t in r.get("tags", [])})
    selected_tags = st.sidebar.multiselect("Tags", all_tags)

    search_query = st.sidebar.text_input("Search", "")

    # Apply filters
    filtered = reviews
    if selected_category != "All":
        filtered = [r for r in filtered if r.get("category") == selected_category]
    if selected_tags:
        filtered = [
            r for r in filtered
            if any(t in r.get("tags", []) for t in selected_tags)
        ]
    if search_query:
        query_lower = search_query.lower()
        filtered = [
            r for r in filtered
            if query_lower in r.get("title", "").lower()
            or query_lower in r.get("content", "").lower()
        ]

    st.markdown(f"Showing **{len(filtered)}** of {len(reviews)} reviews")
    st.markdown("---")

    # Display reviews
    for review in filtered:
        # Build status badge
        status = review.get("status", "")
        status_icon = {"complete": "✅", "in-progress": "🔄", "planned": "📋"}.get(
            status, ""
        )
        header = f"**{review['title']}** ({review.get('year', '?')})"
        if status_icon:
            header = f"{status_icon} {header}"

        with st.expander(header):
            cols = st.columns([3, 1])
            with cols[0]:
                # Ensure content is clean of frontmatter
                content = review.get("content", "")
                content = _strip_frontmatter(content)
                content_preview = content[:800]
                st.markdown(content_preview)
                if len(content) > 800:
                    if st.button(
                        "Show full content",
                        key=f"expand_{review['file_path']}",
                    ):
                        st.markdown(content[800:])
            with cols[1]:
                difficulty = review.get("difficulty", "")
                if difficulty:
                    diff_colors = {
                        "beginner": "🟢",
                        "intermediate": "🟡",
                        "advanced": "🔴",
                    }
                    st.markdown(
                        f"**Difficulty**: {diff_colors.get(difficulty, '')} {difficulty}"
                    )
                st.markdown(f"**Category**: {review.get('category')}")
                st.markdown(f"**Year**: {review.get('year')}")
                tags = review.get("tags", [])
                if tags:
                    tag_html = " ".join(
                        f"`{t}`" for t in tags
                    )
                    st.markdown(f"**Tags**: {tag_html}")
                st.markdown(f"**File**: `{review['file_path']}`")

            # Show key figures if arxiv ID is available
            arxiv_id = review.get("arxiv", "")
            if not arxiv_id:
                # Try to extract arxiv ID from content
                arxiv_match = re.search(r"arxiv[:/](\d{4}\.\d{4,5})", review.get("content", ""), re.I)
                if arxiv_match:
                    arxiv_id = arxiv_match.group(1)
            if arxiv_id:
                render_paper_figures(arxiv_id)


if __name__ == "__main__":
    main()
