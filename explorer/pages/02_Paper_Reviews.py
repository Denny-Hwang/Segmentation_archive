"""Paper Review Explorer - Browse and filter paper analyses."""

import streamlit as st
from pathlib import Path

try:
    import frontmatter
except ImportError:
    frontmatter = None

st.set_page_config(page_title="Paper Reviews - Segmentation Archive", layout="wide")

ARCHIVE_ROOT = Path(__file__).resolve().parent.parent.parent
REVIEWS_DIR = ARCHIVE_ROOT / "01_paper_reviews"


def load_paper_reviews() -> list[dict]:
    """Load all paper review files with their frontmatter metadata."""
    reviews = []
    if not REVIEWS_DIR.exists():
        return reviews

    for md_file in sorted(REVIEWS_DIR.rglob("*.md")):
        if md_file.name.startswith("_") or md_file.name == "README.md":
            continue
        try:
            if frontmatter:
                post = frontmatter.load(md_file)
                meta = dict(post.metadata)
                meta["content"] = post.content
            else:
                meta = {"content": md_file.read_text(encoding="utf-8")}
            meta["file_path"] = str(md_file.relative_to(ARCHIVE_ROOT))
            meta.setdefault("title", md_file.stem.replace("_", " ").title())
            meta.setdefault("category", "uncategorized")
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
            "to `01_paper_reviews/` to populate this page."
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
        with st.expander(f"**{review['title']}** ({review.get('year', '?')})"):
            cols = st.columns([3, 1])
            with cols[0]:
                content_preview = review.get("content", "")[:500]
                st.markdown(content_preview)
                if len(review.get("content", "")) > 500:
                    st.caption("... (truncated)")
            with cols[1]:
                st.markdown(f"**Category**: {review.get('category')}")
                st.markdown(f"**Year**: {review.get('year')}")
                tags = review.get("tags", [])
                if tags:
                    st.markdown(f"**Tags**: {', '.join(tags)}")
                st.markdown(f"**File**: `{review['file_path']}`")


if __name__ == "__main__":
    main()
