"""Paper Review Explorer - Browse and filter paper analyses with enhanced UX."""

import re
import sys

import streamlit as st
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from components.paper_figures import render_paper_figures
from components.frontmatter import strip_frontmatter

st.set_page_config(page_title="Paper Reviews - Segmentation Archive", layout="wide")

ARCHIVE_ROOT = Path(__file__).resolve().parent.parent.parent
REVIEW_DIRS = [
    ARCHIVE_ROOT / "02_unet_family",
    ARCHIVE_ROOT / "03_transformer_segmentation",
    ARCHIVE_ROOT / "04_foundation_models",
]


def _parse_markdown_file(file_path: Path) -> dict:
    raw = file_path.read_text(encoding="utf-8")
    if raw.startswith("\ufeff"):
        raw = raw[1:]

    meta: dict = {}
    match = re.match(r"\A\s*---[ \t]*\r?\n(.*?)\r?\n---[ \t]*\r?\n?", raw, re.DOTALL)
    if match:
        yaml_block = match.group(1)
        content = raw[match.end():]
        for line in yaml_block.splitlines():
            line = line.strip()
            if not line or ":" not in line:
                continue
            key, _, val = line.partition(":")
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if val.startswith("[") and val.endswith("]"):
                val = [v.strip().strip('"').strip("'") for v in val[1:-1].split(",")]
            meta[key] = val
        meta["content"] = content.lstrip("\r\n")
    else:
        meta["content"] = strip_frontmatter(raw)

    return meta


@st.cache_data(show_spinner=False)
def load_paper_reviews() -> list[dict]:
    reviews = []
    for reviews_dir in REVIEW_DIRS:
        if not reviews_dir.exists():
            continue
        for md_file in sorted(reviews_dir.rglob("*.md")):
            if md_file.name.startswith("_") or md_file.name == "README.md":
                continue
            try:
                meta = _parse_markdown_file(md_file)
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

    # ---------- Sidebar Filters ----------
    st.sidebar.subheader("Filters")

    categories = sorted({r.get("category", "uncategorized") for r in reviews})
    selected_category = st.sidebar.selectbox("Category", ["All"] + categories)

    all_tags = sorted({t for r in reviews for t in r.get("tags", []) if isinstance(r.get("tags"), list)})
    selected_tags = st.sidebar.multiselect("Tags", all_tags)

    years = sorted({r.get("year", "unknown") for r in reviews})
    selected_years = st.sidebar.multiselect("Year", years)

    search_query = st.sidebar.text_input("Search", "")

    sort_by = st.sidebar.selectbox(
        "Sort by", ["Title (A-Z)", "Year (newest)", "Year (oldest)", "Category"]
    )

    # Apply filters
    filtered = reviews
    if selected_category != "All":
        filtered = [r for r in filtered if r.get("category") == selected_category]
    if selected_tags:
        filtered = [
            r for r in filtered
            if isinstance(r.get("tags"), list) and any(t in r["tags"] for t in selected_tags)
        ]
    if selected_years:
        filtered = [r for r in filtered if r.get("year") in selected_years]
    if search_query:
        q = search_query.lower()
        filtered = [
            r for r in filtered
            if q in r.get("title", "").lower() or q in r.get("content", "").lower()
        ]

    # Sort
    if sort_by == "Title (A-Z)":
        filtered.sort(key=lambda r: r.get("title", "").lower())
    elif sort_by == "Year (newest)":
        filtered.sort(key=lambda r: str(r.get("year", "0")), reverse=True)
    elif sort_by == "Year (oldest)":
        filtered.sort(key=lambda r: str(r.get("year", "9999")))
    elif sort_by == "Category":
        filtered.sort(key=lambda r: r.get("category", ""))

    st.markdown(f"Showing **{len(filtered)}** of {len(reviews)} reviews")
    st.markdown("---")

    # ---------- Display as cards ----------
    for review in filtered:
        status = review.get("status", "")
        status_icon = {"complete": "✅", "in-progress": "🔄", "planned": "📋"}.get(status, "")
        header = f"**{review['title']}** ({review.get('year', '?')})"
        if status_icon:
            header = f"{status_icon} {header}"

        with st.expander(header):
            cols = st.columns([3, 1])
            with cols[0]:
                content = strip_frontmatter(review.get("content", ""))
                content_preview = content[:1200]
                st.markdown(content_preview)
                if len(content) > 1200:
                    if st.button("Show full content", key=f"expand_{review['file_path']}"):
                        st.markdown(content[1200:])
            with cols[1]:
                difficulty = review.get("difficulty", "")
                if difficulty:
                    diff_colors = {"beginner": "🟢", "intermediate": "🟡", "advanced": "🔴"}
                    st.markdown(f"**Difficulty**: {diff_colors.get(difficulty, '')} {difficulty}")
                st.markdown(f"**Category**: {review.get('category')}")
                st.markdown(f"**Year**: {review.get('year')}")
                tags = review.get("tags", [])
                if isinstance(tags, list) and tags:
                    st.markdown(f"**Tags**: {' '.join(f'`{t}`' for t in tags)}")
                arxiv_id = review.get("arxiv", "")
                if arxiv_id:
                    st.markdown(f"**arXiv**: [{arxiv_id}](https://arxiv.org/abs/{arxiv_id})")
                st.markdown(f"**File**: `{review['file_path']}`")

            # Architecture figures
            if not arxiv_id:
                arxiv_match = re.search(r"arxiv[:/](\d{4}\.\d{4,5})", review.get("content", ""), re.I)
                if arxiv_match:
                    arxiv_id = arxiv_match.group(1)
            if arxiv_id:
                render_paper_figures(arxiv_id)


if __name__ == "__main__":
    main()
