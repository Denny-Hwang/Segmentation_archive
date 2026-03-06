"""Repository Tracker - Monitor GitHub repos and paper citations."""

import json
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

st.set_page_config(page_title="Repo Tracker - Segmentation Archive", layout="wide")

ARCHIVE_ROOT = Path(__file__).resolve().parent.parent.parent
TRACKER_DIR = ARCHIVE_ROOT / "08_repo_tracker"


def load_tracked_repos() -> list[dict]:
    """Load tracked repositories from YAML registry."""
    repos_file = TRACKER_DIR / "tracked_repos.yaml"
    if not repos_file.exists() or yaml is None:
        return []
    with open(repos_file, "r") as f:
        data = yaml.safe_load(f)
    return data.get("repositories", [])


def load_tracked_papers() -> list[dict]:
    """Load tracked papers from YAML registry."""
    papers_file = TRACKER_DIR / "tracked_papers.yaml"
    if not papers_file.exists() or yaml is None:
        return []
    with open(papers_file, "r") as f:
        data = yaml.safe_load(f)
    return data.get("papers", [])


def load_repo_stats() -> list[dict]:
    """Load previously fetched repo stats if available."""
    stats_file = TRACKER_DIR / "repo_stats.json"
    if not stats_file.exists():
        return []
    with open(stats_file, "r") as f:
        return json.load(f)


def main():
    st.title("Repository & Paper Tracker")
    st.markdown(
        "Monitor key segmentation repositories and track paper citation metrics."
    )

    tab1, tab2 = st.tabs(["Repositories", "Papers"])

    # --- Repositories tab ---
    with tab1:
        repos = load_tracked_repos()
        stats = load_repo_stats()

        if not repos:
            st.info("No tracked repositories found. Check `08_repo_tracker/tracked_repos.yaml`.")
        else:
            st.subheader(f"Tracked Repositories ({len(repos)})")

            if pd is not None:
                repo_data = []
                stats_lookup = {s["repo"]: s for s in stats} if stats else {}

                for repo in repos:
                    owner_repo = repo.get("repo", "")
                    stat = stats_lookup.get(owner_repo, {})
                    info = stat.get("info", {})

                    repo_data.append({
                        "Repository": owner_repo,
                        "Category": repo.get("category", ""),
                        "Framework": repo.get("framework", ""),
                        "Stars": info.get("stars", "N/A"),
                        "Forks": info.get("forks", "N/A"),
                        "URL": repo.get("url", ""),
                    })

                df = pd.DataFrame(repo_data)
                st.dataframe(df, use_container_width=True, hide_index=True)
            else:
                for repo in repos:
                    with st.expander(repo.get("name", "Unknown")):
                        st.markdown(f"**Repo**: [{repo.get('repo')}]({repo.get('url')})")
                        st.markdown(f"**Category**: {repo.get('category')}")
                        st.markdown(f"**Framework**: {repo.get('framework')}")
                        st.markdown(f"**Description**: {repo.get('description', '')}")

            if not stats:
                st.caption(
                    "Run `python 08_repo_tracker/scripts/fetch_repo_stats.py` "
                    "to populate star and fork counts."
                )

    # --- Papers tab ---
    with tab2:
        papers = load_tracked_papers()

        if not papers:
            st.info("No tracked papers found. Check `08_repo_tracker/tracked_papers.yaml`.")
        else:
            st.subheader(f"Tracked Papers ({len(papers)})")

            if pd is not None:
                paper_data = []
                for paper in papers:
                    paper_data.append({
                        "Title": paper.get("title", ""),
                        "Year": paper.get("year", ""),
                        "Venue": paper.get("venue", ""),
                        "Category": paper.get("category", ""),
                        "arXiv": paper.get("arxiv", ""),
                    })

                df = pd.DataFrame(paper_data)
                st.dataframe(df, use_container_width=True, hide_index=True)
            else:
                for paper in papers:
                    st.markdown(
                        f"- **{paper.get('title')}** ({paper.get('year')}) "
                        f"- {paper.get('venue', '')} "
                        f"[arXiv:{paper.get('arxiv', '')}]"
                    )


if __name__ == "__main__":
    main()
