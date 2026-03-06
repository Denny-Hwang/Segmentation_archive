#!/usr/bin/env python3
"""Fetch citation counts for tracked papers using the Semantic Scholar API.

Uses the Semantic Scholar Academic Graph API to retrieve citation counts,
influential citation counts, and related metadata for each tracked paper.

Usage:
    python fetch_paper_citations.py [--output citations.json]

Note:
    The Semantic Scholar API has rate limits. An API key can be set via
    the S2_API_KEY environment variable for higher limits.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import requests
except ImportError:
    requests = None

try:
    import yaml
except ImportError:
    yaml = None


TRACKER_DIR = Path(__file__).resolve().parent.parent
PAPERS_FILE = TRACKER_DIR / "tracked_papers.yaml"
DEFAULT_OUTPUT = TRACKER_DIR / "citation_stats.json"

S2_API_BASE = "https://api.semanticscholar.org/graph/v1"
FIELDS = "title,citationCount,influentialCitationCount,year,venue,externalIds,publicationDate"
REQUEST_DELAY = 1.0  # seconds between requests to respect rate limits


def load_tracked_papers(path: Path) -> list[dict]:
    """Load tracked papers from the YAML registry."""
    if yaml is None:
        print("ERROR: PyYAML is required. Install with: pip install pyyaml")
        sys.exit(1)
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return data.get("papers", [])


def get_s2_headers() -> dict[str, str]:
    """Build HTTP headers for Semantic Scholar API."""
    headers = {"Accept": "application/json"}
    api_key = os.environ.get("S2_API_KEY", "")
    if api_key:
        headers["x-api-key"] = api_key
    return headers


def fetch_paper_by_arxiv(arxiv_id: str, headers: dict[str, str]) -> dict[str, Any] | None:
    """Fetch paper metadata from Semantic Scholar using an arXiv ID.

    Args:
        arxiv_id: The arXiv paper identifier (e.g., '1505.04597').
        headers: HTTP headers for the API request.

    Returns:
        Dictionary with paper metadata, or None on failure.
    """
    if requests is None:
        return None

    url = f"{S2_API_BASE}/paper/ARXIV:{arxiv_id}"
    params = {"fields": FIELDS}

    try:
        resp = requests.get(url, headers=headers, params=params, timeout=30)
        if resp.status_code == 404:
            print(f"  Paper not found for arXiv:{arxiv_id}")
            return None
        if resp.status_code == 429:
            print("  Rate limited. Waiting 60 seconds...")
            time.sleep(60)
            resp = requests.get(url, headers=headers, params=params, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        print(f"  Error fetching arXiv:{arxiv_id}: {e}")
        return None


def fetch_all_citations(
    papers: list[dict], headers: dict[str, str]
) -> list[dict[str, Any]]:
    """Fetch citation data for all tracked papers.

    Args:
        papers: List of paper entries from the YAML registry.
        headers: HTTP headers for Semantic Scholar API.

    Returns:
        List of citation result dictionaries.
    """
    results = []

    for paper in papers:
        title = paper.get("title", "Unknown")
        arxiv_id = paper.get("arxiv", "")
        print(f"Fetching citations for: {title}")

        if arxiv_id:
            s2_data = fetch_paper_by_arxiv(arxiv_id, headers)
        else:
            print(f"  No arXiv ID available, skipping.")
            s2_data = None

        result = {
            "title": title,
            "arxiv": arxiv_id,
            "year": paper.get("year"),
            "category": paper.get("category"),
            "citation_count": s2_data.get("citationCount") if s2_data else None,
            "influential_citations": s2_data.get("influentialCitationCount") if s2_data else None,
            "s2_venue": s2_data.get("venue") if s2_data else None,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
        }
        results.append(result)

        time.sleep(REQUEST_DELAY)

    return results


def save_results(results: list[dict[str, Any]], output_path: Path) -> None:
    """Save citation data to a JSON file."""
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved citation data for {len(results)} papers to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Fetch citation counts for tracked segmentation papers"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output JSON file path",
    )
    parser.add_argument(
        "--papers-file",
        type=Path,
        default=PAPERS_FILE,
        help="Path to tracked_papers.yaml",
    )
    args = parser.parse_args()

    papers = load_tracked_papers(args.papers_file)
    if not papers:
        print("No papers found in registry.")
        sys.exit(1)

    print(f"Found {len(papers)} tracked papers.\n")
    headers = get_s2_headers()
    results = fetch_all_citations(papers, headers)
    save_results(results, args.output)


if __name__ == "__main__":
    main()
