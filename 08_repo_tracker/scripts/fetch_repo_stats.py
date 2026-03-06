#!/usr/bin/env python3
"""Fetch GitHub repository statistics for tracked segmentation repos.

Uses the GitHub REST API to collect stars, forks, open issues,
latest release info, and last commit date for each tracked repository.

Usage:
    export GITHUB_TOKEN="ghp_your_token"
    python fetch_repo_stats.py [--output stats.json]
"""

import argparse
import json
import os
import sys
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
REPOS_FILE = TRACKER_DIR / "tracked_repos.yaml"
DEFAULT_OUTPUT = TRACKER_DIR / "repo_stats.json"


def load_tracked_repos(path: Path) -> list[dict]:
    """Load the tracked repositories from the YAML registry."""
    if yaml is None:
        print("ERROR: PyYAML is required. Install with: pip install pyyaml")
        sys.exit(1)
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return data.get("repositories", [])


def get_github_headers() -> dict[str, str]:
    """Build HTTP headers for GitHub API requests."""
    token = os.environ.get("GITHUB_TOKEN", "")
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    else:
        print("WARNING: No GITHUB_TOKEN set. API rate limits will be very low.")
    return headers


def fetch_repo_info(owner_repo: str, headers: dict[str, str]) -> dict[str, Any]:
    """Fetch basic repository information from the GitHub API.

    Args:
        owner_repo: Repository in 'owner/repo' format.
        headers: HTTP headers including auth token.

    Returns:
        Dictionary with repo stats or error information.
    """
    if requests is None:
        return {"error": "requests library not installed"}

    url = f"https://api.github.com/repos/{owner_repo}"
    try:
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        return {
            "full_name": data.get("full_name"),
            "stars": data.get("stargazers_count"),
            "forks": data.get("forks_count"),
            "open_issues": data.get("open_issues_count"),
            "language": data.get("language"),
            "license": data.get("license", {}).get("spdx_id") if data.get("license") else None,
            "last_push": data.get("pushed_at"),
            "created_at": data.get("created_at"),
            "description": data.get("description"),
            "archived": data.get("archived", False),
        }
    except requests.RequestException as e:
        return {"full_name": owner_repo, "error": str(e)}


def fetch_latest_release(owner_repo: str, headers: dict[str, str]) -> dict[str, Any] | None:
    """Fetch the latest release information for a repository.

    Args:
        owner_repo: Repository in 'owner/repo' format.
        headers: HTTP headers including auth token.

    Returns:
        Dictionary with release info, or None if no releases exist.
    """
    if requests is None:
        return None

    url = f"https://api.github.com/repos/{owner_repo}/releases/latest"
    try:
        resp = requests.get(url, headers=headers, timeout=30)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        data = resp.json()
        return {
            "tag": data.get("tag_name"),
            "name": data.get("name"),
            "published_at": data.get("published_at"),
            "prerelease": data.get("prerelease", False),
        }
    except requests.RequestException:
        return None


def fetch_all_stats(repos: list[dict], headers: dict[str, str]) -> list[dict[str, Any]]:
    """Fetch stats for all tracked repositories.

    Args:
        repos: List of tracked repo entries from YAML.
        headers: HTTP headers for GitHub API.

    Returns:
        List of stat dictionaries, one per repository.
    """
    results = []
    for repo_entry in repos:
        owner_repo = repo_entry.get("repo", "")
        print(f"Fetching stats for {owner_repo}...")

        info = fetch_repo_info(owner_repo, headers)
        release = fetch_latest_release(owner_repo, headers)

        results.append({
            "repo": owner_repo,
            "info": info,
            "latest_release": release,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
        })

    return results


def save_results(results: list[dict[str, Any]], output_path: Path) -> None:
    """Save fetched stats to a JSON file."""
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Saved stats for {len(results)} repos to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Fetch GitHub stats for tracked repos")
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output JSON file path",
    )
    parser.add_argument(
        "--repos-file",
        type=Path,
        default=REPOS_FILE,
        help="Path to tracked_repos.yaml",
    )
    args = parser.parse_args()

    repos = load_tracked_repos(args.repos_file)
    if not repos:
        print("No repositories found in registry.")
        sys.exit(1)

    print(f"Found {len(repos)} tracked repositories.")
    headers = get_github_headers()
    results = fetch_all_stats(repos, headers)
    save_results(results, args.output)


if __name__ == "__main__":
    main()
