#!/usr/bin/env python3
"""Check for new releases in tracked repositories.

Compares the latest GitHub releases against previously recorded releases
and reports any new versions found.

Usage:
    export GITHUB_TOKEN="ghp_your_token"
    python check_new_releases.py [--since 2026-03-01]
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
RELEASES_CACHE = TRACKER_DIR / "known_releases.json"


def load_tracked_repos(path: Path) -> list[dict]:
    """Load tracked repositories from the YAML registry."""
    if yaml is None:
        print("ERROR: PyYAML is required. Install with: pip install pyyaml")
        sys.exit(1)
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return data.get("repositories", [])


def load_known_releases(path: Path) -> dict[str, str]:
    """Load previously known release tags from cache.

    Returns:
        Mapping of 'owner/repo' to last known release tag.
    """
    if not path.exists():
        return {}
    with open(path, "r") as f:
        return json.load(f)


def save_known_releases(releases: dict[str, str], path: Path) -> None:
    """Save known release tags to cache."""
    with open(path, "w") as f:
        json.dump(releases, f, indent=2)


def get_github_headers() -> dict[str, str]:
    """Build HTTP headers for GitHub API requests."""
    token = os.environ.get("GITHUB_TOKEN", "")
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def fetch_recent_releases(
    owner_repo: str,
    headers: dict[str, str],
    since: datetime | None = None,
) -> list[dict[str, Any]]:
    """Fetch recent releases for a repository.

    Args:
        owner_repo: Repository in 'owner/repo' format.
        headers: HTTP headers for GitHub API.
        since: Only return releases published after this datetime.

    Returns:
        List of release info dictionaries.
    """
    if requests is None:
        print("ERROR: requests library not installed.")
        return []

    url = f"https://api.github.com/repos/{owner_repo}/releases"
    params = {"per_page": 10}

    try:
        resp = requests.get(url, headers=headers, params=params, timeout=30)
        if resp.status_code == 404:
            return []
        resp.raise_for_status()
        releases = resp.json()
    except requests.RequestException as e:
        print(f"  Error fetching releases for {owner_repo}: {e}")
        return []

    results = []
    for rel in releases:
        published = rel.get("published_at", "")
        if since and published:
            pub_dt = datetime.fromisoformat(published.replace("Z", "+00:00"))
            if pub_dt < since:
                continue
        results.append({
            "tag": rel.get("tag_name"),
            "name": rel.get("name"),
            "published_at": published,
            "prerelease": rel.get("prerelease", False),
            "html_url": rel.get("html_url"),
        })

    return results


def check_all_repos(
    repos: list[dict],
    known: dict[str, str],
    headers: dict[str, str],
    since: datetime | None = None,
) -> list[dict[str, Any]]:
    """Check all tracked repos for new releases.

    Args:
        repos: Tracked repository entries.
        known: Previously known release tags.
        headers: GitHub API headers.
        since: Optional cutoff date for releases.

    Returns:
        List of new release records.
    """
    new_releases = []

    for repo_entry in repos:
        if not repo_entry.get("watch_releases", False):
            continue

        owner_repo = repo_entry.get("repo", "")
        print(f"Checking {owner_repo}...")

        releases = fetch_recent_releases(owner_repo, headers, since)
        last_known = known.get(owner_repo)

        for rel in releases:
            tag = rel.get("tag")
            if tag and tag != last_known:
                new_releases.append({
                    "repo": owner_repo,
                    **rel,
                })

        # Update known releases cache
        if releases:
            known[owner_repo] = releases[0]["tag"]

    return new_releases


def print_results(new_releases: list[dict[str, Any]]) -> None:
    """Print new releases in a readable format."""
    if not new_releases:
        print("\nNo new releases found.")
        return

    print(f"\n{'='*60}")
    print(f"  Found {len(new_releases)} new release(s)")
    print(f"{'='*60}\n")

    for rel in new_releases:
        pre = " [PRE-RELEASE]" if rel.get("prerelease") else ""
        print(f"  {rel['repo']} -> {rel['tag']}{pre}")
        print(f"    Name: {rel.get('name', 'N/A')}")
        print(f"    Published: {rel.get('published_at', 'N/A')}")
        print(f"    URL: {rel.get('html_url', 'N/A')}")
        print()


def main():
    parser = argparse.ArgumentParser(description="Check for new repo releases")
    parser.add_argument(
        "--since",
        type=str,
        default=None,
        help="Only show releases after this date (YYYY-MM-DD)",
    )
    args = parser.parse_args()

    since = None
    if args.since:
        since = datetime.strptime(args.since, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    repos = load_tracked_repos(REPOS_FILE)
    known = load_known_releases(RELEASES_CACHE)
    headers = get_github_headers()

    new_releases = check_all_repos(repos, known, headers, since)
    print_results(new_releases)

    save_known_releases(known, RELEASES_CACHE)
    print(f"Release cache updated at {RELEASES_CACHE}")


if __name__ == "__main__":
    main()
