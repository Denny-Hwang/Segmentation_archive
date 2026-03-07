"""YAML registry parser and Markdown loader for the Segmentation Archive.

Provides utilities to discover, parse, and load archive content including
YAML registry files and Markdown documents with frontmatter.
"""

from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:
    yaml = None

try:
    import frontmatter
except ImportError:
    frontmatter = None


def get_archive_root() -> Path:
    """Return the root path of the Segmentation Archive."""
    return Path(__file__).resolve().parent.parent.parent


def load_yaml(path: Path) -> dict[str, Any]:
    """Load and parse a YAML file.

    Args:
        path: Path to the YAML file.

    Returns:
        Parsed YAML data as a dictionary.

    Raises:
        ImportError: If PyYAML is not installed.
        FileNotFoundError: If the file does not exist.
    """
    if yaml is None:
        raise ImportError("PyYAML is required. Install with: pip install pyyaml")
    if not path.exists():
        raise FileNotFoundError(f"YAML file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def discover_registries(root: Path | None = None) -> list[Path]:
    """Find all _registry.yaml files in the archive.

    Args:
        root: Archive root directory. Defaults to auto-detected root.

    Returns:
        List of paths to registry files.
    """
    if root is None:
        root = get_archive_root()
    return sorted(root.rglob("_registry.yaml"))


def load_all_registries(root: Path | None = None) -> dict[str, Any]:
    """Load all registry files and combine into a single dictionary.

    Args:
        root: Archive root directory.

    Returns:
        Dictionary mapping section names to their registry data.
    """
    if root is None:
        root = get_archive_root()

    registries = {}
    for reg_path in discover_registries(root):
        section = reg_path.parent.name
        try:
            data = load_yaml(reg_path)
            registries[section] = data
        except Exception as e:
            registries[section] = {"_error": str(e)}

    return registries


def load_markdown(path: Path) -> dict[str, Any]:
    """Load a Markdown file, parsing frontmatter if available.

    Uses a self-contained parser to reliably separate ``---`` delimited
    YAML frontmatter from the Markdown body.  The ``python-frontmatter``
    library is **not** used, because it sometimes leaves YAML metadata
    inside the returned content string.

    Args:
        path: Path to the Markdown file.

    Returns:
        Dictionary with 'metadata' (from frontmatter) and 'content' keys.
    """
    import re

    if not path.exists():
        raise FileNotFoundError(f"Markdown file not found: {path}")

    text = path.read_text(encoding="utf-8")

    # Strip BOM
    if text.startswith("\ufeff"):
        text = text[1:]

    metadata: dict[str, Any] = {}
    content = text

    # Match ---\n<yaml>\n--- block at the very start of the file
    match = re.match(
        r"\A\s*---[ \t]*\r?\n(.*?)\r?\n---[ \t]*\r?\n?", text, re.DOTALL
    )
    if match:
        yaml_block = match.group(1)
        content = text[match.end():]
        if yaml is not None:
            try:
                metadata = yaml.safe_load(yaml_block) or {}
            except Exception:
                # Fall back to simple key: value parsing
                metadata = _parse_simple_yaml(yaml_block)
        else:
            metadata = _parse_simple_yaml(yaml_block)

    return {
        "metadata": metadata,
        "content": content.lstrip("\r\n"),
        "path": str(path),
    }


def _parse_simple_yaml(yaml_block: str) -> dict[str, Any]:
    """Minimal key: value parser for YAML frontmatter (no PyYAML needed)."""
    meta: dict[str, Any] = {}
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
    return meta


def discover_markdown_files(
    directory: Path,
    exclude_prefixes: tuple[str, ...] = ("_", "."),
    exclude_names: tuple[str, ...] = ("README.md",),
) -> list[Path]:
    """Find all Markdown files in a directory tree.

    Args:
        directory: Directory to search.
        exclude_prefixes: Skip files starting with these prefixes.
        exclude_names: Skip files with these exact names.

    Returns:
        Sorted list of Markdown file paths.
    """
    if not directory.exists():
        return []

    files = []
    for md_file in directory.rglob("*.md"):
        if md_file.name in exclude_names:
            continue
        if any(md_file.name.startswith(p) for p in exclude_prefixes):
            continue
        files.append(md_file)

    return sorted(files)


def load_section_documents(
    section_dir: Path,
) -> list[dict[str, Any]]:
    """Load all Markdown documents from a section directory.

    Args:
        section_dir: Path to the section directory (e.g., 01_paper_reviews/).

    Returns:
        List of document dictionaries with metadata and content.
    """
    documents = []
    for md_path in discover_markdown_files(section_dir):
        try:
            doc = load_markdown(md_path)
            doc["relative_path"] = str(md_path.relative_to(section_dir.parent))
            documents.append(doc)
        except Exception:
            continue

    return documents
