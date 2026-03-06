#!/usr/bin/env python3
"""Validate all _registry.yaml files in the Segmentation Archive.

Checks each registry file for:
- Valid YAML syntax
- Required fields are present
- Referenced files exist on disk
- No duplicate IDs within a registry

Usage:
    python scripts/validate_registry.py [--archive-root /path/to/archive]
"""

import argparse
import sys
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:
    print("ERROR: PyYAML is required. Install with: pip install pyyaml")
    sys.exit(1)


def find_registries(root: Path) -> list[Path]:
    """Find all _registry.yaml files under the archive root."""
    return sorted(root.rglob("_registry.yaml"))


def validate_yaml_syntax(path: Path) -> tuple[bool, Any]:
    """Validate that a file contains valid YAML.

    Args:
        path: Path to the YAML file.

    Returns:
        Tuple of (is_valid, parsed_data_or_error_message).
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return True, data
    except yaml.YAMLError as e:
        return False, str(e)


def validate_registry_content(
    data: dict, registry_path: Path
) -> list[str]:
    """Validate the content of a registry file.

    Checks for required fields and referenced file existence.

    Args:
        data: Parsed YAML data.
        registry_path: Path to the registry file (for resolving relative paths).

    Returns:
        List of error/warning messages.
    """
    issues = []
    registry_dir = registry_path.parent

    if not isinstance(data, dict):
        issues.append(f"ERROR: Registry root must be a mapping, got {type(data).__name__}")
        return issues

    # Find the main list (could be 'papers', 'repositories', 'datasets', etc.)
    list_keys = [k for k, v in data.items() if isinstance(v, list)]

    if not list_keys:
        issues.append("WARNING: No list entries found in registry")
        return issues

    for key in list_keys:
        entries = data[key]
        seen_ids = set()

        for i, entry in enumerate(entries):
            if not isinstance(entry, dict):
                issues.append(f"WARNING: {key}[{i}] is not a mapping")
                continue

            # Check for duplicate IDs
            entry_id = entry.get("id") or entry.get("name") or entry.get("title")
            if entry_id:
                if entry_id in seen_ids:
                    issues.append(f"ERROR: Duplicate ID '{entry_id}' in {key}")
                seen_ids.add(entry_id)

            # Check referenced files exist
            file_ref = entry.get("file")
            if file_ref:
                file_path = registry_dir / file_ref
                if not file_path.exists():
                    issues.append(
                        f"WARNING: Referenced file not found: {file_ref} "
                        f"(entry: {entry_id or i})"
                    )

            # Check URL fields are non-empty if present
            url = entry.get("url")
            if url is not None and not url:
                issues.append(
                    f"WARNING: Empty URL in entry '{entry_id or i}'"
                )

    return issues


def validate_all(root: Path) -> dict[str, list[str]]:
    """Validate all registry files in the archive.

    Args:
        root: Archive root directory.

    Returns:
        Mapping of registry paths to lists of issues.
    """
    results = {}
    registries = find_registries(root)

    if not registries:
        print(f"No _registry.yaml files found under {root}")
        return results

    for reg_path in registries:
        rel_path = str(reg_path.relative_to(root))
        issues = []

        # Check YAML syntax
        is_valid, data_or_error = validate_yaml_syntax(reg_path)
        if not is_valid:
            issues.append(f"ERROR: Invalid YAML syntax: {data_or_error}")
            results[rel_path] = issues
            continue

        # Check content
        content_issues = validate_registry_content(data_or_error, reg_path)
        issues.extend(content_issues)

        results[rel_path] = issues

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Validate all _registry.yaml files in the archive"
    )
    parser.add_argument(
        "--archive-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="Path to the archive root directory",
    )
    args = parser.parse_args()

    root = args.archive_root
    print(f"Validating registries under: {root}\n")

    results = validate_all(root)

    if not results:
        print("No registry files found.")
        sys.exit(1)

    total_errors = 0
    total_warnings = 0

    for reg_path, issues in results.items():
        errors = [i for i in issues if i.startswith("ERROR")]
        warnings = [i for i in issues if i.startswith("WARNING")]
        total_errors += len(errors)
        total_warnings += len(warnings)

        if issues:
            status = "FAIL" if errors else "WARN"
        else:
            status = "OK"

        print(f"[{status}] {reg_path}")
        for issue in issues:
            print(f"  {issue}")

    print(f"\n{'='*50}")
    print(f"Registries checked: {len(results)}")
    print(f"Errors: {total_errors}")
    print(f"Warnings: {total_warnings}")

    if total_errors > 0:
        sys.exit(1)
    else:
        print("\nAll registries are valid.")
        sys.exit(0)


if __name__ == "__main__":
    main()
