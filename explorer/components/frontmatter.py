"""Shared utility to strip YAML frontmatter from Markdown text.

Every page that loads .md files and renders them via st.markdown() must
call ``strip_frontmatter()`` on the raw text first.  Otherwise the
``---``-delimited YAML block is collapsed into a single paragraph and
rendered as visible text.
"""

import re

# Pattern 1 – standard frontmatter with --- delimiters
_FM_DELIMITED = re.compile(
    r"\A\s*---[ \t]*\r?\n.*?\r?\n---[ \t]*\r?\n?",
    re.DOTALL,
)

# Pattern 2 – YAML-like key: value lines at the very start (no ---)
# Matches consecutive lines that look like  "key: value"
_FM_BARE_YAML = re.compile(
    r"\A([ \t]*[\w][\w-]*[ \t]*:.*\r?\n)+",
)


def strip_frontmatter(text: str) -> str:
    """Remove YAML frontmatter from Markdown text.

    Handles:
    * Full ``---`` / ``---`` delimited frontmatter.
    * Orphaned YAML key-value lines left behind by libraries that strip
      only the ``---`` markers.
    * BOM, CRLF, leading whitespace.

    Always safe to call – returns the original text unchanged when no
    frontmatter is detected.
    """
    if not text:
        return text

    # Strip BOM
    if text[0] == "\ufeff":
        text = text[1:]

    # Case 1: standard --- block
    m = _FM_DELIMITED.search(text)
    if m:
        return text[m.end():].lstrip("\r\n")

    # Case 2: bare YAML lines (no --- delimiters)
    stripped = text.lstrip()
    if stripped and re.match(r"[\w][\w-]*\s*:", stripped):
        lines = stripped.split("\n")
        for i, line in enumerate(lines):
            s = line.strip()
            # Stop at blank line, markdown heading, or code fence
            if not s or s.startswith("#") or s.startswith("```"):
                return "\n".join(lines[i:]).lstrip("\r\n")
            # Stop if line doesn't look like YAML
            if not re.match(r"([\w][\w-]*\s*:|- |\s+[\w])", line):
                return "\n".join(lines[i:]).lstrip("\r\n")
        # All lines look like YAML → return empty
        return ""

    return text
