"""Code block component with syntax highlighting for the explorer."""

import streamlit as st


def render_code_block(
    code: str,
    language: str = "python",
    title: str = "",
    show_line_numbers: bool = True,
) -> None:
    """Render a code block with syntax highlighting.

    Uses Streamlit's built-in code display, with optional Pygments
    highlighting for richer output.

    Args:
        code: Source code string.
        language: Programming language for syntax highlighting.
        title: Optional title displayed above the code block.
        show_line_numbers: Whether to show line numbers.
    """
    if title:
        st.markdown(f"**{title}**")

    st.code(code.strip(), language=language, line_numbers=show_line_numbers)


def render_code_with_annotations(
    code: str,
    annotations: dict[int, str],
    language: str = "python",
    title: str = "",
) -> None:
    """Render code with line-level annotations displayed alongside.

    Args:
        code: Source code string.
        annotations: Mapping of line numbers (1-indexed) to annotation text.
        language: Programming language for highlighting.
        title: Optional title above the code block.
    """
    if title:
        st.markdown(f"**{title}**")

    st.code(code.strip(), language=language, line_numbers=True)

    if annotations:
        st.markdown("**Annotations:**")
        for line_num in sorted(annotations.keys()):
            st.markdown(f"- **Line {line_num}**: {annotations[line_num]}")


def extract_code_blocks(markdown_text: str) -> list[dict]:
    """Extract fenced code blocks from Markdown text.

    Args:
        markdown_text: Raw Markdown content.

    Returns:
        List of dicts with 'language' and 'code' keys.
    """
    blocks = []
    in_code = False
    current_lang = ""
    current_lines = []

    for line in markdown_text.split("\n"):
        stripped = line.strip()
        if stripped.startswith("```") and not in_code:
            in_code = True
            current_lang = stripped[3:].strip() or "text"
            current_lines = []
        elif stripped == "```" and in_code:
            in_code = False
            blocks.append({
                "language": current_lang,
                "code": "\n".join(current_lines),
            })
        elif in_code:
            current_lines.append(line)

    return blocks
