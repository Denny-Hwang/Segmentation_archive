"""Code Analysis Viewer - Explore implementation details and code patterns."""

import streamlit as st
from pathlib import Path

try:
    from pygments import highlight
    from pygments.lexers import PythonLexer
    from pygments.formatters import HtmlFormatter

    HAS_PYGMENTS = True
except ImportError:
    HAS_PYGMENTS = False

st.set_page_config(page_title="Code Analysis - Segmentation Archive", layout="wide")

ARCHIVE_ROOT = Path(__file__).resolve().parent.parent.parent
CODE_DIR = ARCHIVE_ROOT / "05_code_analysis"


def load_code_analyses() -> list[dict]:
    """Load code analysis Markdown files."""
    analyses = []
    if not CODE_DIR.exists():
        return analyses

    for md_file in sorted(CODE_DIR.rglob("*.md")):
        if md_file.name.startswith("_") or md_file.name == "README.md":
            continue
        try:
            content = md_file.read_text(encoding="utf-8")
            analyses.append({
                "name": md_file.stem.replace("_", " ").title(),
                "content": content,
                "file_path": str(md_file.relative_to(ARCHIVE_ROOT)),
                "parent": md_file.parent.name,
            })
        except Exception:
            continue

    return analyses


def render_code_block(code: str, language: str = "python") -> None:
    """Render a code block with syntax highlighting."""
    st.code(code, language=language)


def main():
    st.title("Code Analysis")
    st.markdown(
        "Deep-dive into segmentation model implementations. "
        "Explore code walkthroughs, key patterns, and implementation details."
    )

    analyses = load_code_analyses()

    if not analyses:
        st.info(
            "No code analyses found. Add Markdown files to "
            "`05_code_analysis/` to populate this page."
        )

        # Show placeholder content
        st.markdown("---")
        st.subheader("Common Code Patterns in Segmentation")

        with st.expander("Double Convolution Block (U-Net)", expanded=True):
            st.code(
                '''class DoubleConv(nn.Module):
    """Two consecutive convolution-BN-ReLU blocks."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)''',
                language="python",
            )

        with st.expander("Skip Connection Pattern", expanded=True):
            st.code(
                '''# Encoder path stores features at each scale
encoder_features = []
for encoder_block in self.encoders:
    x = encoder_block(x)
    encoder_features.append(x)
    x = self.pool(x)

# Decoder path concatenates encoder features via skip connections
for i, decoder_block in enumerate(self.decoders):
    x = self.upconv[i](x)
    skip = encoder_features[-(i + 1)]
    x = torch.cat([x, skip], dim=1)
    x = decoder_block(x)''',
                language="python",
            )

        return

    # Navigation
    selected_analysis = st.sidebar.selectbox(
        "Select Analysis",
        [a["name"] for a in analyses],
    )

    analysis = next(a for a in analyses if a["name"] == selected_analysis)

    st.markdown("---")
    st.markdown(analysis["content"])
    st.caption(f"Source: `{analysis['file_path']}`")


if __name__ == "__main__":
    main()
