"""Reliable Mermaid diagram rendering for Streamlit.

Uses st.components.v1.html with Mermaid.js CDN for rendering.
This avoids dependency on streamlit-mermaid which may fail silently
in dark mode or certain Streamlit versions.

Also provides PNG download via mermaid.ink service.
"""

from __future__ import annotations

import base64
import hashlib
import urllib.parse

import streamlit as st
import streamlit.components.v1 as components


def _mermaid_html(code: str, height: int = 450, bg_color: str = "transparent") -> str:
    """Build an HTML page that renders a Mermaid diagram with dark theme support."""
    # Unique ID to prevent collisions when multiple diagrams on one page
    uid = "m" + hashlib.md5(code.encode()).hexdigest()[:10]
    return f"""
    <html>
    <head>
      <script src="https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.min.js"></script>
      <style>
        body {{
          margin: 0;
          padding: 0;
          background: {bg_color};
          display: flex;
          justify-content: center;
          overflow: auto;
        }}
        #{uid} {{
          width: 100%;
        }}
        /* Ensure text is visible on dark backgrounds */
        #{uid} .nodeLabel, #{uid} .edgeLabel,
        #{uid} text, #{uid} .label {{
          font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }}
      </style>
    </head>
    <body>
      <div class="mermaid" id="{uid}">
{code}
      </div>
      <script>
        mermaid.initialize({{
          startOnLoad: true,
          theme: 'dark',
          themeVariables: {{
            darkMode: true,
            background: '{bg_color}',
            primaryColor: '#4A90D9',
            primaryTextColor: '#fff',
            primaryBorderColor: '#5DADE2',
            lineColor: '#AEB6BF',
            secondaryColor: '#50C878',
            tertiaryColor: '#FF6B6B',
            fontSize: '14px',
            fontFamily: '"Segoe UI", Tahoma, Geneva, Verdana, sans-serif'
          }},
          flowchart: {{
            useMaxWidth: true,
            htmlLabels: true
          }},
          securityLevel: 'loose'
        }});
      </script>
    </body>
    </html>
    """


def render_mermaid(code: str, height: int = 450, key: str | None = None) -> None:
    """Render a Mermaid diagram using an embedded HTML component.

    Works reliably in both light and dark Streamlit themes.

    Args:
        code: Mermaid diagram source code.
        height: Component height in pixels.
        key: Optional unique key for the component.
    """
    html = _mermaid_html(code, height=height, bg_color="transparent")
    components.html(html, height=height, scrolling=True)


def render_mermaid_file(path, height: int = 450) -> None:
    """Render a .mermaid file.

    Args:
        path: Path object to the .mermaid file.
        height: Component height in pixels.
    """
    code = path.read_text(encoding="utf-8")
    render_mermaid(code, height=height)


def mermaid_png_download_button(
    code: str,
    filename: str = "diagram.png",
    label: str = "Download PNG",
    key: str | None = None,
) -> None:
    """Show a download button that links to a mermaid.ink PNG rendering.

    Uses the mermaid.ink public service to generate a PNG from diagram source.

    Args:
        code: Mermaid diagram source code.
        filename: Suggested download filename.
        label: Button label.
        key: Streamlit widget key.
    """
    # mermaid.ink accepts base64-encoded Mermaid code
    encoded = base64.urlsafe_b64encode(code.encode("utf-8")).decode("ascii")
    url = f"https://mermaid.ink/img/{encoded}?type=png&bgColor=!23262730"

    st.markdown(
        f'<a href="{url}" download="{filename}" target="_blank">'
        f'<button style="'
        f"background:#4A90D9;color:#fff;border:none;padding:8px 16px;"
        f"border-radius:6px;cursor:pointer;font-size:0.9em;"
        f'">{label}</button></a>',
        unsafe_allow_html=True,
    )
