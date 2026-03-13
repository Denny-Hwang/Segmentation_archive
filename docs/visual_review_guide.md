# Visual Review Package — Image Segmentation Prior Art

This document describes the visual assets created for the Segmentation Archive
and how to use them in papers, presentations, or the Streamlit explorer.

## Assets

| Asset | Location | Source / Reproducibility |
|-------|----------|------------------------|
| Taxonomy diagram | `docs/figures/taxonomy_diagram.mermaid` | Mermaid source; render with `mmdc` or Streamlit |
| Timeline chart (matplotlib) | `docs/figures/timeline_evolution_chart.png` | `scripts/figures/generate_figures.py` |
| Timeline diagram (mermaid) | `docs/figures/timeline_evolution.mermaid` | Mermaid source |
| Model comparison chart | `docs/figures/model_comparison_chart.png` | `scripts/figures/generate_figures.py` |
| Model comparison table | `docs/figures/model_comparison_table.md` | Markdown table |
| Pipeline diagram | `docs/figures/pipeline_diagram.mermaid` | Mermaid source |
| Example images (synthetic) | `assets/examples/*.png` | `scripts/figures/generate_example_images.py` |

## Regenerating Figures

```bash
# Matplotlib-based figures (no special tools needed)
python scripts/figures/generate_figures.py

# Synthetic example images
python scripts/figures/generate_example_images.py

# Mermaid to PNG (requires mermaid-cli)
npm install -g @mermaid-js/mermaid-cli
mmdc -i docs/figures/taxonomy_diagram.mermaid -o docs/figures/taxonomy_diagram.png -w 2048
```

## Using in LaTeX

```latex
\begin{figure}[t]
  \centering
  \includegraphics[width=\linewidth]{figures/taxonomy_diagram.png}
  \caption{Taxonomy of image segmentation approaches. CNN-based methods
  (blue), medical/U-Net family (red), transformer-based (blue-light),
  instance segmentation (gold), universal/panoptic (purple), and
  foundation models (green).}
  \label{fig:taxonomy}
\end{figure}

\begin{figure}[t]
  \centering
  \includegraphics[width=\linewidth]{figures/timeline_evolution_chart.png}
  \caption{Evolution of image segmentation methods (2014--2024).}
  \label{fig:timeline}
\end{figure}

\begin{figure}[t]
  \centering
  \includegraphics[width=\linewidth]{figures/model_comparison_chart.png}
  \caption{Representative benchmark results across three datasets.
  Values from original papers; see Table~\ref{tab:comparison} for details.}
  \label{fig:comparison}
\end{figure>
```

## Upstream / Original Research

This repository is an **independent review and learning resource**.
There is no single upstream research repository. Individual models
and papers are cited within each review document. The bibliography
is maintained in `10_references/bibliography.bib`.

If a specific review document references a GitHub repository
(e.g., `facebookresearch/segment-anything`), that link is included
in the document's metadata or body text.
