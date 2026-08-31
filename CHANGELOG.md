# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- Playground: **Hugging Face Inference API backend** — segment images on the
  Hub instead of locally, so no `torch`/`torchvision` and no model weights are
  needed (token from the sidebar, `HF_TOKEN`, or `.streamlit/secrets.toml`)

### Fixed

- Playground: SegFormer failed to load with *"Could not load any image
  processor class … Missing optional dependencies: torchvision"*. From
  transformers 5.x both SegFormer image processors require torchvision, which
  was missing from `explorer/requirements.txt`; it is now pinned, and load
  failures explain the fix instead of degrading into "model unavailable"

- Initial repository scaffolding with full directory structure
- README files for all sections
- Paper review and code analysis templates
- Registry YAML schemas for Streamlit explorer indexing
- GitHub issue templates (paper review, code analysis, bug report)
- CI workflow configurations (lint, Streamlit deploy)
- Streamlit explorer basic app structure
- Tracked repositories and papers registry (`08_repo_tracker/`)
- Benchmark dataset guides structure (`09_datasets/`)
- Reference materials structure (`10_references/`)
