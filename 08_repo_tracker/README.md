# 08 Repository & Paper Tracker

A system for tracking key open-source repositories and influential papers in the image segmentation field.

## Purpose

- **Repository Tracking**: Monitor GitHub repositories for stars, forks, releases, and activity
- **Paper Tracking**: Track citation counts and impact of key segmentation papers
- **Trend Analysis**: Identify emerging trends from repository and paper metrics

## Structure

```
08_repo_tracker/
├── README.md                  # This file
├── tracked_repos.yaml         # Registry of tracked GitHub repositories
├── tracked_papers.yaml        # Registry of tracked academic papers
├── update_log.md              # Log of tracking updates
└── scripts/
    ├── fetch_repo_stats.py    # Fetch GitHub repository statistics
    ├── fetch_paper_citations.py # Fetch citation counts via Semantic Scholar
    ├── generate_report.py     # Generate monthly tracking reports
    └── check_new_releases.py  # Check for new repository releases
```

## Tracked Repositories

| Repository | Category | Description |
|---|---|---|
| milesial/Pytorch-UNet | Medical/General | Lightweight UNet implementation |
| MIC-DKFZ/nnUNet | Medical | Self-configuring segmentation framework |
| facebookresearch/sam2 | Foundation | Segment Anything Model 2 |
| open-mmlab/mmsegmentation | Framework | OpenMMLab semantic segmentation toolbox |
| qubvel-org/segmentation_models.pytorch | Framework | SMP with pretrained encoders |
| facebookresearch/Mask2Former | Panoptic | Universal image segmentation |
| SHI-Labs/OneFormer | Panoptic | One transformer for all segmentation tasks |
| yingkaisha/keras-unet-collection | Medical/General | Keras UNet variant collection |
| wolny/pytorch-3dunet | Medical/3D | 3D UNet for volumetric segmentation |

## Usage

### Fetch latest stats

```bash
python scripts/fetch_repo_stats.py
python scripts/fetch_paper_citations.py
```

### Generate monthly report

```bash
python scripts/generate_report.py --month 2026-03
```

### Check for new releases

```bash
python scripts/check_new_releases.py
```

## Configuration

Scripts require a GitHub personal access token for API rate limits.
Set the `GITHUB_TOKEN` environment variable before running:

```bash
export GITHUB_TOKEN="ghp_your_token_here"
```
