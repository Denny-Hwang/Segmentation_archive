# 09 Datasets

Reference documentation for key datasets used in image segmentation research and benchmarking.

## Dataset Classification

| Dataset | Domain | Task | Modality | Classes | Size |
|---|---|---|---|---|---|
| Synapse Multi-Organ | Medical | Semantic | CT | 13 organs | 30 cases |
| ACDC | Medical | Semantic | Cardiac MRI | 4 | 100 patients |
| BraTS | Medical | Semantic | Brain MRI | 4 | 2000+ scans |
| ISIC | Medical | Binary/Semantic | Dermoscopy | 2-8 | 25,000+ images |
| Kvasir-SEG | Medical | Binary | Endoscopy | 2 | 1,000 images |
| LiTS | Medical | Semantic | Liver CT | 3 | 201 scans |
| COCO | Natural | Instance/Panoptic | RGB | 80/133 | 330K images |
| ADE20K | Natural | Semantic | RGB | 150 | 25K images |
| Cityscapes | Natural | Semantic/Instance | RGB | 30 | 5K fine + 20K coarse |
| PASCAL VOC | Natural | Semantic/Instance | RGB | 21 | 11,530 images |
| Mapillary Vistas | Natural | Semantic/Panoptic | RGB | 66/37 | 25K images |
| SA-1B | Specialized | Promptable | RGB | N/A | 11M images, 1.1B masks |
| SA-V | Specialized | Video | Video | N/A | 50.9K videos |
| Remote Sensing | Specialized | Semantic | Satellite | Varies | Varies |

## Directory Structure

```
09_datasets/
├── README.md                      # This file
├── _registry.yaml                 # Machine-readable dataset metadata
├── medical/                       # Medical imaging datasets
│   ├── synapse_multi_organ.md
│   ├── acdc.md
│   ├── brats.md
│   ├── isic.md
│   ├── kvasir_seg.md
│   └── lits.md
├── natural/                       # Natural image datasets
│   ├── coco.md
│   ├── ade20k.md
│   ├── cityscapes.md
│   ├── pascal_voc.md
│   └── mapillary_vistas.md
├── specialized/                   # Specialized / foundation model datasets
│   ├── sa1b.md
│   ├── sav.md
│   └── remote_sensing.md
└── _how_to/                       # Practical guides
    ├── download_guide.md
    ├── preprocessing_recipes.md
    └── custom_dataset_guide.md
```

## How to Use

- Browse individual dataset cards for details, download links, and key papers
- See `_how_to/download_guide.md` for step-by-step download instructions
- See `_how_to/preprocessing_recipes.md` for common preprocessing pipelines
- See `_how_to/custom_dataset_guide.md` for creating your own segmentation dataset
