# Dataset Download Guide

Step-by-step instructions for downloading commonly used segmentation datasets.

## General Tips

- Always check the license before using a dataset for your project
- Keep a local copy of the original data with checksums
- Store raw data separately from preprocessed versions
- Use a consistent directory structure across datasets

## Medical Datasets

### Synapse Multi-Organ

```bash
# 1. Register at synapse.org
# 2. Navigate to https://www.synapse.org/#!Synapse:syn3193805
# 3. Accept data use terms, then download via the Synapse client:
pip install synapseclient
synapse get syn3193805 -r
```

### ACDC

```bash
# 1. Register at the ACDC challenge website
# 2. Download from: https://www.creatis.insa-lyon.fr/Challenge/acdc/
# Data is provided in NIfTI (.nii.gz) format
```

### BraTS

```bash
# 1. Register for the BraTS challenge on Synapse
# 2. Download multi-modal MRI volumes
# Each case includes: T1, T1ce, T2, FLAIR + segmentation
```

### ISIC

```bash
# Direct download (ISIC 2018 Task 1):
# Visit https://challenge.isic-archive.com/data/
# Download training images + ground truth masks
```

### Kvasir-SEG

```bash
# Direct download (no registration required):
wget https://datasets.simula.no/downloads/kvasir-seg.zip
unzip kvasir-seg.zip
```

## Natural Image Datasets

### COCO

```bash
# Images
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip

# Annotations
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
wget http://images.cocodataset.org/annotations/panoptic_annotations_trainval2017.zip

# Python API
pip install pycocotools
```

### ADE20K

```bash
# Option 1: Direct download
# Visit https://groups.csail.mit.edu/vision/datasets/ADE20K/

# Option 2: HuggingFace Datasets
pip install datasets
python -c "from datasets import load_dataset; load_dataset('scene_parse_150')"

# Option 3: Via MMSegmentation
python tools/dataset_converters/ade20k.py /path/to/data
```

### Cityscapes

```bash
# 1. Register at https://www.cityscapes-dataset.com/
# 2. Download:
#    - gtFine_trainvaltest.zip (annotations, ~241MB)
#    - leftImg8bit_trainvaltest.zip (images, ~11GB)
```

### PASCAL VOC 2012

```bash
# Direct download
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
tar xf VOCtrainval_11-May-2012.tar

# For augmented set (SBD)
# Download from http://home.bharathh.info/pubs/codes/SBD/
```

## Recommended Directory Layout

```
data/
├── synapse/
│   ├── raw/
│   │   ├── imagesTr/
│   │   └── labelsTr/
│   └── processed/
├── acdc/
│   ├── raw/
│   └── processed/
├── coco/
│   ├── images/
│   │   ├── train2017/
│   │   └── val2017/
│   └── annotations/
├── ade20k/
│   ├── images/
│   └── annotations/
└── cityscapes/
    ├── leftImg8bit/
    └── gtFine/
```
