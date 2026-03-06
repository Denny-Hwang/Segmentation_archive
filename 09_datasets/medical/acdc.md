---
id: acdc
name: Automated Cardiac Diagnosis Challenge
domain: medical
modality: cardiac_mri
task: semantic
classes: 4
size: "100 patients"
license: CC BY-NC-SA 4.0
---

# ACDC - Automated Cardiac Diagnosis Challenge

## Overview

| Field | Details |
|---|---|
| **Name** | ACDC (Automated Cardiac Diagnosis Challenge) |
| **Source** | [CREATIS Lab](https://www.creatis.insa-lyon.fr/Challenge/acdc/) |
| **Size** | 100 patients (training), 50 patients (testing) |
| **Classes** | 4: background, right ventricle (RV), myocardium (MYO), left ventricle (LV) |
| **Modality** | Cine cardiac MRI |
| **Common Use** | Cardiac segmentation, medical transformer evaluation |

## Description

The ACDC dataset was collected from the University Hospital of Dijon and contains cine-MRI recordings from 100 patients, categorized into 5 evenly distributed pathological groups: normal, myocardial infarction, dilated cardiomyopathy, hypertrophic cardiomyopathy, and abnormal right ventricle.

Expert annotations are provided for end-diastolic (ED) and end-systolic (ES) phases. The dataset has become a standard benchmark alongside Synapse for medical image segmentation.

## Download Instructions

1. Visit the [ACDC challenge website](https://www.creatis.insa-lyon.fr/Challenge/acdc/)
2. Register and accept the data use agreement
3. Download the training and testing sets
4. Data is provided in NIfTI format

## Key Papers Using This Dataset

- **TransUNet** (Chen et al., 2021) - Transformer-CNN hybrid
- **Swin-Unet** (Cao et al., 2021) - Pure Swin Transformer encoder-decoder
- **nnU-Net** (Isensee et al., 2021) - Self-configuring segmentation
- **MT-UNet** (Wang et al., 2022) - Mixed transformer U-Net
