---
title: "nnU-Net - Preprocessing Pipeline"
date: 2025-01-15
status: complete
parent: "nnunet/repo_overview.md"
tags: [nnunet, preprocessing, resampling, normalization]
---

# nnU-Net Preprocessing Pipeline

## Pipeline Overview

```
Raw Data (any format)
    |
    +-- Dataset conversion to nnU-Net format
    +-- Fingerprint extraction
    +-- Resampling to target spacing
    +-- Intensity normalization
    +-- Cropping to non-zero region
    |
    +-- Preprocessed .npy / .npz files
```

## Resampling

### Spacing Determination

The target spacing for resampling is determined by the `ExperimentPlanner` based on the median voxel spacing across all training cases. The planner computes the median independently for each axis, so the target spacing can be anisotropic if the dataset is anisotropic. For example, if most cases have spacing `[3.0, 0.7, 0.7]` (thick axial slices), the target spacing will preserve the anisotropy rather than forcing isotropic resampling, which would either waste memory (resampling the thick axis to 0.7mm) or lose resolution (resampling the thin axes to 3.0mm).

For the `3d_lowres` configuration, the target spacing is increased (coarsened) from the full-resolution spacing to ensure the entire image can fit within a reasonable patch size. The lowres spacing is determined by scaling the fullres spacing until the median resampled image size is small enough to be covered by a feasible patch. This means the lowres configuration may use spacing like `[2.0, 2.0, 2.0]` even when the full-resolution target is `[1.0, 0.7, 0.7]`.

The `2d` configuration uses the in-plane spacing (the two highest-resolution axes) as the target, ignoring the through-plane axis since each slice is processed independently. This is particularly beneficial for datasets with highly anisotropic voxels where through-plane information is unreliable.

### Interpolation Methods

nnU-Net uses different interpolation methods for images and labels to preserve data integrity:

```python
# For image data: third-order spline (high quality, smooth)
resampled_image = resample_data_or_seg_to_shape(
    data, new_shape, is_seg=False,
    order=3, order_z=0  # order_z=0 for anisotropic axis
)

# For segmentation labels: nearest-neighbor (preserves discrete values)
resampled_seg = resample_data_or_seg_to_shape(
    seg, new_shape, is_seg=True,
    order=0, order_z=0  # Always nearest for labels
)
```

For anisotropic data, the resampling function treats the low-resolution axis separately (`order_z=0` uses nearest-neighbor interpolation along the thick-slice axis) to avoid introducing interpolation artifacts in the direction where the original sampling was coarse. The high-resolution in-plane axes use cubic spline interpolation for images, which produces smooth results without the blurring of linear interpolation or the ringing of higher-order methods. Labels always use nearest-neighbor interpolation because any other method would create invalid fractional class values at boundaries.

## Normalization

### Modality-Specific Normalization

nnU-Net applies different normalization strategies depending on the imaging modality, as specified in `dataset.json`:

**CT (Computed Tomography)**: Global clipping followed by z-score normalization. CT has a fixed intensity scale (Hounsfield units), so statistics can be computed globally across the entire dataset:

```python
# CT normalization (CTNormalization class):
# 1. Clip to [percentile_00_5, percentile_99_5] of foreground intensities
lower_bound = dataset_fingerprint['foreground_intensity_properties']['percentile_00_5']
upper_bound = dataset_fingerprint['foreground_intensity_properties']['percentile_99_5']
image = np.clip(image, lower_bound, upper_bound)
# 2. Z-score normalize using global foreground mean and std
image = (image - global_mean) / global_std
```

The clipping to 0.5th and 99.5th percentiles removes extreme outliers (e.g., metal artifacts producing very high HU values). The global mean and standard deviation are computed only on foreground voxels (non-zero region) to avoid the large contribution of background air voxels.

**MRI (Magnetic Resonance Imaging)**: Per-image z-score normalization. MRI intensity values are arbitrary and vary between scanners, sequences, and even patients, so global statistics are meaningless:

```python
# MRI normalization (ZScoreNormalization class):
# Per-image, per-channel z-score on foreground voxels only
foreground_mask = image > 0  # or from segmentation
mean = image[foreground_mask].mean()
std = image[foreground_mask].std()
image = (image - mean) / (std + 1e-8)
```

**Other modalities** (RGB images, microscopy, etc.): Default `ZScoreNormalization` is used with per-image statistics, similar to MRI handling. The normalization scheme for each channel is stored in the plans file and can be overridden.

### Normalization Parameters

For CT, the global normalization parameters (mean, std, clip bounds) are stored in the dataset fingerprint file (`dataset_fingerprint.json`) under `foreground_intensity_properties_per_channel`. These are computed once during fingerprint extraction and reused for all preprocessing and inference. For MRI and other modalities, parameters are computed on-the-fly for each image, so no global parameters are stored.

The plans file specifies which normalization scheme to use for each input channel via the `normalization_schemes` list, e.g., `["CTNormalization"]` for CT or `["ZScoreNormalization", "ZScoreNormalization"]` for multi-channel MRI.

## Cropping

Non-zero region cropping is performed as the first preprocessing step by the `crop_to_nonzero()` function. It identifies the bounding box of all non-zero voxels across all channels and crops the image and segmentation to this bounding box, discarding uninformative background:

```python
def crop_to_nonzero(data, seg=None):
    """Crop image and segmentation to the bounding box of non-zero voxels."""
    nonzero_mask = data.sum(axis=0) > 0  # Union across channels
    # Find bounding box
    coords = np.argwhere(nonzero_mask)
    bbox_min = coords.min(axis=0)
    bbox_max = coords.max(axis=0) + 1
    # Crop
    slicer = tuple(slice(mn, mx) for mn, mx in zip(bbox_min, bbox_max))
    cropped_data = data[(slice(None),) + slicer]
    # Store bbox for later restoration
    return cropped_data, slicer, original_shape
```

This step can significantly reduce image size for modalities with large backgrounds (e.g., abdominal CT where the body occupies only a fraction of the full scan volume). The cropping bounding box is stored alongside the preprocessed data so that predictions can be placed back into the original coordinate system during inference.

## Data Format

### Input Format

nnU-Net expects data organized in a specific directory structure under `nnUNet_raw/`:

```
nnUNet_raw/
  Dataset001_BrainTumour/
    dataset.json          # Metadata: modalities, labels, num_training, file_endings
    imagesTr/
      BrainTumour_001_0000.nii.gz   # Case 001, channel 0 (e.g., T1)
      BrainTumour_001_0001.nii.gz   # Case 001, channel 1 (e.g., T2)
      BrainTumour_002_0000.nii.gz   # Case 002, channel 0
    labelsTr/
      BrainTumour_001.nii.gz        # Segmentation for case 001
      BrainTumour_002.nii.gz        # Segmentation for case 002
    imagesTs/                        # Optional test images
```

The `_XXXX` suffix on image files indicates the channel index (zero-padded to 4 digits). The `dataset.json` file must specify `channel_names`, `labels`, `numTraining`, and `file_ending`. Supported file formats include NIfTI (`.nii.gz`), NRRD, MHA, and other formats readable by SimpleITK.

### Output Format

Preprocessed data is stored under `nnUNet_preprocessed/DatasetXXX/` as compressed NumPy arrays:

```
nnUNet_preprocessed/
  Dataset001_BrainTumour/
    nnUNetPlans.json           # The experiment plan
    dataset_fingerprint.json   # Cached fingerprint
    nnUNetPlans_2d/            # Preprocessed for 2D config
      BrainTumour_001.npz      # Contains 'data' and 'seg' arrays
      BrainTumour_001.pkl      # Properties: original spacing, shape, crop bbox
    nnUNetPlans_3d_fullres/    # Preprocessed for 3D fullres config
      ...
    nnUNetPlans_3d_lowres/     # Only if lowres was planned
      ...
```

Each `.npz` file contains the preprocessed image data (float32, shape `[C, D, H, W]`) and segmentation (integer, same spatial shape). The accompanying `.pkl` file stores case-specific properties needed for inference (original spacing, shape, cropping bounding box, normalization parameters for MRI).

## Custom Preprocessing

Users can customize preprocessing by subclassing the default preprocessor or by modifying the plans file. The main extension points are:

```python
from nnunetv2.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor

class MyPreprocessor(DefaultPreprocessor):
    def run_case_npy(self, data, seg, properties):
        # Call parent for standard preprocessing
        data, seg, properties = super().run_case_npy(data, seg, properties)
        # Add custom steps (e.g., histogram equalization)
        data = custom_histogram_equalization(data)
        return data, seg, properties
```

Common customizations include adding custom intensity normalization (e.g., histogram matching for histopathology), applying domain-specific preprocessing (e.g., skull stripping for brain MRI), or modifying the resampling target spacing. The plans file can also be edited directly to change normalization schemes without subclassing, by replacing `"ZScoreNormalization"` with a custom class name that is registered in nnU-Net's normalization scheme registry.
