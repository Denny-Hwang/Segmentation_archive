---
title: "nnU-Net - Experiment Planner Analysis"
date: 2025-01-15
status: planned
parent: "nnunet/repo_overview.md"
tags: [nnunet, experiment-planning, auto-configuration]
---

# nnU-Net Experiment Planner

## Overview

The `ExperimentPlanner` (in `nnunetv2/experiment_planning/experiment_planners/default_experiment_planner.py`) is the core of nnU-Net's self-configuring philosophy. Given only a dataset, it automatically determines: patch size, batch size, network topology (depth, channels, kernel sizes), normalization scheme, resampling strategy, and training schedule. The planner analyzes a **dataset fingerprint** and applies heuristic rules to produce a `plans.json` file that fully specifies the experiment.

The entry point is `nnUNetv2_plan_and_preprocess`, which runs:
1. `DatasetFingerprintExtractor` -- analyzes the raw dataset
2. `ExperimentPlanner.plan_experiment()` -- generates plans from the fingerprint
3. Preprocessing -- applies the plan to produce training-ready data

## Dataset Fingerprint

### Fingerprint Extraction

The `DatasetFingerprintExtractor` (in `experiment_planning/dataset_fingerprint/fingerprint_extractor.py`) computes:

- **Spacings**: Per-image voxel spacing (from NIfTI/DICOM headers)
- **Image sizes**: Spatial dimensions of each image (after optional cropping)
- **Intensity statistics**: Per-modality mean, std, min, max, percentiles (0.5th, 99.5th) computed on foreground voxels only
- **Class frequencies**: Number of voxels per class across the dataset
- **Number of cases**: Total training samples
- **Modality information**: From `dataset.json` -- CT, MRI, etc.

Statistics are computed on a subset of cases (all by default) and cached to avoid recomputation.

### Fingerprint Contents

The fingerprint is a dictionary stored as `dataset_fingerprint.json`:

```python
{
    "spacings": [[1.0, 0.7, 0.7], [1.0, 0.8, 0.8], ...],  # per-case spacings
    "shapes_after_crop": [[128, 256, 256], ...],
    "foreground_intensity_properties_per_channel": {
        "0": {"mean": 45.2, "std": 30.1, "percentile_00_5": -200, "percentile_99_5": 350, ...}
    },
    "num_training_cases": 50,
    "dataset_properties": { ... }
}
```

## Planning Algorithm

### Patch Size Selection

The planner iterates to find the largest patch that fits in GPU memory:

1. **Start with the median image shape** (after resampling to target spacing)
2. **Reduce dimensions iteratively**: If the patch exceeds the memory budget, reduce the largest dimension by a factor (rounding down to a valid size)
3. **Minimum patch size**: At least 2x the downsampling factor per axis (e.g., if 5 pooling stages, minimum is 32 per axis)
4. **Anisotropy handling**: For anisotropic data (e.g., thick CT slices), the patch size in the anisotropic axis is smaller, and pooling is delayed in that axis

```python
# Simplified logic:
patch_size = median_shape_after_resampling
while estimate_vram(patch_size, network_topology) > gpu_memory_budget:
    largest_axis = argmax(patch_size)
    patch_size[largest_axis] //= 2
```

### Batch Size Selection

Batch size is determined by the remaining GPU memory after accounting for the model and one sample:

```python
# Approximate logic:
reference_vram = 8_000_000_000  # 8 GB reference GPU
estimated_per_sample = estimate_memory(patch_size, network_topology)
batch_size = max(2, reference_vram // estimated_per_sample)
```

- Minimum batch size is 2 (for BatchNorm/InstanceNorm to function)
- The planner targets a reference GPU with ~8-10 GB VRAM
- Larger GPUs do not automatically get larger batches (by design, for reproducibility)

### Network Topology

The topology is derived from the patch size:

1. **Number of stages** = `floor(log2(min(patch_size)))` -- pooling until the smallest dimension reaches ~4-8
2. **Channels per stage**: Start at 32, double each stage, cap at 320 (for 3D) or 512 (for 2D)
3. **Kernel sizes per stage**: 3x3x3 for isotropic axes, 1xNxN if one axis is too small for pooling
4. **Pool sizes**: 2x2x2 by default, but axes with dimension < min_threshold use pool size 1

```python
# Channel progression (3D, default):
# Stage 0: 32
# Stage 1: 64
# Stage 2: 128
# Stage 3: 256
# Stage 4: 320 (capped)
# Stage 5: 320 (capped)
```

### Configuration Cascade

nnU-Net plans up to four configurations:

| Configuration | Description | When Used |
|--------------|-------------|-----------|
| **2D** | 2D U-Net on individual slices | Always planned for 3D data |
| **3D_fullres** | 3D U-Net at original spacing | Always planned |
| **3D_lowres** | 3D U-Net at reduced resolution | Planned if median image shape > 4x the patch size |
| **3D_cascade** | 3D_lowres -> 3D_fullres refinement | Planned alongside 3D_lowres |

The cascade first predicts at low resolution, then uses the low-res prediction as an additional input channel to a full-resolution network for refinement.

## Plans File Format

The output `nnUNetPlans.json` contains:

```python
{
    "dataset_name": "Dataset001_BrainTumour",
    "plans_name": "nnUNetPlans",
    "original_median_spacing_after_transp": [1.0, 1.0, 1.0],
    "original_median_shape_after_transp": [140, 170, 140],
    "configurations": {
        "2d": {
            "data_identifier": "nnUNetPlans_2d",
            "patch_size": [256, 256],
            "batch_size": 12,
            "UNet_class_name": "PlainConvUNet",
            "n_conv_per_stage_encoder": [2, 2, 2, 2, 2, 2],
            "n_conv_per_stage_decoder": [2, 2, 2, 2, 2],
            "conv_kernel_sizes": [[3,3], [3,3], [3,3], [3,3], [3,3], [3,3]],
            "pool_op_kernel_sizes": [[1,1], [2,2], [2,2], [2,2], [2,2], [2,2]],
            "UNet_base_num_features": 32,
            "unet_max_num_features": 512,
            "resampling_fn_data": "resample_data_or_seg_to_shape",
            "normalization_schemes": ["ZScoreNormalization"],
            "spacing": [1.0, 1.0],
        },
        "3d_fullres": {
            "patch_size": [128, 128, 128],
            "batch_size": 2,
            "unet_max_num_features": 320,
            # ... similar structure
        },
        # "3d_lowres" and "3d_cascade_fullres" if applicable
    }
}
```

## Customizing the Planner

Override the planner by subclassing `ExperimentPlanner`:

```python
from nnunetv2.experiment_planning.experiment_planners.default_experiment_planner import ExperimentPlanner

class MyCustomPlanner(ExperimentPlanner):
    def __init__(self, dataset_name_or_id, gpu_memory_target_in_gb=24):
        super().__init__(dataset_name_or_id, gpu_memory_target_in_gb)

    def determine_fullres_target_spacing(self, *args):
        return [0.5, 0.5, 0.5]  # Force isotropic 0.5mm spacing

    def get_plans_for_configuration(self, *args):
        plans = super().get_plans_for_configuration(*args)
        plans['batch_size'] = 4  # Override batch size
        return plans
```

Run with: `nnUNetv2_plan_and_preprocess -d DATASET_ID -pl MyCustomPlanner`

Key overridable methods:
- `determine_fullres_target_spacing()` -- target resampling spacing
- `determine_normalization_scheme()` -- per-channel normalization
- `get_plans_for_configuration()` -- full plan for one configuration
- `determine_transpose()` -- axis ordering

## Key Heuristics

1. **Target spacing = median spacing** of the training dataset (robust to outliers)
2. **Anisotropy threshold**: If any spacing axis is > 3x the smallest, the data is treated as anisotropic; that axis gets delayed pooling and smaller patch extent
3. **Memory budget**: Targets ~8 GB VRAM, making plans reproducible across different GPUs (actual training can use gradient accumulation on smaller GPUs)
4. **Max features cap**: 320 for 3D (memory), 512 for 2D (less memory pressure)
5. **Base features = 32**: Starting channel count, doubling at each stage
6. **2 convolutions per stage** in encoder and decoder (configurable)
7. **Patch overlap at inference**: 50% overlap with Gaussian-weighted stitching for smooth predictions
8. **Cascade threshold**: 3D_lowres is planned when the median image volume exceeds 4x the maximum feasible patch volume, indicating that full-resolution patches cannot cover enough context
