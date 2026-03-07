---
title: "MMSegmentation - Config System Analysis"
date: 2025-01-15
status: complete
parent: "mmsegmentation/repo_overview.md"
tags: [mmsegmentation, config, inheritance, mmengine]
---

# MMSegmentation Config System

## Overview

MMSegmentation uses MMEngine's hierarchical configuration system to define all aspects of a training experiment: model architecture, dataset pipeline, training schedule, and runtime settings. Configurations are written as Python files (`.py`), which offers more flexibility than YAML or JSON because they support arbitrary Python expressions, function calls, and conditional logic. The `Config` class from MMEngine parses these files into nested dictionaries that are used throughout the framework to instantiate objects via the registry system.

The config system's most powerful feature is **inheritance**: a config file can declare one or more `_base_` configs to inherit from, then override or extend specific fields. This enables a modular, DRY (Don't Repeat Yourself) approach where base configs define common patterns (e.g., a standard ResNet backbone, a standard 80K iteration schedule) and experiment configs only specify the differences. A typical experiment config might be only 10-20 lines long while fully specifying a complex training pipeline by inheriting from 4 base configs.

## Config Inheritance

### Base Configs

Base configs in MMSegmentation are organized into four categories, stored in the `configs/_base_/` directory:

```
configs/_base_/
    models/          # Architecture definitions (backbone + head)
    datasets/        # Dataset and dataloader configurations
    schedules/       # Optimizer, scheduler, iteration count
    default_runtime.py  # Logging, checkpointing, hooks
```

Each base config defines one aspect of the training pipeline. For example, `configs/_base_/models/fcn_r50-d8.py` defines an FCN model with a ResNet-50 backbone:

```python
# configs/_base_/models/fcn_r50-d8.py
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    backbone=dict(
        type='ResNetV1c',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 2, 4),
        strides=(1, 2, 1, 1),
        norm_cfg=norm_cfg,
        norm_eval=False,
        style='pytorch',
        contract_dilation=True),
    decode_head=dict(
        type='FCNHead',
        in_channels=2048,
        in_index=3,
        channels=512,
        num_convs=2,
        concat_input=True,
        num_classes=19,
        norm_cfg=norm_cfg,
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
)
```

### Config Merging

A child config inherits from base configs using the `_base_` list and overrides specific values. MMEngine's merging follows a recursive dict update strategy: the child's values replace the parent's values at matching keys, and new keys are added:

```python
# configs/fcn/fcn_r101-d8_4xb4-80k_cityscapes-512x1024.py
_base_ = [
    '../_base_/models/fcn_r50-d8.py',
    '../_base_/datasets/cityscapes.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_80k.py'
]

# Override: change backbone from ResNet-50 to ResNet-101
model = dict(
    backbone=dict(depth=101),  # Only override depth; all other fields inherited
    decode_head=dict(num_classes=19),
)
```

The merging is recursive, meaning `model.backbone.depth=101` only overrides the `depth` field while preserving all other backbone settings (norm_cfg, dilations, strides, etc.) from the base config. This is implemented by `Config._merge_a_into_b()` in MMEngine.

### Variable References

MMEngine supports cross-references within configs using the `{{_base_.variable}}` syntax, allowing one config to reference values defined in another:

```python
# Reference a variable from a base config
_base_ = ['../_base_/models/segformer_mit-b0.py']

# Use a value defined in the base
model = dict(
    decode_head=dict(
        num_classes={{_base_.model.decode_head.num_classes}}
    )
)
```

In practice, variable references are less commonly used than simple dict overrides. The more common pattern is to define shared variables (like `norm_cfg` or `num_classes`) as top-level variables in the config and reference them locally within the same file.

## Config Structure

### Model Config

The model configuration specifies the full segmentation architecture as a nested dictionary. The top-level `type` is typically `'EncoderDecoder'`, which composes a backbone, optional neck, decode head, and optional auxiliary head:

```python
model = dict(
    type='EncoderDecoder',
    pretrained='open-mmlab://resnet50_v1c',
    backbone=dict(type='ResNetV1c', depth=50, ...),
    decode_head=dict(
        type='PSPHead',
        in_channels=2048,
        in_index=3,
        channels=512,
        pool_scales=(1, 2, 3, 6),
        num_classes=150,
        loss_decode=dict(type='CrossEntropyLoss', loss_weight=1.0),
    ),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=1024,
        in_index=2,
        channels=256,
        num_classes=150,
        loss_decode=dict(type='CrossEntropyLoss', loss_weight=0.4),
    ),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'),
)
```

The `in_index` parameter controls which backbone output stage feeds into each head. The `auxiliary_head` provides deep supervision during training (gradient from an earlier stage) and is typically disabled during inference.

### Dataset Config

Dataset configuration defines the data pipeline including loading, transforms, and batching:

```python
dataset_type = 'CityscapesDataset'
data_root = 'data/cityscapes/'

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='RandomResize', scale=(2048, 1024), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=(512, 1024), cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs'),
]

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='leftImg8bit/train', seg_map_path='gtFine/train'),
        pipeline=train_pipeline,
    )
)
```

The pipeline is a list of transform dictionaries, each specifying a `type` that maps to a registered transform class. Transforms are applied sequentially, and each transform receives and returns a data dictionary containing the image, annotations, and metadata.

### Schedule Config

The schedule config defines the optimizer, learning rate scheduler, and total training iterations:

```python
# configs/_base_/schedules/schedule_80k.py
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer, clip_grad=None)
param_scheduler = [
    dict(type='PolyLR', eta_min=1e-4, power=0.9, begin=0, end=80000, by_epoch=False),
]
train_cfg = dict(type='IterBasedTrainLoop', max_iters=80000, val_interval=8000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
```

MMSegmentation uses iteration-based training (not epoch-based) by default, which is standard practice for segmentation where datasets can be very large. The `PolyLR` scheduler (polynomial decay) is the most commonly used scheduler in segmentation, gradually reducing the learning rate following `lr = base_lr * (1 - iter/max_iter)^power`.

### Runtime Config

The runtime config controls logging, checkpointing, environment settings, and hooks:

```python
# configs/_base_/default_runtime.py
default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)
log_level = 'INFO'
log_processor = dict(by_epoch=False)
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=8000),
    sampler_seed=dict(type='DistSamplerSeedHook'),
)
```

## Config-to-Code Resolution

The journey from a config file to a running model follows this path:

```python
# 1. Parse config file (resolves inheritance, merges dicts)
from mmengine.config import Config
cfg = Config.fromfile('configs/pspnet/pspnet_r50-d8_4xb4-80k_cityscapes-512x1024.py')

# 2. Build model from config dict (uses MODELS registry)
from mmseg.registry import MODELS
model = MODELS.build(cfg.model)
# Internally: EncoderDecoder(backbone=ResNetV1c(...), decode_head=PSPHead(...), ...)

# 3. Build dataset from config (uses DATASETS registry)
from mmseg.registry import DATASETS
dataset = DATASETS.build(cfg.train_dataloader.dataset)

# 4. Build runner (orchestrates training)
from mmengine.runner import Runner
runner = Runner.from_cfg(cfg)
runner.train()
```

The `MODELS.build(cfg)` call recursively resolves nested `type` fields: it looks up `'EncoderDecoder'` in the MODELS registry, instantiates it, and passes the remaining dict fields as keyword arguments. Each nested component (backbone, decode_head) is similarly resolved via its own `type` field.

## Custom Config Tips

When creating experiment configs, follow these patterns for maintainability:

```python
# Pattern 1: Minimal override config
_base_ = './pspnet_r50-d8_4xb4-80k_cityscapes-512x1024.py'
model = dict(backbone=dict(depth=101))  # Only change backbone depth

# Pattern 2: Multi-dataset with shared model
_base_ = ['../_base_/models/pspnet_r50-d8.py', '../_base_/default_runtime.py']
# Define dataset and schedule inline for custom experiments

# Pattern 3: Hyperparameter sweep
_base_ = './baseline.py'
optim_wrapper = dict(optimizer=dict(lr=0.005))  # Override learning rate only

# Pattern 4: Delete inherited fields with _delete_=True
model = dict(
    backbone=dict(
        _delete_=True,
        type='SwinTransformer',
        embed_dims=96,
        depths=[2, 2, 6, 2],
    )
)
```

The `_delete_=True` pattern is essential when switching between fundamentally different architectures (e.g., CNN to Transformer), since recursive merging would otherwise keep stale fields from the parent config that are invalid for the new architecture. Always validate configs with `python tools/print_config.py <config_path>` to see the fully resolved configuration before training.
