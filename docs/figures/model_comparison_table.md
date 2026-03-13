<!-- Model Comparison Table — Representative benchmarks -->
<!-- Sources: original papers; values are approximate/reported; not all directly comparable -->

# Model Comparison Table

## Synapse Multi-Organ (CT, 8 organs)

| Model | Year | Backbone | mDSC (%) | mHD95 (mm) | Params (M) | Notes |
|-------|------|----------|----------|------------|------------|-------|
| U-Net | 2015 | Conv | 76.85 | 31.10 | ~31 | Baseline encoder-decoder |
| Attention U-Net | 2018 | Conv | 77.77 | 29.20 | ~34 | + attention gates |
| UNet++ | 2018 | Conv | 78.30 | 28.50 | ~36 | Nested dense skips |
| TransUNet | 2021 | R50+ViT | 77.48 | 31.69 | ~105 | CNN-Transformer hybrid |
| Swin-Unet | 2021 | Swin-T | 79.13 | 21.55 | ~27 | Pure transformer |
| nnU-Net | 2021 | Conv (auto) | 82.50 | 15.20 | ~31 | Self-configuring |
| HiFormer | 2023 | Swin+CNN | 80.39 | 20.77 | ~25 | Dual-branch |

## ADE20K (Scene Parsing, 150 classes)

| Model | Year | Backbone | mIoU (%) | Params (M) | Notes |
|-------|------|----------|----------|------------|-------|
| PSPNet | 2017 | R-101 | 44.39 | ~65 | Pyramid pooling |
| DeepLab v3+ | 2018 | R-101 | 45.47 | ~63 | ASPP + decoder |
| SegFormer-B5 | 2021 | MiT-B5 | 51.80 | ~84 | Efficient transformer |
| Mask2Former | 2022 | Swin-L | 56.01 | ~216 | Universal masked attn |
| OneFormer | 2023 | Swin-L | 57.40 | ~220 | Task-conditioned |

## Cityscapes val (Urban Scenes, 19 classes)

| Model | Year | Backbone | mIoU (%) | Notes |
|-------|------|----------|----------|-------|
| PSPNet | 2017 | R-101 | 79.70 | |
| DeepLab v3+ | 2018 | R-101 | 80.90 | |
| SegFormer-B5 | 2021 | MiT-B5 | 84.00 | |
| Mask2Former | 2022 | Swin-L | 83.30 | |
| OneFormer | 2023 | Swin-L | 84.40 | |

> **Note:** Benchmark values are taken from the respective original papers
> or well-known reproducibility studies. Direct comparison across rows should
> be done cautiously due to differences in training recipes, augmentation,
> and evaluation protocols.
