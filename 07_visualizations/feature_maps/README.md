---
title: "Feature Map Visualization Methodology"
date: 2025-01-15
status: planned
tags: [feature-maps, visualization, interpretability]
---

# Feature Map Visualization

## Purpose

Visualizing intermediate feature maps helps understand what different layers of a segmentation model learn. Early layers typically detect edges and textures, while deeper layers capture semantic concepts.

## Methodology

### 1. Hook-Based Feature Extraction

Register forward hooks on target layers to capture activations during inference:

```python
activations = {}

def get_activation(name):
    def hook(model, input, output):
        activations[name] = output.detach()
    return hook

# Register hooks
model.encoder.layer1.register_forward_hook(get_activation("encoder_stage1"))
model.encoder.layer2.register_forward_hook(get_activation("encoder_stage2"))
```

### 2. Visualization Techniques

| Technique | Description | Best For |
|-----------|-------------|----------|
| Individual channel display | Show each feature map channel as a grayscale image | Understanding what individual filters detect |
| Channel-wise mean | Average across all channels at a stage | Overall activation pattern at each scale |
| Top-k activated channels | Show the k channels with highest mean activation | Finding the most relevant features |
| PCA projection | Project high-dimensional feature maps to 3 channels (RGB) | Compact multi-channel visualization |
| Grad-CAM | Gradient-weighted class activation maps | Class-specific region importance |

### 3. Recommended Layers to Visualize

For a U-Net-style architecture:
- Encoder stage 1 output (high resolution, low-level features)
- Encoder stage 3 output (medium resolution, mid-level features)
- Bottleneck output (lowest resolution, highest-level features)
- Decoder stage 1 output (after first skip connection)
- Final layer before classification head

## Output Format

Feature map visualizations should be saved as:
- Individual PNG images per layer per input
- Grid montages showing multiple channels side by side
- Comparison grids across different models for the same input

## Tools

- **PyTorch hooks**: For feature extraction
- **matplotlib**: For rendering
- **torchvision.utils.make_grid**: For creating channel montages

## Planned Visualizations

TODO: Generate feature maps for the following model-input combinations:
- U-Net on sample segmentation images
- SegFormer attention maps
- SAM 2 image encoder features
