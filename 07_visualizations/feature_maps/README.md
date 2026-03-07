---
title: "Feature Map Visualization Methodology"
date: 2025-01-15
status: planned
tags: [feature-maps, visualization, interpretability]
---

# Feature Map Visualization

## Purpose

Visualizing intermediate feature maps helps understand what different layers of a segmentation
model learn. Early layers typically detect edges and textures, while deeper layers capture
semantic concepts. For segmentation models specifically, feature map visualization reveals how
spatial information is preserved (or lost) through the encoder, and how it is recovered in the
decoder via skip connections and upsampling.

Understanding feature maps is valuable for:

- **Debugging models** that produce unexpected segmentation outputs.
- **Comparing architectures** to see how U-Net, SegFormer, and other models represent the
  same input at different stages.
- **Building intuition** about what neural networks learn, which is essential for designing
  better architectures.
- **Identifying failure modes**, such as when a model fails to activate on a target region.

---

## Methodology

### 1. Hook-Based Feature Extraction

PyTorch's forward hooks allow you to intercept and store intermediate activations without
modifying the model code. This is the standard technique for feature map extraction.

```python
import torch
import torch.nn as nn

class FeatureExtractor:
    """Extract intermediate feature maps from a model using forward hooks."""

    def __init__(self, model, target_layers):
        """
        Args:
            model: The PyTorch model.
            target_layers: List of strings specifying layer names to hook.
                Example: ["encoder.layer1", "encoder.layer3", "decoder.block1"]
        """
        self.model = model
        self.activations = {}
        self.hooks = []

        for name in target_layers:
            layer = self._get_layer_by_name(model, name)
            hook = layer.register_forward_hook(self._make_hook(name))
            self.hooks.append(hook)

    def _get_layer_by_name(self, model, name):
        """Navigate nested modules using dot-separated names."""
        parts = name.split(".")
        module = model
        for part in parts:
            module = getattr(module, part)
        return module

    def _make_hook(self, name):
        def hook_fn(module, input, output):
            # Handle cases where output is a tuple (e.g., transformers)
            if isinstance(output, tuple):
                output = output[0]
            self.activations[name] = output.detach().cpu()
        return hook_fn

    def extract(self, input_tensor):
        """Run a forward pass and return the captured activations."""
        self.activations = {}
        with torch.no_grad():
            _ = self.model(input_tensor)
        return self.activations

    def remove_hooks(self):
        """Clean up hooks to avoid memory leaks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
```

**Usage example:**

```python
import segmentation_models_pytorch as smp

model = smp.Unet("resnet34", encoder_weights="imagenet", classes=1)
model.eval()

extractor = FeatureExtractor(model, [
    "encoder.layer1",    # 64 channels, 1/2 resolution
    "encoder.layer2",    # 128 channels, 1/4 resolution
    "encoder.layer3",    # 256 channels, 1/8 resolution
    "encoder.layer4",    # 512 channels, 1/16 resolution
    "decoder.blocks.0",  # First decoder block (after deepest skip)
    "decoder.blocks.3",  # Last decoder block (highest resolution)
])

# Run extraction
image_tensor = preprocess(image).unsqueeze(0)  # [1, 3, H, W]
features = extractor.extract(image_tensor)

# features["encoder.layer1"] has shape [1, 64, H/2, W/2]
# features["encoder.layer4"] has shape [1, 512, H/16, W/16]

extractor.remove_hooks()  # Always clean up
```

---

### 2. Visualization Techniques

#### Individual Channel Display

Show each feature map channel as a separate grayscale image. This is the most direct way to
see what individual convolutional filters respond to.

```python
import matplotlib.pyplot as plt
import numpy as np

def plot_feature_channels(feature_map, layer_name, num_channels=16, ncols=8):
    """
    Plot individual channels of a feature map.

    Args:
        feature_map: Tensor of shape [1, C, H, W] or [C, H, W].
        layer_name: String name for the plot title.
        num_channels: Number of channels to display.
        ncols: Number of columns in the grid.
    """
    if feature_map.dim() == 4:
        feature_map = feature_map[0]  # Remove batch dimension

    num_channels = min(num_channels, feature_map.shape[0])
    nrows = (num_channels + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2, nrows * 2))
    fig.suptitle(f"{layer_name} ({feature_map.shape[0]} channels, "
                 f"{feature_map.shape[1]}x{feature_map.shape[2]})", fontsize=14)

    for idx in range(nrows * ncols):
        ax = axes.flat[idx] if nrows > 1 else axes[idx]
        if idx < num_channels:
            channel = feature_map[idx].numpy()
            ax.imshow(channel, cmap="viridis")
            ax.set_title(f"Ch {idx}", fontsize=8)
        ax.axis("off")

    plt.tight_layout()
    return fig
```

#### Channel-Wise Mean Activation

Average across all channels at a given stage to see the overall activation pattern. This
provides a quick summary of "where does this layer pay attention."

```python
def plot_mean_activation(feature_map, layer_name, original_image=None):
    """
    Plot the mean activation across all channels, optionally overlaid on the input image.

    Args:
        feature_map: Tensor of shape [1, C, H, W].
        layer_name: String name for the plot title.
        original_image: Optional numpy array [H, W, 3] of the input image.
    """
    if feature_map.dim() == 4:
        feature_map = feature_map[0]

    mean_activation = feature_map.mean(dim=0).numpy()  # [H, W]

    fig, axes = plt.subplots(1, 2 if original_image is not None else 1,
                              figsize=(10, 5))

    if original_image is not None:
        axes[0].imshow(original_image)
        axes[0].set_title("Input Image")
        axes[0].axis("off")
        ax = axes[1]
    else:
        ax = axes if not hasattr(axes, '__len__') else axes[0]

    im = ax.imshow(mean_activation, cmap="jet")
    ax.set_title(f"{layer_name} - Mean Activation")
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046)

    plt.tight_layout()
    return fig
```

#### Top-K Activated Channels

Select the channels with the highest mean activation value. These are the most "excited"
filters for a given input, and often correspond to the most task-relevant features.

```python
def plot_topk_channels(feature_map, layer_name, k=8):
    """Plot the k channels with highest mean activation."""
    if feature_map.dim() == 4:
        feature_map = feature_map[0]

    # Compute mean activation per channel
    channel_means = feature_map.mean(dim=(1, 2))  # [C]
    topk_indices = torch.topk(channel_means, k).indices

    fig, axes = plt.subplots(1, k, figsize=(k * 2.5, 2.5))
    fig.suptitle(f"{layer_name} - Top {k} Activated Channels", fontsize=12)

    for i, ch_idx in enumerate(topk_indices):
        channel = feature_map[ch_idx].numpy()
        axes[i].imshow(channel, cmap="inferno")
        axes[i].set_title(f"Ch {ch_idx.item()}\n(mean={channel_means[ch_idx]:.2f})",
                          fontsize=8)
        axes[i].axis("off")

    plt.tight_layout()
    return fig
```

#### PCA Projection (Multi-Channel to RGB)

Project the high-dimensional feature map (e.g., 512 channels) down to 3 dimensions using
PCA, then map to RGB. This gives a compact, colorful summary of the feature space.

```python
from sklearn.decomposition import PCA

def pca_feature_to_rgb(feature_map):
    """
    Project a feature map to 3 channels using PCA for RGB visualization.

    Args:
        feature_map: Tensor of shape [C, H, W].

    Returns:
        RGB numpy array of shape [H, W, 3], values in [0, 1].
    """
    C, H, W = feature_map.shape
    # Reshape to [H*W, C] for PCA
    pixels = feature_map.permute(1, 2, 0).reshape(-1, C).numpy()

    pca = PCA(n_components=3)
    projected = pca.fit_transform(pixels)  # [H*W, 3]

    # Normalize each component to [0, 1]
    projected -= projected.min(axis=0)
    projected /= (projected.max(axis=0) + 1e-8)

    rgb = projected.reshape(H, W, 3)
    return rgb
```

#### Grad-CAM for Segmentation

Gradient-weighted Class Activation Mapping highlights which regions the model considers most
important for predicting a specific class. For segmentation, we compute gradients of the
output mask with respect to a target layer's activations.

```python
def compute_gradcam(model, input_tensor, target_layer, target_class=1):
    """
    Compute Grad-CAM heatmap for a segmentation model.

    Args:
        model: Segmentation model.
        input_tensor: Input image tensor [1, 3, H, W].
        target_layer: The layer to compute Grad-CAM for.
        target_class: Class index for which to compute the heatmap.

    Returns:
        Heatmap numpy array of shape [H, W], values in [0, 1].
    """
    gradients = {}
    activations = {}

    def forward_hook(module, input, output):
        activations["value"] = output

    def backward_hook(module, grad_input, grad_output):
        gradients["value"] = grad_output[0]

    fh = target_layer.register_forward_hook(forward_hook)
    bh = target_layer.register_full_backward_hook(backward_hook)

    # Forward pass
    output = model(input_tensor)  # [1, num_classes, H, W]

    # Backward pass: use the sum of target class logits as the scalar
    model.zero_grad()
    target_score = output[0, target_class].sum()
    target_score.backward()

    # Compute Grad-CAM
    grads = gradients["value"][0]       # [C, h, w]
    acts = activations["value"][0]      # [C, h, w]
    weights = grads.mean(dim=(1, 2))    # [C] -- global average pooling of gradients

    cam = (weights[:, None, None] * acts).sum(dim=0)  # [h, w]
    cam = torch.relu(cam)               # Only positive contributions
    cam = cam / (cam.max() + 1e-8)      # Normalize to [0, 1]

    fh.remove()
    bh.remove()

    return cam.detach().cpu().numpy()
```

---

### 3. Recommended Layers to Visualize

For a U-Net-style architecture, the most informative layers to visualize are:

| Layer | Resolution | What to Look For |
|-------|-----------|-----------------|
| Encoder stage 1 output | 1/2 of input | Edge detectors, texture filters, color blobs |
| Encoder stage 2 output | 1/4 of input | Texture combinations, corners, simple patterns |
| Encoder stage 3 output | 1/8 of input | Object parts, regional patterns |
| Bottleneck output | 1/16 of input | High-level semantic features, global context |
| Decoder stage 1 output | 1/8 of input | How high-level features begin recovering spatial detail |
| Decoder final stage | 1/2 of input | Near-output features, should resemble the mask |
| Skip connection features | Various | Compare encoder features before and after skip connection fusion |

For **SegFormer** and other transformer models, also visualize:

- **Attention maps** from each transformer block (reshape from [B, H, N, N] to spatial grids).
- **Multi-scale features** from each encoder stage before the MLP decoder fuses them.

---

### 4. Multi-Layer Comparison Grid

A particularly useful visualization is a grid showing the mean activation map at every
encoder and decoder stage, arranged to mirror the U-Net's U-shaped architecture:

```python
def plot_unet_feature_grid(features, original_image, figsize=(18, 8)):
    """
    Create a U-shaped grid showing feature maps at each stage of a U-Net.

    Args:
        features: Dict mapping layer names to tensors.
            Expected keys: enc1, enc2, enc3, enc4, bottleneck, dec4, dec3, dec2, dec1
        original_image: Input image as numpy array [H, W, 3].
    """
    stages = {
        "top_left": ["enc1", "enc2", "enc3", "enc4"],
        "bottom": ["bottleneck"],
        "top_right": ["dec4", "dec3", "dec2", "dec1"],
    }

    fig, axes = plt.subplots(2, 5, figsize=figsize)
    fig.suptitle("Feature Maps Across U-Net Stages", fontsize=16)

    # Encoder stages (top row, left side)
    for i, key in enumerate(stages["top_left"]):
        if key in features:
            mean_act = features[key][0].mean(dim=0).numpy()
            axes[0, i].imshow(mean_act, cmap="viridis")
            axes[0, i].set_title(f"Enc {i+1}\n{features[key].shape[1]}ch", fontsize=9)
        axes[0, i].axis("off")

    # Bottleneck (bottom center)
    if "bottleneck" in features:
        mean_act = features["bottleneck"][0].mean(dim=0).numpy()
        axes[1, 2].imshow(mean_act, cmap="viridis")
        axes[1, 2].set_title(f"Bottleneck\n{features['bottleneck'].shape[1]}ch", fontsize=9)
    for j in [0, 1, 3, 4]:
        axes[1, j].axis("off")
    axes[1, 2].axis("off")

    # Decoder stages (top row, right side -- we reuse the 5th column)
    # This is simplified; a full implementation would use a custom layout
    axes[0, 4].imshow(original_image)
    axes[0, 4].set_title("Input", fontsize=9)
    axes[0, 4].axis("off")

    plt.tight_layout()
    return fig
```

---

### 5. Interpretation Guidance

When examining feature maps, here are key patterns to look for:

**Encoder features (early layers):**
- Individual channels should respond to specific visual patterns: horizontal edges, vertical
  edges, color gradients, textures.
- If all channels look similar or blank, the model may not be learning diverse features
  (possible issue with initialization or training).

**Encoder features (deep layers):**
- Activations should roughly correspond to semantic regions in the image. For a cell
  segmentation model, you might see channels that activate specifically on cell interiors,
  cell boundaries, or background.
- The spatial resolution is low, so features look "blobby"---this is expected.

**Bottleneck features:**
- The most compressed representation. Should encode the highest-level semantics.
- Strong, localized activations often correspond to the target objects.
- If the bottleneck shows no structure, the model may have insufficient capacity.

**Decoder features:**
- Should progressively sharpen as resolution increases.
- After skip connection fusion, features should be noticeably sharper than before---this
  confirms the skip connections are working.
- The final decoder stage should closely resemble the predicted segmentation mask.

**Comparing models:**
- CNN models typically show more spatially localized, edge-like features in early layers.
- Transformer models tend to show more globally coherent features even in early stages.
- If two models have similar accuracy but different feature maps, they may complement each
  other well in an ensemble.

---

## Output Format

Feature map visualizations should be saved as:

- **Individual PNG images** per layer per input: `{model}_{layer}_{image_id}.png`
- **Grid montages** showing multiple channels: `{model}_{layer}_{image_id}_grid.png`
- **Comparison grids** across different models: `comparison_{image_id}_{layer_depth}.png`
- **Animated GIFs** cycling through channels: `{model}_{layer}_{image_id}.gif`

Recommended settings:
- DPI: 150 for review, 300 for publication.
- Colormap: `viridis` (perceptually uniform) for general use, `jet` for heatmap overlays,
  `inferno` for high-contrast single-channel views.

---

## Tools

- **PyTorch hooks**: For feature extraction (as shown above).
- **matplotlib**: For rendering static images and grids.
- **torchvision.utils.make_grid**: For creating channel montages quickly.
- **captum**: Facebook's interpretability library, provides Grad-CAM and other attribution
  methods out of the box.
- **tensorboard**: Log feature maps during training for real-time monitoring.

---

## Planned Visualizations

Generate feature maps for the following model-input combinations:

1. **U-Net on Kvasir-SEG polyp images** -- show how encoder features capture polyp boundaries
   and how decoder features recover spatial detail through skip connections.
2. **SegFormer attention maps on ADE20K** -- visualize the self-attention patterns at each
   transformer stage, showing how global context is captured.
3. **SAM 2 image encoder features on medical images** -- compare zero-shot features with
   fine-tuned features to see how LoRA adaptation changes the encoder's representations.
4. **U-Net vs. Attention U-Net skip features** -- compare skip connection features before and
   after the attention gate to show what the attention mechanism suppresses.
