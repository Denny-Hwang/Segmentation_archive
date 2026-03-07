---
title: "Attention Map Visualization in Attention U-Net"
date: 2025-03-06
status: complete
tags: [attention-maps, visualization, interpretability, attention-gate]
difficulty: beginner
---

# Attention Map Visualization

## Overview

Attention gates produce spatial attention maps at each skip connection level, providing interpretable visualizations of what the model focuses on. These maps can be extracted and visualized to understand model behavior, debug predictions, and build clinical trust.

## Extracting Attention Maps

Attention maps can be captured using forward hooks in PyTorch:

```python
attention_maps = {}

def hook_fn(name):
    def hook(module, input, output):
        attention_maps[name] = module.sigmoid(module.psi(
            module.relu(module.W_g(input[0]) + module.W_x(input[1]))
        )).detach().cpu()
    return hook

# Register hooks on attention gates
for name, module in model.named_modules():
    if isinstance(module, AttentionGate):
        module.register_forward_hook(hook_fn(name))
```

## Visualization

```python
import matplotlib.pyplot as plt

def show_attention_maps(image, attention_maps, prediction, ground_truth):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes[0, 0].imshow(image, cmap='gray')
    axes[0, 0].set_title('Input Image')
    
    for i, (name, attn) in enumerate(attention_maps.items()):
        ax = axes[(i+1)//3, (i+1)%3]
        # Upsample attention map to input resolution
        attn_up = F.interpolate(attn, size=image.shape, mode='bilinear')
        ax.imshow(image, cmap='gray', alpha=0.5)
        ax.imshow(attn_up.squeeze(), cmap='jet', alpha=0.5)
        ax.set_title(f'Attention Level {i+1}')
    
    axes[1, 1].imshow(prediction, cmap='viridis')
    axes[1, 1].set_title('Prediction')
    axes[1, 2].imshow(ground_truth, cmap='viridis')
    axes[1, 2].set_title('Ground Truth')
    plt.tight_layout()
```

## Interpretation Guidelines

- **Level 1 (highest resolution)**: Shows fine-grained spatial focus, typically highlighting boundaries
- **Level 2-3 (middle)**: Shows organ/structure-level focus, most informative for understanding model behavior
- **Level 4 (lowest resolution)**: Shows coarse regional focus, often covers the entire target area plus surroundings

High attention values (red/yellow in jet colormap) indicate regions the model considers important. Low values (blue) indicate suppressed regions. Comparing attention maps across correctly and incorrectly segmented cases can reveal failure modes.
