---
title: "Recurrent Convolution Blocks in R2U-Net"
date: 2025-03-06
status: complete
tags: [recurrent-convolution, feature-accumulation, r2u-net]
difficulty: intermediate
---

# Recurrent Convolution Blocks

## Overview

Recurrent convolution blocks are the core building block of R2U-Net. Instead of applying a convolution once (as in standard U-Net) or twice (as in DoubleConv), the same convolution kernel is applied t times in a recurrent fashion. Each iteration accumulates features from the previous iteration, effectively expanding the receptive field and enriching the feature representation without adding parameters.

## Mechanism

At each time step t, the output is:

```
x_o^t = f(W_f * x_i + W_r * x_o^{t-1} + b)
```

where:
- `x_i` is the original input (constant across time steps)
- `x_o^{t-1}` is the output from the previous time step (t-1)
- `W_f` are the feedforward convolution weights
- `W_r` are the recurrent convolution weights (crucially, shared across time steps)
- `f` is ReLU activation

For t=0, x_o^0 = f(W_f * x_i + b) (standard convolution).

## Feature Accumulation

At each iteration, the recurrent term `W_r * x_o^{t-1}` incorporates information from the previous step. This creates a feature accumulation effect:

- **t=0**: Standard convolution with receptive field = kernel size
- **t=1**: Features enriched with context from t=0, effective RF ≈ 2× kernel size
- **t=2**: Further enriched, effective RF ≈ 3× kernel size

The shared weights mean the same convolution "refines" its own output, learning to iteratively improve the feature representation.

## Implementation

```python
class RecurrentBlock(nn.Module):
    def __init__(self, channels, t=2):
        super().__init__()
        self.t = t
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        x_r = self.conv(x)  # t=0
        for _ in range(self.t - 1):
            x_r = self.conv(x + x_r)  # t=1, 2, ...
        return x_r
```

Note that `x` (the original input) is added at each step, not `x_r`. This ensures the original input information is preserved at every iteration.

## Comparison with Standard Convolutions

| Approach | Params | FLOPs | Effective RF |
|----------|--------|-------|-------------|
| Single conv | C²K² | 1× | K |
| DoubleConv (U-Net) | 2C²K² | 2× | 2K-1 |
| RecurrentConv (t=2) | C²K² | 2× | ~2K |
| RecurrentConv (t=3) | C²K² | 3× | ~3K |

Recurrent convolution achieves a similar effective receptive field as DoubleConv with half the parameters, at the cost of sequential computation (iterations cannot be parallelized).
