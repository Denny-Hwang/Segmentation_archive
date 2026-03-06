---
title: "Attention Gate Mechanism"
date: 2025-03-06
status: planned
tags:
  - attention-gate
  - soft-attention
  - gating-signal
parent: attention_unet/review.md
---

# Attention Gate Mechanism

## Overview

_TODO: Describe how attention gates selectively highlight salient features from skip connections using contextual information from the gating signal._

---

## Attention Gate Architecture

_TODO: Diagram and step-by-step explanation of the attention gate._

### Inputs

- _TODO: Feature map from skip connection (x_l) -- high resolution, local features_
- _TODO: Gating signal (g) -- low resolution, contextual features from deeper layer_

### Operations

1. _TODO: Linear transformations W_x and W_g to map both inputs to the same channel dimension_
2. _TODO: Element-wise addition of the transformed features_
3. _TODO: ReLU activation_
4. _TODO: 1x1 convolution (psi) to produce a single attention map_
5. _TODO: Sigmoid to obtain attention coefficients alpha in [0, 1]_
6. _TODO: Element-wise multiplication of alpha with the skip connection features_

---

## Mathematical Formulation

_TODO: Write out the equations:_

$$
q_{att}^l = \psi^T (\sigma_1 (W_x^T x_i^l + W_g^T g_i + b_g)) + b_\psi
$$

$$
\alpha_i^l = \sigma_2(q_{att}^l)
$$

---

## Additive vs Multiplicative Attention

_TODO: Compare the additive attention used here with multiplicative (dot-product) attention used in transformers._

---

## Grid-Based Gating

_TODO: Explain the grid-based gating signal strategy for computational efficiency._

---

## Computational Cost

_TODO: Analyze the additional parameters and FLOPs introduced by attention gates._

---

## Visualizing Attention Maps

_TODO: Reference [attention_visualization.md](./attention_visualization.md) for examples of learned attention maps._
