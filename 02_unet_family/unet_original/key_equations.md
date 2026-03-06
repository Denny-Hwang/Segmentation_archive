---
title: "U-Net Key Equations"
date: 2025-03-06
status: planned
tags:
  - loss-function
  - weight-map
  - equations
parent: unet_original/review.md
---

# U-Net Key Equations

## 1. Pixel-wise Softmax

The network output is passed through a pixel-wise softmax over the final feature map to produce class probabilities:

$$
p_k(\mathbf{x}) = \frac{\exp(a_k(\mathbf{x}))}{\sum_{k'=1}^{K} \exp(a_{k'}(\mathbf{x}))}
$$

Where:
- $a_k(\mathbf{x})$ is the activation in feature channel $k$ at pixel position $\mathbf{x}$
- $K$ is the number of classes

---

## 2. Cross-Entropy Loss with Weight Map

The loss function is a weighted pixel-wise cross-entropy:

$$
E = \sum_{\mathbf{x} \in \Omega} w(\mathbf{x}) \log(p_{\ell(\mathbf{x})}(\mathbf{x}))
$$

Where:
- $\Omega$ is the set of all pixels
- $\ell(\mathbf{x})$ is the ground truth label at pixel $\mathbf{x}$
- $w(\mathbf{x})$ is the weight map that assigns importance to each pixel

---

## 3. Weight Map Computation

The weight map pre-computed for each ground truth segmentation forces the network to learn border pixels between touching objects:

$$
w(\mathbf{x}) = w_c(\mathbf{x}) + w_0 \cdot \exp\left(-\frac{(d_1(\mathbf{x}) + d_2(\mathbf{x}))^2}{2\sigma^2}\right)
$$

Where:
- $w_c(\mathbf{x})$ is the class frequency balancing weight
- $d_1(\mathbf{x})$ is the distance to the border of the nearest cell
- $d_2(\mathbf{x})$ is the distance to the border of the second nearest cell
- $w_0 = 10$ and $\sigma \approx 5$ pixels (as set in the paper)

---

## 4. Weight Initialization

Weights are drawn from a Gaussian distribution adapted for the network depth:

$$
W \sim \mathcal{N}\left(0, \sqrt{\frac{2}{N}}\right)
$$

Where $N$ is the number of incoming nodes of one neuron. This follows the He initialization strategy.

---

## 5. Overlap-Tile Strategy

_TODO: Describe the mirror-padding extrapolation used for seamless tiling of large images. Not an equation per se but a key algorithmic detail._

---

## 6. Data Augmentation Transformations

_TODO: Document the elastic deformation field applied during training:_

$$
\Delta x(i,j) \sim \text{Uniform}(-1, 1) * \alpha, \quad \text{smoothed by Gaussian with } \sigma
$$

_TODO: Detail the values of alpha and sigma used._

---

## Notes

- _TODO: Discuss the relationship between the weight map and morphological operations._
- _TODO: Compare this loss formulation with Dice loss (used in V-Net) and focal loss._
