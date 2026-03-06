---
title: "U-Net Architecture Description"
date: 2025-03-06
status: planned
tags:
  - architecture
  - encoder-decoder
  - skip-connections
parent: unet_original/review.md
---

# U-Net Architecture

## Overview

The U-Net architecture is a symmetric encoder-decoder network with skip connections. Its name derives from the U-shaped structure when visualized: the left side contracts (encodes), the bottom is the bottleneck, and the right side expands (decodes).

---

## Contracting Path (Encoder)

_TODO: Detail each encoder block._

| Stage | Input Size | Operations | Output Channels |
|-------|-----------|------------|-----------------|
| Enc-1 | 572 x 572 | 2x (3x3 conv + ReLU), 2x2 max pool | 64 |
| Enc-2 | 284 x 284 | 2x (3x3 conv + ReLU), 2x2 max pool | 128 |
| Enc-3 | 140 x 140 | 2x (3x3 conv + ReLU), 2x2 max pool | 256 |
| Enc-4 | 68 x 68   | 2x (3x3 conv + ReLU), 2x2 max pool | 512 |

---

## Bottleneck

_TODO: Describe the bottom of the U -- 1024 channels, two 3x3 convolutions._

---

## Expansive Path (Decoder)

_TODO: Detail each decoder block with up-convolution and concatenation._

| Stage | Operations | Output Channels |
|-------|-----------|-----------------|
| Dec-4 | 2x2 up-conv, concat with Enc-4, 2x (3x3 conv + ReLU) | 512 |
| Dec-3 | 2x2 up-conv, concat with Enc-3, 2x (3x3 conv + ReLU) | 256 |
| Dec-2 | 2x2 up-conv, concat with Enc-2, 2x (3x3 conv + ReLU) | 128 |
| Dec-1 | 2x2 up-conv, concat with Enc-1, 2x (3x3 conv + ReLU) | 64 |

---

## Final Layer

_TODO: 1x1 convolution mapping 64 feature channels to the desired number of classes._

---

## Skip Connections

_TODO: Explain the crop-and-concatenate mechanism. Discuss why concatenation is preferred over addition (as used in FCN)._

---

## Key Design Decisions

- _TODO: No padding in convolutions (original paper) -- output is smaller than input_
- _TODO: Mirror padding at image borders for seamless tiling_
- _TODO: Overlap-tile strategy for large images_

---

## Parameter Count

_TODO: Estimate and tabulate learnable parameters per layer._

---

## Diagram

See [architecture_diagram.mermaid](./architecture_diagram.mermaid) for the full Mermaid diagram.
