---
title: "SAM 2 - Prompt Encoder Analysis"
date: 2025-01-15
status: planned
parent: "sam2/repo_overview.md"
tags: [sam2, prompt-encoder, points, boxes, masks]
---

# SAM 2 Prompt Encoder

## Overview

TODO: Analyze the prompt encoder in `sam2/modeling/prompt_encoder.py`

## Prompt Types

### Point Prompts
TODO: How (x, y) coordinates with positive/negative labels are encoded

### Box Prompts
TODO: How bounding boxes (top-left, bottom-right) are encoded

### Mask Prompts
TODO: How dense mask inputs are downsampled and encoded

## Positional Encoding

TODO: How spatial positions are encoded (learned vs sinusoidal)

## Sparse vs Dense Embeddings

TODO: How point/box (sparse) and mask (dense) prompts produce different embedding types

## No-Prompt Case

TODO: How the model handles inference with no prompt (automatic mask generation)

## Prompt Combination

TODO: How multiple prompts of different types are combined
