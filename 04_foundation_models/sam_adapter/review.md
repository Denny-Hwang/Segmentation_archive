---
title: "SAM-Adapter: Adapting SAM in Underperformed Scenes"
date: 2025-03-06
status: planned
tags: [adapter, parameter-efficient, domain-adaptation, sam]
difficulty: intermediate
---

# SAM-Adapter

## Meta Information

| Field | Value |
|-------|-------|
| **Paper Title** | SAM Fails to Segment Anything? -- SAM-Adapter: Adapting SAM in Underperformed Scenes |
| **Authors** | Chen, T., Zhu, L., Deng, C., Cao, R., Wang, Y., Zhang, S., Li, Z., Sun, L., Zang, Y., Mao, P. |
| **Year** | 2023 |
| **Venue** | arXiv |
| **arXiv** | [2304.09148](https://arxiv.org/abs/2304.09148) |
| **Difficulty** | Intermediate |

## One-Line Summary

SAM-Adapter inserts lightweight adapter modules into SAM's image encoder to enable parameter-efficient domain adaptation for challenging scenes where vanilla SAM underperforms.

## Motivation and Problem Statement

<!-- Where and why SAM fails, motivating the need for adaptation -->

## Architecture Overview

<!-- How adapters are inserted into SAM's architecture -->

### Key Components

- **Adapter Tuning**: See [adapter_tuning.md](adapter_tuning.md)

## Technical Details

### Adapter Module Design

<!-- Architecture of the adapter modules -->

### Insertion Points

<!-- Where adapters are placed in SAM's encoder -->

### Frozen vs. Trainable Parameters

<!-- What is frozen and what is trained -->

### Task-Specific Heads

<!-- How task-specific outputs are generated -->

### Training Strategy

<!-- Training procedure and hyperparameters -->

## Experiments and Results

### Target Domains

<!-- Challenging scenes evaluated (e.g., shadow detection, camouflaged objects) -->

### Key Results

<!-- Main quantitative results -->

### Comparison with Full Fine-Tuning

<!-- Adapter efficiency vs. full fine-tuning performance -->

## Strengths

<!-- List key strengths of the approach -->

## Limitations

<!-- List key limitations -->

## Connections

<!-- How this work relates to SAM and other adaptation papers -->

## References

<!-- Key references cited in the paper -->
