---
title: "nnU-Net - Experiment Planner Analysis"
date: 2025-01-15
status: planned
parent: "nnunet/repo_overview.md"
tags: [nnunet, experiment-planning, auto-configuration]
---

# nnU-Net Experiment Planner

## Overview

TODO: Analyze how `ExperimentPlanner` in `experiment_planning/` automatically determines all training hyperparameters.

## Dataset Fingerprint

### Fingerprint Extraction
TODO: What statistics are computed from the dataset

### Fingerprint Contents
TODO: Document the fingerprint data structure

## Planning Algorithm

### Patch Size Selection
TODO: How optimal patch size is determined

### Batch Size Selection
TODO: How batch size is computed from available GPU memory

### Network Topology
TODO: How the number of stages, channels per stage, etc. are determined

### Configuration Cascade
TODO: How 2D, 3D_fullres, 3D_lowres, and 3D_cascade are planned

## Plans File Format

TODO: Document the structure of the generated plans file

## Customizing the Planner

TODO: How to override automatic decisions

## Key Heuristics

TODO: Document the rules of thumb embedded in the planning code
