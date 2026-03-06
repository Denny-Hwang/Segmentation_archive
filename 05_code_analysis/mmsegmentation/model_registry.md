---
title: "MMSegmentation - Model Registry Analysis"
date: 2025-01-15
status: planned
parent: "mmsegmentation/repo_overview.md"
tags: [mmsegmentation, registry, models, decorator-pattern]
---

# MMSegmentation Model Registry

## Overview

TODO: Analyze how MMSegmentation uses MMEngine's registry to manage models

## Registry Architecture

### MODELS Registry
TODO: How `@MODELS.register_module()` works

### Build from Config
TODO: How `MODELS.build(cfg)` instantiates a model from a config dict

## Segmentor Types

### EncoderDecoder
TODO: How the standard encoder-decoder segmentor composes backbone + head

### CascadeEncoderDecoder
TODO: How cascade segmentors work

## Adding Custom Components

### Custom Backbone
TODO: Step-by-step guide to registering a custom backbone

### Custom Decode Head
TODO: Step-by-step guide to registering a custom decode head

## Available Model Inventory

TODO: List all registered backbones and decode heads with their paper references
