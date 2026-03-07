---
title: "Task-Conditioned Training in OneFormer"
date: 2025-03-06
status: complete
tags: [task-conditioning, multi-task, task-token, universal-segmentation]
difficulty: advanced
---

# Task-Conditioned Training

## Overview

Task-conditioned training is OneFormer's strategy for training a single model on all three segmentation tasks simultaneously. Rather than multi-task training where all tasks are optimized in every iteration (which causes interference), OneFormer samples one task per iteration and conditions the model on that task via a task token. This simple approach prevents negative transfer between tasks while allowing shared feature learning.

## Task Token Design

Three task tokens are defined: `{semantic, instance, panoptic}`. Each is represented as a learnable embedding vector of dimension d (matching the query dimension). At each training iteration, one task is sampled uniformly at random, and the corresponding token is provided as an additional input to the model. At inference, the user specifies which task to perform by providing the appropriate token.

## Conditioning Mechanism

The task token influences the model through task-conditioned query initialization:

1. The task token is passed through a 2-layer MLP to produce a task bias vector
2. This bias is added to all N learnable query embeddings: `Q_init = Q_learnable + MLP(task_token)`
3. The conditioned queries are then processed by the transformer decoder as usual

This means the same set of queries behave differently depending on the task. For semantic segmentation, queries are biased toward covering semantic regions (potentially merging instances). For instance segmentation, queries are biased toward detecting individual objects. The conditioning is lightweight — only the query initialization changes, not the architecture.

## Single Model, Multiple Tasks

The key advantage is that one model replaces three. During training, each iteration processes one task, so the batch contains ground truth annotations for that task only. The model learns shared low-level features (edges, textures, object boundaries) across all tasks while developing task-specific behavior through the conditioned queries.

Shared parameters: backbone, pixel decoder, transformer decoder weights, mask prediction head, class prediction head. Task-specific: only the query initialization bias (via MLP) and the task token embeddings.

## Training Protocol

Each training iteration: (1) sample a task uniformly from {semantic, instance, panoptic}; (2) load a batch with that task's ground truth; (3) condition queries with the task token; (4) compute predictions; (5) apply Hungarian matching and loss for that task only. The data loader cycles through all three tasks' annotations for the same images. Training runs for 160k iterations on COCO with AdamW optimizer, lr=1e-4 with polynomial decay.

## Ablation Results

| Configuration | ADE20K PQ | ADE20K mIoU | COCO AP |
|--------------|-----------|-------------|---------|
| Task-conditioned (proposed) | 49.8 | 58.0 | 49.0 |
| No task token | 48.6 | 57.2 | 48.2 |
| All tasks every iteration | 48.1 | 56.8 | 47.5 |
| Separate models (Mask2Former) | 48.1 | 57.8 | 50.1 |

Task conditioning adds +1.2 PQ over no conditioning. Training all tasks every iteration is worse due to gradient interference. The single OneFormer model matches three separate Mask2Former models on panoptic/semantic while using 3× less compute.

## Comparison with Multi-Head Approaches

Traditional multi-task learning uses shared backbones with task-specific heads. This creates larger models and doesn't share decoder knowledge across tasks. OneFormer's approach is more parameter-efficient: the only task-specific parameters are the task token embeddings (3 × d) and the conditioning MLP (~2d² parameters). Everything else — including the decoder — is fully shared. This is possible because masked attention already provides query-level specialization, and the task token simply biases which specialization pattern emerges.

## Implementation Notes

The task token is a learnable `nn.Embedding(3, d)` indexed by task ID. The conditioning MLP is `nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d))`. During inference, switching between tasks requires only changing the task token index — no model reloading or architectural changes. This makes deployment straightforward: one model serves all three segmentation endpoints.
