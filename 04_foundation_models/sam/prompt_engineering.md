---
title: "Prompt Engineering for SAM"
date: 2025-03-06
status: planned
tags: [prompt-engineering, points, boxes, masks, text-prompts]
difficulty: intermediate
---

# Prompt Engineering for SAM

## Overview

SAM's prompt engineering system provides a flexible interface that allows users to specify segmentation targets through multiple input modalities. The prompt encoder converts these diverse inputs into a unified embedding representation that the mask decoder can process alongside image features. This design draws on the concept of "prompting" from NLP foundation models, where task specification is separated from model architecture. In SAM, prompts serve as the primary mechanism for resolving ambiguity about which object or region to segment, making the quality and type of prompt a critical determinant of output quality.

The prompt encoder handles two categories of inputs: sparse prompts (points and boxes) and dense prompts (masks). Sparse prompts are represented as sets of positional encoding vectors summed with learned type embeddings, while dense prompts pass through a small convolutional network to produce spatial feature maps. All prompt representations are designed to be compatible with the 256-dimensional embedding space used by the mask decoder, enabling seamless combination of different prompt types.

## Point Prompts

Point prompts are the simplest and most lightweight form of interaction with SAM. Each point is classified as either a foreground (positive) point indicating part of the target object, or a background (negative) point indicating a region to exclude. Internally, each point is represented as a positional encoding computed at its (x, y) coordinate using a Fourier feature encoding scheme, summed with a learned embedding that distinguishes foreground from background semantics.

Single-point prompts are inherently ambiguous because a single location can correspond to multiple valid segmentation scales (e.g., clicking on a person's eye could mean the eye, the face, or the full person). SAM addresses this through its multi-mask prediction mechanism, outputting three masks at different granularity levels. Adding more points progressively resolves this ambiguity: with 2-3 well-placed foreground and background points, SAM typically converges to a single high-quality mask. Empirically, performance improves substantially from 1 to 3 points, with diminishing returns beyond 5-6 points. Negative points are particularly valuable for excluding nearby objects or background regions that the model incorrectly includes.

## Box Prompts

Bounding box prompts specify a rectangular region that tightly encloses the target object. The box is encoded as two points (top-left and bottom-right corners), each represented with positional encodings and distinct learned type embeddings. Box prompts are generally the most informative single-prompt type because they simultaneously convey location, scale, and approximate extent of the target object, substantially reducing ambiguity.

In practice, box prompts yield the highest single-prompt accuracy among all prompt types, typically achieving 5-10 mIoU points higher than single-point prompts. This makes box prompts the preferred choice for automated pipelines where bounding boxes are available from object detectors. The effectiveness of box prompts is particularly notable for objects with complex shapes or multiple disconnected components, where the bounding box provides spatial context that points alone cannot convey. However, box prompts can struggle when multiple distinct objects are enclosed within the same bounding box, as the model must determine which object is the intended target.

## Mask Prompts

Mask prompts allow users to provide an existing (potentially coarse or incomplete) segmentation mask as input. This is especially useful for iterative refinement workflows, where an initial prediction from SAM is fed back as a mask prompt to obtain a more precise segmentation. The mask prompt is processed through a lightweight convolutional network (two stride-2 convolutions followed by a 1x1 convolution) that downsamples the input mask to 64x64 resolution and produces a 256-dimensional dense embedding, which is added element-wise to the image embedding before entering the mask decoder.

Mask prompts enable a "refinement loop" where SAM progressively improves its output across iterations. In the first iteration, a point or box prompt generates an initial mask; in subsequent iterations, the predicted mask is fed back along with any corrective point prompts. This iterative refinement typically converges within 2-3 iterations and can recover from significant errors in the initial prediction. Mask prompts also enable use cases such as segmentation editing, where a user provides a partial mask and SAM completes or adjusts the boundaries.

## Text Prompts

SAM's architecture includes support for text prompts through CLIP's text encoder, which maps natural language descriptions to embedding vectors aligned with the visual feature space. In principle, text prompts enable open-vocabulary segmentation where users can describe the target object in natural language (e.g., "the red car on the left"). The text embedding is treated as a sparse prompt token and processed alongside any other prompt tokens in the mask decoder.

However, the text prompting capability was not extensively trained or evaluated in the original SAM release, and its performance remains limited compared to point and box prompts. The primary challenge is that CLIP's text-image alignment was designed for image-level matching rather than pixel-level localization, making it difficult to precisely associate text descriptions with specific spatial regions. Subsequent works such as Grounding SAM (combining Grounding DINO with SAM) have achieved more robust text-based segmentation by using a dedicated grounding model to convert text to box prompts, which are then passed to SAM.

## Prompt Combinations

One of SAM's most powerful capabilities is the ability to combine multiple prompt types in a single forward pass. For example, a user might provide a bounding box to specify the general region, a positive point on the target, and a negative point on a nearby distractor. The prompt encoder processes each input independently and concatenates the resulting tokens, which are then jointly processed by the mask decoder through self-attention and cross-attention operations.

Common effective combinations include: (1) box + point, where the box provides spatial context and the point disambiguates which object within the box is targeted; (2) box + negative points, where the box defines the region and negative points exclude background clutter or neighboring objects; (3) mask + corrective points, used in iterative refinement where previous predictions are refined based on user corrections. The flexibility of prompt combinations is essential for SAM's role as an interactive annotation tool, as it allows annotators to progressively refine segmentations through a natural workflow.

## Best Practices

When using SAM for practical applications, several guidelines improve results. For single-object segmentation, a tight bounding box generally produces the best results with minimal user effort. If a bounding box is unavailable, placing a foreground point near the center of the target object (rather than near edges) yields more stable predictions. Including at least one negative point near confusing background regions significantly reduces false positive area.

For batch or automated processing, using detected bounding boxes as prompts is recommended, as this pipeline (detector + SAM) consistently outperforms single-point approaches. When annotating ambiguous objects, starting with a coarse prompt and iteratively refining using the mask refinement loop provides the best quality-effort trade-off. For objects with thin or elongated structures, multiple points along the structure (rather than a single point) help SAM capture the full extent. Finally, when the multi-mask output is enabled, selecting the mask with the highest predicted IoU score is a reliable heuristic for automated selection.

## Prompt Sensitivity Analysis

SAM exhibits varying degrees of sensitivity to prompt quality depending on the prompt type. Point prompts are the most sensitive to placement: shifting a single foreground point by just 10-20 pixels can change the predicted mask substantially, especially near object boundaries or at the junction of multiple objects. This sensitivity is expected given the ambiguity inherent in point-based specification, and it diminishes rapidly as additional points are provided.

Box prompts are more robust, with moderate tolerance to box looseness (adding 10-20% padding beyond the tight bounding box typically does not degrade performance significantly, and can even improve it slightly by providing context). However, boxes that are excessively loose or that shift to partially exclude the target object cause noticeable degradation. Mask prompts show the most graceful degradation: even substantially incorrect input masks (e.g., covering only 50% of the target) can still guide SAM toward a reasonable output, as the model learns to treat the mask as a soft spatial prior rather than a hard constraint.

Across all prompt types, SAM is generally more robust on natural images with common object categories than on specialized domains (medical, satellite, etc.), where the model's learned visual priors are less calibrated to the domain-specific appearance distributions.

## Implementation Notes

The prompt encoder is implemented as a lightweight module with minimal computational overhead. Sparse prompts use a Fourier feature positional encoding with 128 frequency bands, producing 256-dimensional embeddings. The type embeddings (foreground point, background point, top-left corner, bottom-right corner) are stored as learned parameters initialized randomly. The dense prompt convolutional network has approximately 100K parameters, negligible compared to the image encoder.

When no prompt is provided (as in automatic mask generation), SAM uses a default "no-mask" embedding -- a learned token that signals the absence of a mask prompt. For batched inference, multiple sets of prompts can be processed in parallel against the same image embedding, with the prompt encoder and mask decoder executing independently for each prompt set. This parallelism is key to the efficiency of SAM's automatic mask generation pipeline, where a 32x32 grid of point prompts is evaluated simultaneously.
