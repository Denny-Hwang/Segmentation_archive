---
title: "SAM 2 - Prompt Encoder Analysis"
date: 2025-01-15
status: complete
parent: "sam2/repo_overview.md"
tags: [sam2, prompt-encoder, points, boxes, masks]
---

# SAM 2 Prompt Encoder

## Overview

The prompt encoder in SAM 2, defined in `sam2/modeling/sam/prompt_encoder.py`, converts user-provided prompts (points, bounding boxes, and dense masks) into embedding vectors that the mask decoder can process. The encoder handles two categories of output: **sparse embeddings** (from points and boxes, represented as a sequence of tokens) and **dense embeddings** (from mask inputs, represented as a spatial feature map). The prompt encoder is lightweight and contains relatively few parameters compared to the image encoder, as its primary role is to translate geometric prompts into the same embedding space used by the transformer-based mask decoder.

The `PromptEncoder` class maintains learned embedding tables for different prompt types and uses positional encoding to convert spatial coordinates into high-dimensional vectors. It supports arbitrary combinations of prompt types in a single call, allowing users to provide, for example, a bounding box with positive and negative point refinements simultaneously.

## Prompt Types

### Point Prompts

Point prompts consist of (x, y) coordinates with associated labels indicating positive (foreground) or negative (background) intent. The encoding process converts each point into a token embedding by combining a positional encoding of the spatial location with a learned label embedding:

```python
class PromptEncoder(nn.Module):
    def __init__(self, embed_dim=256, ...):
        # Learned embeddings for point labels
        self.point_embeddings = nn.ModuleList([
            nn.Embedding(1, embed_dim) for _ in range(4)
            # 0: background/negative, 1: foreground/positive,
            # 2: top-left corner, 3: bottom-right corner
        ])

    def _embed_points(self, points, labels, pad):
        # points: [B, N, 2] - (x, y) coordinates normalized to [0, 1]
        # labels: [B, N] - 0 for negative, 1 for positive
        points = points + 0.5  # Shift to center of pixel
        point_embedding = self.pe_layer.forward_with_coords(points, self.input_image_size)
        # Add learned label embedding
        point_embedding[labels == 0] += self.point_embeddings[0].weight
        point_embedding[labels == 1] += self.point_embeddings[1].weight
        return point_embedding  # [B, N, embed_dim]
```

The positional encoding converts raw pixel coordinates into a 256-dimensional vector using a random Fourier feature approach (also known as positional encoding with learned or fixed frequency bands). This ensures that points at different spatial locations produce distinct embeddings, and nearby points produce similar embeddings.

Negative points are particularly important for interactive refinement: after an initial mask prediction, users can click on false-positive regions to indicate "not this object." The negative point embedding tells the decoder to suppress mask predictions in that area.

### Box Prompts

Bounding box prompts are encoded as two special points: the top-left corner and the bottom-right corner. Each corner receives the same positional encoding as regular points but uses distinct learned label embeddings (indices 2 and 3) to differentiate them from foreground/background points:

```python
def _embed_boxes(self, boxes):
    # boxes: [B, 4] - (x1, y1, x2, y2) normalized coordinates
    boxes = boxes + 0.5  # Shift to center of pixel
    coords = boxes.reshape(-1, 2, 2)  # [B, 2, 2] - two corners
    corner_embedding = self.pe_layer.forward_with_coords(coords, self.input_image_size)
    # Add corner-specific learned embeddings
    corner_embedding[:, 0, :] += self.point_embeddings[2].weight  # Top-left
    corner_embedding[:, 1, :] += self.point_embeddings[3].weight  # Bottom-right
    return corner_embedding  # [B, 2, embed_dim]
```

Box prompts are treated as sparse embeddings and concatenated with any point embeddings before being passed to the mask decoder. A box prompt is semantically stronger than a single point because it constrains the object's spatial extent. The decoder typically switches to single-mask mode when a box prompt is provided, since the box significantly reduces ambiguity.

### Mask Prompts

Dense mask prompts allow users to provide a coarse mask (e.g., from a previous iteration or a different model) as input. The mask is downsampled and encoded through a small convolutional network to produce a dense embedding with the same spatial dimensions as the image embedding:

```python
class PromptEncoder(nn.Module):
    def __init__(self, embed_dim=256, mask_in_chans=16, ...):
        self.mask_downscaling = nn.Sequential(
            nn.Conv2d(1, mask_in_chans // 4, kernel_size=2, stride=2),
            LayerNorm2d(mask_in_chans // 4),
            nn.GELU(),
            nn.Conv2d(mask_in_chans // 4, mask_in_chans, kernel_size=2, stride=2),
            LayerNorm2d(mask_in_chans),
            nn.GELU(),
            nn.Conv2d(mask_in_chans, embed_dim, kernel_size=1),
        )

    def _embed_masks(self, masks):
        # masks: [B, 1, H, W] - input mask at image resolution
        mask_embedding = self.mask_downscaling(masks)
        return mask_embedding  # [B, embed_dim, H/4, W/4]
```

The mask input is downsampled by 4x through two strided convolutions, matching the stride of the image encoder's finest feature map. The resulting dense embedding is added element-wise to the image embedding in the mask decoder, effectively biasing the decoder's spatial attention toward regions indicated by the input mask.

## Positional Encoding

SAM 2 uses a **random Fourier feature** positional encoding (also called sinusoidal positional encoding) implemented in the `PositionEmbeddingRandom` class:

```python
class PositionEmbeddingRandom(nn.Module):
    def __init__(self, num_pos_feats=128, scale=None):
        super().__init__()
        # Fixed random matrix for Fourier features
        self.register_buffer('positional_encoding_gaussian_matrix',
                           torch.randn(2, num_pos_feats))

    def forward_with_coords(self, coords_input, image_size):
        # Normalize coordinates to [0, 1]
        coords = coords_input.clone()
        coords[:, :, 0] = coords[:, :, 0] / image_size[1]  # x / width
        coords[:, :, 1] = coords[:, :, 1] / image_size[0]  # y / height
        # Map to Fourier features: [B, N, 2] @ [2, num_pos_feats] -> [B, N, num_pos_feats]
        coords = 2 * coords - 1  # Scale to [-1, 1]
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        # Concatenate sin and cos: [B, N, 2*num_pos_feats]
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)
```

The positional encoding matrix is randomly initialized and **frozen** (not trained), following the random Fourier features paradigm. With `num_pos_feats=128`, the output dimension is 256 (128 sin + 128 cos), matching the transformer embedding dimension. The use of random Fourier features instead of learned positional embeddings provides smooth spatial interpolation and works for any input resolution without retraining.

## Sparse vs Dense Embeddings

The prompt encoder produces two distinct types of output that are consumed differently by the mask decoder:

**Sparse embeddings** (`[B, N, embed_dim]`): Produced by point and box prompts. These are a variable-length sequence of token embeddings that are concatenated with the mask decoder's output tokens and processed through the transformer's self-attention and cross-attention layers. Sparse embeddings directly influence the mask decoder's token processing and, through the two-way attention mechanism, also modulate the image features.

**Dense embeddings** (`[B, embed_dim, H/4, W/4]`): Produced by mask prompts (or a zero tensor if no mask is provided). These have the same spatial dimensions as the image embedding and are added element-wise to the image features before the mask decoder processes them. Dense embeddings provide pixel-level spatial guidance that is more fine-grained than what sparse point/box prompts can express.

```python
def forward(self, points=None, boxes=None, masks=None):
    sparse_embeddings = torch.empty((bs, 0, self.embed_dim))

    if points is not None:
        point_emb = self._embed_points(points, labels, pad=(boxes is None))
        sparse_embeddings = torch.cat([sparse_embeddings, point_emb], dim=1)

    if boxes is not None:
        box_emb = self._embed_boxes(boxes)
        sparse_embeddings = torch.cat([sparse_embeddings, box_emb], dim=1)

    if masks is not None:
        dense_embeddings = self._embed_masks(masks)
    else:
        dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
            bs, -1, *self.image_embedding_size
        )

    return sparse_embeddings, dense_embeddings
```

## No-Prompt Case

When no prompt is provided (used in automatic mask generation mode), the prompt encoder produces a special "no-prompt" embedding. For sparse embeddings, an empty token sequence is returned. For dense embeddings, a learned `no_mask_embed` embedding is broadcast across all spatial positions:

```python
# No-mask embedding: a single learned vector broadcast spatially
self.no_mask_embed = nn.Embedding(1, embed_dim)

# When masks=None:
dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
    bs, -1, image_embedding_size[0], image_embedding_size[1]
)
```

The `no_mask_embed` serves as a learned "default" spatial bias, telling the decoder "no mask guidance was provided." This is important for distinguishing between "the user provided an all-zeros mask" (which means "nothing is foreground") and "the user provided no mask at all" (which means "use your best judgment"). In automatic mask generation mode (grid-based point prompting across the entire image), the no-mask embedding is used for every candidate region.

## Prompt Combination

Multiple prompt types can be freely combined in a single prediction. The encoding process is additive -- all sparse embeddings are concatenated into a single token sequence, and the dense embedding is the mask embedding (or no-mask embedding if no mask is provided):

```python
# Example: box + 2 positive points + 1 negative point + mask
sparse = concat([
    box_embed,        # [B, 2, 256] - two corner tokens
    point_embed,      # [B, 3, 256] - three point tokens (2 pos + 1 neg)
])  # Total: [B, 5, 256]

dense = mask_embed   # [B, 256, 64, 64] - from input mask

# These are passed together to the mask decoder
masks, iou = mask_decoder(image_emb, sparse, dense)
```

The order of sparse tokens matters slightly because the self-attention in the transformer can develop position-dependent behavior during training. In practice, the order is: IoU token, mask output tokens, then prompt tokens (points and boxes). This consistent ordering helps the model learn to distinguish between its own output tokens and user-provided prompt tokens. Multiple refinement iterations are supported by feeding the previous prediction's mask back as the mask prompt for the next iteration.
