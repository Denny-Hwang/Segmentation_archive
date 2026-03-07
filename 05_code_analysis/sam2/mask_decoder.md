---
title: "SAM 2 - Mask Decoder Analysis"
date: 2025-01-15
status: complete
parent: "sam2/repo_overview.md"
tags: [sam2, mask-decoder, transformer, segmentation-head]
---

# SAM 2 Mask Decoder

## Overview

The mask decoder in SAM 2, defined in `sam2/modeling/sam/mask_decoder.py`, is a lightweight transformer-based module that takes image embeddings (optionally conditioned on memory), prompt embeddings, and output tokens, and produces segmentation masks along with IoU (Intersection over Union) quality scores. The decoder is intentionally small (only 2 transformer layers by default) because the heavy lifting is done by the image encoder and memory attention; the decoder's job is to combine prompt information with image features to produce the final mask.

The decoder follows the same two-way attention design introduced in SAM 1, where both the output tokens attend to the image embeddings and the image embeddings attend to the output tokens. This bidirectional attention allows the decoder to refine both the mask tokens (which will be projected to masks) and the image features (which will be spatially decoded into pixel-level predictions) simultaneously.

## Architecture

### Transformer Decoder Layers

The mask decoder uses a `TwoWayTransformer` with the following configuration:

```python
class MaskDecoder(nn.Module):
    def __init__(self, transformer_dim=256, num_multimask_outputs=3, ...):
        self.transformer = TwoWayTransformer(
            depth=2,            # 2 transformer layers
            embedding_dim=256,  # Token dimension
            num_heads=8,        # Attention heads
            mlp_dim=2048,       # MLP hidden dimension
        )
        # Output tokens: learnable embeddings for each mask
        self.num_mask_tokens = num_multimask_outputs + 1  # 3 multi + 1 single
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)
        self.iou_token = nn.Embedding(1, transformer_dim)
```

Each transformer layer consists of four sub-operations in sequence: (1) self-attention on the token sequence (prompt tokens + output tokens), (2) cross-attention from tokens to image embeddings, (3) a feed-forward MLP on the tokens, and (4) cross-attention from image embeddings back to the tokens. This "two-way" design (steps 2 and 4) is the distinguishing feature of the decoder architecture.

### Two-Way Attention

The two-way attention mechanism allows information to flow in both directions between the sparse token sequence and the dense image embedding:

```python
class TwoWayAttentionBlock(nn.Module):
    def forward(self, queries, keys, query_pe, key_pe):
        # queries = prompt + output tokens, keys = image embeddings

        # 1. Self-attention on queries (tokens attend to each other)
        q = queries + query_pe
        attn_out = self.self_attn(q=q, k=q, v=queries)
        queries = queries + attn_out

        # 2. Cross-attention: tokens attend to image
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out

        # 3. MLP on tokens
        queries = queries + self.mlp(queries)

        # 4. Cross-attention: image attends to tokens
        q = keys + key_pe
        k = queries + query_pe
        attn_out = self.cross_attn_image_to_token(q=q, k=k, v=queries)
        keys = keys + attn_out

        return queries, keys
```

Step 4 (image-to-token attention) is crucial: it allows the image embedding to be spatially modified based on the prompt information. For example, if a positive point prompt is provided, the image features near that point are enhanced while features far from the prompt are suppressed. This creates a spatially-aware image embedding that, when upsampled, naturally produces a mask centered on the prompted region.

### Multi-Mask Output

The decoder produces multiple mask candidates to handle ambiguous prompts. When a single point click is provided, it could refer to multiple valid objects (a part vs. the whole, an object vs. a group). The decoder addresses this by generating 3 mask candidates plus 1 "single-mask" prediction:

```python
class MaskDecoder(nn.Module):
    def predict_masks(self, image_embeddings, image_pe, sparse_prompt_embeddings,
                      dense_prompt_embeddings):
        # Concatenate output tokens with prompt tokens
        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)
        tokens = torch.cat([output_tokens, sparse_prompt_embeddings], dim=1)

        # Run two-way transformer
        hs, src = self.transformer(tokens, image_embeddings)

        # Extract mask tokens and IoU token from output
        iou_token_out = hs[:, 0, :]           # IoU prediction token
        mask_tokens_out = hs[:, 1:1+self.num_mask_tokens, :]  # Mask tokens

        # Upscale image features: 64x64 -> 256x256
        src = src.transpose(1, 2).view(b, c, h, w)
        upscaled = self.output_upscaling(src)  # ConvTranspose2d stack

        # Generate masks via dot product: mask_tokens x upscaled_features
        masks = []
        for i in range(self.num_mask_tokens):
            mask = (mask_tokens_out[:, i, :] @ self.output_hypernetworks_mlps[i](
                mask_tokens_out[:, i, :])).view(b, 1, h*4, w*4)
            masks.append(mask)
        masks = torch.stack(masks, dim=1)  # [B, num_masks, H*4, W*4]

        return masks, iou_token_out
```

The 4 output masks correspond to: mask 0 (whole object), mask 1 (part-level), mask 2 (sub-part-level), and mask 3 (single-mask mode, used when `multimask_output=False`). During training, all masks are supervised, but during inference, the mask with the highest predicted IoU is typically selected.

## Mask Prediction Head

The mask prediction head converts transformer output tokens into spatial mask predictions through a combination of upsampling and hypernetwork-based projection:

```python
# Output upscaling: 4x spatial upsampling via transposed convolutions
self.output_upscaling = nn.Sequential(
    nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
    LayerNorm2d(transformer_dim // 4),
    nn.GELU(),
    nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
)

# Per-mask hypernetwork MLPs
self.output_hypernetworks_mlps = nn.ModuleList([
    MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
    for _ in range(self.num_mask_tokens)
])
```

The upscaled image features have shape `[B, 32, 256, 256]` (for 1024x1024 input). Each mask token is projected through its own MLP to produce a 32-dimensional vector, which is then used as a per-pixel classifier via dot product with the upscaled features. This hypernetwork approach allows each mask to have its own spatial classification weights without explicitly storing a full spatial mask.

## IoU Prediction Head

The IoU prediction head estimates the quality of each generated mask, enabling automatic mask selection during inference:

```python
self.iou_prediction_head = MLP(transformer_dim, iou_head_hidden_dim,
                                self.num_mask_tokens, iou_head_depth)

# During forward:
iou_pred = self.iou_prediction_head(iou_token_out)  # [B, num_mask_tokens]
```

The IoU head is a 3-layer MLP that takes the IoU output token (refined through the transformer) and predicts a scalar IoU score for each of the mask candidates. During training, the predicted IoU is supervised with the actual IoU between the predicted mask and ground truth. During inference, the mask with the highest predicted IoU is automatically selected as the best output. This self-scoring mechanism is essential for the promptable segmentation paradigm, where the model must autonomously judge which of its multiple outputs is most appropriate.

## Ambiguity Resolution

SAM 2's multi-mask output is specifically designed to handle prompt ambiguity. A single point click on an image is inherently ambiguous -- it could refer to a small sub-part, a larger part, or the entire object. Rather than forcing the model to make a single prediction, the decoder produces three masks at different granularity levels:

```python
# During inference with a single point prompt:
masks, iou_scores, _ = mask_decoder(
    image_embeddings=img_emb,
    sparse_prompt_embeddings=point_emb,
    multimask_output=True  # Returns 3 masks
)
# masks shape: [B, 3, 256, 256]
# iou_scores shape: [B, 3]

# Select best mask based on predicted IoU
best_mask_idx = iou_scores.argmax(dim=1)
best_mask = masks[torch.arange(B), best_mask_idx]
```

When multiple prompts are provided (e.g., two points, or a box), ambiguity is reduced and the model switches to single-mask mode (`multimask_output=False`), using the 4th mask token that is trained specifically for unambiguous scenarios. This heuristic is applied in the predictor classes: if the number of prompt points exceeds 1, or if a box prompt is provided, single-mask mode is used automatically.

## Comparison with SAM 1 Decoder

SAM 2's mask decoder closely follows SAM 1's design, with several refinements:

The core architecture (two-way transformer, multi-mask output, IoU prediction) is essentially identical between SAM 1 and SAM 2. The `TwoWayTransformer` class is reused with minimal changes. The key differences are in how the decoder interacts with the rest of the system rather than in the decoder itself. In SAM 2, the decoder receives memory-conditioned features (from the memory attention module) instead of raw encoder features, enabling temporal consistency in video mode. Additionally, SAM 2's decoder outputs an "object pointer" token that is stored in the memory bank for future frame conditioning, which is a new output not present in SAM 1.

SAM 2 also adds support for occlusion prediction: the decoder includes an additional output head that predicts whether the tracked object is occluded in the current frame. When occlusion is detected, the model can suppress the mask output and skip memory updates for that frame, preventing error propagation through the memory bank.
