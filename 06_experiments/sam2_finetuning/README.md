---
title: "Experiment: SAM 2 Fine-Tuning"
date: 2025-01-15
status: planned
tags: [experiment, sam2, fine-tuning, foundation-model, transfer-learning]
---

# SAM 2 Fine-Tuning Experiment

## Objective

Evaluate the effectiveness of fine-tuning SAM 2 on a domain-specific segmentation dataset,
comparing zero-shot performance, mask decoder fine-tuning, LoRA-based adaptation, and full
fine-tuning strategies. The goal is to determine the most cost-effective approach for adapting
a foundation segmentation model to specialized domains where zero-shot performance is
insufficient.

---

## Background

SAM 2 (Segment Anything Model 2) by Ravi et al. (2024) extends the original SAM to handle
both image and video segmentation. It consists of three main components:

1. **Image Encoder:** A Vision Transformer (Hiera) that produces image embeddings. This is
   the largest component and contains the bulk of the learned visual representations.
2. **Prompt Encoder:** Encodes sparse prompts (points, boxes) and dense prompts (masks) into
   embedding vectors.
3. **Mask Decoder:** A lightweight transformer decoder that combines image embeddings and
   prompt embeddings to produce output masks and confidence scores.

SAM 2 performs remarkably well zero-shot on natural images but often struggles on specialized
domains such as medical imaging, satellite imagery, and industrial inspection, where the
visual characteristics differ substantially from the web-scale training data.

---

## Fine-Tuning Strategies

| Strategy | Frozen Components | Trainable Components | Expected GPU Memory | Training Speed |
|----------|------------------|---------------------|-------------------|----------------|
| Zero-shot | All | None | ~4 GB (inference only) | N/A |
| Decoder-only | Image encoder, prompt encoder | Mask decoder (~4M params) | ~8 GB | Fast (~1 hr) |
| LoRA (rank=4) | Most of image encoder | LoRA adapters + mask decoder (~6M params) | ~12 GB | Moderate (~3 hrs) |
| LoRA (rank=16) | Most of image encoder | LoRA adapters + mask decoder (~10M params) | ~14 GB | Moderate (~4 hrs) |
| Partial encoder FT | Early encoder layers | Last 6 encoder blocks + decoder (~50M params) | ~20 GB | Slow (~8 hrs) |
| Full fine-tuning | None | All (~300M params) | ~24+ GB | Slowest (~12 hrs) |

### Strategy 1: Decoder-Only Fine-Tuning

The simplest approach. Freeze the image encoder and prompt encoder entirely, and only update
the mask decoder weights. This works well when the domain shift is moderate---the encoder
features are still informative, but the decoder needs to learn new mask patterns.

```python
# Freeze encoder
for param in model.image_encoder.parameters():
    param.requires_grad = False
for param in model.prompt_encoder.parameters():
    param.requires_grad = False

# Only decoder is trainable
for param in model.mask_decoder.parameters():
    param.requires_grad = True
```

### Strategy 2: LoRA-Based Adaptation

Low-Rank Adaptation (LoRA) injects small trainable rank decomposition matrices into the
attention layers of the image encoder. This allows the encoder to adapt to the new domain
without modifying the original weights, preserving the pretrained representations while
adding domain-specific capability.

```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=4,                          # Rank of the adaptation matrices
    lora_alpha=8,                 # Scaling factor
    target_modules=[              # Apply LoRA to attention projections
        "qkv",                    # Query-Key-Value projection in Hiera
        "proj",                   # Output projection
    ],
    lora_dropout=0.1,
    bias="none",
)

model.image_encoder = get_peft_model(model.image_encoder, lora_config)

# Also unfreeze the mask decoder
for param in model.mask_decoder.parameters():
    param.requires_grad = True
```

**Rank selection guidance:**
- Rank 4: Minimal adaptation, best for small domain shifts. ~2M additional parameters.
- Rank 8: Moderate adaptation, good default choice. ~4M additional parameters.
- Rank 16: Aggressive adaptation, for large domain shifts. ~8M additional parameters.
- Rank 32+: Approaching full fine-tuning capacity; diminishing returns.

### Strategy 3: Full Fine-Tuning

All parameters are trainable. Use a lower learning rate for the encoder (1e-6 to 1e-5) and
a higher rate for the decoder (1e-4) to avoid catastrophic forgetting of pretrained features.

```python
# Differential learning rates
optimizer = torch.optim.AdamW([
    {"params": model.image_encoder.parameters(), "lr": 1e-6},
    {"params": model.prompt_encoder.parameters(), "lr": 1e-5},
    {"params": model.mask_decoder.parameters(), "lr": 1e-4},
], weight_decay=0.01)
```

---

## Dataset Preparation

### Recommended Domains

Choose a domain where SAM 2 zero-shot performance is known to be limited:

| Domain | Dataset | Images | Classes | Why SAM Struggles |
|--------|---------|--------|---------|-------------------|
| Medical (polyps) | Kvasir-SEG | 1,000 | 1 (binary) | Low contrast, ambiguous boundaries |
| Medical (organs) | Synapse Multi-Organ | ~3,700 slices | 8 | Grayscale, subtle tissue boundaries |
| Satellite | iSAID | 2,806 | 15 | Overhead perspective, tiny objects |
| Industrial | MVTec AD | ~5,000 | 1 (anomaly) | Subtle surface defects |

### Primary Dataset: Kvasir-SEG (Gastrointestinal Polyp Segmentation)

- **Images:** 1,000 colonoscopy images with pixel-level polyp annotations
- **Resolution:** Variable (original), resize to 1024x1024 for SAM 2
- **Split:** 800 train / 100 val / 100 test
- **Source:** [SimulaLab Kvasir-SEG](https://datasets.simula.no/kvasir-seg/)

### Data Format for SAM 2

SAM 2 expects inputs in a specific format. The dataloader must provide:

```python
class SAM2Dataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.images = sorted(glob(f"{image_dir}/*.jpg"))
        self.masks = sorted(glob(f"{mask_dir}/*.png"))
        self.transform = transform

    def __getitem__(self, idx):
        image = cv2.imread(self.images[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks[idx], cv2.IMREAD_GRAYSCALE)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image, mask = augmented["image"], augmented["mask"]

        # Generate prompts from ground truth mask
        # Option A: Random point prompts
        foreground_coords = np.argwhere(mask > 0)
        if len(foreground_coords) > 0:
            point_idx = np.random.randint(len(foreground_coords))
            point_coord = foreground_coords[point_idx]  # (y, x)
            point_label = 1  # foreground
        else:
            point_coord = np.array([mask.shape[0] // 2, mask.shape[1] // 2])
            point_label = 0  # background

        # Option B: Bounding box prompt from mask
        ys, xs = np.where(mask > 0)
        if len(ys) > 0:
            bbox = np.array([xs.min(), ys.min(), xs.max(), ys.max()])
            # Add random jitter to simulate imprecise prompts
            jitter = np.random.randint(-10, 10, size=4)
            bbox = np.clip(bbox + jitter, 0, max(mask.shape))
        else:
            bbox = np.array([0, 0, mask.shape[1], mask.shape[0]])

        return {
            "image": image,
            "mask": mask,
            "point_coords": point_coord,
            "point_labels": point_label,
            "bbox": bbox,
        }
```

### Augmentation Pipeline

```python
train_transform = A.Compose([
    A.Resize(1024, 1024),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
    A.GaussianBlur(blur_limit=(3, 7), p=0.2),
    # Note: Do NOT apply normalization here -- SAM 2 has its own preprocessor
])

val_transform = A.Compose([
    A.Resize(1024, 1024),
])
```

---

## Training Configuration

```yaml
experiment:
  name: sam2_finetuning
  seed: 42
  device: cuda

model:
  checkpoint: "facebook/sam2-hiera-large"  # or base, small, tiny
  strategy: lora  # Options: decoder_only, lora, partial_encoder, full
  lora_rank: 8
  lora_alpha: 16
  lora_dropout: 0.1

dataset:
  name: kvasir_seg
  root: data/kvasir_seg/
  image_size: 1024
  train_split: 0.8
  val_split: 0.1
  test_split: 0.1
  prompt_type: point  # Options: point, box, point_and_box

training:
  epochs: 50
  batch_size: 2          # SAM 2 is memory-intensive at 1024x1024
  accumulation_steps: 8  # Effective batch size = 16
  optimizer: adamw
  lr: 5e-5               # LoRA/decoder LR
  encoder_lr: 1e-6       # Only used for full/partial fine-tuning
  weight_decay: 0.01
  scheduler: cosine
  warmup_epochs: 5
  loss: bce_dice          # 0.5 * BCE + 0.5 * Dice
  mixed_precision: true
  early_stopping_patience: 10

evaluation:
  metrics: [dice, iou, hausdorff95, boundary_f1]
  prompt_types: [point_1, point_3, box, point_and_box]
```

---

## Evaluation

### Metrics

| Metric | Description |
|--------|-------------|
| Dice coefficient | Primary metric, standard for medical segmentation |
| IoU (Jaccard) | Complementary overlap metric |
| Hausdorff Distance (95th) | Boundary quality metric |
| Boundary F1 (2px tolerance) | Boundary precision metric |

### Evaluation Protocol

Each model is evaluated under multiple prompt conditions to assess prompt sensitivity:

1. **Single point (foreground center).** One point at the centroid of the ground truth mask.
2. **Three points (foreground).** Three random foreground points.
3. **Bounding box.** Tight bounding box around the ground truth mask.
4. **Box + point.** Bounding box plus one foreground point.
5. **No prompt (automatic mode).** Generate all masks without prompts (SAM 2 automatic mode).

### Prompt Sensitivity Analysis

For each test image, evaluate with 10 different random point placements. Report mean and
standard deviation of Dice score. A well-fine-tuned model should have lower variance (less
sensitive to exact prompt placement).

### Baselines

| Baseline | Description |
|----------|-------------|
| SAM 2 zero-shot (point) | Unmodified SAM 2 with 1 foreground point prompt |
| SAM 2 zero-shot (box) | Unmodified SAM 2 with bounding box prompt |
| U-Net (trained from scratch) | Standard U-Net trained on same data, no prompts |
| U-Net + ResNet34 (pretrained) | Pretrained encoder U-Net on same data, no prompts |

---

## Expected Results

### Kvasir-SEG Polyp Segmentation (Dice Score)

| Method | Point (1) | Point (3) | Box | Box + Point |
|--------|-----------|-----------|-----|-------------|
| SAM 2 zero-shot | ~0.72 | ~0.78 | ~0.82 | ~0.84 |
| SAM 2 decoder-only FT | ~0.82 | ~0.86 | ~0.88 | ~0.89 |
| SAM 2 LoRA (r=4) | ~0.85 | ~0.88 | ~0.90 | ~0.91 |
| SAM 2 LoRA (r=16) | ~0.86 | ~0.89 | ~0.91 | ~0.92 |
| SAM 2 full FT | ~0.87 | ~0.90 | ~0.91 | ~0.92 |
| U-Net (scratch) | ~0.83 (no prompt) | - | - | - |
| U-Net + ResNet34 | ~0.87 (no prompt) | - | - | - |

**Expected conclusions:**

1. Zero-shot SAM 2 underperforms trained-from-scratch U-Net on this medical domain, but
   even decoder-only fine-tuning closes most of the gap.
2. LoRA (r=4-8) achieves near-full-fine-tuning performance at a fraction of the compute and
   memory cost, making it the recommended default strategy.
3. Full fine-tuning provides marginal improvement over LoRA but requires 2-3x more GPU
   memory and training time, with increased risk of overfitting on small datasets.
4. Prompted models (SAM 2) can outperform unprompted models (U-Net) when box prompts are
   available, as the prompt constrains the search space.

---

## How to Run

```bash
# Step 1: Prepare dataset
python prepare_data.py --dataset kvasir_seg --output data/kvasir_seg/

# Step 2: Evaluate zero-shot baseline
python evaluate_zeroshot.py \
    --checkpoint facebook/sam2-hiera-large \
    --dataset data/kvasir_seg/ \
    --prompt-type point box \
    --output results/zeroshot/

# Step 3: Fine-tune with different strategies
for strategy in decoder_only lora_r4 lora_r8 lora_r16 full; do
    python finetune_sam2.py \
        --strategy $strategy \
        --config configs/${strategy}.yaml \
        --dataset data/kvasir_seg/ \
        --output results/${strategy}/
done

# Step 4: Train baselines
python train_unet.py --dataset data/kvasir_seg/ --output results/unet_baseline/

# Step 5: Evaluate all models
python evaluate_all.py --results-dir results/ --output analysis/

# Step 6: Prompt sensitivity analysis
python prompt_sensitivity.py --model results/lora_r8/ --dataset data/kvasir_seg/ --n-trials 10

# Step 7: Generate comparison visualizations
python visualize_results.py --results-dir results/ --num-samples 15
```

---

## Implementation Notes

### Installing SAM 2

```bash
pip install segment-anything-2
# or from source:
git clone https://github.com/facebookresearch/segment-anything-2.git
cd segment-anything-2 && pip install -e .
```

### Loading and Modifying the Model

```python
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# Load pretrained model
sam2_model = build_sam2("sam2_hiera_l.yaml", "checkpoints/sam2_hiera_large.pt")
predictor = SAM2ImagePredictor(sam2_model)

# For fine-tuning, access components directly:
image_encoder = sam2_model.image_encoder    # Hiera backbone
prompt_encoder = sam2_model.sam_prompt_encoder
mask_decoder = sam2_model.sam_mask_decoder
```

### Training Loop Skeleton

```python
for epoch in range(num_epochs):
    model.train()
    for batch in train_loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        points = batch["point_coords"].to(device)
        labels = batch["point_labels"].to(device)

        # Forward pass
        with torch.cuda.amp.autocast():
            image_embeddings = model.image_encoder(images)
            sparse_emb, dense_emb = model.sam_prompt_encoder(
                points=(points, labels), boxes=None, masks=None
            )
            pred_masks, iou_preds = model.sam_mask_decoder(
                image_embeddings=image_embeddings,
                image_pe=model.sam_prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_emb,
                dense_prompt_embeddings=dense_emb,
                multimask_output=False,
            )

            # Compute loss
            loss = dice_bce_loss(pred_masks, masks)

        # Backward pass
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
```

### Common Pitfalls

- **Input preprocessing.** SAM 2 expects images normalized with its own pixel mean/std. Do
  not apply ImageNet normalization manually---use the built-in preprocessor.
- **Prompt coordinate format.** Points are in (x, y) format, not (row, col). Bounding boxes
  are in (x1, y1, x2, y2) format.
- **Multi-mask output.** SAM 2 can output multiple masks per prompt. During fine-tuning, set
  `multimask_output=False` for single-mask prediction, or use the mask with the highest IoU
  prediction score.
- **Memory management.** At 1024x1024, SAM 2 is memory-intensive. Use gradient accumulation,
  mixed precision, and gradient checkpointing to fit within GPU memory.

---

## References

1. Ravi, N., et al. (2024). SAM 2: Segment Anything in Images and Videos. arXiv:2408.00714.
2. Kirillov, A., et al. (2023). Segment Anything. ICCV 2023.
3. Hu, E. J., et al. (2022). LoRA: Low-Rank Adaptation of Large Language Models. ICLR 2022.
4. Jha, D., et al. (2020). Kvasir-SEG: A Segmented Polyp Dataset. MMM 2020.
5. Ma, J., et al. (2024). Segment Anything in Medical Images. Nature Communications, 15, 654.
6. Zhang, K., & Liu, D. (2023). Customized Segment Anything Model for Medical Image
   Segmentation. arXiv:2304.13785.
