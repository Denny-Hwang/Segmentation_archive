---
title: "nnU-Net - Postprocessing"
date: 2025-01-15
status: complete
parent: "nnunet/repo_overview.md"
tags: [nnunet, postprocessing, ensembling, inference]
---

# nnU-Net Postprocessing

## Postprocessing Pipeline

nnU-Net's postprocessing pipeline is fully automatic and determined empirically on the validation set during cross-validation. After inference produces raw softmax predictions, the pipeline applies: (1) ensembling of multiple folds and/or configurations, (2) optional test-time augmentation (TTA), (3) resampling back to original image spacing, (4) argmax to convert probabilities to discrete labels, and (5) optional connected component filtering. Each step is designed to maximize the final evaluation metric (typically Dice score) without manual intervention.

The postprocessing logic lives in `nnunetv2/postprocessing/` and is orchestrated by the `determine_postprocessing` function, which evaluates all candidate postprocessing strategies on the cross-validation results and selects the one that yields the best mean Dice score. The selected strategy is saved alongside the trained model and applied automatically during inference with `nnUNetv2_predict`.

## Connected Component Analysis

Connected component analysis is applied per-class to remove small spurious predictions. For each foreground class, nnU-Net evaluates whether keeping only the largest connected component improves the Dice score on the validation set:

```python
# Simplified connected component logic
from scipy.ndimage import label as scipy_label

def apply_cc_filtering(prediction, class_id, min_size=None):
    binary_mask = (prediction == class_id)
    labeled_array, num_features = scipy_label(binary_mask)
    if num_features <= 1:
        return prediction
    # Find largest component
    component_sizes = np.bincount(labeled_array.ravel())[1:]  # Skip background
    largest_component = np.argmax(component_sizes) + 1
    # Remove all other components
    prediction[binary_mask & (labeled_array != largest_component)] = 0
    return prediction
```

The decision of whether to apply connected component filtering is made independently for each class. For some classes (e.g., liver, which is a single connected organ), keeping only the largest component consistently improves results by removing false positives in distant regions. For other classes (e.g., lymph nodes, which are naturally multi-instance), connected component filtering would harm performance and is automatically disabled. The planner stores the per-class filtering decisions in a `postprocessing.pkl` file.

nnU-Net also supports minimum component size thresholds: components smaller than a fraction of the expected organ size are removed. This is more conservative than keeping only the largest component and works well for structures that may have legitimate multi-component predictions.

## Ensembling

### Cross-Validation Ensemble

nnU-Net trains 5-fold cross-validation by default, producing 5 independently trained models. During inference, predictions from all 5 folds are averaged at the softmax probability level before applying argmax:

```python
# Fold ensembling (simplified)
ensemble_probs = np.zeros_like(fold_0_probs)
for fold_idx in range(5):
    model = load_model(f'fold_{fold_idx}/checkpoint_final.pth')
    fold_probs = predict_with_model(model, image)
    ensemble_probs += fold_probs
ensemble_probs /= 5.0
final_prediction = np.argmax(ensemble_probs, axis=0)
```

This approach is equivalent to averaging the posterior probabilities and is more robust than majority voting because it preserves confidence information. Regions where 3 out of 5 folds predict class A with high confidence and 2 predict class B with low confidence will correctly resolve to class A, whereas majority voting might give each fold equal weight regardless of confidence.

### Configuration Ensemble

nnU-Net can also ensemble across configurations (e.g., combining 2D and 3D_fullres predictions, or 3D_fullres and 3D_cascade). The `nnUNetv2_find_best_configuration` command evaluates all individual configurations and all pairwise configuration ensembles on the cross-validation results, selecting the combination that maximizes mean Dice:

```python
# Configuration ensembling
configs_to_try = ['2d', '3d_fullres', '3d_lowres', '3d_cascade_fullres']
best_score = 0
best_ensemble = None
for combo in all_pairs_and_singles(configs_to_try):
    score = evaluate_ensemble(combo, validation_predictions)
    if score > best_score:
        best_score = score
        best_ensemble = combo
```

Cross-configuration ensembling is powerful because different configurations capture complementary information: 2D models excel at in-plane detail, 3D_fullres captures local 3D context, and 3D_cascade provides global context. The ensemble typically outperforms any single configuration by 0.5-2% Dice.

## Test-Time Augmentation

nnU-Net applies mirroring-based test-time augmentation during inference. For 3D data, the input is flipped along each combination of axes (x, y, z), producing 8 augmented versions (including the original). Each version is passed through the model, the predictions are flipped back to the original orientation, and the results are averaged:

```python
# TTA with mirroring (3D case - 8 augmentations)
mirror_axes = [(0,), (1,), (2,), (0,1), (0,2), (1,2), (0,1,2)]
all_predictions = [model(image)]  # Original
for axes in mirror_axes:
    flipped_input = torch.flip(image, dims=[d+2 for d in axes])  # +2 for batch+channel
    pred = model(flipped_input)
    pred = torch.flip(pred, dims=[d+2 for d in axes])  # Flip prediction back
    all_predictions.append(pred)
avg_prediction = torch.stack(all_predictions).mean(dim=0)
```

For 2D data, only x and y mirroring is applied (4 augmentations). TTA is enabled by default during inference and typically improves Dice by 0.2-0.5%. It can be disabled with `--disable_tta` for faster inference at the cost of slightly lower accuracy.

nnU-Net does not use rotation or scaling TTA by default, as mirroring is computationally cheap (no interpolation needed) and empirically captures most of the TTA benefit. The limitation is that mirroring only helps with left-right/up-down symmetry, not rotational invariance.

## Resampling Back to Original Space

After inference produces predictions at the preprocessed resolution and spacing, the results must be mapped back to the original image space for clinical use:

```python
# Reverse resampling (simplified)
def resample_prediction_to_original(prediction, original_shape, original_spacing, target_spacing):
    # Resample from target_spacing back to original_spacing
    resampled = resample_data_or_seg_to_shape(
        prediction[None],  # Add channel dim
        original_shape,
        is_seg=True,  # Use nearest-neighbor
        order=0
    )
    # Pad back to original image size (undo cropping)
    result = np.zeros(full_original_shape, dtype=prediction.dtype)
    result[crop_bbox_slicing] = resampled
    return result
```

Nearest-neighbor interpolation is used for resampling predictions (as with labels during preprocessing) to maintain discrete class values. The cropping bounding box stored during preprocessing is used to place the prediction back into the correct spatial location within the full original image volume. The output is saved in the same format as the input (typically NIfTI), preserving the original affine matrix for correct spatial alignment.

## Automatic Postprocessing Selection

The automatic postprocessing selection (`determine_postprocessing`) is run after cross-validation completes. It exhaustively evaluates postprocessing strategies on the validation predictions:

1. **No postprocessing** (baseline): Evaluate raw argmax predictions
2. **Per-class connected component filtering**: For each foreground class independently, evaluate whether keeping only the largest connected component improves the mean Dice
3. **All-class connected component filtering**: Evaluate keeping only the largest connected component across all foreground classes simultaneously

The strategy that produces the highest mean Dice score on the validation set is selected and saved. This is a conservative approach: postprocessing is only applied when it demonstrably helps, avoiding the risk of removing valid predictions. The selected strategy is stored in `postprocessing.pkl` and automatically loaded during inference.

## Submission Formatting

For challenge submissions, nnU-Net produces prediction files in the same format as the input labels (typically NIfTI `.nii.gz`). The `nnUNetv2_predict` command handles the full pipeline from raw test images to final predictions:

```bash
nnUNetv2_predict -i INPUT_FOLDER -o OUTPUT_FOLDER \
    -d DATASET_ID -c CONFIGURATION \
    -f 0 1 2 3 4  # Use all 5 folds
    --save_probabilities  # Optional: save softmax probabilities for ensembling
```

When `--save_probabilities` is specified, both the final segmentation and the softmax probability maps are saved. The probability maps (stored as `.npz` files) can be used for later cross-configuration ensembling. For challenge submissions, only the final segmentation files (argmax of ensembled probabilities, with postprocessing applied) are needed. The output file names match the input file names, making it straightforward to submit predictions to challenge evaluation servers.
