---
title: "Segment Anything in Medical Images"
date: 2025-03-06
status: planned
tags: [foundation-model, medical-segmentation, domain-adaptation, sam]
difficulty: intermediate
---

# MedSAM

## Meta Information

| Field | Value |
|-------|-------|
| **Paper Title** | Segment Anything in Medical Images |
| **Authors** | Ma, J., He, Y., Li, F., Han, L., You, C., Wang, B. |
| **Year** | 2024 |
| **Venue** | Nature Communications |
| **arXiv** | N/A |
| **Difficulty** | Intermediate |

## One-Line Summary

MedSAM fine-tunes SAM on a large-scale medical image dataset spanning multiple modalities and anatomies, demonstrating that domain-specific adaptation significantly improves segmentation performance on medical images.

## Motivation and Problem Statement

While SAM demonstrated impressive zero-shot segmentation on natural images, its performance degraded substantially on medical images. Medical images differ fundamentally from natural photographs: they exhibit low contrast between structures, contain noise patterns specific to each imaging modality, and display anatomical structures with subtle boundaries that require domain expertise to delineate. Evaluations showed that SAM's zero-shot performance on medical benchmarks was 10-30 IoU points below specialized medical segmentation models like nnU-Net, making it impractical for clinical use without adaptation.

The authors identified a critical opportunity: SAM's architecture and learned visual representations could serve as a strong foundation for medical image segmentation if properly adapted. Rather than training a medical segmentation model from scratch, fine-tuning SAM on medical data could leverage the rich visual features learned from 1.1 billion natural image masks while adapting the model's output to the specific visual characteristics of medical imaging. This transfer learning approach promised better data efficiency and broader generalization across medical domains than training specialized models independently for each imaging modality.

## Architecture Overview

MedSAM retains SAM's three-component architecture (image encoder, prompt encoder, mask decoder) without structural modifications. The key change is in the training: the entire model is fine-tuned end-to-end on a curated dataset of 1,570,263 medical image-mask pairs covering 10 imaging modalities and over 30 anatomical structures. The prompt interface is simplified to bounding box prompts only, as boxes are the most reliable and informative prompt type and are straightforward for clinicians to provide. Point and text prompts are not used during MedSAM training or inference.

### Key Components

- **Medical Adaptation**: See [medical_adaptation.md](medical_adaptation.md)

## Technical Details

### Training Data

MedSAM was trained on a curated dataset of approximately 1.57 million medical image-mask pairs assembled from over 30 publicly available medical imaging datasets. The dataset spans 10 imaging modalities: CT (computed tomography), MRI (magnetic resonance imaging), ultrasound, X-ray, dermoscopy, endoscopy, fundus photography, mammography, OCT (optical coherence tomography), and pathology. CT and MRI contribute the largest share of training data, as 3D volumes are sliced into individual 2D images for training. The dataset covers over 30 major anatomical structures including organs (liver, kidney, spleen, heart), tumors, and vascular structures.

### Fine-Tuning Strategy

The entire SAM model (image encoder, prompt encoder, and mask decoder) is fine-tuned end-to-end using the medical training data. The ViT-B variant of SAM (91M parameters) was used as the base model rather than ViT-H, as it provides a better balance between performance and computational cost for the medical domain. Training used the AdamW optimizer with a learning rate of 1e-4, cosine learning rate decay, a batch size of 16, and ran for 100 epochs. The loss function combined binary cross-entropy loss and dice loss with equal weighting, following standard medical segmentation practice.

### Modality Coverage

The 10 supported modalities represent a broad cross-section of clinical imaging. CT provides high-resolution 3D anatomical detail and represents the largest portion of training data (approximately 40%). MRI covers T1, T2, FLAIR, and contrast-enhanced sequences across brain, cardiac, and abdominal regions. Ultrasound images present unique challenges including speckle noise and operator-dependent quality. Dermoscopy and fundus photography are specialized 2D modalities for skin and retinal imaging respectively. Pathology images at high magnification differ dramatically from radiological images, testing the model's ability to generalize across vastly different visual domains.

### Prompt Strategy for Medical Images

MedSAM exclusively uses bounding box prompts during both training and inference. During training, bounding boxes are derived from ground-truth masks by computing the tight bounding box and adding random perturbation (jittering by up to 20 pixels in each direction) to simulate imprecise clinical inputs. This jittering strategy is critical for robustness: it trains the model to handle boxes that are not perfectly tight, which is realistic for clinical use where radiologists may draw approximate boxes.

The choice of box-only prompts is deliberate. In clinical workflows, drawing a bounding box around a region of interest is a natural and fast interaction that clinicians are accustomed to. Point prompts were excluded because their sensitivity to placement is problematic in medical images where boundaries between structures can be ambiguous. Text prompts were excluded due to their limited maturity in SAM's original implementation.

## Experiments and Results

### Datasets and Modalities

MedSAM was evaluated on 86 internal validation datasets spanning all 10 training modalities, plus 19 external validation datasets from modalities and anatomies not seen during training. The internal benchmarks include standard challenges such as BTCV (abdominal CT), ACDC (cardiac MRI), ISIC (dermoscopy), and REFUGE (fundus photography). External benchmarks include unseen tumor types, rare anatomical structures, and imaging protocols not represented in the training set.

### Key Results

MedSAM achieved a mean dice score of 87.2% across the 86 internal validation datasets, substantially outperforming vanilla SAM (which achieved approximately 62.4% on the same benchmarks with box prompts). On the 19 external validation datasets, MedSAM achieved a mean dice of 78.9%, demonstrating meaningful generalization to unseen domains. Performance was strongest on CT (90.1% dice) and MRI (88.3% dice), moderate on ultrasound (82.5%) and dermoscopy (84.7%), and lowest on pathology (74.2%) and OCT (76.8%).

### Comparison with Vanilla SAM

The fine-tuning improved performance by approximately 25 dice points on average compared to zero-shot SAM. The improvement was largest on modalities most dissimilar from natural images: CT (+32 points), MRI (+28 points), and ultrasound (+26 points). On modalities closer to natural images such as dermoscopy (+15 points) and endoscopy (+18 points), the improvement was smaller but still substantial. These results demonstrate that SAM's learned visual features provide a useful starting point, but significant domain adaptation is necessary for medical applications.

## Strengths

MedSAM demonstrates that a single model can handle diverse medical imaging modalities, eliminating the need to train separate models for CT, MRI, ultrasound, etc. The bounding box prompt interface is clinician-friendly and integrates naturally into radiological workflows. The large-scale training dataset (1.57M image-mask pairs) ensures broad coverage of anatomical structures and imaging conditions. Published in Nature Communications, MedSAM provides a validated, reproducible baseline for medical image segmentation with foundation models.

## Limitations

MedSAM is limited to 2D segmentation, processing each slice independently without volumetric context, which is suboptimal for 3D imaging modalities (CT, MRI). The bounding box-only prompt interface is less flexible than SAM's full prompt suite; point prompts and iterative refinement are not supported. Performance on rare structures and uncommon imaging protocols remains below specialized models like nnU-Net that can be trained on task-specific data. The ViT-B backbone, while more efficient than ViT-H, still requires GPU inference, limiting deployment in resource-constrained clinical settings.

## Connections

MedSAM directly builds on SAM (Kirillov et al. 2023) by adapting its architecture to the medical domain through full fine-tuning. MedSAM-2 (Zhu et al. 2024) extends this approach by leveraging SAM 2's video capabilities for volumetric medical segmentation. SAM-Adapter (Chen et al. 2023) offers an alternative parameter-efficient adaptation strategy using adapter modules rather than full fine-tuning. The nnU-Net framework (Isensee et al. 2021) remains the primary comparison point for specialized medical segmentation. MedSAM's approach of large-scale multi-modality training connects to broader trends in medical foundation models.

## References

- Kirillov et al., "Segment Anything," ICCV 2023 (SAM foundation).
- Isensee et al., "nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation," Nature Methods 2021 (primary comparison).
- Ronneberger et al., "U-Net: Convolutional Networks for Biomedical Image Segmentation," MICCAI 2015 (foundational medical segmentation).
- He et al., "Masked Autoencoders Are Scalable Vision Learners," CVPR 2022 (MAE pre-training used in SAM).
- Zhu et al., "MedSAM-2: Segment Medical Images As Video Via Segment Anything Model 2," arXiv 2024 (successor work).
