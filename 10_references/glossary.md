# Glossary of Segmentation Terms

An alphabetical reference of key terms used in image segmentation research.

---

**Attention Mechanism**: A neural network component that learns to weight different spatial or channel features by importance, allowing the model to focus on relevant regions. Used in Attention U-Net, transformers, etc.

**Atrous Convolution (Dilated Convolution)**: A convolution with gaps (dilation) between kernel elements, enlarging the receptive field without increasing parameters. Core component of DeepLab architectures.

**Backbone**: The encoder network (e.g., ResNet, Swin Transformer) that extracts feature representations from input images. Typically pretrained on ImageNet.

**Boundary Loss**: A loss function that penalizes errors at object boundaries rather than region overlap. Useful for thin or elongated structures.

**Class Imbalance**: A common segmentation problem where some classes occupy far fewer pixels than others, leading to biased predictions toward majority classes.

**Cross-Entropy Loss**: A standard classification loss applied per-pixel in segmentation. Often combined with Dice loss for improved performance.

**Decoder**: The upsampling path of an encoder-decoder architecture that gradually recovers spatial resolution from compressed feature maps.

**Dice Coefficient (F1 Score)**: A spatial overlap metric defined as 2|A intersection B| / (|A| + |B|). Ranges from 0 (no overlap) to 1 (perfect overlap).

**Dice Loss**: A loss function derived from the Dice coefficient: 1 - Dice. Effective for class-imbalanced segmentation tasks.

**Encoder**: The downsampling path that progressively extracts hierarchical features while reducing spatial resolution.

**Encoder-Decoder Architecture**: A network design with a contracting encoder and expanding decoder, connected by a bottleneck. U-Net is the canonical example.

**Feature Pyramid Network (FPN)**: A multi-scale feature extraction architecture that builds a top-down pathway with lateral connections for detecting objects at different scales.

**Foundation Model**: A large-scale model pretrained on broad data that can be adapted to many downstream tasks. SAM is a foundation model for segmentation.

**Ground Truth**: The reference annotation (mask) that represents the correct segmentation, typically created by human experts.

**Hausdorff Distance (HD95)**: A boundary-based metric measuring the maximum distance between predicted and ground truth boundaries. HD95 uses the 95th percentile to reduce sensitivity to outliers.

**Ignore Index**: A special label value (typically 255) used to mark pixels that should be excluded from loss computation during training.

**Instance Segmentation**: The task of detecting and delineating each individual object instance with a unique mask. Distinguishes between separate objects of the same class.

**IoU (Intersection over Union)**: Also called Jaccard Index. Computed as |A intersection B| / |A union B|. The standard segmentation evaluation metric.

**Latent Space**: The compressed feature representation at the bottleneck of an encoder-decoder network.

**Mask**: A pixel-level annotation assigning each pixel to a class or object. Can be binary (foreground/background) or multi-class.

**Mean IoU (mIoU)**: The average IoU computed across all classes. The primary metric for semantic segmentation benchmarks.

**Multi-Scale Processing**: Techniques that process images at multiple resolutions to capture both fine details and global context.

**Panoptic Quality (PQ)**: The evaluation metric for panoptic segmentation, combining recognition quality (RQ) and segmentation quality (SQ).

**Panoptic Segmentation**: A unified segmentation task that assigns both a semantic label and an instance ID to every pixel, combining semantic and instance segmentation.

**Patch-Based Training**: Processing image patches (crops) rather than full images during training, commonly used in medical imaging for high-resolution volumes.

**Pixel Accuracy**: The fraction of correctly classified pixels. A simple but often misleading metric due to class imbalance sensitivity.

**Pooling**: A downsampling operation (max pooling, average pooling) that reduces spatial dimensions while retaining important features.

**Promptable Segmentation**: A segmentation paradigm (introduced by SAM) where the model segments objects based on user prompts such as points, bounding boxes, or text.

**Receptive Field**: The region of the input image that influences a particular neuron's output. Larger receptive fields capture more context.

**Semantic Segmentation**: The task of assigning a class label to every pixel in an image. Does not distinguish between instances of the same class.

**Skip Connection**: A direct connection between encoder and decoder layers that preserves spatial information lost during downsampling. A defining feature of U-Net.

**Softmax**: An activation function that converts logits to a probability distribution over classes. Applied per-pixel in segmentation to produce class probabilities.

**Stuff vs. Things**: In panoptic segmentation, "things" are countable objects (car, person) and "stuff" is amorphous regions (sky, road, grass).

**Transposed Convolution (Deconvolution)**: A learnable upsampling operation used in decoders to increase spatial resolution.

**Vision Transformer (ViT)**: A transformer architecture that processes images as sequences of patches, applying self-attention across patch tokens.

**Volumetric Segmentation**: 3D segmentation of volumetric data such as CT or MRI scans, where predictions are made for each voxel.

**Zero-Shot Segmentation**: The ability to segment objects from classes not seen during training, typically enabled by foundation models or language-guided approaches.
