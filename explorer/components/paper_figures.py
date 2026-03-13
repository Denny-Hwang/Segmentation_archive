"""Paper figure / key architecture diagrams for visual review.

Provides Mermaid-based architecture diagrams for well-known segmentation papers
so reviewers can quickly see the high-level structure.
"""

import streamlit as st
from pathlib import Path

try:
    from streamlit_mermaid import st_mermaid

    HAS_MERMAID = True
except ImportError:
    HAS_MERMAID = False

# Map arXiv IDs to key architecture diagrams (Mermaid) and captions
PAPER_FIGURES: dict[str, list[dict[str, str]]] = {
    "1505.04597": [  # U-Net
        {
            "caption": "Fig 1. U-Net encoder-decoder architecture with skip connections",
            "mermaid": """graph TD
    Input["Input Image<br/>572×572"] --> E1["Conv 3×3 + ReLU ×2<br/>64 ch"]
    E1 --> Pool1["Max Pool 2×2"]
    Pool1 --> E2["Conv 3×3 + ReLU ×2<br/>128 ch"]
    E2 --> Pool2["Max Pool 2×2"]
    Pool2 --> E3["Conv 3×3 + ReLU ×2<br/>256 ch"]
    E3 --> Pool3["Max Pool 2×2"]
    Pool3 --> E4["Conv 3×3 + ReLU ×2<br/>512 ch"]
    E4 --> Pool4["Max Pool 2×2"]
    Pool4 --> BN["Bottleneck<br/>1024 ch"]
    BN --> Up4["Up-conv 2×2"]
    Up4 --> D4["Conv 3×3 + ReLU ×2<br/>512 ch"]
    D4 --> Up3["Up-conv 2×2"]
    Up3 --> D3["Conv 3×3 + ReLU ×2<br/>256 ch"]
    D3 --> Up2["Up-conv 2×2"]
    Up2 --> D2["Conv 3×3 + ReLU ×2<br/>128 ch"]
    D2 --> Up1["Up-conv 2×2"]
    Up1 --> D1["Conv 3×3 + ReLU ×2<br/>64 ch"]
    D1 --> Out["1×1 Conv → Segmentation Map"]
    E1 -.->|skip connection| D1
    E2 -.->|skip connection| D2
    E3 -.->|skip connection| D3
    E4 -.->|skip connection| D4

    style BN fill:#FFD700,stroke:#333
    style Out fill:#50C878,stroke:#333""",
        },
    ],
    "1411.4038": [  # FCN
        {
            "caption": "Fig 1. FCN: Fully Convolutional Network for pixel-wise prediction",
            "mermaid": """graph LR
    Input["Input Image"] --> Conv1["Conv1<br/>+ Pool"]
    Conv1 --> Conv2["Conv2<br/>+ Pool"]
    Conv2 --> Conv3["Conv3<br/>+ Pool"]
    Conv3 --> Conv4["Conv4<br/>+ Pool"]
    Conv4 --> Conv5["Conv5<br/>+ Pool"]
    Conv5 --> FC6["Conv 1×1<br/>(FC6)"]
    FC6 --> FC7["Conv 1×1<br/>(FC7)"]
    FC7 --> Score["Score<br/>1×1 Conv"]
    Score --> Up32["Upsample ×32"]
    Up32 --> Output["FCN-32s"]
    Conv4 -.->|"+ score"| Up16["Fuse + Up ×16"]
    Score --> Up16
    Up16 --> Out16["FCN-16s"]
    Conv3 -.->|"+ score"| Up8["Fuse + Up ×8"]
    Up16 --> Up8
    Up8 --> Out8["FCN-8s"]

    style Out8 fill:#50C878,stroke:#333
    style Score fill:#FFD700,stroke:#333""",
        },
    ],
    "1412.7062": [  # DeepLab v1
        {
            "caption": "Fig 1. DeepLab: Atrous convolution + CRF post-processing",
            "mermaid": """graph LR
    Input["Input Image"] --> VGG["VGG-16 Backbone<br/>(atrous conv)"]
    VGG --> ASPP["Atrous Spatial<br/>Pyramid Pooling"]
    ASPP --> Score["Coarse<br/>Score Map"]
    Score --> Bilinear["Bilinear<br/>Upsampling"]
    Bilinear --> CRF["Dense CRF<br/>Post-processing"]
    CRF --> Output["Final<br/>Segmentation"]

    style ASPP fill:#DA70D6,stroke:#333
    style CRF fill:#FFD700,stroke:#333
    style Output fill:#50C878,stroke:#333""",
        },
    ],
    "1807.10165": [  # UNet++
        {
            "caption": "Fig 1. UNet++ nested dense skip connections",
            "mermaid": """graph TD
    X00["X⁰·⁰"] --> X10["X¹·⁰"]
    X10 --> X20["X²·⁰"]
    X20 --> X30["X³·⁰"]
    X30 --> X40["X⁴·⁰"]
    X40 --> X31["X³·¹"]
    X30 --> X31
    X31 --> X22["X²·²"]
    X20 --> X21["X²·¹"]
    X21 --> X22
    X22 --> X13["X¹·³"]
    X10 --> X11["X¹·¹"]
    X11 --> X12["X¹·²"]
    X12 --> X13
    X13 --> X04["X⁰·⁴"]
    X00 --> X01["X⁰·¹"]
    X01 --> X02["X⁰·²"]
    X02 --> X03["X⁰·³"]
    X03 --> X04
    X04 --> Out["Output"]

    style X40 fill:#FFD700,stroke:#333
    style Out fill:#50C878,stroke:#333""",
        },
    ],
    "1804.03999": [  # Attention U-Net
        {
            "caption": "Fig 1. Attention Gate mechanism in skip connections",
            "mermaid": """graph TD
    subgraph Attention Gate
        G["Gating Signal g"] --> Wg["Wg × g"]
        X["Skip Feature x"] --> Wx["Wx × x"]
        Wg --> Add["+ Add"]
        Wx --> Add
        Add --> ReLU["ReLU σ₁"]
        ReLU --> Psi["ψ Conv"]
        Psi --> Sigmoid["Sigmoid σ₂"]
        Sigmoid --> Alpha["α attention<br/>coefficients"]
    end
    X --> Mul["× Multiply"]
    Alpha --> Mul
    Mul --> Output["Attended<br/>Feature x̂"]

    style Alpha fill:#FFD700,stroke:#333
    style Output fill:#50C878,stroke:#333""",
        },
    ],
    "2010.11929": [  # ViT
        {
            "caption": "Fig 1. Vision Transformer (ViT) architecture",
            "mermaid": """graph LR
    Input["Input Image"] --> Patch["Split into<br/>16×16 Patches"]
    Patch --> Embed["Linear Projection<br/>+ Position Embed"]
    CLS["[CLS] Token"] --> Embed
    Embed --> T1["Transformer<br/>Encoder ×L"]
    T1 --> MLP["MLP Head"]
    MLP --> Output["Classification"]

    subgraph "Transformer Encoder"
        LN1["LayerNorm"] --> MSA["Multi-Head<br/>Self-Attention"]
        MSA --> Res1["+ Residual"]
        Res1 --> LN2["LayerNorm"]
        LN2 --> FFN["MLP (FFN)"]
        FFN --> Res2["+ Residual"]
    end

    style T1 fill:#4A90D9,stroke:#333
    style Output fill:#50C878,stroke:#333""",
        },
    ],
    "2102.04306": [  # TransUNet
        {
            "caption": "Fig 1. TransUNet: CNN + Transformer hybrid",
            "mermaid": """graph TD
    Input["Input Image"] --> CNN["CNN Encoder<br/>(ResNet-50)"]
    CNN --> Patch["Patch Embed<br/>+ Position"]
    Patch --> Trans["Transformer<br/>Encoder ×12"]
    Trans --> Reshape["Reshape to<br/>Feature Map"]
    Reshape --> Up1["Upsample + Conv"]
    CNN -.->|"Skip 1"| Up1
    Up1 --> Up2["Upsample + Conv"]
    CNN -.->|"Skip 2"| Up2
    Up2 --> Up3["Upsample + Conv"]
    CNN -.->|"Skip 3"| Up3
    Up3 --> Out["Segmentation<br/>Output"]

    style Trans fill:#4A90D9,stroke:#333
    style CNN fill:#FF8C00,stroke:#333
    style Out fill:#50C878,stroke:#333""",
        },
    ],
    "2105.15203": [  # SegFormer
        {
            "caption": "Fig 1. SegFormer: Hierarchical Transformer + MLP decoder",
            "mermaid": """graph TD
    Input["Input Image"] --> S1["Stage 1<br/>Overlap Patch Embed<br/>+ Transformer ×N₁"]
    S1 --> S2["Stage 2<br/>Overlap Patch Embed<br/>+ Transformer ×N₂"]
    S2 --> S3["Stage 3<br/>Overlap Patch Embed<br/>+ Transformer ×N₃"]
    S3 --> S4["Stage 4<br/>Overlap Patch Embed<br/>+ Transformer ×N₄"]
    S1 --> MLP1["MLP<br/>Upsample"]
    S2 --> MLP2["MLP<br/>Upsample"]
    S3 --> MLP3["MLP<br/>Upsample"]
    S4 --> MLP4["MLP<br/>Upsample"]
    MLP1 --> Concat["Concat"]
    MLP2 --> Concat
    MLP3 --> Concat
    MLP4 --> Concat
    Concat --> Fuse["MLP Fusion"]
    Fuse --> Out["Segmentation"]

    style Fuse fill:#4A90D9,stroke:#333
    style Out fill:#50C878,stroke:#333""",
        },
    ],
    "2112.01527": [  # Mask2Former
        {
            "caption": "Fig 1. Mask2Former: Universal architecture for segmentation",
            "mermaid": """graph TD
    Input["Input Image"] --> Backbone["Backbone<br/>(Swin/ResNet)"]
    Backbone --> PD["Pixel Decoder<br/>(Multi-scale Deformable Attn)"]
    PD --> MF["Multi-scale<br/>Features"]
    Queries["N Learnable<br/>Queries"] --> TD["Transformer<br/>Decoder ×L"]
    MF --> TD
    TD --> MaskEmbed["Mask Embeddings"]
    TD --> ClassPred["Class Predictions"]
    MaskEmbed --> DotProd["Dot Product<br/>with Pixel Features"]
    MF --> DotProd
    DotProd --> Masks["N Binary Masks"]
    ClassPred --> Out["Final Segmentation<br/>(Semantic/Instance/Panoptic)"]
    Masks --> Out

    style TD fill:#4A90D9,stroke:#333
    style Out fill:#50C878,stroke:#333""",
        },
    ],
    "2304.02643": [  # SAM
        {
            "caption": "Fig 1. Segment Anything Model (SAM) architecture",
            "mermaid": """graph LR
    Input["Input Image"] --> ImgEnc["Image Encoder<br/>(ViT-H)"]
    ImgEnc --> ImgEmbed["Image<br/>Embedding"]

    Points["Point<br/>Prompts"] --> PromptEnc["Prompt<br/>Encoder"]
    Boxes["Box<br/>Prompts"] --> PromptEnc
    Text["Text<br/>Prompts"] --> PromptEnc
    PromptEnc --> PromptEmbed["Prompt<br/>Embedding"]

    ImgEmbed --> MaskDec["Lightweight<br/>Mask Decoder<br/>(2-layer Transformer)"]
    PromptEmbed --> MaskDec
    MaskDec --> Masks["Valid Masks<br/>+ IoU Scores"]

    style ImgEnc fill:#4A90D9,stroke:#333
    style MaskDec fill:#FFD700,stroke:#333
    style Masks fill:#50C878,stroke:#333""",
        },
    ],
    "1606.06650": [  # 3D U-Net
        {
            "caption": "Fig 1. 3D U-Net with sparse annotation training",
            "mermaid": """graph TD
    Input["3D Volume<br/>Input"] --> E1["3D Conv ×2<br/>+ BN + ReLU<br/>32 ch"]
    E1 --> Pool1["3D Max Pool"]
    Pool1 --> E2["3D Conv ×2<br/>64 ch"]
    E2 --> Pool2["3D Max Pool"]
    Pool2 --> E3["3D Conv ×2<br/>128 ch"]
    E3 --> Pool3["3D Max Pool"]
    Pool3 --> BN["Bottleneck<br/>256 ch"]
    BN --> Up3["3D Up-conv"]
    Up3 --> D3["3D Conv ×2<br/>128 ch"]
    D3 --> Up2["3D Up-conv"]
    Up2 --> D2["3D Conv ×2<br/>64 ch"]
    D2 --> Up1["3D Up-conv"]
    Up1 --> D1["3D Conv ×2<br/>32 ch"]
    D1 --> Out["1×1×1 Conv →<br/>Dense Segmentation"]

    E1 -.->|"3D skip"| D1
    E2 -.->|"3D skip"| D2
    E3 -.->|"3D skip"| D3

    Sparse["Sparse Annotations<br/>(few slices)"] -.->|"masked loss"| Out

    style BN fill:#FFD700,stroke:#333
    style Sparse fill:#FF6B6B,stroke:#333
    style Out fill:#50C878,stroke:#333""",
        },
    ],
    "1809.10486": [  # nnU-Net
        {
            "caption": "Fig 1. nnU-Net self-configuring pipeline",
            "mermaid": """graph TD
    Data["Input Dataset"] --> FP["Fingerprint<br/>Extraction"]
    FP --> Rules["Rule-based<br/>Configuration"]
    Rules --> Arch["Architecture<br/>Selection"]
    Rules --> PP["Preprocessing<br/>Plan"]
    Rules --> Train["Training<br/>Scheme"]
    Arch --> C2D["2D U-Net"]
    Arch --> C3D["3D U-Net<br/>(full res)"]
    Arch --> Cas["3D U-Net<br/>(cascade)"]
    C2D --> CV["5-Fold<br/>Cross-Validation"]
    C3D --> CV
    Cas --> CV
    CV --> Ensemble["Postprocessing<br/>+ Ensemble"]
    Ensemble --> Best["Best Model<br/>Selection"]

    style FP fill:#4A90D9,stroke:#333
    style Best fill:#50C878,stroke:#333""",
        },
    ],
}


def render_paper_figures(arxiv_id: str) -> bool:
    """Render key architecture figures for a paper if available.

    Args:
        arxiv_id: The arXiv paper ID (e.g. "1505.04597").

    Returns:
        True if figures were rendered, False otherwise.
    """
    figures = PAPER_FIGURES.get(arxiv_id)
    if not figures:
        return False

    from components.mermaid_render import render_mermaid

    st.markdown("**Key Figures:**")
    for fig in figures:
        st.caption(fig["caption"])
        render_mermaid(fig["mermaid"], height=350)
    return True


def render_paper_figures_inline(arxiv_id: str) -> bool:
    """Render a compact figure for inline use (e.g. in reading roadmap).

    Args:
        arxiv_id: The arXiv paper ID.

    Returns:
        True if figures were rendered.
    """
    figures = PAPER_FIGURES.get(arxiv_id)
    if not figures:
        return False

    from components.mermaid_render import render_mermaid

    for fig in figures:
        with st.expander(f"🔍 {fig['caption']}"):
            render_mermaid(fig["mermaid"], height=300)
    return True
