"""Classical segmentation algorithms for the interactive Segmentation Lab.

Each algorithm is registered in :data:`ALGORITHMS` with UI parameter specs
and teaching notes (how it works, strengths, weaknesses). All algorithms
take an RGB uint8 array and return an integer label map of the same
height/width. Implementations rely only on numpy / scipy / scikit-image,
so they run in a couple of seconds on CPU - fast enough for live
experimentation in the browser.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np


@dataclass
class Param:
    """UI specification for one tunable algorithm parameter."""

    name: str
    label: str
    ptype: str  # "int" | "float" | "choice"
    default: Any
    min: Any = None
    max: Any = None
    step: Any = None
    choices: list | None = None
    help: str = ""


@dataclass
class Algorithm:
    """A registered segmentation algorithm with metadata for the UI."""

    key: str
    name: str
    category: str
    year: str
    fn: Callable[..., np.ndarray]
    summary: str
    how_it_works: str
    strengths: list[str] = field(default_factory=list)
    weaknesses: list[str] = field(default_factory=list)
    params: list[Param] = field(default_factory=list)


def _to_gray(image: np.ndarray) -> np.ndarray:
    from skimage.color import rgb2gray

    return rgb2gray(image)


# ---------------------------------------------------------------------------
# Algorithm implementations
# ---------------------------------------------------------------------------


def run_otsu(
    image: np.ndarray, blur_sigma: float = 1.0, invert: bool = False
) -> np.ndarray:
    """Global Otsu thresholding on the grayscale image."""
    from scipy.ndimage import gaussian_filter
    from skimage.filters import threshold_otsu

    gray = _to_gray(image)
    if blur_sigma > 0:
        gray = gaussian_filter(gray, blur_sigma)
    t = threshold_otsu(gray)
    labels = (gray > t).astype(np.int32)
    if invert:
        labels = 1 - labels
    return labels


def run_multi_otsu(
    image: np.ndarray, classes: int = 3, blur_sigma: float = 1.0
) -> np.ndarray:
    """Multi-level Otsu thresholding into N intensity classes."""
    from scipy.ndimage import gaussian_filter
    from skimage.filters import threshold_multiotsu

    gray = _to_gray(image)
    if blur_sigma > 0:
        gray = gaussian_filter(gray, blur_sigma)
    try:
        thresholds = threshold_multiotsu(gray, classes=classes)
    except ValueError:
        # Fewer distinct gray levels than requested classes
        return run_otsu(image, blur_sigma=blur_sigma)
    return np.digitize(gray, bins=thresholds).astype(np.int32)


def run_kmeans(
    image: np.ndarray,
    k: int = 4,
    color_space: str = "Lab",
    spatial_weight: float = 0.0,
    seed: int = 42,
) -> np.ndarray:
    """K-Means clustering of pixels in color (+ optional spatial) space."""
    from scipy.cluster.vq import kmeans2
    from skimage.color import rgb2lab

    h, w = image.shape[:2]
    if color_space == "Lab":
        feats = rgb2lab(image).reshape(-1, 3)
        feats = feats / np.array([100.0, 128.0, 128.0])  # roughly normalize
    else:
        feats = image.reshape(-1, 3).astype(np.float64) / 255.0

    if spatial_weight > 0:
        yy, xx = np.mgrid[0:h, 0:w]
        coords = np.stack([yy.ravel() / h, xx.ravel() / w], axis=1)
        feats = np.hstack([feats, coords * spatial_weight])

    rng = np.random.default_rng(seed)
    _, labels = kmeans2(feats, k, minit="++", seed=rng)
    return labels.reshape(h, w).astype(np.int32)


def run_slic(
    image: np.ndarray, n_segments: int = 100, compactness: float = 10.0
) -> np.ndarray:
    """SLIC superpixels - K-Means in joint color-spatial (labxy) space."""
    from skimage.segmentation import slic

    return slic(
        image,
        n_segments=n_segments,
        compactness=compactness,
        start_label=0,
        channel_axis=-1,
    ).astype(np.int32)


def run_felzenszwalb(
    image: np.ndarray, scale: float = 100.0, sigma: float = 0.8, min_size: int = 50
) -> np.ndarray:
    """Felzenszwalb-Huttenlocher efficient graph-based segmentation."""
    from skimage.segmentation import felzenszwalb

    return felzenszwalb(image, scale=scale, sigma=sigma, min_size=min_size).astype(
        np.int32
    )


def run_watershed(
    image: np.ndarray,
    min_distance: int = 20,
    blur_sigma: float = 2.0,
    invert: bool = False,
) -> np.ndarray:
    """Marker-controlled watershed on the distance transform.

    Foreground is extracted with Otsu, markers are placed at distance-
    transform peaks, then watershed floods from the markers - the
    classic recipe for separating touching objects.
    """
    from scipy import ndimage as ndi
    from skimage.feature import peak_local_max
    from skimage.filters import threshold_otsu
    from skimage.segmentation import watershed

    gray = _to_gray(image)
    if blur_sigma > 0:
        gray = ndi.gaussian_filter(gray, blur_sigma)
    t = threshold_otsu(gray)
    fg = gray > t
    if invert:
        fg = ~fg
    # Ensure "foreground" is the minority region (objects, not background)
    if fg.mean() > 0.5:
        fg = ~fg

    distance = ndi.distance_transform_edt(fg)
    peaks = peak_local_max(distance, min_distance=min_distance, labels=fg)
    markers = np.zeros_like(gray, dtype=np.int32)
    for i, (r, c) in enumerate(peaks, start=1):
        markers[r, c] = i
    if markers.max() == 0:
        return fg.astype(np.int32)
    labels = watershed(-distance, markers, mask=fg)
    return labels.astype(np.int32)  # 0 = background, 1..N = instances


def run_chan_vese(
    image: np.ndarray, mu: float = 0.1, num_iter: int = 100, blur_sigma: float = 1.0
) -> np.ndarray:
    """Chan-Vese active contours without edges (region-based energy)."""
    from scipy.ndimage import gaussian_filter
    from skimage.segmentation import chan_vese

    gray = _to_gray(image)
    if blur_sigma > 0:
        gray = gaussian_filter(gray, blur_sigma)
    seg = chan_vese(gray, mu=mu, max_num_iter=num_iter, extended_output=False)
    return seg.astype(np.int32)


def run_quickshift(
    image: np.ndarray, kernel_size: int = 5, max_dist: int = 10, ratio: float = 0.5
) -> np.ndarray:
    """Quickshift mode-seeking segmentation in color-(x,y) space."""
    from skimage.segmentation import quickshift

    return quickshift(
        np.ascontiguousarray(image),
        kernel_size=kernel_size,
        max_dist=max_dist,
        ratio=ratio,
        rng=42,
    ).astype(np.int32)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

ALGORITHMS: dict[str, Algorithm] = {
    a.key: a
    for a in [
        Algorithm(
            key="otsu",
            name="Otsu Threshold",
            category="Thresholding",
            year="1979",
            fn=run_otsu,
            summary="Single global threshold that maximizes between-class variance.",
            how_it_works=(
                "Builds the grayscale histogram and picks the threshold that "
                "maximizes the variance *between* the two resulting classes "
                "(equivalently, minimizes within-class variance). Purely "
                "histogram-based: spatial layout is ignored entirely."
            ),
            strengths=[
                "Extremely fast (one histogram pass) and parameter-free",
                "Optimal when the histogram is bimodal with balanced classes",
                "Great baseline: if Otsu works, you may not need anything fancier",
            ],
            weaknesses=[
                "Fails under illumination gradients (one threshold for the whole image)",
                "Binary only - cannot handle multiple classes",
                "Ignores spatial context: noise produces salt-and-pepper output",
            ],
            params=[
                Param(
                    "blur_sigma",
                    "Pre-blur σ",
                    "float",
                    1.0,
                    0.0,
                    5.0,
                    0.5,
                    help="Gaussian smoothing before thresholding suppresses noise.",
                ),
                Param(
                    "invert",
                    "Invert foreground",
                    "choice",
                    False,
                    choices=[False, True],
                    help="Swap which side of the threshold is foreground.",
                ),
            ],
        ),
        Algorithm(
            key="multi_otsu",
            name="Multi-Otsu Threshold",
            category="Thresholding",
            year="2001",
            fn=run_multi_otsu,
            summary="Otsu generalized to N intensity classes.",
            how_it_works=(
                "Searches for N-1 thresholds that jointly maximize between-"
                "class variance across N classes, then assigns each pixel to "
                "an intensity band. Still histogram-only - no spatial term."
            ),
            strengths=[
                "Handles multiple classes while staying nearly as fast as Otsu",
                "Deterministic and easy to reason about",
            ],
            weaknesses=[
                "Requires choosing the number of classes in advance",
                "Same blindness to illumination gradients and spatial context",
                "Classes with overlapping intensity ranges cannot be separated",
            ],
            params=[
                Param("classes", "Number of classes", "int", 3, 2, 5, 1),
                Param("blur_sigma", "Pre-blur σ", "float", 1.0, 0.0, 5.0, 0.5),
            ],
        ),
        Algorithm(
            key="kmeans",
            name="K-Means Clustering",
            category="Clustering",
            year="1967",
            fn=run_kmeans,
            summary="Cluster pixels by color (optionally + position) into K groups.",
            how_it_works=(
                "Each pixel becomes a feature vector (color, optionally with "
                "x,y coordinates). K-Means alternates between assigning "
                "pixels to the nearest centroid and re-estimating centroids "
                "until convergence. Lab color space makes distances closer "
                "to human color perception."
            ),
            strengths=[
                "Uses full color information, not just intensity",
                "Spatial weight interpolates between pure color clustering and superpixels",
                "Simple, well-understood, widely applicable",
            ],
            weaknesses=[
                "K must be chosen (wrong K merges or splits classes)",
                "No spatial coherence at weight 0: produces scattered fragments",
                "Sensitive to initialization; distinct objects with the same color merge",
            ],
            params=[
                Param("k", "K (clusters)", "int", 4, 2, 10, 1),
                Param(
                    "color_space",
                    "Color space",
                    "choice",
                    "Lab",
                    choices=["Lab", "RGB"],
                ),
                Param(
                    "spatial_weight",
                    "Spatial weight",
                    "float",
                    0.0,
                    0.0,
                    2.0,
                    0.1,
                    help="0 = color only; higher values make clusters spatially compact.",
                ),
            ],
        ),
        Algorithm(
            key="slic",
            name="SLIC Superpixels",
            category="Superpixels",
            year="2012",
            fn=run_slic,
            summary="Fast K-Means in labxy space producing compact superpixels.",
            how_it_works=(
                "Runs localized K-Means in the 5-D (L, a, b, x, y) space with "
                "cluster search restricted to a local neighborhood, yielding "
                "hundreds of compact, boundary-adherent regions. Superpixels "
                "over-segment by design - they are a preprocessing step that "
                "later stages (or a classifier) can merge."
            ),
            strengths=[
                "Excellent boundary adherence at low compute cost",
                "Controllable region count and compactness",
                "Standard preprocessing for graph-based and interactive methods",
            ],
            weaknesses=[
                "Deliberate over-segmentation: not a final semantic result",
                "Compactness trades boundary accuracy for regular shape",
                "Metrics after best-overlap mapping look inflated (see Lab notes)",
            ],
            params=[
                Param("n_segments", "Superpixels", "int", 100, 20, 600, 20),
                Param(
                    "compactness",
                    "Compactness",
                    "float",
                    10.0,
                    0.1,
                    50.0,
                    0.1,
                    help="Higher = squarer regions, lower = follows color edges.",
                ),
            ],
        ),
        Algorithm(
            key="felzenszwalb",
            name="Felzenszwalb Graph",
            category="Graph-based",
            year="2004",
            fn=run_felzenszwalb,
            summary="Greedy graph merging with an adaptive region contrast criterion.",
            how_it_works=(
                "Treats the image as a graph with color-difference edge "
                "weights, then merges regions greedily unless the boundary "
                "evidence between them exceeds each region's internal "
                "variation (plus a scale-dependent tolerance). Adapts to "
                "local contrast: keeps low-contrast boundaries in smooth "
                "areas while merging noisy textured regions."
            ),
            strengths=[
                "Adaptive to local contrast - robust across very different scenes",
                "Near-linear time; region count adapts to image content",
                "Classic pre-deep-learning baseline (used in Selective Search / R-CNN)",
            ],
            weaknesses=[
                "Scale parameter is unintuitive and image-dependent",
                "Greedy merging can leak through weak boundaries",
                "Region count is not directly controllable",
            ],
            params=[
                Param(
                    "scale",
                    "Scale",
                    "float",
                    100.0,
                    10.0,
                    1000.0,
                    10.0,
                    help="Larger scale = fewer, larger regions.",
                ),
                Param("sigma", "Pre-blur σ", "float", 0.8, 0.0, 3.0, 0.1),
                Param("min_size", "Min region size", "int", 50, 10, 500, 10),
            ],
        ),
        Algorithm(
            key="watershed",
            name="Watershed (markers)",
            category="Region-based",
            year="1991",
            fn=run_watershed,
            summary="Flood regions from distance-transform peaks to split touching objects.",
            how_it_works=(
                "Extracts foreground with Otsu, computes the distance "
                "transform (distance to background), places one marker per "
                "local peak, then 'floods' the inverted distance map from "
                "the markers. Watershed lines form where floods meet - "
                "exactly at the narrow necks between touching objects."
            ),
            strengths=[
                "The classic solution for separating touching convex objects (cells!)",
                "Produces instance labels, not just semantic masks",
                "Marker control prevents the raw watershed's chronic over-segmentation",
            ],
            weaknesses=[
                "Quality depends entirely on marker placement (min-distance parameter)",
                "Assumes roughly convex objects; elongated shapes get split",
                "Inherits any failure of the initial foreground threshold",
            ],
            params=[
                Param(
                    "min_distance",
                    "Marker min distance",
                    "int",
                    20,
                    5,
                    80,
                    5,
                    help="Smaller = more markers = more (possibly spurious) instances.",
                ),
                Param("blur_sigma", "Pre-blur σ", "float", 2.0, 0.0, 5.0, 0.5),
                Param(
                    "invert",
                    "Invert foreground",
                    "choice",
                    False,
                    choices=[False, True],
                ),
            ],
        ),
        Algorithm(
            key="chan_vese",
            name="Chan-Vese Active Contour",
            category="Energy-based",
            year="2001",
            fn=run_chan_vese,
            summary="Evolves a contour minimizing region variance - no edges needed.",
            how_it_works=(
                "Minimizes the Mumford-Shah-style energy: inside/outside "
                "intensity variance plus a contour-length penalty (μ), using "
                "level sets. Because the energy is region-based rather than "
                "gradient-based, it finds objects with weak or smooth "
                "boundaries where edge detectors fail."
            ),
            strengths=[
                "Works on blurred / weak-edged objects (medical imaging staple)",
                "Length penalty μ gives explicit control over boundary smoothness",
                "Topology-flexible: contours can split and merge during evolution",
            ],
            weaknesses=[
                "Iterative and slow compared to thresholding",
                "Binary (two-phase) in its classic form",
                "Assumes roughly homogeneous foreground/background intensities",
            ],
            params=[
                Param(
                    "mu",
                    "μ (smoothness)",
                    "float",
                    0.1,
                    0.0,
                    1.0,
                    0.05,
                    help="Higher μ = smoother, shorter boundary; small details vanish.",
                ),
                Param("num_iter", "Iterations", "int", 100, 20, 300, 20),
                Param("blur_sigma", "Pre-blur σ", "float", 1.0, 0.0, 3.0, 0.5),
            ],
        ),
        Algorithm(
            key="quickshift",
            name="Quickshift",
            category="Clustering",
            year="2008",
            fn=run_quickshift,
            summary="Mode-seeking (mean-shift family) in joint color-spatial space.",
            how_it_works=(
                "Estimates the density of pixels in (color, x, y) space and "
                "links each pixel to its nearest neighbor with higher "
                "density; cutting links longer than max_dist yields "
                "segments around density modes. A faster relative of "
                "mean-shift that needs no cluster count."
            ),
            strengths=[
                "No need to choose the number of segments",
                "Finds arbitrarily-shaped clusters around color modes",
            ],
            weaknesses=[
                "Slowest of the classical methods shown here",
                "Sensitive to kernel_size / max_dist interplay",
                "Like SLIC, usually over-segments - needs a merging stage",
            ],
            params=[
                Param("kernel_size", "Kernel size", "int", 5, 3, 12, 1),
                Param("max_dist", "Max link distance", "int", 10, 2, 30, 2),
                Param("ratio", "Color/space ratio", "float", 0.5, 0.0, 1.0, 0.1),
            ],
        ),
    ]
}


def run_algorithm(key: str, image: np.ndarray, params: dict) -> np.ndarray:
    """Run a registered algorithm and return its integer label map."""
    return ALGORITHMS[key].fn(image, **params)
