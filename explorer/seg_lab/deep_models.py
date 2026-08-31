"""Deep segmentation backends for the Playground page.

The same SegFormer checkpoints are reachable two ways:

``local``
    A ``transformers`` pipeline. Weights are downloaded once and run on the
    CPU of whatever machine serves the app. Needs ``torch`` **and**
    ``torchvision``: from transformers 5.x every image processor is built on
    a torchvision or PIL backend, and both SegFormer variants
    (``SegformerImageProcessor`` / ``SegformerImageProcessorPil``) are
    declared ``@requires(backends=("torch", "torchvision"))``. Without
    torchvision installed, ``AutoImageProcessor`` finds no importable class
    and model loading fails with "Could not load any image processor class".

``hf_api``
    The Hugging Face Inference API. The image is posted to the Hub and only
    the masks come back, so no torch, no torchvision and no model weights are
    needed locally - which also keeps memory-capped hosts (Streamlit
    Community Cloud) able to run the 84M-parameter B5 checkpoint.

Both backends return what the ``image-segmentation`` pipeline produces -
``[{"label": str, "score": float | None, "mask": PIL.Image}]`` - so the page
renders, overlays and scores them identically.
"""

from __future__ import annotations

import importlib.util
import os
import time
from typing import Any

LOCAL = "local"
HF_API = "hf_api"

BACKEND_LABELS = {
    LOCAL: "Local (transformers)",
    HF_API: "Hugging Face API (remote)",
}

# Environment variables checked for an Inference API token, in order.
TOKEN_ENV_VARS = ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HF_API_TOKEN")

MODEL_INFO: dict[str, dict] = {
    "SegFormer-B0 (ADE20K)": {
        "hf_model": "nvidia/segformer-b0-finetuned-ade-512-512",
        "task": "image-segmentation",
        "description": "Lightweight SegFormer (3.7M params). Fast on CPU.",
        "weight": "light",
    },
    "SegFormer-B1 (ADE20K)": {
        "hf_model": "nvidia/segformer-b1-finetuned-ade-512-512",
        "task": "image-segmentation",
        "description": "Mid-size SegFormer (13.7M params). Good accuracy/speed trade-off.",
        "weight": "mid",
    },
    "SegFormer-B5 (ADE20K)": {
        "hf_model": "nvidia/segformer-b5-finetuned-ade-640-640",
        "task": "image-segmentation",
        "description": "Large SegFormer (84M params). High accuracy, slower.",
        "weight": "heavy",
    },
}


class BackendError(RuntimeError):
    """A backend could not run - message is written for the end user."""


def _installed(module: str) -> bool:
    """True if `module` can be imported without actually importing it."""
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def missing_local_packages() -> list[str]:
    """Packages the local backend needs that are not installed."""
    required = {
        "transformers": "transformers",
        "torch": "torch",
        "torchvision": "torchvision",
        "PIL": "Pillow",
        "numpy": "numpy",
    }
    return [pkg for module, pkg in required.items() if not _installed(module)]


def missing_api_packages() -> list[str]:
    """Packages the Hugging Face API backend needs that are not installed."""
    required = {"huggingface_hub": "huggingface-hub", "PIL": "Pillow"}
    return [pkg for module, pkg in required.items() if not _installed(module)]


def local_available() -> bool:
    return not missing_local_packages()


def api_available() -> bool:
    return not missing_api_packages()


def token_from_env() -> str | None:
    """First Hugging Face token found in the environment, if any."""
    for var in TOKEN_ENV_VARS:
        value = os.environ.get(var, "").strip()
        if value:
            return value
    return None


def install_hint(packages: list[str]) -> str:
    return "pip install " + " ".join(packages)


# ---------------------------------------------------------------------------
# Local backend - transformers pipeline on CPU
# ---------------------------------------------------------------------------

_TORCHVISION_FIX = (
    "transformers 5.x builds every image processor on a torchvision or PIL "
    "backend and both SegFormer variants require torchvision, so preprocessing "
    "cannot start without it. Either install it "
    "(`pip install torchvision`, or `pip install -r explorer/requirements.txt`) "
    "or switch the sidebar backend to **Hugging Face API (remote)**, which "
    "needs no local torch/torchvision."
)


def _friendly_local_error(model_key: str, exc: Exception) -> BackendError:
    """Turn a transformers loading failure into actionable guidance."""
    text = str(exc)
    if "image processor" in text or "torchvision" in text:
        return BackendError(
            f"Failed to load **{model_key}**: {text}\n\n{_TORCHVISION_FIX}"
        )
    return BackendError(f"Failed to load **{model_key}**: {text}")


def load_local_pipeline(model_key: str):
    """Build a CPU ``image-segmentation`` pipeline for `model_key`.

    Raises `BackendError` with an actionable message on any failure; the
    caller is expected to cache the returned pipeline across reruns.
    """
    missing = missing_local_packages()
    if missing:
        raise BackendError(
            f"Local inference needs `{'`, `'.join(missing)}`. "
            f"Install it with `{install_hint(missing)}`, or switch the sidebar "
            "backend to **Hugging Face API (remote)**."
        )

    info = MODEL_INFO[model_key]
    try:
        from transformers import pipeline

        return pipeline(
            task=info["task"],
            model=info["hf_model"],
            device=-1,  # CPU; change to 0 for GPU
        )
    except Exception as exc:  # noqa: BLE001 - surfaced to the user as-is
        raise _friendly_local_error(model_key, exc) from exc


def run_local(pipe, image_bytes: bytes) -> tuple[list[dict], float]:
    """Run a loaded pipeline on PNG/JPEG bytes."""
    import io

    from PIL import Image

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    t0 = time.perf_counter()
    results = pipe(image)
    return normalize(results), time.perf_counter() - t0


# ---------------------------------------------------------------------------
# Remote backend - Hugging Face Inference API
# ---------------------------------------------------------------------------


def build_api_client(token: str | None, timeout: float = 120.0):
    """Create an `InferenceClient` for the Hub's image-segmentation task."""
    missing = missing_api_packages()
    if missing:
        raise BackendError(
            f"The Hugging Face API backend needs `{'`, `'.join(missing)}`. "
            f"Install it with `{install_hint(missing)}`."
        )
    if not token:
        raise BackendError(
            "No Hugging Face access token found. Create a free read token at "
            "https://huggingface.co/settings/tokens, then either paste it in "
            "the sidebar, export `HF_TOKEN=...`, or add it to "
            '`.streamlit/secrets.toml` as `HF_TOKEN = "hf_..."`.'
        )

    from huggingface_hub import InferenceClient

    return InferenceClient(token=token, timeout=timeout)


def run_hf_api(client, model_key: str, image_bytes: bytes) -> tuple[list[dict], float]:
    """Segment `image_bytes` on the Hub instead of locally."""
    import io

    from PIL import Image

    model_id = MODEL_INFO[model_key]["hf_model"]
    t0 = time.perf_counter()
    try:
        response = client.image_segmentation(image_bytes, model=model_id)
    except Exception as exc:  # noqa: BLE001 - surfaced to the user as-is
        raise _friendly_api_error(model_key, model_id, exc) from exc
    elapsed = time.perf_counter() - t0

    # The local pipeline always returns masks at input resolution; providers
    # may not, so pin them here and keep both backends interchangeable. A
    # size we cannot read is not worth failing a good response over.
    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            input_size = image.size
    except Exception:  # noqa: BLE001 - undecodable input, skip the resize
        input_size = None

    results = normalize(response, size=input_size)
    if not results:
        raise BackendError(
            f"The Inference API returned no segments for **{model_key}**. "
            "The provider may not serve this checkpoint - try SegFormer-B0, "
            "or switch to the local backend."
        )
    return results, elapsed


def _friendly_api_error(model_key: str, model_id: str, exc: Exception) -> BackendError:
    """Map Inference API failures onto what the user can actually do."""
    text = str(exc)
    status = getattr(getattr(exc, "response", None), "status_code", None)

    if status in (401, 403) or "401" in text or "Invalid credentials" in text:
        hint = (
            "The token was rejected. Check that it is a valid **read** token "
            "from https://huggingface.co/settings/tokens and that it has "
            "'Make calls to Inference Providers' enabled."
        )
    elif status == 404 or "not supported" in text or "No Inference Provider" in text:
        hint = (
            f"No Inference provider currently serves `{model_id}`. Try another "
            "model in the sidebar, or run it with the local backend "
            "(`pip install -r explorer/requirements.txt`)."
        )
    elif status == 429 or "rate limit" in text.lower():
        hint = (
            "Rate limit reached for this token. Wait a minute, or run the "
            "model with the local backend."
        )
    elif status == 503 or "loading" in text.lower():
        hint = (
            "The model is still loading on the Hub (cold start). Wait ~30s "
            "and run it again."
        )
    else:
        hint = "Check network access to huggingface.co, or switch to the local backend."
    return BackendError(
        f"Hugging Face API call for **{model_key}** failed: {text}\n\n{hint}"
    )


# ---------------------------------------------------------------------------
# Shared output normalisation
# ---------------------------------------------------------------------------


def _as_field(item: Any, key: str):
    if isinstance(item, dict):
        return item.get(key)
    return getattr(item, key, None)


def normalize(response: Any, size: tuple[int, int] | None = None) -> list[dict]:
    """Coerce either backend's output into pipeline-shaped dicts.

    The local pipeline yields dicts; `InferenceClient.image_segmentation`
    yields `ImageSegmentationOutputElement` dataclasses. Both carry a `label`,
    an optional `score` and a PIL mask, which is all the page needs. Masks are
    nearest-resized to `size` (width, height) when one is given.
    """
    from PIL import Image

    segments = []
    for i, item in enumerate(response or []):
        mask = _as_field(item, "mask")
        if mask is None:
            continue
        mask = mask.convert("L")
        if size is not None and mask.size != size:
            mask = mask.resize(size, Image.NEAREST)
        score = _as_field(item, "score")
        try:
            score = float(score) if score is not None else None
        except (TypeError, ValueError):
            score = None
        segments.append(
            {
                "label": _as_field(item, "label") or f"segment_{i}",
                "score": score,
                "mask": mask,
            }
        )
    return segments
