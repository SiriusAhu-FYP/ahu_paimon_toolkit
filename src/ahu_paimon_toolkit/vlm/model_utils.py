"""Model detection, vision model identification, and vLLM readiness check."""

from __future__ import annotations

import time

from loguru import logger as lg
from openai import OpenAI

_VLM_KEYWORDS = {"vl", "vision", "visual"}

_KNOWN_VLM_PREFIXES = {
    "qwen/qwen3.5",
    "qwen/qwen3-vl",
    "qwen/qwen2.5-vl",
    "qwen/qwen2-vl",
    "opengvlab/internvl",
    "deepseek-ai/deepseek-vl",
    "mistralai/ministral",
    "microsoft/phi-3",
    "huggingfacetb/smolvlm",
}


def detect_model(client: OpenAI) -> str:
    """Auto-detect the currently running model via /v1/models."""
    models = client.models.list()
    if models.data:
        model_id = models.data[0].id
        lg.info("Detected model: {}", model_id)
        return model_id
    raise RuntimeError("No running model detected")


def detect_model_from_url(base_url: str = "http://localhost:8000/v1") -> str:
    """Create a throwaway client and detect the running model."""
    client = OpenAI(base_url=base_url, api_key="EMPTY")
    return detect_model(client)


def model_short_name(model_id: str) -> str:
    """Convert 'Org/Model-Name' to 'Org_Model-Name' for directory naming."""
    return model_id.replace("/", "_").replace(" ", "_")


def is_vision_model(model_id: str) -> bool:
    """Determine if a model supports image input.

    Strategy (by priority):
    1. Known VLM family prefix match
    2. Model name contains vl/vision/visual keywords
    3. Default to True (conservative: avoid dropping image data)
    """
    model_lower = model_id.lower()

    for prefix in _KNOWN_VLM_PREFIXES:
        if model_lower.startswith(prefix):
            return True

    parts = model_lower.replace("/", "-").replace("_", "-").split("-")
    if set(parts) & _VLM_KEYWORDS:
        return True

    lg.info("Model '{}' not matched to known VLM pattern, assuming VLM", model_id)
    return True


def wait_for_vllm_ready(
    base_url: str = "http://localhost:8000/v1",
    timeout_s: float = 300.0,
    poll_interval_s: float = 10.0,
) -> str:
    """Poll vLLM /v1/models until it responds. Returns the detected model ID."""
    start = time.monotonic()
    last_err = None
    while time.monotonic() - start < timeout_s:
        try:
            return detect_model_from_url(base_url)
        except Exception as e:
            last_err = e
            elapsed = time.monotonic() - start
            lg.debug("vLLM not ready ({:.0f}s / {:.0f}s): {}", elapsed, timeout_s, e)
            time.sleep(poll_interval_s)
    raise TimeoutError(f"vLLM not ready after {timeout_s}s: {last_err}")
