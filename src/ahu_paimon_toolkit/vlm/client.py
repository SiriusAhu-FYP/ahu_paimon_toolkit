"""Async VLM client: OpenAI-compatible chat/completions interface."""

from __future__ import annotations

import httpx
from loguru import logger as lg

from ahu_paimon_toolkit.models import FrameDescription, KeyFrame
from ahu_paimon_toolkit.vlm.model_utils import is_vision_model


class AsyncVLMClient:
    """Async vLLM inference client.

    Calls a local vLLM OpenAI-compatible chat/completions endpoint.
    Sends image + text for VLM models, text-only for non-VLM models.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        model: str = "Qwen/Qwen3-VL-2B-Instruct",
        prompt: str = "Describe this game screenshot in detail.",
        *,
        timeout: float = 60.0,
        max_tokens: int = 512,
        temperature: float = 0.1,
    ) -> None:
        self._base_url = base_url
        self._model = model
        self._prompt = prompt
        self._timeout = timeout
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._client: httpx.AsyncClient | None = None
        self._is_vlm = is_vision_model(model)

    @property
    def model(self) -> str:
        return self._model

    @property
    def base_url(self) -> str:
        return self._base_url

    async def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                timeout=httpx.Timeout(self._timeout),
            )
        return self._client

    async def describe_frame(self, keyframe: KeyFrame) -> FrameDescription:
        """Call VLM to describe a single keyframe."""
        client = await self._ensure_client()

        if self._is_vlm:
            user_content: list[dict] | str = [
                {"type": "text", "text": self._prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{keyframe.base64_image}",
                    },
                },
            ]
        else:
            user_content = self._prompt

        payload = {
            "model": self._model,
            "messages": [{"role": "user", "content": user_content}],
            "max_tokens": self._max_tokens,
            "temperature": self._temperature,
        }

        response = await client.post("/chat/completions", json=payload)
        response.raise_for_status()
        data = response.json()
        content = data["choices"][0]["message"]["content"]

        lg.info("VLM response for frame #{} ({} chars)", keyframe.frame_id, len(content))

        return FrameDescription(
            frame_id=keyframe.frame_id,
            timestamp_ms=keyframe.timestamp_ms,
            description=content,
        )

    async def chat(
        self,
        messages: list[dict],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        """Generic chat completion call. Returns the response content string."""
        client = await self._ensure_client()
        payload = {
            "model": self._model,
            "messages": messages,
            "max_tokens": max_tokens or self._max_tokens,
            "temperature": temperature if temperature is not None else self._temperature,
        }
        response = await client.post("/chat/completions", json=payload)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]

    async def close(self) -> None:
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
            lg.debug("VLM client closed")
