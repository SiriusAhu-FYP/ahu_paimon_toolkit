"""DeepSeek-based video summarization: aggregate frame descriptions into a coherent summary."""

from __future__ import annotations

import httpx
from loguru import logger as lg

from ahu_paimon_toolkit.models import FrameDescription, VideoSummary

DEFAULT_SUMMARIZER_PROMPT = (
    "你是一个专业的视频内容分析师。下面提供了一系列按时间顺序排列的视频帧描述"
    "（每条描述对应视频中的一个关键画面），请根据这些帧描述生成一段连贯的视频总结。\n\n"
    "要求：\n"
    "1. **时间线叙述**: 按照时间顺序，概述视频中发生了什么\n"
    "2. **关键事件**: 提炼出视频中最重要的事件或转折点\n"
    "3. **整体主题**: 总结视频的主题或核心内容\n"
    "4. **简洁明了**: 总结控制在 3-5 句话之间，避免逐帧复述\n\n"
    "请直接输出总结文本，不要添加额外的格式标记。"
)


def build_frame_text(descriptions: list[FrameDescription]) -> str:
    """Join timestamped frame descriptions into structured text."""
    lines: list[str] = []
    for desc in descriptions:
        ts_s = desc.timestamp_ms / 1000.0
        lines.append(f"[{ts_s:.1f}s] Frame#{desc.frame_id}: {desc.description}")
    return "\n".join(lines)


class Summarizer:
    """DeepSeek summarization client.

    Collects frame descriptions from VLM and produces a coherent video summary.
    """

    def __init__(
        self,
        api_key: str,
        api_base_url: str = "https://api.deepseek.com",
        model: str = "deepseek-chat",
        prompt: str | None = None,
        timeout: float = 120.0,
    ) -> None:
        self._api_key = api_key
        self._api_base_url = api_base_url
        self._model = model
        self._prompt = prompt or DEFAULT_SUMMARIZER_PROMPT
        self._timeout = timeout
        self._client: httpx.AsyncClient | None = None

    async def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self._api_base_url,
                timeout=httpx.Timeout(self._timeout),
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
            )
        return self._client

    async def summarize(
        self,
        descriptions: list[FrameDescription],
        duration_s: float,
    ) -> VideoSummary:
        """Send all frame descriptions to DeepSeek and return a VideoSummary."""
        if not descriptions:
            lg.warning("No frame descriptions to summarize")
            return VideoSummary(
                frame_descriptions=[],
                summary_text="No valid frame data for summarization.",
                total_keyframes=0,
                duration_s=duration_s,
            )

        frame_text = build_frame_text(descriptions)
        user_message = (
            f"{self._prompt}\n\n"
            f"--- Frame data ({len(descriptions)} frames, {duration_s:.1f}s video) ---\n"
            f"{frame_text}"
        )

        client = await self._ensure_client()
        payload = {
            "model": self._model,
            "messages": [{"role": "user", "content": user_message}],
            "max_tokens": 1024,
            "temperature": 0.3,
        }

        lg.info("Summarizing {} frame descriptions ({:.1f}s video)", len(descriptions), duration_s)
        response = await client.post("/v1/chat/completions", json=payload)
        response.raise_for_status()
        data = response.json()
        summary_text = data["choices"][0]["message"]["content"]

        lg.info("Summary complete ({} chars)", len(summary_text))

        return VideoSummary(
            frame_descriptions=descriptions,
            summary_text=summary_text,
            total_keyframes=len(descriptions),
            duration_s=duration_s,
        )

    async def close(self) -> None:
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
            lg.debug("Summarizer client closed")
