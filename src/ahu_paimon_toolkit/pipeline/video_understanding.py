"""Encapsulated video understanding pipeline: capture + VLM + summary."""

from __future__ import annotations

import asyncio
import signal
import time

from loguru import logger as lg

from ahu_paimon_toolkit.capture.window_capture import WindowNotFoundError, run_capture_loop
from ahu_paimon_toolkit.config import ToolkitSettings
from ahu_paimon_toolkit.models import (
    FrameDescription,
    OnFrameSampledCallback,
    PipelineResult,
)
from ahu_paimon_toolkit.pipeline.queue_manager import KeyFrameQueue
from ahu_paimon_toolkit.pipeline.summarizer import Summarizer
from ahu_paimon_toolkit.vlm.client import AsyncVLMClient


class VideoUnderstandingPipeline:
    """Encapsulated pipeline: screen capture -> frame diff -> VLM -> summary.

    This is the main "video capture + understanding" class requested for reuse
    across projects. All dependencies are injected via constructor parameters.
    """

    def __init__(
        self,
        vlm_client: AsyncVLMClient,
        summarizer: Summarizer | None = None,
        *,
        window_keyword: str = "PotPlayer",
        capture_interval_ms: int = 500,
        max_frame_size: int = 512,
        diff_method: str = "mse",
        diff_threshold: float = 500.0,
        queue_max_size: int = 50,
        queue_expiry_ms: int = 10000,
        recording_duration_s: int = 20,
    ) -> None:
        self._vlm_client = vlm_client
        self._summarizer = summarizer
        self._window_keyword = window_keyword
        self._capture_interval_ms = capture_interval_ms
        self._max_frame_size = max_frame_size
        self._diff_method = diff_method
        self._diff_threshold = diff_threshold
        self._queue_max_size = queue_max_size
        self._queue_expiry_ms = queue_expiry_ms
        self._recording_duration_s = recording_duration_s

    @classmethod
    def from_settings(
        cls,
        settings: ToolkitSettings,
        vlm_client: AsyncVLMClient,
        summarizer: Summarizer | None = None,
    ) -> VideoUnderstandingPipeline:
        """Create a pipeline from a ToolkitSettings instance."""
        return cls(
            vlm_client=vlm_client,
            summarizer=summarizer,
            window_keyword=settings.capture.window_title_keyword,
            capture_interval_ms=settings.capture.screenshot_interval_ms,
            max_frame_size=settings.capture.max_size,
            diff_method=settings.algorithm.method,
            diff_threshold=settings.algorithm.diff_threshold,
            queue_max_size=settings.queue.max_size,
            queue_expiry_ms=settings.queue.expiry_time_ms,
            recording_duration_s=settings.capture.recording_duration_s,
        )

    async def run(
        self,
        stop_event: asyncio.Event | None = None,
        on_frame_sampled: OnFrameSampledCallback | None = None,
        skip_summary: bool = False,
    ) -> PipelineResult:
        """Run the full pipeline: capture -> VLM -> summary."""
        lg.info("=" * 60)
        lg.info("VideoUnderstandingPipeline started")
        lg.info(
            "duration={}s | interval={}ms | diff={} | threshold={}",
            self._recording_duration_s,
            self._capture_interval_ms,
            self._diff_method,
            self._diff_threshold,
        )
        lg.info("=" * 60)

        if stop_event is None:
            stop_event = asyncio.Event()

        queue = KeyFrameQueue(
            max_size=self._queue_max_size,
            expiry_time_ms=self._queue_expiry_ms,
        )
        results: list[FrameDescription] = []
        loop = asyncio.get_running_loop()
        self._register_signal(stop_event, loop)

        try:
            capture_task = asyncio.ensure_future(
                asyncio.to_thread(
                    run_capture_loop,
                    queue.inner,
                    loop,
                    stop_event,
                    window_keyword=self._window_keyword,
                    interval_ms=self._capture_interval_ms,
                    max_size=self._max_frame_size,
                    diff_method=self._diff_method,
                    diff_threshold=self._diff_threshold,
                    on_frame_sampled=on_frame_sampled,
                )
            )
        except WindowNotFoundError as e:
            lg.error("Window not found: {}", e)
            return PipelineResult()

        consumer_task = asyncio.ensure_future(
            self._consume_frames(queue, self._vlm_client, results, stop_event)
        )
        timer_task = asyncio.ensure_future(
            self._timer(self._recording_duration_s, stop_event)
        )

        await capture_task
        lg.info("Capture thread exited")

        try:
            await asyncio.wait_for(consumer_task, timeout=120.0)
        except asyncio.TimeoutError:
            lg.warning("Consumer timeout, forcing stop")
            consumer_task.cancel()

        timer_task.cancel()

        duration_s = float(self._recording_duration_s)
        summary = None

        if not skip_summary and results and self._summarizer is not None:
            lg.info("Summarizing {} descriptions | dropped: {}", len(results), queue.dropped_count)
            try:
                summary = await self._summarizer.summarize(results, duration_s=duration_s)
                lg.info("Summary:\n{}", summary.summary_text)
            except Exception:
                lg.exception("Summarization failed")

        lg.info("Pipeline finished")

        return PipelineResult(
            descriptions=results,
            summary=summary,
            total_keyframes=len(results),
            total_dropped=queue.dropped_count,
            duration_s=duration_s,
        )

    @staticmethod
    async def _consume_frames(
        queue: KeyFrameQueue,
        vlm_client: AsyncVLMClient,
        results: list[FrameDescription],
        stop_event: asyncio.Event,
    ) -> None:
        while True:
            if stop_event.is_set() and queue.empty():
                break
            try:
                keyframe = await asyncio.wait_for(queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            if keyframe is None:
                continue
            try:
                desc = await vlm_client.describe_frame(keyframe)
                results.append(desc)
            except Exception:
                lg.exception("VLM inference failed for frame #{}", keyframe.frame_id)
            finally:
                queue.task_done()

    @staticmethod
    async def _timer(duration_s: int, stop_event: asyncio.Event) -> None:
        start = time.monotonic()
        while not stop_event.is_set():
            if time.monotonic() - start >= duration_s:
                lg.info("Recording time reached ({:.1f}s), stopping", duration_s)
                stop_event.set()
                break
            await asyncio.sleep(0.5)

    @staticmethod
    def _register_signal(stop_event: asyncio.Event, loop: asyncio.AbstractEventLoop) -> None:
        def handler() -> None:
            lg.warning("Interrupt signal received, shutting down gracefully...")
            stop_event.set()

        try:
            loop.add_signal_handler(signal.SIGINT, handler)
        except NotImplementedError:
            signal.signal(signal.SIGINT, lambda *_: handler())
