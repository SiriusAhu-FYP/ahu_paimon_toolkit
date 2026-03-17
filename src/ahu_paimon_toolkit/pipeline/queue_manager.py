"""Async queue with expiry-based eviction for keyframes.

Core principle: prefer dropping stale frames over processing outdated data.
"""

from __future__ import annotations

import asyncio
import time

from loguru import logger as lg

from ahu_paimon_toolkit.models import KeyFrame


class KeyFrameQueue:
    """Wraps asyncio.Queue with automatic expiry eviction.

    Args:
        max_size: Maximum queue capacity.
        expiry_time_ms: Keyframe validity period (ms); older frames are dropped.
    """

    def __init__(self, max_size: int = 50, expiry_time_ms: int = 10000) -> None:
        self._max_size = max_size
        self._expiry_time_ms = expiry_time_ms
        self._queue: asyncio.Queue[KeyFrame] = asyncio.Queue(maxsize=max_size)
        self._dropped_count = 0

    @property
    def qsize(self) -> int:
        return self._queue.qsize()

    @property
    def dropped_count(self) -> int:
        return self._dropped_count

    @property
    def inner(self) -> asyncio.Queue[KeyFrame]:
        """Expose the underlying queue for thread-safe enqueue via run_coroutine_threadsafe."""
        return self._queue

    def put_nowait(self, item: KeyFrame) -> None:
        self._queue.put_nowait(item)

    async def put(self, item: KeyFrame) -> None:
        await self._queue.put(item)

    async def get(self) -> KeyFrame | None:
        """Get a keyframe, automatically skipping expired ones."""
        while True:
            item = await self._queue.get()
            age_ms = (time.monotonic() - item.created_at) * 1000
            if age_ms > self._expiry_time_ms:
                self._dropped_count += 1
                lg.warning(
                    "Dropping expired keyframe #{} | age={:.0f}ms > expiry={}ms | total dropped: {}",
                    item.frame_id, age_ms, self._expiry_time_ms, self._dropped_count,
                )
                self._queue.task_done()
                continue
            return item

    def task_done(self) -> None:
        self._queue.task_done()

    async def join(self) -> None:
        await self._queue.join()

    def empty(self) -> bool:
        return self._queue.empty()
