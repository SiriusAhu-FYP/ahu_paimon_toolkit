"""Window-based screen capture with mss and pywin32."""

from __future__ import annotations

import asyncio
import base64
import time
from typing import TYPE_CHECKING

import cv2
import mss
import numpy as np
import win32gui
from loguru import logger as lg

from ahu_paimon_toolkit.models import KeyFrame, OnFrameSampledCallback

if TYPE_CHECKING:
    from numpy.typing import NDArray


class WindowNotFoundError(Exception):
    pass


def _enum_windows() -> list[tuple[int, str]]:
    """Enumerate all visible windows, returning (hwnd, title) pairs."""
    results: list[tuple[int, str]] = []

    def callback(hwnd: int, _: object) -> bool:
        if win32gui.IsWindowVisible(hwnd):
            title = win32gui.GetWindowText(hwnd)
            if title:
                results.append((hwnd, title))
        return True

    win32gui.EnumWindows(callback, None)
    return results


def find_window(keyword: str) -> tuple[int, str]:
    """Find window by fuzzy title match. Returns (hwnd, title)."""
    windows = _enum_windows()
    keyword_lower = keyword.lower()

    candidates = [
        (hwnd, title)
        for hwnd, title in windows
        if keyword_lower in title.lower()
    ]

    if not candidates:
        raise WindowNotFoundError(
            f"No window found with title containing '{keyword}'. "
            f"Visible windows: {[t for _, t in windows[:10]]}"
        )

    best = min(candidates, key=lambda x: len(x[1]))
    lg.info("Window matched: hwnd={}, title='{}'", best[0], best[1])
    return best


def capture_window(hwnd: int, max_size: int) -> NDArray[np.uint8]:
    """Capture a window screenshot, resized to max_size long edge."""
    left, top, right, bottom = win32gui.GetWindowRect(hwnd)
    width = right - left
    height = bottom - top

    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid window size: {width}x{height}")

    with mss.mss() as sct:
        monitor = {"left": left, "top": top, "width": width, "height": height}
        screenshot = sct.grab(monitor)
        frame = np.array(screenshot, dtype=np.uint8)[:, :, :3]

    return _resize_frame(frame, max_size)


def _resize_frame(frame: NDArray[np.uint8], max_size: int) -> NDArray[np.uint8]:
    """Proportionally resize so the long edge does not exceed max_size."""
    h, w = frame.shape[:2]
    long_edge = max(h, w)
    if long_edge <= max_size:
        return frame
    scale = max_size / long_edge
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


def frame_to_base64(frame: NDArray[np.uint8], quality: int = 85) -> str:
    """Encode a BGR numpy array as a JPEG Base64 string."""
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    success, buffer = cv2.imencode(".jpg", frame, encode_params)
    if not success:
        raise RuntimeError("JPEG encoding failed")
    return base64.b64encode(buffer.tobytes()).decode("ascii")


def run_capture_loop(
    queue: asyncio.Queue[KeyFrame],
    loop: asyncio.AbstractEventLoop,
    stop_event: asyncio.Event,
    *,
    window_keyword: str = "PotPlayer",
    interval_ms: int = 500,
    max_size: int = 512,
    diff_method: str = "mse",
    diff_threshold: float = 500.0,
    on_frame_sampled: OnFrameSampledCallback | None = None,
) -> None:
    """Capture loop: runs in a separate thread, pushes keyframes to async queue.

    All parameters are explicit -- no implicit config dependency.
    """
    interval_s = interval_ms / 1000.0

    hwnd, title = find_window(window_keyword)
    lg.info(
        "Capture loop started | window='{}' | interval={}ms | method={} | threshold={}",
        title, interval_ms, diff_method, diff_threshold,
    )

    last_keyframe: NDArray[np.uint8] | None = None
    frame_id = 0
    sample_idx = 0
    start_time = time.monotonic()

    while not stop_event.is_set():
        iter_start = time.perf_counter()

        try:
            frame = capture_window(hwnd, max_size)
        except Exception:
            lg.warning("Capture failed, window may be closed or minimized")
            time.sleep(interval_s)
            continue

        elapsed_ms = int((time.monotonic() - start_time) * 1000)

        if last_keyframe is not None:
            from ahu_paimon_toolkit.capture.frame_diff import compute_diff

            diff = compute_diff(last_keyframe, frame, diff_method)
            if diff < diff_threshold:
                reason = f"diff {diff:.2f} < threshold {diff_threshold:.2f}"
                if on_frame_sampled is not None:
                    on_frame_sampled(sample_idx, elapsed_ms, frame, diff, False, reason)
                sample_idx += 1
                elapsed = time.perf_counter() - iter_start
                time.sleep(max(0, interval_s - elapsed))
                continue
            reason = f"diff {diff:.2f} >= threshold {diff_threshold:.2f}"
        else:
            diff = None
            reason = "first frame"

        if on_frame_sampled is not None:
            on_frame_sampled(sample_idx, elapsed_ms, frame, diff, True, reason)
        sample_idx += 1

        last_keyframe = frame.copy()
        b64 = frame_to_base64(frame)

        keyframe = KeyFrame(
            frame_id=frame_id,
            timestamp_ms=elapsed_ms,
            base64_image=b64,
        )
        frame_id += 1

        future = asyncio.run_coroutine_threadsafe(
            _safe_put(queue, keyframe), loop
        )
        try:
            future.result(timeout=2.0)
        except Exception:
            lg.warning("Queue full, dropping keyframe #{}", keyframe.frame_id)

        elapsed = time.perf_counter() - iter_start
        time.sleep(max(0, interval_s - elapsed))

    lg.info("Capture loop ended | sampled {} frames, extracted {} keyframes", sample_idx, frame_id)


async def _safe_put(queue: asyncio.Queue[KeyFrame], item: KeyFrame) -> None:
    """Non-blocking enqueue; raises QueueFull if full."""
    try:
        queue.put_nowait(item)
    except asyncio.QueueFull:
        raise
