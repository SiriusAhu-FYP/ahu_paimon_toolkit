"""OpenCV-based video frame extraction."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from loguru import logger as lg
from numpy.typing import NDArray

from ahu_paimon_toolkit.capture.frame_diff import compute_diff
from ahu_paimon_toolkit.capture.window_capture import frame_to_base64
from ahu_paimon_toolkit.models import KeyFrame


class VideoFrameExtractor:
    """Extract keyframes from a video file using frame differencing."""

    def __init__(
        self,
        video_path: Path,
        *,
        sample_interval_ms: int = 500,
        max_size: int = 512,
        diff_method: str = "mse",
        diff_threshold: float = 500.0,
    ) -> None:
        self.video_path = video_path
        self.sample_interval_ms = sample_interval_ms
        self.max_size = max_size
        self.diff_method = diff_method
        self.diff_threshold = diff_threshold

    def get_duration_s(self) -> float:
        """Get video duration in seconds."""
        cap = cv2.VideoCapture(str(self.video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.release()
        if fps <= 0:
            raise ValueError(f"Cannot read video FPS: {self.video_path}")
        return frame_count / fps

    def extract_keyframes(self) -> list[KeyFrame]:
        """Extract keyframes from the video file.

        Returns a list of KeyFrame objects with Base64-encoded images.
        """
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {self.video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = max(1, int(fps * self.sample_interval_ms / 1000))

        last_keyframe: NDArray[np.uint8] | None = None
        keyframes: list[KeyFrame] = []
        frame_idx = 0
        keyframe_id = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_interval != 0:
                frame_idx += 1
                continue

            timestamp_ms = int(frame_idx / fps * 1000)

            h, w = frame.shape[:2]
            long_edge = max(h, w)
            if long_edge > self.max_size:
                scale = self.max_size / long_edge
                frame = cv2.resize(
                    frame,
                    (int(w * scale), int(h * scale)),
                    interpolation=cv2.INTER_AREA,
                )

            if last_keyframe is not None:
                diff = compute_diff(last_keyframe, frame, self.diff_method)
                if diff < self.diff_threshold:
                    frame_idx += 1
                    continue

            last_keyframe = frame.copy()
            b64 = frame_to_base64(frame)

            keyframes.append(KeyFrame(
                frame_id=keyframe_id,
                timestamp_ms=timestamp_ms,
                base64_image=b64,
            ))
            keyframe_id += 1
            frame_idx += 1

        cap.release()
        lg.info(
            "Extracted {} keyframes from {} (total {} frames sampled)",
            len(keyframes), self.video_path.name, frame_idx,
        )
        return keyframes
