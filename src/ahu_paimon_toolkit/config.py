"""Toolkit-level configuration helpers."""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Literal

from loguru import logger as lg
from pydantic import BaseModel, Field


class CaptureConfig(BaseModel):
    window_title_keyword: str = "PotPlayer"
    screenshot_interval_ms: int = 500
    max_size: int = 512
    recording_duration_s: int = 20


class AlgorithmConfig(BaseModel):
    method: Literal["mse", "ssim"] = "mse"
    diff_threshold: float = 500.0


class QueueConfig(BaseModel):
    max_size: int = 50
    expiry_time_ms: int = 10000


class LLMConfig(BaseModel):
    vllm_base_url: str = "http://localhost:8000/v1"
    vllm_model: str = "Qwen/Qwen3-VL-2B-Instruct"


class LogConfig(BaseModel):
    log_dir: str = "logs"
    console_level: str = "INFO"
    file_level: str = "DEBUG"


class DeepSeekConfig(BaseModel):
    api_key: str = ""
    api_base_url: str = "https://api.deepseek.com"
    model: str = "deepseek-chat"


class ToolkitSettings(BaseModel):
    """Aggregated settings for the toolkit."""

    capture: CaptureConfig = Field(default_factory=CaptureConfig)
    algorithm: AlgorithmConfig = Field(default_factory=AlgorithmConfig)
    queue: QueueConfig = Field(default_factory=QueueConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    log: LogConfig = Field(default_factory=LogConfig)
    deepseek: DeepSeekConfig = Field(default_factory=DeepSeekConfig)


def setup_logging(
    cfg: LogConfig | None = None,
    *,
    log_root: Path | None = None,
) -> Path | None:
    """Configure loguru: console + rotating file output.

    Returns the log file path, or None if file logging was not set up.
    """
    if cfg is None:
        cfg = LogConfig()

    lg.remove()
    lg.add(
        sys.stderr,
        level=cfg.console_level.upper(),
        format="<green>{time:HH:mm:ss.SSS}</green> | "
               "<level>{level:<7}</level> | "
               "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
               "<level>{message}</level>",
    )

    log_path = Path(cfg.log_dir)
    if log_root is not None:
        log_dir = log_root / cfg.log_dir if not log_path.is_absolute() else log_path
    else:
        log_dir = log_path

    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = log_dir / f"{timestamp}.log"

    lg.add(
        str(log_file),
        level=cfg.file_level.upper(),
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<7} | "
               "{name}:{function}:{line} - {message}",
        rotation="10 MB",
        encoding="utf-8",
    )

    lg.info("Log system initialized -> {}", log_file)
    return log_file
