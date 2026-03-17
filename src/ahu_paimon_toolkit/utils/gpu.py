"""GPU information and memory query utilities."""

from __future__ import annotations

import subprocess


def get_gpu_memory_mb() -> int:
    """Query current GPU memory usage (MB) via nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return int(result.stdout.strip().split("\n")[0])
    except Exception:
        return 0


def get_gpu_info() -> dict:
    """Return a dict with GPU name, total/free/used memory."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.free,memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        parts = [p.strip() for p in result.stdout.strip().split(",")]
        return {
            "name": parts[0],
            "memory_total_mb": int(parts[1]),
            "memory_free_mb": int(parts[2]),
            "memory_used_mb": int(parts[3]),
        }
    except Exception:
        return {
            "name": "unknown",
            "memory_total_mb": 0,
            "memory_free_mb": 0,
            "memory_used_mb": 0,
        }
