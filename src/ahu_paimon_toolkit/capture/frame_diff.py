"""Frame differencing algorithms: MSE and SSIM."""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray


def compute_mse(frame_a: NDArray[np.uint8], frame_b: NDArray[np.uint8]) -> float:
    """Mean Squared Error between two frames. Higher = more different."""
    diff = frame_a.astype(np.float64) - frame_b.astype(np.float64)
    return float(np.mean(diff ** 2))


def compute_ssim(frame_a: NDArray[np.uint8], frame_b: NDArray[np.uint8]) -> float:
    """Structural Similarity Index. Returns 1-SSIM so higher = more different."""
    gray_a = cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY)

    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2

    mu_a = cv2.GaussianBlur(gray_a.astype(np.float64), (11, 11), 1.5)
    mu_b = cv2.GaussianBlur(gray_b.astype(np.float64), (11, 11), 1.5)

    mu_a_sq = mu_a ** 2
    mu_b_sq = mu_b ** 2
    mu_ab = mu_a * mu_b

    sigma_a_sq = cv2.GaussianBlur(gray_a.astype(np.float64) ** 2, (11, 11), 1.5) - mu_a_sq
    sigma_b_sq = cv2.GaussianBlur(gray_b.astype(np.float64) ** 2, (11, 11), 1.5) - mu_b_sq
    sigma_ab = (
        cv2.GaussianBlur(
            gray_a.astype(np.float64) * gray_b.astype(np.float64), (11, 11), 1.5
        )
        - mu_ab
    )

    numerator = (2 * mu_ab + c1) * (2 * sigma_ab + c2)
    denominator = (mu_a_sq + mu_b_sq + c1) * (sigma_a_sq + sigma_b_sq + c2)
    ssim_map = numerator / denominator

    return float(1.0 - np.mean(ssim_map))


def compute_diff(
    frame_a: NDArray[np.uint8],
    frame_b: NDArray[np.uint8],
    method: str = "mse",
) -> float:
    """Compute frame difference using the specified algorithm."""
    if frame_a.shape != frame_b.shape:
        h, w = frame_a.shape[:2]
        frame_b = cv2.resize(frame_b, (w, h))
    if method == "mse":
        return compute_mse(frame_a, frame_b)
    elif method == "ssim":
        return compute_ssim(frame_a, frame_b)
    else:
        raise ValueError(f"Unknown diff method: {method}, supported: 'mse', 'ssim'")
