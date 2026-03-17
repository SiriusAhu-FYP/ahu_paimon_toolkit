"""Score aggregation and statistics utilities for LLM-as-Judge results."""

from __future__ import annotations

import math
from statistics import mean, stdev

from ahu_paimon_toolkit.models import JudgeScore


class ScoreAggregator:
    """Aggregate multiple JudgeScore results and compute statistics."""

    def __init__(self, scores: list[JudgeScore] | None = None) -> None:
        self._scores: list[JudgeScore] = list(scores) if scores else []

    def add(self, score: JudgeScore) -> None:
        self._scores.append(score)

    @property
    def count(self) -> int:
        return len(self._scores)

    def mean_total(self) -> float:
        if not self._scores:
            return 0.0
        return mean(s.total_score for s in self._scores)

    def stdev_total(self) -> float:
        if len(self._scores) < 2:
            return 0.0
        return stdev(s.total_score for s in self._scores)

    def ci_95_total(self) -> tuple[float, float]:
        """95% confidence interval for total score (assumes normal distribution)."""
        if len(self._scores) < 2:
            m = self.mean_total()
            return (m, m)
        m = self.mean_total()
        s = self.stdev_total()
        n = len(self._scores)
        margin = 1.96 * s / math.sqrt(n)
        return (m - margin, m + margin)

    def mean_per_dimension(self) -> dict[str, float]:
        """Compute mean score per dimension across all results."""
        if not self._scores:
            return {}
        dim_values: dict[str, list[int]] = {}
        for score in self._scores:
            for dim_name, dim_score in score.dimension_scores.items():
                dim_values.setdefault(dim_name, []).append(dim_score)
        return {name: mean(vals) for name, vals in dim_values.items()}

    def summary_dict(self) -> dict:
        """Return a summary dictionary suitable for reports."""
        ci = self.ci_95_total()
        return {
            "count": self.count,
            "mean_total": round(self.mean_total(), 2),
            "stdev_total": round(self.stdev_total(), 2),
            "ci_95_lower": round(ci[0], 2),
            "ci_95_upper": round(ci[1], 2),
            "mean_per_dimension": {
                k: round(v, 2) for k, v in self.mean_per_dimension().items()
            },
        }

    def filter_by_model(self, model_id: str) -> ScoreAggregator:
        return ScoreAggregator([s for s in self._scores if s.model_id == model_id])

    def filter_by_asset(self, asset_id: str) -> ScoreAggregator:
        return ScoreAggregator([s for s in self._scores if s.asset_id == asset_id])

    def filter_by_prompt_mode(self, prompt_mode: str) -> ScoreAggregator:
        return ScoreAggregator([s for s in self._scores if s.prompt_mode == prompt_mode])
