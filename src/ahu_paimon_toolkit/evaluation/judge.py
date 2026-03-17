"""LLM-as-Judge: evaluate VLM responses using a cloud LLM with structured rubrics."""

from __future__ import annotations

import json

import httpx
from loguru import logger as lg

from ahu_paimon_toolkit.models import JudgeScore


class LLMJudge:
    """Evaluate VLM outputs by sending them alongside a scoring rubric to a judge LLM.

    The judge model (e.g. DeepSeek) scores the response on multiple dimensions
    and returns structured feedback.
    """

    def __init__(
        self,
        api_key: str,
        api_base_url: str = "https://api.deepseek.com",
        model: str = "deepseek-chat",
        timeout: float = 120.0,
    ) -> None:
        self._api_key = api_key
        self._api_base_url = api_base_url
        self._model = model
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

    async def evaluate(
        self,
        *,
        asset_id: str,
        model_id: str,
        prompt_mode: str,
        vlm_response: str,
        task_definition: dict,
        reference_answer: dict,
        scoring_rubric: dict,
        grading_prompt: str,
    ) -> JudgeScore:
        """Send VLM response + rubric to the judge model and parse structured scores.

        Args:
            asset_id: ID of the test asset (e.g. "01_Minecraft").
            model_id: ID of the tested VLM.
            prompt_mode: "A_description" or "B_assistant".
            vlm_response: The VLM's raw text response to evaluate.
            task_definition: Dict with scene_summary and evaluation_intent.
            reference_answer: Dict with minimum_expected_points, good_additional_points, etc.
            scoring_rubric: Dict with scale, dimensions, hard_penalties.
            grading_prompt: System-level instruction for the judge.
        """
        user_message = self._build_grading_message(
            vlm_response=vlm_response,
            task_definition=task_definition,
            reference_answer=reference_answer,
            scoring_rubric=scoring_rubric,
        )

        client = await self._ensure_client()
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": grading_prompt},
                {"role": "user", "content": user_message},
            ],
            "max_tokens": 2048,
            "temperature": 0.1,
            "response_format": {"type": "json_object"},
        }

        lg.info("Sending evaluation request for {} / {} / {}", asset_id, model_id, prompt_mode)
        response = await client.post("/v1/chat/completions", json=payload)
        response.raise_for_status()
        data = response.json()
        raw_content = data["choices"][0]["message"]["content"]

        return self._parse_judge_response(
            raw_content,
            asset_id=asset_id,
            model_id=model_id,
            prompt_mode=prompt_mode,
        )

    @staticmethod
    def _build_grading_message(
        *,
        vlm_response: str,
        task_definition: dict,
        reference_answer: dict,
        scoring_rubric: dict,
    ) -> str:
        """Build the user message containing all context for the judge."""
        dimensions_text = ""
        for dim in scoring_rubric.get("dimensions", []):
            dimensions_text += f"\n### {dim['name']}: {dim['description']}\n"
            for score_val, criteria in dim.get("criteria", {}).items():
                dimensions_text += f"  - Score {score_val}: {criteria}\n"

        penalties_text = "\n".join(
            f"  - {p}" for p in scoring_rubric.get("hard_penalties", [])
        )

        min_points = "\n".join(
            f"  - {p}" for p in reference_answer.get("minimum_expected_points", [])
        )
        good_points = "\n".join(
            f"  - {p}" for p in reference_answer.get("good_additional_points", [])
        )

        return f"""## Task Context

**Scene Summary**: {task_definition.get('scene_summary', 'N/A')}
**Evaluation Intent**: {task_definition.get('evaluation_intent', 'N/A')}

## Reference Answer

### Minimum Expected Points:
{min_points}

### Good Additional Points:
{good_points}

## Scoring Rubric
{dimensions_text}

## Hard Penalties (automatic score reduction):
{penalties_text}

## VLM Response to Evaluate:

{vlm_response}

---

Please score the above VLM response strictly according to the rubric. Return a JSON object with these fields:
- "dimension_scores": {{"dimension_name": score, ...}} (each score is 0, 1, or 2)
- "total_score": sum of all dimension scores (integer)
- "strengths": [list of strengths]
- "weaknesses": [list of weaknesses]
- "missing_points": [key points from the reference that the response missed]
- "hallucinations": [any unsupported claims made by the response]
"""

    @staticmethod
    def _parse_judge_response(
        raw_content: str,
        *,
        asset_id: str,
        model_id: str,
        prompt_mode: str,
    ) -> JudgeScore:
        """Parse the judge's JSON response into a JudgeScore."""
        try:
            parsed = json.loads(raw_content)
        except json.JSONDecodeError:
            lg.warning("Failed to parse judge JSON, attempting extraction")
            start = raw_content.find("{")
            end = raw_content.rfind("}") + 1
            if start >= 0 and end > start:
                parsed = json.loads(raw_content[start:end])
            else:
                return JudgeScore(
                    asset_id=asset_id,
                    model_id=model_id,
                    prompt_mode=prompt_mode,
                    raw_response=raw_content,
                )

        dimension_scores = parsed.get("dimension_scores", {})
        total = parsed.get("total_score", sum(dimension_scores.values()))

        return JudgeScore(
            asset_id=asset_id,
            model_id=model_id,
            prompt_mode=prompt_mode,
            dimension_scores=dimension_scores,
            total_score=total,
            max_score=10,
            strengths=parsed.get("strengths", []),
            weaknesses=parsed.get("weaknesses", []),
            missing_points=parsed.get("missing_points", []),
            hallucinations=parsed.get("hallucinations", []),
            raw_response=raw_content,
        )

    async def close(self) -> None:
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
            lg.debug("LLMJudge client closed")
