"""Public orchestration API."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .catalog import SemanticCatalog
from .executor import AnalyticsExecutor
from .feedback import FeedbackStore
from .llm import LLMPlanner
from .loaders import load_csv, load_json
from .planner import HybridPlanner
from .validator import PlanValidationError, PlanValidator


class AnalyticsEngine:
    def __init__(
        self,
        sales_path: str | Path,
        targets_path: str | Path,
        dictionary_path: str | Path,
        feedback_path: str | Path | None = None,
        llm_url: str | None = None,
        llm_model: str | None = None,
        llm_api_key: str | None = None,
    ):
        sales = load_csv(sales_path)
        targets = load_csv(targets_path)
        dictionary = load_json(dictionary_path)
        if not sales:
            raise ValueError("sales dataset is empty")
        self.catalog = SemanticCatalog.build(dictionary, sales)
        self.feedback = FeedbackStore(feedback_path)
        llm = LLMPlanner(llm_url, llm_model, llm_api_key) if llm_url and llm_model else None
        self.planner = HybridPlanner(self.catalog, self.feedback, llm)
        self.validator = PlanValidator(self.catalog)
        self.executor = AnalyticsExecutor(sales, targets)

    def answer(self, query: str) -> dict[str, Any]:
        if not query or not query.strip():
            raise ValueError("query must not be empty")
        plan = self.planner.plan(query)
        try:
            self.validator.validate(plan)
        except PlanValidationError as error:
            if plan.source != "llm":
                raise
            plan = self.planner._heuristic_plan(query)
            plan.assumptions.append(f"Rejected unsafe or invalid GenAI plan: {error}")
            plan.semantic_confidence = min(plan.semantic_confidence, 0.78)
            self.validator.validate(plan)

        compiled = self.executor.compile(plan)
        result = self.executor.execute(compiled)
        confidence = self._final_confidence(plan.semantic_confidence, plan.analysis_type, result)
        params_note = f"\n-- parameters: {json.dumps(compiled.params, ensure_ascii=False)}" if compiled.params else ""
        return {
            "query": query,
            "generated_logic": compiled.sql + params_note,
            "result": result,
            "confidence_score": confidence,
            "explanation": self._explanation(plan, compiled.description, result),
        }

    @staticmethod
    def _final_confidence(base: float, analysis_type: str, result: list[dict[str, Any]]) -> float:
        score = base + 0.04  # successful validation and execution
        if not result:
            score -= 0.12
        if analysis_type == "yoy" and not any(row.get("yoy_growth_pct") is not None for row in result):
            score -= 0.18
        return round(max(0.0, min(1.0, score)), 2)

    @staticmethod
    def _explanation(plan: Any, execution: str, result: list[dict[str, Any]]) -> str:
        understood = f"Understood as {plan.analysis_type.replace('_', ' ')} analysis of {plan.metric}"
        if plan.dimensions:
            understood += f" by {', '.join(plan.dimensions)}"
        if plan.filters:
            rendered = ", ".join(f"{item.field} {item.operator} {item.value}" for item in plan.filters)
            understood += f", filtered to {rendered}"
        text = f"{understood}. Generated a validated, parameterized SQLite query to {execution.lower()}."
        if plan.source == "llm":
            text += " The semantic plan was produced by the configured GenAI model and passed the allowlist validator."
        elif plan.source == "feedback":
            text += " An accepted correction from the feedback log supplied the semantic plan."
        else:
            text += " The deterministic fallback planner supplied the semantic plan; configure GenAI for broader language coverage."
        if plan.analysis_type == "yoy" and not any(row.get("yoy_growth_pct") is not None for row in result):
            text += " A growth rate cannot be calculated because the filtered data contains no prior-year comparison."
        if plan.assumptions:
            text += " Assumptions: " + " ".join(plan.assumptions)
        return text

