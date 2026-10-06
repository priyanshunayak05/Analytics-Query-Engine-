"""Hybrid planner: feedback correction, GenAI plan, then deterministic fallback."""

from __future__ import annotations

import calendar
import re
from datetime import date
from typing import Any

from .catalog import SemanticCatalog
from .feedback import FeedbackStore
from .llm import LLMPlanner
from .models import FilterSpec, QueryPlan


MONTHS = {name.lower(): index for index, name in enumerate(calendar.month_name) if name}
MONTHS.update({name.lower(): index for index, name in enumerate(calendar.month_abbr) if name})


class HybridPlanner:
    def __init__(
        self,
        catalog: SemanticCatalog,
        feedback: FeedbackStore,
        llm: LLMPlanner | None = None,
    ):
        self.catalog = catalog
        self.feedback = feedback
        self.llm = llm

    def plan(self, query: str) -> QueryPlan:
        correction = self.feedback.corrected_plan(query)
        if correction:
            plan = QueryPlan.from_dict(correction, source="feedback")
            plan.semantic_confidence = max(plan.semantic_confidence, 0.98)
            return plan

        if self.llm:
            try:
                payload = self.llm.plan(query, self.catalog, self.feedback.prompt_examples())
                return QueryPlan.from_dict(payload, source="llm")
            except Exception as exc:  # fallback is intentional and surfaced as an assumption
                plan = self._heuristic_plan(query)
                plan.assumptions.append(f"GenAI planner unavailable; deterministic fallback used ({type(exc).__name__}).")
                plan.semantic_confidence = min(plan.semantic_confidence, 0.80)
                return plan
        return self._heuristic_plan(query)

    def _heuristic_plan(self, query: str) -> QueryPlan:
        text = " ".join(query.lower().strip().split())
        metric = self._metric(text)
        filters, time_assumptions = self._filters(text)
        dimensions = self._dimensions(text)
        limit_match = re.search(r"\b(?:top|bottom)\s+(\d+)\b", text)
        limit = int(limit_match.group(1)) if limit_match else None
        strongest = bool(re.search(r"\b(highest|highest-selling|most|maximum|max|best|greatest)\b", text))
        weakest = bool(re.search(r"\b(lowest|least|minimum|min|worst|smallest)\b", text))
        assumptions = list(time_assumptions)

        if "target" in text:
            plan = QueryPlan("target_comparison", "revenue", ["region"], filters,
                             comparison="<" if "miss" in text or "below" in text else ">=")
        elif "contribution" in text or "share" in text or "percent" in text or "%" in text:
            dimension = self._first_group_dimension(text, dimensions) or "product_category"
            plan = QueryPlan("contribution", metric, [dimension], filters)
        elif "yoy" in text or "year over year" in text:
            plan = QueryPlan("yoy", metric, ["order_year"], filters, time_grain="year")
        elif ("top" in text or "bottom" in text) and ("each" in text or "per region" in text or "per country" in text):
            partition = self._partition(text)
            rank_dimension = self._rank_dimension(text)
            analysis_type = "nested_top_aggregate" if "customer" in text and "revenue of" in text else "top_n_per_group"
            plan = QueryPlan(
                analysis_type, metric, partition, filters, limit or 1,
                rank_dimension=rank_dimension, partition_by=partition,
            )
        elif "top" in text or "bottom" in text or strongest or weakest:
            dimension = self._rank_dimension(text)
            plan = QueryPlan(
                "rank", metric, [dimension], filters, limit or (1 if (strongest or weakest) else 10),
                rank_dimension=dimension, comparison="asc" if ("bottom" in text or weakest) else "desc",
            )
        else:
            plan = QueryPlan("aggregate", metric, dimensions, filters)

        plan.source = "heuristic"
        plan.assumptions.extend(assumptions)
        plan.semantic_confidence = self._confidence(plan, text)
        return plan

    def _metric(self, text: str) -> str:
        candidates = {key: key for key in self.catalog.metrics}
        candidates.update({key.lower(): value for key, value in self.catalog.synonyms.items()})
        candidates.update({"average order value": "avg_order_value", "order value": "avg_order_value"})
        for term in sorted(candidates, key=len, reverse=True):
            if re.search(rf"\b{re.escape(term)}\b", text):
                mapped = candidates[term]
                return "orders" if "count(order_id)" in mapped else mapped
        return "revenue"

    def _dimensions(self, text: str) -> list[str]:
        aliases = {
            "category": "product_category", "subcategory": "product_subcategory",
            "segment": "customer_segment", "customer": "customer_id",
            "product": "product_name", "month": "order_month", "year": "order_year",
        }
        found: list[tuple[int, str]] = []
        for dimension in self.catalog.dimensions:
            term = dimension.replace("_", " ")
            match = re.search(rf"\b{re.escape(term)}s?\b", text)
            if match:
                found.append((match.start(), dimension))
        for term, dimension in aliases.items():
            if term == "product" and re.search(r"\bproduct\s+(?:sub)?category\b", text):
                continue
            if term == "category" and re.search(r"\bproduct\s+category\b", text):
                # The physical semantic field is product_category; this alias is redundant.
                continue
            match = re.search(rf"\b{term}s?\b", text)
            if match and not any(item[1] == dimension for item in found):
                found.append((match.start(), dimension))
        return [dimension for _, dimension in sorted(found)]

    def _filters(self, text: str) -> tuple[list[FilterSpec], list[str]]:
        filters: list[FilterSpec] = []
        assumptions: list[str] = []
        for field, values in self.catalog.values.items():
            for value in sorted(values, key=len, reverse=True):
                if re.search(rf"(?<!\w){re.escape(value.lower())}(?!\w)", text):
                    filters.append(FilterSpec(field, "=", value))
                    break

        explicit = re.search(r"\b(20\d{2})[-/](0?[1-9]|1[0-2])\b", text)
        month_number = None
        for name, number in MONTHS.items():
            if re.search(rf"\b{re.escape(name)}\b", text):
                month_number = number
                break
        if explicit:
            filters.append(FilterSpec("order_month", "=", f"{explicit.group(1)}-{int(explicit.group(2)):02d}"))
        elif month_number:
            year_match = re.search(r"\b(20\d{2})\b", text)
            year = int(year_match.group(1)) if year_match else self._latest_year()
            filters.append(FilterSpec("order_month", "=", f"{year}-{month_number:02d}"))
            if not year_match:
                assumptions.append(f"Interpreted month as {year}-{month_number:02d}, using the latest data year.")
        elif "last month" in text:
            latest = self._latest_date()
            first = latest.replace(day=1)
            previous = date(first.year - (first.month == 1), 12 if first.month == 1 else first.month - 1, 1)
            filters.append(FilterSpec("order_month", "=", previous.strftime("%Y-%m")))
            assumptions.append("Resolved 'last month' relative to the latest date in the dataset.")
        else:
            quarter = re.search(r"\bq([1-4])(?:\s+(20\d{2}))?\b", text)
            if "this quarter" in text:
                latest = self._latest_date()
                quarter_number, year = (latest.month - 1) // 3 + 1, latest.year
                assumptions.append("Resolved 'this quarter' relative to the latest date in the dataset.")
            elif quarter:
                quarter_number = int(quarter.group(1))
                year = int(quarter.group(2) or self._latest_year())
                if not quarter.group(2):
                    assumptions.append(f"Interpreted Q{quarter_number} as {year}, using the latest data year.")
            else:
                quarter_number = None
                year = None
            if quarter_number and year:
                start_month = (quarter_number - 1) * 3 + 1
                filters.extend([
                    FilterSpec("order_month", ">=", f"{year}-{start_month:02d}"),
                    FilterSpec("order_month", "<=", f"{year}-{start_month + 2:02d}"),
                ])
            else:
                year_match = re.search(r"\b(?:in|during|for)\s+(20\d{2})\b", text)
                if year_match:
                    filters.append(FilterSpec("order_year", "=", year_match.group(1)))
        return self._dedupe_filters(filters), assumptions

    def _latest_date(self) -> date:
        return date.fromisoformat(self.catalog.latest_date) if self.catalog.latest_date else date.today()

    def _latest_year(self) -> int:
        return self._latest_date().year

    @staticmethod
    def _dedupe_filters(filters: list[FilterSpec]) -> list[FilterSpec]:
        seen = set()
        result = []
        for item in filters:
            key = (item.field, item.operator, str(item.value))
            if key not in seen:
                seen.add(key)
                result.append(item)
        return result

    @staticmethod
    def _first_group_dimension(text: str, dimensions: list[str]) -> str | None:
        for dimension in dimensions:
            if dimension not in {"order_month", "order_year"}:
                return dimension
        return None

    @staticmethod
    def _partition(text: str) -> list[str]:
        if "country" in text:
            return ["country"]
        if "category" in text and "each category" in text:
            return ["product_category"]
        return ["region"]

    @staticmethod
    def _rank_dimension(text: str) -> str:
        if "customer" in text:
            return "customer_id"
        if re.search(r"\bcit(?:y|ies)\b", text):
            return "city"
        if "categor" in text:
            return "product_category"
        if "subcategor" in text:
            return "product_subcategory"
        if "country" in text and "per country" not in text:
            return "country"
        if "region" in text and "per region" not in text and "each region" not in text:
            return "region"
        return "product_name"

    @staticmethod
    def _confidence(plan: QueryPlan, text: str) -> float:
        score = 0.76
        if plan.analysis_type in {"target_comparison", "contribution", "yoy"}:
            score += 0.08
        if plan.analysis_type in {"rank", "top_n_per_group", "nested_top_aggregate"} and plan.limit:
            score += 0.06
        if plan.filters:
            score += 0.05
        if plan.metric == "revenue" and not any(term in text for term in ("revenue", "sales", "income")):
            score -= 0.08
        return round(max(0.0, min(0.94, score)), 2)
