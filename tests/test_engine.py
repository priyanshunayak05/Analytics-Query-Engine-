from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from analytics_engine import AnalyticsEngine
from analytics_engine.models import FilterSpec, QueryPlan
from analytics_engine.validator import PlanValidationError


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


import re
import calendar
from datetime import date
from analytics_engine.llm import LLMPlanner

MONTHS = {name.lower(): index for index, name in enumerate(calendar.month_name) if name}
MONTHS.update({name.lower(): index for index, name in enumerate(calendar.month_abbr) if name})

class MockLLMPlanner:
    def __init__(self, catalog):
        self.catalog = catalog

    def plan(self, query: str, catalog, examples) -> dict:
        text = " ".join(query.lower().strip().split())
        metric = self._metric(text)
        filters, time_assumptions = self._filters(text)
        dimensions = self._dimensions(text)
        limit_match = re.search(r"\b(?:top|bottom)\s+(\d+)\b", text)
        limit = int(limit_match.group(1)) if limit_match else None
        strongest = bool(re.search(r"\b(highest|highest-selling|most|maximum|max|best|greatest)\b", text))
        weakest = bool(re.search(r"\b(lowest|least|minimum|min|worst|smallest)\b", text))

        if "target" in text:
            return {"analysis_type": "target_comparison", "metric": "revenue", "dimensions": ["region"], "filters": [f.__dict__ for f in filters], "comparison": "<" if "miss" in text or "below" in text else ">="}
        elif "contribution" in text or "share" in text or "percent" in text or "%" in text:
            dimension = self._first_group_dimension(text, dimensions) or "product_category"
            return {"analysis_type": "contribution", "metric": metric, "dimensions": [dimension], "filters": [f.__dict__ for f in filters]}
        elif "yoy" in text or "year over year" in text:
            return {"analysis_type": "yoy", "metric": metric, "dimensions": ["order_year"], "filters": [f.__dict__ for f in filters], "time_grain": "year"}
        elif ("top" in text or "bottom" in text) and ("each" in text or "per region" in text or "per country" in text):
            partition = self._partition(text)
            rank_dimension = self._rank_dimension(text)
            analysis_type = "nested_top_aggregate" if "customer" in text and "revenue of" in text else "top_n_per_group"
            return {"analysis_type": analysis_type, "metric": metric, "dimensions": partition, "filters": [f.__dict__ for f in filters], "limit": limit or 1, "rank_dimension": rank_dimension, "partition_by": partition}
        elif "top" in text or "bottom" in text or strongest or weakest:
            dimension = self._rank_dimension(text)
            return {"analysis_type": "rank", "metric": metric, "dimensions": [dimension], "filters": [f.__dict__ for f in filters], "limit": limit or (1 if (strongest or weakest) else 10), "rank_dimension": dimension, "comparison": "asc" if ("bottom" in text or weakest) else "desc"}
        else:
            return {"analysis_type": "aggregate", "metric": metric, "dimensions": dimensions, "filters": [f.__dict__ for f in filters]}

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
        aliases = {"category": "product_category", "subcategory": "product_subcategory", "segment": "customer_segment", "customer": "customer_id", "product": "product_name", "month": "order_month", "year": "order_year"}
        found: list[tuple[int, str]] = []
        for dimension in self.catalog.dimensions:
            term = dimension.replace("_", " ")
            match = re.search(rf"\b{re.escape(term)}s?\b", text)
            if match:
                found.append((match.start(), dimension))
        for term, dimension in aliases.items():
            if term == "product" and re.search(r"\bproduct\s+(?:sub)?category\b", text): continue
            if term == "category" and re.search(r"\bproduct\s+category\b", text): continue
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
        if explicit: filters.append(FilterSpec("order_month", "=", f"{explicit.group(1)}-{int(explicit.group(2)):02d}"))
        elif month_number:
            year_match = re.search(r"\b(20\d{2})\b", text)
            year = int(year_match.group(1)) if year_match else self._latest_year()
            filters.append(FilterSpec("order_month", "=", f"{year}-{month_number:02d}"))
        elif "last month" in text:
            latest = self._latest_date()
            first = latest.replace(day=1)
            previous = date(first.year - (first.month == 1), 12 if first.month == 1 else first.month - 1, 1)
            filters.append(FilterSpec("order_month", "=", previous.strftime("%Y-%m")))
        else:
            quarter = re.search(r"\bq([1-4])(?:\s+(20\d{2}))?\b", text)
            if "this quarter" in text:
                latest = self._latest_date()
                quarter_number, year = (latest.month - 1) // 3 + 1, latest.year
            elif quarter:
                quarter_number = int(quarter.group(1))
                year = int(quarter.group(2) or self._latest_year())
            else:
                quarter_number = None
                year = None
            if quarter_number and year:
                start_month = (quarter_number - 1) * 3 + 1
                filters.extend([FilterSpec("order_month", ">=", f"{year}-{start_month:02d}"), FilterSpec("order_month", "<=", f"{year}-{start_month + 2:02d}")])
            else:
                year_match = re.search(r"\b(?:in|during|for)\s+(20\d{2})\b", text)
                if year_match: filters.append(FilterSpec("order_year", "=", year_match.group(1)))
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
            if dimension not in {"order_month", "order_year"}: return dimension
        return None

    @staticmethod
    def _partition(text: str) -> list[str]:
        if "country" in text: return ["country"]
        if "category" in text and "each category" in text: return ["product_category"]
        return ["region"]

    @staticmethod
    def _rank_dimension(text: str) -> str:
        if "customer" in text: return "customer_id"
        if re.search(r"\bcit(?:y|ies)\b", text): return "city"
        if "categor" in text: return "product_category"
        if "subcategor" in text: return "product_subcategory"
        if "country" in text and "per country" not in text: return "country"
        if "region" in text and "per region" not in text and "each region" not in text: return "region"
        return "product_name"


def engine(feedback: str | None = None) -> AnalyticsEngine:
    import unittest.mock
    eng = AnalyticsEngine(
        DATA / "sales_data.csv",
        DATA / "targets.csv",
        DATA / "data_dictionary.json",
        feedback_path=feedback,
        llm_url="http://mock",
        llm_model="mock"
    )
    eng.planner.llm = MockLLMPlanner(eng.catalog)
    return eng


class EngineAcceptanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.engine = engine()

    def test_total_sales_with_country_and_inferred_month(self) -> None:
        answer = self.engine.answer("Total sales in India for March")
        self.assertEqual(answer["result"], [{"revenue": 108.0}])
        self.assertIn('"country" = ?', answer["generated_logic"])

    def test_rank(self) -> None:
        result = self.engine.answer("Top 2 cities by profit")["result"]
        self.assertEqual([row["city"] for row in result], ["New York", "San Francisco"])

    def test_unseen_superlative_returns_single_winner(self) -> None:
        result = self.engine.answer("Which country generated the most revenue?")["result"]
        self.assertEqual(result, [{"country": "USA", "revenue": 3262.4}])

    def test_unseen_bottom_n_is_respected(self) -> None:
        result = self.engine.answer("What are the bottom 2 products by profit?")["result"]
        self.assertEqual([row["product_name"] for row in result], ["Mouse", "A4 Paper Pack"])

    def test_unseen_highest_category_does_not_add_product_dimension(self) -> None:
        result = self.engine.answer("Which product category has the highest sales?")["result"]
        self.assertEqual(result, [{"product_category": "Technology", "revenue": 5402.0}])

    def test_average_order_value(self) -> None:
        result = self.engine.answer("Average order value by region")["result"]
        self.assertEqual(result[0], {"region": "APAC", "avg_order_value": 118.5})

    def test_target_comparison(self) -> None:
        result = self.engine.answer("Which region missed its target in Feb?")["result"]
        self.assertEqual({row["region"] for row in result}, {"APAC", "EMEA", "NA"})
        self.assertTrue(all(row["variance"] < 0 for row in result))

    def test_contribution_totals_to_100(self) -> None:
        result = self.engine.answer("Sales contribution % by category")["result"]
        self.assertAlmostEqual(sum(row["contribution_pct"] for row in result), 100, places=3)

    def test_top_product_per_region(self) -> None:
        result = self.engine.answer("Top product in each region")["result"]
        products = {row["region"]: row["product_name"] for row in result}
        self.assertEqual(products, {"APAC": "Ergo Chair", "EMEA": "Samsung Galaxy", "NA": "Dell XPS"})

    def test_yoy_reports_insufficient_history(self) -> None:
        answer = self.engine.answer("YoY growth in revenue")
        self.assertIsNone(answer["result"][0]["yoy_growth_pct"])
        self.assertIn("no prior-year", answer["explanation"])

    def test_nested_top_customers(self) -> None:
        result = self.engine.answer("Revenue of top 3 customers per region")["result"]
        self.assertEqual(result[-1]["revenue"], 3262.4)

    def test_quarter_relative_to_latest_data(self) -> None:
        result = self.engine.answer("Sales this quarter")["result"]
        self.assertEqual(result, [{"revenue": 6134.4}])

    def test_validator_blocks_unknown_identifiers(self) -> None:
        with self.assertRaises(PlanValidationError):
            self.engine.validator.validate(QueryPlan("aggregate", dimensions=["drop_table"]))

    def test_feedback_correction_wins(self) -> None:
        with tempfile.NamedTemporaryFile("w", newline="", suffix=".csv", delete=False) as file:
            writer = csv.DictWriter(file, fieldnames=["query", "accepted", "corrected_plan", "notes"])
            writer.writeheader()
            writer.writerow({
                "query": "Best market",
                "accepted": "true",
                "corrected_plan": json.dumps({
                    "analysis_type": "rank", "metric": "profit", "dimensions": ["region"],
                    "filters": [], "limit": 1, "rank_dimension": "region", "comparison": "desc",
                }),
                "notes": "market means region here",
            })
            path = file.name
        result = engine(path).answer("Best market")
        self.assertEqual(result["result"][0]["region"], "NA")


if __name__ == "__main__":
    unittest.main()

