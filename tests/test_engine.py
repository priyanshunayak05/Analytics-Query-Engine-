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


def engine(feedback: str | None = None) -> AnalyticsEngine:
    return AnalyticsEngine(
        DATA / "sales_data.csv",
        DATA / "targets.csv",
        DATA / "data_dictionary.json",
        feedback_path=feedback,
    )


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

