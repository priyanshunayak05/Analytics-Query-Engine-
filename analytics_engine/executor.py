"""Safe plan-to-SQL compiler and in-memory SQLite executor."""

from __future__ import annotations

import sqlite3
from typing import Any

from .models import CompiledQuery, QueryPlan


METRIC_SQL = {
    "revenue": "SUM(revenue)",
    "profit": "SUM(profit)",
    "orders": "COUNT(DISTINCT order_id)",
    "avg_order_value": "SUM(revenue) / NULLIF(COUNT(DISTINCT order_id), 0)",
}


class AnalyticsExecutor:
    SALES_COLUMNS = [
        "order_id", "order_date", "region", "country", "city", "customer_id",
        "customer_segment", "product_category", "product_subcategory", "product_name",
        "quantity", "unit_price", "discount", "shipping_cost", "profit",
    ]
    NUMERIC_COLUMNS = {"quantity", "unit_price", "discount", "shipping_cost", "profit"}

    def __init__(self, sales: list[dict[str, str]], targets: list[dict[str, str]]):
        self.connection = sqlite3.connect(":memory:")
        self.connection.row_factory = sqlite3.Row
        self._load(sales, targets)

    def _load(self, sales: list[dict[str, str]], targets: list[dict[str, str]]) -> None:
        columns = ", ".join(
            f'"{column}" {"REAL" if column in self.NUMERIC_COLUMNS else "TEXT"}'
            for column in self.SALES_COLUMNS
        )
        self.connection.execute(f"CREATE TABLE sales ({columns}, revenue REAL, order_month TEXT, order_year TEXT)")
        placeholders = ",".join("?" for _ in range(len(self.SALES_COLUMNS) + 3))
        for row in sales:
            values: list[Any] = []
            for column in self.SALES_COLUMNS:
                value: Any = row.get(column)
                if column in self.NUMERIC_COLUMNS:
                    value = float(value or 0)
                values.append(value)
            quantity = float(row.get("quantity") or 0)
            unit_price = float(row.get("unit_price") or 0)
            discount = float(row.get("discount") or 0)
            revenue = quantity * unit_price * (1 - discount)
            order_date = row.get("order_date", "")
            values.extend([revenue, order_date[:7], order_date[:4]])
            self.connection.execute(f"INSERT INTO sales VALUES ({placeholders})", values)

        self.connection.execute("CREATE TABLE targets (region TEXT, month TEXT, target_revenue REAL)")
        for row in targets:
            self.connection.execute(
                "INSERT INTO targets VALUES (?, ?, ?)",
                (row.get("region"), row.get("month"), float(row.get("target_revenue") or 0)),
            )
        self.connection.commit()

    def compile(self, plan: QueryPlan) -> CompiledQuery:
        if plan.analysis_type == "target_comparison":
            return self._target_comparison(plan)
        if plan.analysis_type == "contribution":
            return self._contribution(plan)
        if plan.analysis_type == "top_n_per_group":
            return self._top_n_per_group(plan)
        if plan.analysis_type == "nested_top_aggregate":
            return self._nested_top_aggregate(plan)
        if plan.analysis_type == "yoy":
            return self._yoy(plan)
        return self._aggregate_or_rank(plan)

    def execute(self, compiled: CompiledQuery) -> list[dict[str, Any]]:
        cursor = self.connection.execute(compiled.sql, compiled.params)
        return [self._clean_row(dict(row)) for row in cursor.fetchall()]

    @staticmethod
    def _clean_row(row: dict[str, Any]) -> dict[str, Any]:
        for key, value in row.items():
            if isinstance(value, float):
                row[key] = round(value, 4)
        return row

    def _where(self, plan: QueryPlan, alias: str = "") -> tuple[str, list[Any]]:
        clauses = []
        params: list[Any] = []
        prefix = f"{alias}." if alias else ""
        for item in plan.filters:
            operator = item.operator.lower()
            if operator == "in":
                values = item.value if isinstance(item.value, list) else [item.value]
                clauses.append(f'{prefix}"{item.field}" IN ({",".join("?" for _ in values)})')
                params.extend(values)
            else:
                clauses.append(f'{prefix}"{item.field}" {item.operator} ?')
                params.append(item.value)
        return (" WHERE " + " AND ".join(clauses) if clauses else ""), params

    def _aggregate_or_rank(self, plan: QueryPlan) -> CompiledQuery:
        dimensions = plan.dimensions
        select_dims = ", ".join(f'"{item}"' for item in dimensions)
        metric = METRIC_SQL[plan.metric]
        where, params = self._where(plan)
        select = f"{select_dims}, {metric} AS {plan.metric}" if select_dims else f"{metric} AS {plan.metric}"
        sql = f"SELECT {select} FROM sales{where}"
        if dimensions:
            sql += " GROUP BY " + ", ".join(f'"{item}"' for item in dimensions)
        if plan.analysis_type == "rank":
            direction = "ASC" if plan.comparison == "asc" else "DESC"
            sql += f" ORDER BY {plan.metric} {direction}, " + ", ".join(f'"{item}" ASC' for item in dimensions)
            sql += " LIMIT ?"
            params.append(plan.limit or 10)
        elif dimensions:
            sql += " ORDER BY " + ", ".join(f'"{item}" ASC' for item in dimensions)
        return CompiledQuery(sql, params, f"Aggregate {plan.metric}" + (f" by {', '.join(dimensions)}" if dimensions else ""))

    def _contribution(self, plan: QueryPlan) -> CompiledQuery:
        dimension = plan.dimensions[0]
        metric = METRIC_SQL[plan.metric]
        where, params = self._where(plan)
        sql = f"""WITH grouped AS (
  SELECT "{dimension}" AS dimension_value, {metric} AS metric_value
  FROM sales{where}
  GROUP BY "{dimension}"
)
SELECT dimension_value AS "{dimension}", ROUND(metric_value, 4) AS {plan.metric},
       ROUND(100.0 * metric_value / NULLIF(SUM(metric_value) OVER (), 0), 4) AS contribution_pct
FROM grouped
ORDER BY contribution_pct DESC, dimension_value ASC"""
        return CompiledQuery(sql, params, f"Compute each {dimension}'s percentage contribution to total {plan.metric}")

    def _target_comparison(self, plan: QueryPlan) -> CompiledQuery:
        where, sales_params = self._where(plan, "s")
        target_clauses = []
        target_params: list[Any] = []
        for item in plan.filters:
            if item.field == "order_month" and item.operator == "=":
                target_clauses.append("t.month = ?")
                target_params.append(item.value)
            elif item.field == "region" and item.operator == "=":
                target_clauses.append("t.region = ?")
                target_params.append(item.value)
        target_scope = (" AND ".join(target_clauses) + " AND ") if target_clauses else ""
        having = "COALESCE(a.actual_revenue, 0) < t.target_revenue" if plan.comparison == "<" else "COALESCE(a.actual_revenue, 0) >= t.target_revenue"
        sql = f"""WITH actual AS (
  SELECT s.region, s.order_month AS month, SUM(s.revenue) AS actual_revenue
  FROM sales s{where}
  GROUP BY s.region, s.order_month
)
SELECT t.region, t.month, ROUND(COALESCE(a.actual_revenue, 0), 4) AS actual_revenue,
       ROUND(t.target_revenue, 4) AS target_revenue,
       ROUND(COALESCE(a.actual_revenue, 0) - t.target_revenue, 4) AS variance,
       ROUND(100.0 * (COALESCE(a.actual_revenue, 0) - t.target_revenue) / NULLIF(t.target_revenue, 0), 4) AS variance_pct
FROM targets t
LEFT JOIN actual a ON t.region = a.region AND t.month = a.month
WHERE {target_scope}{having}
ORDER BY t.region"""
        return CompiledQuery(sql, sales_params + target_params, "Compare regional monthly revenue with the target table, including zero-sales targets")

    def _top_n_per_group(self, plan: QueryPlan) -> CompiledQuery:
        partition = plan.partition_by[0]
        rank_dim = plan.rank_dimension or "product_name"
        metric = METRIC_SQL[plan.metric]
        where, params = self._where(plan)
        sql = f"""WITH grouped AS (
  SELECT "{partition}", "{rank_dim}", {metric} AS metric_value
  FROM sales{where}
  GROUP BY "{partition}", "{rank_dim}"
), ranked AS (
  SELECT *, DENSE_RANK() OVER (
    PARTITION BY "{partition}" ORDER BY metric_value DESC, "{rank_dim}" ASC
  ) AS rank
  FROM grouped
)
SELECT "{partition}", "{rank_dim}", ROUND(metric_value, 4) AS {plan.metric}, rank
FROM ranked
WHERE rank <= ?
ORDER BY "{partition}", rank"""
        params.append(plan.limit or 1)
        return CompiledQuery(sql, params, f"Rank {rank_dim} by {plan.metric} within each {partition}")

    def _nested_top_aggregate(self, plan: QueryPlan) -> CompiledQuery:
        partition = plan.partition_by[0]
        rank_dim = plan.rank_dimension or "customer_id"
        metric = METRIC_SQL[plan.metric]
        where, params = self._where(plan)
        sql = f"""WITH entity_metric AS (
  SELECT "{partition}", "{rank_dim}", {metric} AS metric_value
  FROM sales{where}
  GROUP BY "{partition}", "{rank_dim}"
), ranked AS (
  SELECT *, ROW_NUMBER() OVER (
    PARTITION BY "{partition}" ORDER BY metric_value DESC, "{rank_dim}" ASC
  ) AS rank
  FROM entity_metric
), top_entities AS (
  SELECT * FROM ranked WHERE rank <= ?
)
SELECT "{partition}", ROUND(SUM(metric_value), 4) AS {plan.metric},
       COUNT(*) AS entities_included
FROM top_entities
GROUP BY "{partition}"
ORDER BY "{partition}"
"""
        params.append(plan.limit or 3)
        return CompiledQuery(sql, params, f"Rank {rank_dim} within {partition}, keep top {plan.limit or 3}, then aggregate {plan.metric}")

    def _yoy(self, plan: QueryPlan) -> CompiledQuery:
        where, params = self._where(plan)
        metric = METRIC_SQL[plan.metric]
        sql = f"""WITH annual AS (
  SELECT order_year, {metric} AS metric_value
  FROM sales{where}
  GROUP BY order_year
), compared AS (
  SELECT order_year, metric_value,
         LAG(metric_value) OVER (ORDER BY order_year) AS previous_year_value
  FROM annual
)
SELECT order_year, ROUND(metric_value, 4) AS {plan.metric},
       ROUND(previous_year_value, 4) AS previous_year_{plan.metric},
       ROUND(100.0 * (metric_value - previous_year_value) / NULLIF(previous_year_value, 0), 4) AS yoy_growth_pct
FROM compared
ORDER BY order_year"""
        return CompiledQuery(sql, params, f"Aggregate {plan.metric} by year and compare with the prior year using LAG")
