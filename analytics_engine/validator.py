"""Strict validation boundary between model output and executable SQL."""

from __future__ import annotations

from .catalog import SemanticCatalog
from .models import QueryPlan


ANALYSIS_TYPES = {
    "aggregate", "rank", "contribution", "target_comparison",
    "top_n_per_group", "nested_top_aggregate", "yoy",
}
OPERATORS = {"=", "!=", ">", ">=", "<", "<=", "in"}


class PlanValidationError(ValueError):
    pass


class PlanValidator:
    def __init__(self, catalog: SemanticCatalog):
        self.catalog = catalog
        self.metrics = set(catalog.metrics) | {"revenue", "profit", "orders", "avg_order_value"}
        self.fields = set(catalog.dimensions) | {"order_month", "order_year"}

    def validate(self, plan: QueryPlan) -> QueryPlan:
        errors: list[str] = []
        if plan.analysis_type not in ANALYSIS_TYPES:
            errors.append(f"unsupported analysis_type: {plan.analysis_type}")
        if plan.metric not in self.metrics:
            errors.append(f"unsupported metric: {plan.metric}")
        for field in plan.dimensions + plan.partition_by:
            if field not in self.fields:
                errors.append(f"unsupported dimension: {field}")
        if plan.rank_dimension and plan.rank_dimension not in self.fields:
            errors.append(f"unsupported rank dimension: {plan.rank_dimension}")
        for item in plan.filters:
            if item.field not in self.fields:
                errors.append(f"unsupported filter field: {item.field}")
            if item.operator.lower() not in OPERATORS:
                errors.append(f"unsupported operator: {item.operator}")
        if plan.limit is not None and not 1 <= int(plan.limit) <= 1000:
            errors.append("limit must be between 1 and 1000")
        if plan.analysis_type in {"rank", "top_n_per_group", "nested_top_aggregate"} and not plan.rank_dimension:
            errors.append("ranking analysis requires rank_dimension")
        if plan.analysis_type in {"top_n_per_group", "nested_top_aggregate"} and not plan.partition_by:
            errors.append("within-group ranking requires partition_by")
        if errors:
            raise PlanValidationError("; ".join(errors))
        return plan

