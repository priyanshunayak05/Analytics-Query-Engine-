"""Typed domain models used by the planner and executor."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class FilterSpec:
    field: str
    operator: str
    value: Any


@dataclass
class QueryPlan:
    analysis_type: str
    metric: str = "revenue"
    dimensions: list[str] = field(default_factory=list)
    filters: list[FilterSpec] = field(default_factory=list)
    limit: int | None = None
    rank_dimension: str | None = None
    partition_by: list[str] = field(default_factory=list)
    time_grain: str | None = None
    comparison: str | None = None
    source: str = "heuristic"
    semantic_confidence: float = 0.0
    assumptions: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, payload: dict[str, Any], source: str = "llm") -> "QueryPlan":
        filters = [
            item if isinstance(item, FilterSpec) else FilterSpec(**item)
            for item in payload.get("filters", [])
        ]
        allowed = {f.name for f in cls.__dataclass_fields__.values()}
        values = {k: v for k, v in payload.items() if k in allowed and k != "filters"}
        values["filters"] = filters
        values["source"] = source
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CompiledQuery:
    sql: str
    params: list[Any]
    description: str

