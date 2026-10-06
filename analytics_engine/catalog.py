"""Semantic catalog derived from the dictionary and physical data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class SemanticCatalog:
    metrics: dict[str, str]
    dimensions: list[str]
    synonyms: dict[str, str]
    time_mappings: dict[str, str]
    values: dict[str, list[str]]
    latest_date: str

    @classmethod
    def build(cls, dictionary: dict[str, Any], sales: list[dict[str, str]]) -> "SemanticCatalog":
        physical = list(sales[0]) if sales else []
        non_dimensions = {"quantity", "unit_price", "discount", "shipping_cost", "profit"}
        dimensions = list(dict.fromkeys(
            list(dictionary.get("dimensions", []))
            + [column for column in physical if column not in non_dimensions]
            + ["order_month", "order_year"]
        ))
        values: dict[str, list[str]] = {}
        for dimension in dimensions:
            if dimension in physical and dimension not in {"order_id", "order_date"}:
                values[dimension] = sorted({row.get(dimension, "") for row in sales if row.get(dimension)})
        dates = sorted(row.get("order_date", "") for row in sales if row.get("order_date"))
        return cls(
            metrics=dict(dictionary.get("metrics", {})),
            dimensions=dimensions,
            synonyms=dict(dictionary.get("synonyms", {})),
            time_mappings=dict(dictionary.get("time_mappings", {})),
            values=values,
            latest_date=dates[-1] if dates else "",
        )

    def prompt_context(self) -> dict[str, Any]:
        return {
            "metrics": self.metrics,
            "dimensions": self.dimensions,
            "synonyms": self.synonyms,
            "time_mappings": self.time_mappings,
            "known_values": {key: vals[:30] for key, vals in self.values.items()},
            "latest_data_date": self.latest_date,
        }

