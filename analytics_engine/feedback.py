"""Feedback ingestion and exact-query correction memory."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .loaders import load_csv


class FeedbackStore:
    def __init__(self, path: str | Path | None = None):
        self.rows = load_csv(path) if path and Path(path).exists() else []

    def corrected_plan(self, query: str) -> dict[str, Any] | None:
        wanted = " ".join(query.lower().split())
        for row in reversed(self.rows):
            if " ".join(row.get("query", "").lower().split()) != wanted:
                continue
            raw = row.get("corrected_plan") or row.get("plan")
            if raw:
                try:
                    return json.loads(raw)
                except json.JSONDecodeError:
                    continue
        return None

    def prompt_examples(self, limit: int = 8) -> list[dict[str, Any]]:
        examples = []
        for row in reversed(self.rows):
            raw = row.get("corrected_plan") or row.get("plan")
            if not row.get("query") or not raw:
                continue
            try:
                examples.append({"query": row["query"], "correct_plan": json.loads(raw)})
            except json.JSONDecodeError:
                pass
            if len(examples) >= limit:
                break
        return examples

