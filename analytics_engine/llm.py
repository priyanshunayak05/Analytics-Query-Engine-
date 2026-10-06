"""Provider-neutral GenAI planner using a user-supplied chat-completions URL."""

from __future__ import annotations

import json
import urllib.request
from typing import Any

from .catalog import SemanticCatalog


SYSTEM_PROMPT = """You are the semantic planning layer of an analytics engine.
Translate the user's question into ONE JSON query plan. Never emit SQL or prose.
Use only catalog metrics, dimensions, known values, and these analysis_type values:
aggregate, rank, contribution, target_comparison, top_n_per_group,
nested_top_aggregate, yoy.

Plan shape:
{
  "analysis_type": "aggregate",
  "metric": "revenue",
  "dimensions": [],
  "filters": [{"field":"country","operator":"=","value":"India"}],
  "limit": null,
  "rank_dimension": null,
  "partition_by": [],
  "time_grain": null,
  "comparison": null,
  "semantic_confidence": 0.0,
  "assumptions": []
}

Resolve relative periods against latest_data_date, not today's date. Use order_month
as YYYY-MM and order_year as YYYY. For "top X in/within each Y", use
top_n_per_group. For "revenue of top customers per region", use
nested_top_aggregate. Set a calibrated confidence from 0 to 1 and list assumptions.
"""


class LLMPlanner:
    def __init__(self, url: str, model: str, api_key: str | None = None, timeout: int = 45):
        self.url = url
        self.model = model
        self.api_key = api_key
        self.timeout = timeout

    def plan(
        self,
        query: str,
        catalog: SemanticCatalog,
        feedback_examples: list[dict[str, Any]],
    ) -> dict[str, Any]:
        user_payload = {
            "catalog": catalog.prompt_context(),
            "feedback_examples": feedback_examples,
            "query": query,
        }
        body = json.dumps({
            "model": self.model,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(user_payload)},
            ],
        }).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(self.url, data=body, headers=headers, method="POST")
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
        content = payload["choices"][0]["message"]["content"]
        if isinstance(content, list):
            content = "".join(item.get("text", "") for item in content)
        return json.loads(content)

