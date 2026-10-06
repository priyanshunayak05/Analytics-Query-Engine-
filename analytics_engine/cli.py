"""Command-line entry point."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .engine import AnalyticsEngine
from .loaders import load_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Natural-language analytics query engine")
    parser.add_argument("--sales", required=True, help="Path to sales CSV")
    parser.add_argument("--targets", required=True, help="Path to targets CSV")
    parser.add_argument("--dictionary", required=True, help="Path to data dictionary JSON")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--query", help="One natural-language query")
    source.add_argument("--queries", help="JSON file containing query objects or strings")
    parser.add_argument("--output", help="Write the JSON result to this file")
    parser.add_argument("--feedback", help="Optional feedback_log.csv")
    parser.add_argument("--llm-url", help="Full chat-completions endpoint supplied by the user")
    parser.add_argument("--llm-model", help="Model identifier accepted by that endpoint")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    engine = AnalyticsEngine(
        args.sales,
        args.targets,
        args.dictionary,
        feedback_path=args.feedback,
        llm_url=args.llm_url,
        llm_model=args.llm_model,
        llm_api_key=os.environ.get("ANALYTICS_LLM_API_KEY"),
    )
    queries = [args.query] if args.query else _read_queries(args.queries)
    try:
        answers = [engine.answer(query) for query in queries]
        payload = answers[0] if args.query else answers
        rendered = json.dumps(payload, indent=2, ensure_ascii=False)
        if args.output:
            Path(args.output).write_text(rendered + "\n", encoding="utf-8")
        else:
            print(rendered)
        return 0
    except (ValueError, KeyError) as error:
        print(json.dumps({"error": str(error)}), file=sys.stderr)
        return 2


def _read_queries(path: str) -> list[str]:
    payload = load_json(path)
    queries = [item.get("query") if isinstance(item, dict) else item for item in payload]
    return [item for item in queries if isinstance(item, str) and item.strip()]


if __name__ == "__main__":
    raise SystemExit(main())

