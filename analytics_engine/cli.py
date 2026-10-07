# Command-line entry point

import argparse
import json
import os
import sys
from pathlib import Path

from .engine import AnalyticsEngine
from .loaders import load_json

def _load_env():
    # A simple, dependency-free .env loader
    env_file = Path(".env")
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                key, val = line.split("=", 1)
                os.environ[key.strip()] = val.strip().strip("\"'")

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Natural-language analytics query engine")
    
    # We set default from environment if available, otherwise it's required
    def env_default(env_var, default=None):
        return os.environ.get(env_var, default)
        
    parser.add_argument("--sales", default=env_default("ANALYTICS_SALES", "data/sales_data.csv"), help="Path to sales CSV")
    parser.add_argument("--targets", default=env_default("ANALYTICS_TARGETS", "data/targets.csv"), help="Path to targets CSV")
    parser.add_argument("--dictionary", default=env_default("ANALYTICS_DICTIONARY", "data/data_dictionary.json"), help="Path to data dictionary JSON")
    
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--query", help="One natural-language query")
    source.add_argument("--queries", default=env_default("ANALYTICS_QUERIES"), help="JSON file containing query objects or strings")
    
    parser.add_argument("--output", help="Write the JSON result to this file")
    parser.add_argument("--feedback", default=env_default("ANALYTICS_FEEDBACK"), help="Optional feedback_log.csv")
    
    # LLM Settings from env
    parser.add_argument("--llm-url", default=env_default("ANALYTICS_LLM_URL"), required=not bool(env_default("ANALYTICS_LLM_URL")), help="Full chat-completions endpoint")
    parser.add_argument("--llm-model", default=env_default("ANALYTICS_LLM_MODEL"), required=not bool(env_default("ANALYTICS_LLM_MODEL")), help="Model identifier")
    
    return parser

def main(argv: list[str] | None = None) -> int:
    _load_env() # load our .env file first
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

