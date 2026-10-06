"""Tolerant input loaders, including repair of the supplied over-quoted files."""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import Any


def load_json(path: str | Path) -> Any:
    text = Path(path).read_text(encoding="utf-8-sig")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        repaired = []
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith('"') and stripped.endswith('"'):
                prefix = line[: len(line) - len(line.lstrip())]
                stripped = stripped[1:-1].replace('""', '"')
                line = prefix + stripped
            repaired.append(line)
        return json.loads("\n".join(repaired))


def load_csv(path: str | Path) -> list[dict[str, str]]:
    text = Path(path).read_text(encoding="utf-8-sig")
    rows = list(csv.reader(io.StringIO(text)))
    if not rows:
        return []
    # Some spreadsheet exports wrap an entire CSV row in another quoted cell.
    if len(rows[0]) == 1 and "," in rows[0][0]:
        rows = [next(csv.reader([row[0]])) for row in rows if row]
    header = [value.strip() for value in rows[0]]
    return [dict(zip(header, row, strict=False)) for row in rows[1:] if row]


def coerce_number(value: str) -> int | float | str:
    try:
        number = float(value)
        return int(number) if number.is_integer() else number
    except (TypeError, ValueError):
        return value

