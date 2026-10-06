"""Read prediction CSV text literally, without missing-value or numeric coercion."""
from __future__ import annotations

import csv
from pathlib import Path


def read_prediction_rows(path: Path) -> list[dict[str, str]]:
    """Preserve the saved human truth and predictions exactly as CSV strings."""
    csv.field_size_limit(10_000_000)
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not {"id", "truth", "prediction"}.issubset(reader.fieldnames or []):
            raise ValueError(f"Missing prediction CSV columns: {path}")
        rows = list(reader)
    for row in rows:
        if not isinstance(row["truth"], str) or not row["truth"].strip():
            raise ValueError(f"Empty or missing finalized reference for {row['id']}")
    return rows
