"""
What this file is for:
Entry point for the per-child analysis: per-child WER, then the statistics.

High-level role in the pipeline:
Pipeline stage 6. Reads predictions stage 4 already wrote and writes only into
results_dir/analysis/. It can also be run by itself after inference.

    python -u scripts/run_analysis.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import per_child_wer
import run_anova


def main() -> None:
    print("=" * 72)
    print("A1. Per-child WER")
    print("=" * 72)
    per_child_wer.main()

    print()
    print("=" * 72)
    print("A2. Group x model statistics")
    print("=" * 72)
    run_anova.main()


if __name__ == "__main__":
    main()
