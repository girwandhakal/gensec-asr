"""
What this file is for:
Entry point for the per-child analysis: per-child WER, then the statistics.

High-level role in the pipeline:
Deliberately NOT part of run_pipeline.py. The pipeline is hours on a GPU; this
reads predictions stage 4 already wrote and finishes in seconds on a laptop.
Keeping them apart also means this can never touch wer_report.txt - everything
here is written into results_dir/analysis/.

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
