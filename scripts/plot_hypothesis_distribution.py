"""Plot retained Whisper hypothesis counts by TD/CD group.

Run from any directory: python scripts/plot_hypothesis_distribution.py
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter


ROOT = Path(__file__).resolve().parents[1]


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nbest", type=Path, default=ROOT / "data/utterance_id_to_nbest_greedy_sample10.json")
    parser.add_argument("--metadata", type=Path, default=ROOT / "data/utterance_metadata.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "participant_audit/hypothesis_distribution")
    args = parser.parse_args()
    hypotheses, metadata = read_json(args.nbest), read_json(args.metadata)
    counts = {group: Counter({n: 0 for n in range(1, 6)}) for group in ("TD", "CD")}
    excluded = Counter()
    aliases = {"TD": "TD", "LT": "CD", "CD": "CD", "SLI": "CD"}
    for utterance_id, entry in hypotheses.items():
        if utterance_id not in metadata:
            excluded["missing_metadata"] += 1
            continue
        label = metadata[utterance_id].get("group")
        if label not in aliases:
            raise ValueError(f"Unknown group {label!r} for {utterance_id}")
        # Count the retained n-best list; 1best_text is a separate baseline field.
        candidates = {(item.get("text") or "").strip() for item in entry.get("nbest", [])}
        candidates.discard("")
        n = len(candidates)
        if n == 0:
            excluded["no_nonempty_hypotheses"] += 1
            continue
        if n > 5:
            raise ValueError(f"Expected at most five hypotheses, found {n} for {utterance_id}")
        counts[aliases[label]][n] += 1

    assert sum(sum(c.values()) for c in counts.values()) + sum(excluded.values()) == len(hypotheses)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "pdf.fonttype": 42, "ps.fonttype": 42})

    def draw(ax, group):
        values = [counts[group][n] for n in range(1, 6)]
        bars = ax.bar(range(1, 6), values, width=0.85, color={"TD": "#2878a1", "CD": "#ce7540"}[group])
        ax.bar_label(bars, labels=[f"{v:,}" for v in values], padding=4, fontsize=10)
        ax.set(title=f"{group} (n = {sum(values):,} utterances)", xlabel="Number of hypotheses", ylabel="Frequency (utterances)")
        ax.set_xticks(range(1, 6))
        ax.set_xlim(0.4, 5.6)
        ax.set_ylim(0, max(values, default=0) * 1.18 or 1)
        ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_axisbelow(True)
        ax.grid(axis="y", alpha=0.2)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), layout="constrained")
    for ax, group in zip(axes, counts):
        draw(ax, group)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(args.output_dir / f"whisper_hypothesis_distribution.{suffix}", dpi=300)
    plt.close(fig)
    for group in counts:
        fig, ax = plt.subplots(figsize=(5.5, 4.5), layout="constrained")
        draw(ax, group)
        for suffix in ("png", "pdf"):
            fig.savefig(args.output_dir / f"whisper_hypothesis_distribution_{group}.{suffix}", dpi=300)
        plt.close(fig)

    with (args.output_dir / "frequency_counts.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["group", "number_of_hypotheses", "utterance_frequency", "percentage_within_group"])
        for group, frequencies in counts.items():
            total = sum(frequencies.values())
            for n in range(1, 6):
                writer.writerow([group, n, frequencies[n], f"{100 * frequencies[n] / total:.4f}" if total else "0.0000"])

    lines = ["# Whisper hypothesis distribution", "",
             "Frequency of retained distinct, nonempty Whisper hypotheses in each utterance's nbest list, grouped by TD and CD. Each utterance contributes to exactly one bin (1–5). The separate 1best_text baseline field is not counted again. These are retained candidates, not the number of raw sampling attempts. CD pools metadata labels LT, CD, and SLI; the current data uses LT. Panels use separate y-axis scales. All generated utterances with recognized group metadata and at least one nonempty candidate are included, across dataset splits.", "",
             f"Hypothesis source: `{args.nbest.resolve()}`", f"Group source: `{args.metadata.resolve()}`", "",
             "| Hypotheses per utterance | TD frequency | CD frequency |", "|---:|---:|---:|"]
    lines.extend(f"| {n} | {counts['TD'][n]:,} | {counts['CD'][n]:,} |" for n in range(1, 6))
    lines.append(f"| Total utterances | {sum(counts['TD'].values()):,} | {sum(counts['CD'].values()):,} |")
    lines.extend(["", f"Input utterances: {len(hypotheses):,}. Excluded: {dict(excluded)}.", "",
                  "Suggested caption: Distribution of the number of distinct retained Whisper hypotheses per utterance for typically developing (TD) children and children with communication disorders (CD). The x-axis indicates the number of hypotheses (1–5), and the y-axis indicates the number of utterances with that many hypotheses. Panels use separate y-axis scales."])
    (args.output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
