"""Stage 7. Draw publication figures from saved evaluation metrics."""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

from config import load_config
from text import normalize_prediction_for_scoring


PLOT_STYLE = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 11,
    "mathtext.fontset": "dejavuserif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": False,
}


def figure_rows(metrics: dict, mode: str) -> list[dict]:
    """Validate the two systems before putting them on the same plot."""
    if "by_reference_length" not in metrics:
        raise ValueError("Reference-length metrics are missing; rerun scripts/evaluate.py first.")

    corrected_name = f"gensec_{mode}"
    rows = []
    for bucket in metrics["by_reference_length"]:
        baseline = bucket["systems"]["whisper_1best"]
        corrected = bucket["systems"][corrected_name]
        if (baseline["utterances"], baseline["reference_words"]) != (
            corrected["utterances"], corrected["reference_words"]
        ):
            raise ValueError(f"Systems have different reference counts in {bucket['label']}")

        if not baseline["utterances"]:
            continue

        values = [baseline["wer"] * 100, corrected["wer"] * 100]
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError(f"Invalid WER in {bucket['label']}")

        rows.append({
            "label": bucket["label"].replace("-", "–"),
            "utterances": baseline["utterances"],
            "whisper": values[0],
            "corrected": values[1],
        })

    if not rows:
        raise ValueError("No evaluated utterances are available for the figure.")

    for name in ("whisper_1best", corrected_name):
        if sum(row["utterances"] for row in rows) != metrics[name]["utterances"]:
            raise ValueError(f"Reference-length counts do not cover all {name} utterances")

    return rows


def save_reference_length_figure(rows: list[dict], output_path: Path, model_label: str = "Whisper + FLAN-T5") -> None:
    """Export a grayscale APA-style figure as a 600-dpi PNG."""
    import matplotlib

    # Slurm jobs have no display. Select the file-only backend before pyplot.
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator

    with plt.rc_context(PLOT_STYLE):
        figure, axes = plt.subplots(figsize=(7.2, 4.5))
        # Figure numbers and titles belong in the manuscript, outside the image.
        figure.subplots_adjust(left=0.20, right=0.97, top=0.86, bottom=0.15)

        for position, row in enumerate(rows):
            axes.plot(
                [row["whisper"], row["corrected"]], [position, position],
                color="0.55", linewidth=1, zorder=2,
            )
            axes.scatter(
                row["whisper"], position, s=38, facecolor="0.15", edgecolor="0.15",
                label="Whisper" if position == 0 else None, zorder=3,
            )
            axes.scatter(
                row["corrected"], position, s=38, facecolor="white", edgecolor="0.15",
                linewidth=1.2, label=model_label if position == 0 else None,
                zorder=3,
            )

            # Separate the value labels vertically when the two estimates are close.
            for key, offset, alignment in (
                ("whisper", 8, "bottom"), ("corrected", -8, "top")
            ):
                axes.annotate(
                    f"{row[key]:.2f}", (row[key], position), xytext=(0, offset),
                    textcoords="offset points", ha="center", va=alignment, fontsize=9,
                )

        labels = [f"{row['label']}\n($n$ = {row['utterances']:,})" for row in rows]
        axes.set_yticks(range(len(rows)), labels)
        axes.set_ylim(len(rows) - 0.4, -0.6)
        axes.tick_params(axis="y", length=0, pad=9)
        axes.set_xlabel("Word error rate (%)", labelpad=8)

        # Insertions can push WER past 100%; leave room for the largest label.
        largest = max(max(row["whisper"], row["corrected"]) for row in rows)
        axes.set_xlim(0, max(20, math.ceil((largest + 8) / 10) * 10))
        axes.xaxis.set_major_locator(MultipleLocator(20))
        axes.spines["bottom"].set_color("0.6")
        axes.grid(axis="y", color="0.9", linewidth=0.5)
        axes.set_axisbelow(True)
        axes.legend(
            loc="lower right", bbox_to_anchor=(1, 1.05), frameon=False,
            ncol=2, fontsize=9, handletextpad=0.4, columnspacing=1.3,
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            figure.savefig(output_path, dpi=600, facecolor="white")
        finally:
            plt.close(figure)


def error_type_rows(metrics: dict, mode: str, model_label: str = "Whisper + FLAN-T5") -> list[dict]:
    """Check the stacked counts against the totals reported by evaluation."""
    rows = []
    for name, label in (
        ("whisper_1best", "Whisper"), (f"gensec_{mode}", model_label)
    ):
        result = metrics[name]
        counts = [result[key] for key in ("substitutions", "deletions", "insertions")]
        if any(type(count) is not int or count < 0 for count in counts):
            raise ValueError(f"Invalid word-error counts for {name}")
        if sum(counts) != result["total_errors"]:
            raise ValueError(f"Word-error counts do not sum to the total for {name}")

        rows.append({"label": label, "counts": counts, "total": result["total_errors"]})

    baseline, corrected = metrics["whisper_1best"], metrics[f"gensec_{mode}"]
    if (baseline["utterances"], baseline["reference_words"]) != (
        corrected["utterances"], corrected["reference_words"]
    ):
        raise ValueError("Systems have different reference counts for the word-error figure")

    return rows


def save_error_type_figure(rows: list[dict], output_path: Path) -> None:
    """Stack substitutions, deletions, and insertions for the two systems."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, MaxNLocator

    with plt.rc_context(PLOT_STYLE):
        figure, axes = plt.subplots(figsize=(7.2, 3.4))
        figure.subplots_adjust(left=0.25, right=0.97, top=0.80, bottom=0.22)

        # Keep the same shades in the bars and legend; light segments use dark text.
        categories = [
            ("Substitutions", "0.15", "white"),
            ("Deletions", "0.48", "white"),
            ("Insertions", "0.78", "0.15"),
        ]
        for position, row in enumerate(rows):
            left = 0
            for count, (label, shade, text_color) in zip(row["counts"], categories):
                axes.barh(
                    position, count, left=left, height=0.5, color=shade,
                    edgecolor="white", linewidth=0.5,
                    label=label if position == 0 else None,
                )
                if count:
                    axes.text(
                        left + count / 2, position, f"{count:,}",
                        ha="center", va="center", color=text_color, fontsize=9,
                    )
                left += count

            axes.annotate(
                f"{row['total']:,}", (row["total"], position), xytext=(7, 0),
                textcoords="offset points", va="center", weight="bold", fontsize=10,
            )

        largest = max(row["total"] for row in rows)
        axes.set_xlim(0, max(1, largest * 1.18))
        axes.set_ylim(len(rows) - 0.4, -0.6)
        axes.set_yticks(range(len(rows)), [row["label"] for row in rows])
        axes.tick_params(axis="y", length=0, pad=9)
        axes.set_xlabel("Word errors", labelpad=8)
        axes.xaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
        axes.xaxis.set_major_formatter(FuncFormatter(
            lambda value, _: f"{value / 1000:g}k" if largest >= 1000 else f"{value:,.0f}"
        ))
        axes.spines["bottom"].set_color("0.6")
        axes.grid(axis="x", color="0.9", linewidth=0.5)
        axes.set_axisbelow(True)
        axes.legend(
            loc="lower left", bbox_to_anchor=(0, 1.05), frameon=False,
            ncol=3, fontsize=9, handlelength=1, handletextpad=0.5, columnspacing=1.2,
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            figure.savefig(output_path, dpi=600, facecolor="white")
        finally:
            plt.close(figure)


def corpus_rows(metrics: dict, mode: str) -> list[dict]:
    """Order corpora by absolute improvement, with all test utterances included."""
    if not metrics.get("by_corpus"):
        raise ValueError("Corpus metrics are missing; rerun scripts/evaluate.py first.")

    rows = []
    for corpus in metrics["by_corpus"]:
        baseline = corpus["systems"]["whisper_1best"]
        corrected = corpus["systems"][f"gensec_{mode}"]
        if (baseline["utterances"], baseline["reference_words"]) != (
            corrected["utterances"], corrected["reference_words"]
        ):
            raise ValueError(f"Systems have different reference counts in {corpus['corpus']}")

        reduction = 100 * (baseline["wer"] - corrected["wer"])
        if not math.isfinite(reduction):
            raise ValueError(f"Invalid WER change in {corpus['corpus']}")
        rows.append({
            "corpus": corpus["corpus"], "children": corpus["children"],
            "utterances": baseline["utterances"],
            "reference_words": baseline["reference_words"], "reduction": reduction,
        })

    baseline = metrics["whisper_1best"]
    if (sum(row["utterances"] for row in rows), sum(row["reference_words"] for row in rows)) != (
        baseline["utterances"], baseline["reference_words"]
    ):
        raise ValueError("Corpus counts do not cover all evaluated utterances and reference words")

    return sorted(rows, key=lambda row: (-row["reduction"], row["corpus"]))


def save_corpus_figure(rows: list[dict], output_path: Path) -> None:
    """Show pooled WER reductions; negative bars indicate worse corrected WER."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    # Give each corpus enough vertical space even when the dataset grows.
    height = max(3.4, 0.4 * len(rows) + 1.2)
    with plt.rc_context(PLOT_STYLE):
        figure, axes = plt.subplots(figsize=(7.2, height))
        figure.subplots_adjust(left=0.43, right=0.97, top=0.96, bottom=0.18)

        for position, row in enumerate(rows):
            value = row["reduction"]
            axes.barh(
                position, value, height=0.65,
                color="0.15" if value >= 0 else "0.8", edgecolor="0.2", linewidth=0.5,
            )
            axes.annotate(
                f"{value:+.1f}", (value, position),
                xytext=(5 if value >= 0 else -5, 0), textcoords="offset points",
                ha="left" if value >= 0 else "right", va="center", fontsize=9,
            )

        labels = [
            f"{row['corpus']} ({row['children']} "
            f"{'child' if row['children'] == 1 else 'children'}; "
            f"{row['utterances']:,} utterances)"
            for row in rows
        ]
        axes.set_yticks(range(len(rows)), labels, fontsize=9)
        axes.tick_params(axis="y", length=0, pad=8)
        axes.set_ylim(len(rows) - 0.4, -0.6)

        # Keep zero visible and leave room for signed labels on both sides.
        low = min(0, min(row["reduction"] for row in rows))
        high = max(0, max(row["reduction"] for row in rows))
        padding = max(0.8, (high - low) * 0.15)
        axes.set_xlim(low - padding, high + padding)
        axes.axvline(0, color="0.2", linewidth=0.8)
        axes.xaxis.set_major_locator(MaxNLocator(nbins=5))
        axes.set_xlabel("WER reduction (percentage points)", labelpad=8)
        axes.spines["bottom"].set_color("0.6")
        axes.grid(axis="x", color="0.9", linewidth=0.5)
        axes.set_axisbelow(True)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            figure.savefig(output_path, dpi=600, facecolor="white")
        finally:
            plt.close(figure)


def inspect_sample_pools(config: dict) -> tuple[Counter, dict]:
    """Return histogram counts only when every decoded clip has a full pool."""
    nbest_path = config.get("nbest_path")
    samples_path = config.get("whisper_samples_path")
    if not nbest_path or not samples_path or not nbest_path.is_file() or not samples_path.is_file():
        return Counter(), {"available": False, "reason": "raw_sample_cache_unavailable"}

    nbest = json.loads(nbest_path.read_text(encoding="utf-8"))
    samples = json.loads(samples_path.read_text(encoding="utf-8"))
    expected = config["asr_sample_pool_size"]
    missing = nbest.keys() - samples.keys()
    incomplete = sum(
        not isinstance(samples[uid].get("sampled_texts"), list)
        or len(samples[uid]["sampled_texts"]) != expected
        for uid in nbest.keys() & samples.keys()
    )
    status = {
        "available": not missing and not incomplete,
        "decoded_clips": len(nbest),
        "complete_raw_sample_pools": len(nbest) - len(missing) - incomplete,
        "missing_raw_sample_pools": len(missing),
        "incomplete_raw_sample_pools": incomplete,
        "sample_pool_size": expected,
    }
    if not status["available"]:
        status["reason"] = "raw_sample_coverage_incomplete"
        return Counter(), status

    counts = Counter()
    for utterance_id in nbest:
        pool = samples[utterance_id]["sampled_texts"]
        # Use the same transcript equivalence rule as n-best selection.
        distinct = {normalize_prediction_for_scoring(text) for text in pool}
        counts[len(distinct - {""})] += 1
    return counts, status


def unique_sample_counts(config: dict) -> Counter:
    """Strict accessor for analyses that require complete raw sampling."""
    counts, status = inspect_sample_pools(config)
    if not status["available"]:
        raise ValueError(
            f"Raw Whisper samples are missing for {status.get('missing_raw_sample_pools', 0):,} "
            f"decoded clips; {status.get('incomplete_raw_sample_pools', 0):,} pools are incomplete."
        )
    return counts


def save_unique_sample_histogram(counts: Counter, pool_size: int, output_path: Path) -> None:
    """Draw the number of utterances with each possible unique-sample count."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    values = [counts.get(number, 0) for number in range(pool_size + 1)]
    with plt.rc_context(PLOT_STYLE):
        figure, axes = plt.subplots(figsize=(7.2, 4.5))
        figure.subplots_adjust(left=0.15, right=0.97, top=0.96, bottom=0.16)
        axes.bar(range(pool_size + 1), values, color="0.35", edgecolor="0.15", width=0.75)
        axes.set_xticks(range(pool_size + 1))
        axes.set_xlabel(f"Unique hypotheses among {pool_size} Whisper samples")
        axes.set_ylabel("Utterances")
        axes.yaxis.set_major_locator(MaxNLocator(integer=True))
        axes.grid(axis="y", color="0.9", linewidth=0.5)
        axes.set_axisbelow(True)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            figure.savefig(output_path, dpi=600, facecolor="white")
        finally:
            plt.close(figure)


def main(config: dict | None = None) -> None:
    config = config or load_config()
    metrics_path = config["results_dir"] / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    mode = config["inference_modes"][0]
    model_label = config.get("model_display_name", "Whisper + FLAN-T5")
    rows = figure_rows(metrics, mode)
    error_rows = error_type_rows(metrics, mode, model_label)
    corpus_data = corpus_rows(metrics, mode)
    output_path = config["figure_dir"] / "wer_by_reference_length.png"
    save_reference_length_figure(rows, output_path, model_label)
    print(f"Wrote {output_path}")

    output_path = config["figure_dir"] / "word_errors_by_type.png"
    save_error_type_figure(error_rows, output_path)
    print(f"Wrote {output_path}")

    output_path = config["figure_dir"] / "wer_change_by_corpus.png"
    save_corpus_figure(corpus_data, output_path)
    print(f"Wrote {output_path}")

    unique_counts, sample_status = inspect_sample_pools(config)
    output_path = config["figure_dir"] / "unique_hypotheses_histogram.png"
    counts_path = config["figure_dir"] / "unique_hypotheses_counts.json"
    if sample_status["available"]:
        save_unique_sample_histogram(unique_counts, config["asr_sample_pool_size"], output_path)
        print(f"Wrote {output_path}")
        counts_path.write_text(json.dumps({
            "utterances": sum(unique_counts.values()),
            "sample_pool_size": config["asr_sample_pool_size"],
            "counts": {str(number): unique_counts.get(number, 0)
                       for number in range(config["asr_sample_pool_size"] + 1)},
        }, indent=2), encoding="utf-8")
        print(f"Wrote {counts_path}")
    else:
        output_path.unlink(missing_ok=True)
        counts_path.unlink(missing_ok=True)
        print("Skipping raw-sample histogram: "
              f"{sample_status.get('missing_raw_sample_pools', 'unknown')} missing and "
              f"{sample_status.get('incomplete_raw_sample_pools', 'unknown')} incomplete pools")
    status_path = config["figure_dir"] / "unique_hypotheses_status.json"
    status_path.write_text(json.dumps(sample_status, indent=2), encoding="utf-8")
    print(f"Wrote {status_path}")


if __name__ == "__main__":
    main()
