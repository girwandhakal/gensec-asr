"""Check that publication figures use pooled scores and consistent sample counts."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import evaluate
import generate_figures


class FigureTests(unittest.TestCase):
    def test_unique_whisper_histogram_counts_raw_ten_sample_pools(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            nbest_path = root / "nbest.json"
            samples_path = root / "samples.json"
            nbest_path.write_text(json.dumps({"repeated": {}, "varied": {}}))
            samples_path.write_text(json.dumps({
                "repeated": {"sampled_texts": ["Hello!"] * 5 + ["hello"] * 5},
                "varied": {"sampled_texts": [f"word {number}" for number in range(10)]},
            }))
            config = {
                "nbest_path": nbest_path,
                "whisper_samples_path": samples_path,
                "asr_sample_pool_size": 10,
            }
            counts = generate_figures.unique_sample_counts(config)
            self.assertEqual(counts, {1: 1, 10: 1})

            samples_path.write_text(json.dumps({"repeated": {"sampled_texts": ["hello"] * 10}}))
            with self.assertRaisesRegex(ValueError, "missing for 1 decoded clips"):
                generate_figures.unique_sample_counts(config)

    def metrics(self):
        # The 3–5-word bucket has one error in eight words: 12.5%, not the
        # 16.7% obtained by averaging the two utterance-level WERs.
        systems = {
            "whisper_1best": [
                ("short", "hello", "hello extra extra"),
                ("three", "a b c", "a b wrong"),
                ("five", "a b c d e", "a b c d e"),
            ],
            "gensec_zero_shot": [
                ("short", "hello", "hello extra extra extra"),
                ("three", "a b c", "a b c"),
                ("five", "a b c d e", "a b c d e"),
            ],
        }
        metadata = {
            "short": {"corpus": "A", "child_id": "A::child"},
            "three": {"corpus": "B", "child_id": "B::child"},
            "five": {"corpus": "B", "child_id": "B::child"},
        }
        return {
            **{name: evaluate.score(rows) for name, rows in systems.items()},
            "by_reference_length": evaluate.reference_length_metrics(systems),
            "by_corpus": evaluate.corpus_metrics(systems, metadata),
        }

    def test_pooled_scores_empty_buckets_and_wer_over_100(self):
        rows = generate_figures.figure_rows(self.metrics(), "zero_shot")
        self.assertEqual([row["utterances"] for row in rows], [1, 2])
        self.assertEqual(rows[0]["corrected"], 300)
        self.assertEqual(rows[1]["whisper"], 12.5)

        corpora = generate_figures.corpus_rows(self.metrics(), "zero_shot")
        self.assertEqual([row["corpus"] for row in corpora], ["B", "A"])
        self.assertEqual(corpora[0]["children"], 1)
        self.assertEqual(corpora[0]["utterances"], 2)
        self.assertEqual(corpora[0]["reduction"], 12.5)
        self.assertEqual(corpora[1]["reduction"], -100)

    def test_mismatched_denominators_are_rejected(self):
        metrics = self.metrics()
        metrics["by_reference_length"][0]["systems"]["gensec_zero_shot"]["reference_words"] += 1
        with self.assertRaisesRegex(ValueError, "different reference counts"):
            generate_figures.figure_rows(metrics, "zero_shot")

        metrics = self.metrics()
        metrics["gensec_zero_shot"]["reference_words"] += 1
        with self.assertRaisesRegex(ValueError, "different reference counts"):
            generate_figures.error_type_rows(metrics, "zero_shot")

        metrics = self.metrics()
        metrics["gensec_zero_shot"]["insertions"] += 1
        with self.assertRaisesRegex(ValueError, "do not sum"):
            generate_figures.error_type_rows(metrics, "zero_shot")

    def test_missing_or_incomplete_metrics_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "rerun scripts/evaluate.py"):
            generate_figures.figure_rows({}, "zero_shot")

        metrics = self.metrics()
        metrics["whisper_1best"]["utterances"] += 1
        with self.assertRaisesRegex(ValueError, "do not cover all"):
            generate_figures.figure_rows(metrics, "zero_shot")

        with self.assertRaisesRegex(ValueError, "Corpus counts do not cover"):
            generate_figures.corpus_rows(metrics, "zero_shot")

    def test_standalone_stage_writes_only_high_resolution_pngs(self):
        from PIL import Image

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "metrics.json").write_text(json.dumps(self.metrics()), encoding="utf-8")
            config = {
                "results_dir": root,
                "figure_dir": root / "figure",
                "inference_modes": ["zero_shot"],
            }
            generate_figures.main(config)

            outputs = {
                "wer_by_reference_length.png": (4320, 2700),
                "word_errors_by_type.png": (4320, 2040),
                "wer_change_by_corpus.png": (4320, 2040),
            }
            self.assertEqual({p.name for p in config["figure_dir"].iterdir()}, set(outputs))
            for name, dimensions in outputs.items():
                with Image.open(config["figure_dir"] / name) as image:
                    self.assertEqual(image.size, dimensions)
                    self.assertAlmostEqual(image.info["dpi"][0], 600, places=1)


if __name__ == "__main__":
    unittest.main()
