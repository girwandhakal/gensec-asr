"""Regression checks for the upstream CSV / unchanged-reference boundary."""
import ast
import csv
import importlib.util
import json
import hashlib
import types
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT.parent / "asr-dataset-pipelines/src/media_pipeline"


def module(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


ground_truth = module(UPSTREAM / "ground_truth.py")
importer = module(ROOT / "scripts/build_reference_map.py")


class HandoffTests(unittest.TestCase):
    def test_cleaning_keeps_spoken_scopes_forms_compounds_and_repetitions(self):
        cases = {
            "<I want> [/] I want snow+man .": "i want i want snowman",
            "Mummy@f wants hot+dog .": "mummy wants hotdog",
            "(be)cause it's mine .": "because it's mine",
            "yes [= explanation [nested]] yes .": "yes yes",
            "&-um &=laughs &+ss +... xxx yyy www 0det .": "",
            "Ｉ don’t know .": "i don't know",
            "hmm mhm uhuh uhhuh .": "mm mm uhuh uhhuh",
            "sheâ€™s here .": "she's here",
        }
        for raw, expected in cases.items():
            with self.subTest(raw=raw):
                self.assertEqual(ground_truth.clean_ground_truth(raw), expected)

    def test_report_finalization_is_idempotent_and_import_is_exact(self):
        with tempfile.TemporaryDirectory() as directory:
            media = Path(directory)
            clip = media / "Corpus/TD/child/abc_100_900_1.mp3"
            clip.parent.mkdir(parents=True)
            clip.write_bytes(b"fixture-audio-not-decoded")
            row = dict(output=r"C:\old\media\Corpus\TD\child\abc_100_900_1.mp3",
                       utterance="<Mummy@f> [/] snow+man .", status="already_exists",
                       corpus="Corpus", group="TD", child_id="child")
            rows = ground_truth.finalize_rows([row], media)
            self.assertEqual(rows, ground_truth.finalize_rows(rows, media))
            self.assertEqual(rows[0]["audio_path"], "Corpus/TD/child/abc_100_900_1.mp3")
            # The reader must preserve text, even if future upstream conventions
            # change case or whitespace. It validates, rather than recleaning.
            rows[0]["utterance"] = "Mummy  snowman"
            report = media / "media_download_report.csv"
            with report.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=ground_truth.REPORT_COLUMNS)
                writer.writeheader()
                writer.writerows({key: row.get(key, "") for key in ground_truth.REPORT_COLUMNS}
                                 for row in rows)
            references, metadata = importer.build_reference_map(
                {"reference_report_csv": report, "media_dir": media})
            self.assertEqual(references[clip.stem], "Mummy  snowman")
            self.assertEqual(metadata[clip.stem]["audio_path"], rows[0]["audio_path"])
            self.assertEqual(clip.read_bytes(), b"fixture-audio-not-decoded")

    def test_export_columns_and_adjacent_timestamps(self):
        columns = ground_truth.REPORT_COLUMNS
        self.assertEqual(len(columns), 11)
        self.assertEqual(columns[columns.index("start_seconds") + 1], "end_seconds")
        self.assertFalse(set(columns) & {"output", "reference_cleaning_version",
            "reference_exclusion_reason", "reference_status", "source_url", "status"})

    def test_prediction_csv_preserves_missing_value_words_numbers_and_whitespace(self):
        io = module(ROOT / "scripts/reference_io.py")
        truths = ["NA", "null", "nan", "001", "Mummy  snowman", "  hello  "]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["id", "truth", "prediction"])
                writer.writeheader()
                writer.writerows({"id": str(i), "truth": truth, "prediction": ""}
                                 for i, truth in enumerate(truths))
            rows = io.read_prediction_rows(path)
            self.assertEqual([row["truth"] for row in rows], truths)
            self.assertTrue(all(row["prediction"] == "" for row in rows))

    def test_prediction_csv_rejects_empty_reference_without_rewriting(self):
        io = module(ROOT / "scripts/reference_io.py")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.csv"
            path.write_text("id,truth,prediction\none,,hello\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Empty or missing finalized reference"):
                io.read_prediction_rows(path)

    def test_exclusions_duplicates_and_path_escape(self):
        with tempfile.TemporaryDirectory() as directory:
            media = Path(directory)
            base = dict(output="", utterance="xxx .", status="skipped")
            result = ground_truth.finalize_rows([base], media)[0]
            self.assertEqual(result["reference_exclusion_reason"], "empty_cleaned_reference")
            result = ground_truth.finalize_rows([dict(base, utterance="hello")], media)[0]
            self.assertEqual(result["reference_exclusion_reason"], "no_audio_output")
            bad = dict(base, output="../outside.mp3", utterance="hello")
            with self.assertRaises(ValueError):
                ground_truth.finalize_rows([bad], media)
            duplicate = dict(base, output="same.mp3", utterance="hello")
            with self.assertRaises(ValueError):
                ground_truth.finalize_rows([duplicate, duplicate], media)

    def test_raw_only_csv_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            media = Path(directory)
            report = media / "report.csv"
            report.write_text("utterance,output,status\nhello,clip.mp3,created\n")
            with self.assertRaisesRegex(ValueError, "not finalized"):
                importer.build_reference_map({"reference_report_csv": report, "media_dir": media})

    def test_clip_identity_must_match_source_and_timestamps(self):
        with tempfile.TemporaryDirectory() as directory:
            media = Path(directory)
            source = "transcripts/Corpus/child.cha"
            prefix = hashlib.blake2s(f"Corpus|{source}".encode(), digest_size=4).hexdigest()
            row = dict(corpus="Corpus", filepath=source, start_seconds="0.1",
                       end_seconds="0.9", output=f"{prefix}_100_900_1.mp3",
                       utterance="hello", status="already_exists")
            (media / row["output"]).write_bytes(b"audio")
            self.assertEqual(ground_truth.finalize_rows([row], media)[0]["reference_status"], "ready")
            with self.assertRaisesRegex(ValueError, "timestamp mismatch"):
                ground_truth.finalize_rows([dict(row, end_seconds="1.5")], media)

    def test_builder_preserves_reference(self):
        # Execute pure builder functions without importing model/runtime packages.
        tree = ast.parse((ROOT / "scripts/build_gensec_dataset.py").read_text())
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                     and node.name in {"collect_hypotheses", "build_dataset"}]
        namespace = {"json": json}
        exec(compile(ast.Module(body=functions, type_ignores=[]), "builder", "exec"), namespace)
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            nbest, references = directory / "nbest.json", directory / "references.json"
            nbest.write_text(json.dumps({"one": {"1best_text": "mummy", "nbest": []}}))
            references.write_text(json.dumps({"one": "Mummy  snowman"}))
            kept, dropped = namespace["build_dataset"]({"nbest_path": nbest,
                "reference_map_path": references, "max_hypotheses": 5, "min_hypotheses": 1})
            self.assertEqual(kept[0]["output"], "Mummy  snowman")
            self.assertEqual(dropped, [])

    def test_training_preserves_target_and_evaluation_does_not_reclean_it(self):
        text = module(ROOT / "scripts/text.py")
        class Frame:
            def __init__(self, rows):
                self.rows = rows
            def to_dict(self, orient):
                return self.rows
            def __len__(self):
                return len(self.rows)
        frame = Frame([{"id": "one", "output": "Mummy  snowman", "input": ["MUMMY"]}])
        fake_pd = types.SimpleNamespace(read_json=lambda path: frame, DataFrame=Frame)
        tree = ast.parse((ROOT / "scripts/train.py").read_text())
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                     and node.name in {"normalize_hypothesis", "load_examples"}]
        namespace = {"pd": fake_pd, "json": json, "collapse_whitespace": text.collapse_whitespace}
        exec(compile(ast.Module(body=functions, type_ignores=[]), "training", "exec"), namespace)
        fake_path = types.SimpleNamespace(read_text=lambda **kwargs: json.dumps(frame.rows))
        result = namespace["load_examples"]({"processed_path": fake_path, "min_hypotheses": 1})
        self.assertEqual(result.rows[0]["output"], "Mummy  snowman")
        tree = ast.parse((ROOT / "scripts/evaluate.py").read_text())
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                     and node.name in {"align", "count_errors", "score"}]
        namespace = {"normalize_prediction_for_scoring": text.normalize_prediction_for_scoring, "WORST_EXAMPLES": 10}
        exec(compile(ast.Module(body=functions, type_ignores=[]), "evaluation", "exec"), namespace)
        self.assertEqual(namespace["score"]([("one", "Mummy snowman", "mummy snowman")])["substitutions"], 1)
        self.assertEqual(namespace["score"]([("one", "mummy snowman", "MUMMY snowman!")])["wer"], 0)


if __name__ == "__main__":
    unittest.main()
