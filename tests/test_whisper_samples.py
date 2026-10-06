"""Exercise raw-sample capture without downloading Whisper or decoding audio."""
import ast
import contextlib
import importlib.util
import io
import json
import tempfile
import time
import types
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[1]


def generation_functions():
    spec = importlib.util.spec_from_file_location("sample_text", ROOT / "scripts/text.py")
    text = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(text)
    tree = ast.parse((ROOT / "scripts/generate_nbest.py").read_text(encoding="utf-8"))
    nodes = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.Assign))]
    namespace = {
        "json": json, "Path": Path, "Counter": Counter, "time": time,
        "clean_whisper_text": text.clean_whisper_text,
        "normalize_prediction_for_scoring": text.normalize_prediction_for_scoring,
        "clip_seconds": lambda uid: 1.0,
        "torch": types.SimpleNamespace(inference_mode=contextlib.nullcontext,
                                       manual_seed=Mock()),
        "librosa": types.SimpleNamespace(load=lambda *args, **kwargs: ([0] * 16000, 16000)),
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "generate_nbest", "exec"), namespace)
    return namespace


class WhisperSamplesTests(unittest.TestCase):
    def test_batch_preserves_order_duplicates_empty_and_uncleaned_text(self):
        ns = generation_functions()
        pools = [[" Hello! ", " Hello! ", "", "MUMMY"] + ["hello"] * 6,
                 [" café ", "", "yes!", "yes!"] + ["yes"] * 6]
        features = types.SimpleNamespace(to=lambda **kwargs: "features")
        processor = Mock(return_value=types.SimpleNamespace(input_features=features))
        processor.batch_decode.side_effect = [["Hello!", "yes"], sum(pools, [])]
        model = Mock()
        config = dict(sample_rate=16000, asr_tokens_per_second=8, asr_token_margin=16,
                      asr_sample_pool_size=10, num_return_sequences=5,
                      temperature=0.8, top_p=0.95, top_k=50)
        entries = ns["transcribe_batch"]([[0] * 16000] * 2, processor, model,
                                         "cpu", "float32", config)
        for entry, pool in zip(entries, pools):
            self.assertEqual(entry["sampled_texts"], pool)
            self.assertLessEqual(len(entry["nbest"]), 5)
        self.assertEqual(entries[0]["nbest"],
                         ns["unique_hypotheses"]("Hello!", pools[0], 5))
        self.assertEqual(model.generate.call_count, 2)
        self.assertEqual(model.generate.call_args.kwargs["num_return_sequences"], 10)

    def config(self, directory):
        path = Path(directory)
        config = dict(nbest_path=path / "nbest.json",
                      whisper_samples_path=path / "samples.json",
                      metadata_path=path / "metadata.json",
                      reference_map_path=path / "references.json", media_dir=path,
                      min_hypotheses=1, asr_limit=None, asr_model_id="fixture", seed=42,
                      sample_rate=16000, min_clip_seconds=0.1, max_clip_seconds=30,
                      asr_batch_size=2, asr_save_every=2, asr_sample_pool_size=10)
        config["metadata_path"].write_text(json.dumps({uid: {"audio_path": uid + ".mp3"}
                                                      for uid in ("one", "two")}))
        config["reference_map_path"].write_text(json.dumps({"one": "hello", "two": "yes"}))
        return config

    def test_single_retry_checkpoints_samples_and_resume_does_not_decode(self):
        ns = generation_functions()
        entry = {"1best_text": "hello", "nbest": [{"rank": 1, "text": "hello"}],
                 "sampled_texts": [" Hello! ", ""] + ["hello"] * 8}
        ns["load_model"] = Mock(return_value=(None, None, "cpu", None))
        ns["transcribe_batch"] = Mock(side_effect=[RuntimeError("batch failed"),
                                                   [dict(entry)], [dict(entry)]])
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            config = self.config(directory)
            ns["generate_nbest"](config)
            samples = json.loads(config["whisper_samples_path"].read_text(encoding="utf-8"))
            nbest = json.loads(config["nbest_path"].read_text(encoding="utf-8"))
            self.assertEqual(set(samples), {"one", "two"})
            for uid in samples:
                self.assertEqual(samples[uid]["sampled_texts"], entry["sampled_texts"])
                self.assertEqual(samples[uid]["greedy_text"], "hello")
                self.assertEqual(nbest[uid], {key: entry[key] for key in ("1best_text", "nbest")})
            self.assertTrue(config["whisper_samples_path"].with_suffix(".signature.json").is_file())
            ns["load_model"].reset_mock()
            ns["transcribe_batch"].reset_mock()
            ns["generate_nbest"](config)
            ns["load_model"].assert_not_called()
            ns["transcribe_batch"].assert_not_called()
            self.assertEqual(json.loads(config["whisper_samples_path"].read_text()), samples)

    def test_legacy_cache_reports_missing_samples_without_fabricating_or_decoding(self):
        ns = generation_functions()
        ns["load_model"] = Mock()
        with tempfile.TemporaryDirectory() as directory:
            config = self.config(directory)
            config["nbest_path"].write_text(json.dumps({uid: {"nbest": [{"text": "hello"}]}
                                                       for uid in ("one", "two")}))
            config["nbest_path"].with_suffix(".signature.json").write_text(ns["decode_signature"](config))
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                ns["generate_nbest"](config)
            self.assertIn("Raw samples unavailable for 2 cached clips", output.getvalue())
            self.assertEqual(json.loads(config["whisper_samples_path"].read_text()), {})
            ns["load_model"].assert_not_called()


if __name__ == "__main__":
    unittest.main()
