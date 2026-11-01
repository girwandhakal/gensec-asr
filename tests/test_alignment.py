"""The shared audio filter returns the expected exclusions."""

import ast
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


def load_function(path, name, namespace):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == name)
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


class AlignmentTests(unittest.TestCase):
    def test_audio_filter_reasons(self):
        def seconds(uid):
            parts = uid.split("_")
            return (int(parts[-2]) - int(parts[-3])) / 1000

        audio_filter = load_function(ROOT / "scripts/audio_filter.py", "audio_filter",
                                     {"clip_seconds": seconds})
        config = dict(min_clip_seconds=0.5, max_reference_words_per_second=4,
                      long_single_word_seconds=5)
        for uid, reference, expected in [
            ("abcd_0_100_1", "hello", "point_bullet_timestamp"),
            ("abcd_0_400_1", "hello", "clip_too_short"),
            ("abcd_0_1000_1", "one two three four five", "impossible_speech_rate"),
            ("abcd_0_6000_1", "yes", "long_clip_short_reference"),
            ("abcd_0_1000_1", "hello there", ""),
        ]:
            with self.subTest(uid=uid, reference=reference):
                self.assertEqual(audio_filter(uid, reference, config), expected)


if __name__ == "__main__":
    unittest.main()
