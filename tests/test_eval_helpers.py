import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]


def load_module(name, relative_path, fake_modules=None):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules, fake_modules or {}):
        spec.loader.exec_module(module)
    return module


class EvalHelperTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        openai = types.ModuleType("openai")
        openai.OpenAI = object
        cls.infer = load_module("infer", "eval/infer.py", {"openai": openai})
        cls.judge = load_module("judge_qwenlm", "eval/judge_qwenlm.py")
        cls.accuracy = load_module("cal_acc", "eval/cal_acc.py")
        cls.prepare = load_module("prepare_data", "eval/prepare_data.py")

    def test_answer_prefix_does_not_become_option_a(self):
        answer = self.judge.extract_answer("Answer: C")
        self.assertEqual(answer, "C")
        self.assertTrue(self.judge.first_letter_match("C", answer))
        self.assertFalse(self.judge.first_letter_match("A", answer))
        self.assertFalse(self.judge.first_letter_match("A", "According to the image, C is correct."))
        self.assertTrue(self.judge.first_letter_match("C", "The answer is C."))

    def test_thinking_and_answer_tags_are_removed(self):
        raw = "<think>reasoning</think>\n<answer>D</answer>"
        self.assertEqual(self.infer.normalize_model_answer(raw), "D")
        self.assertEqual(self.judge.extract_answer(raw), "D")

    def test_judge_error_cannot_be_counted_as_an_incorrect_answer(self):
        with self.assertRaisesRegex(RuntimeError, "Judge output contains an error"):
            self.accuracy.is_correct({"judge": "[JUDGE_ERROR] TimeoutError"})

    def test_duplicate_numeric_indices_get_distinct_uids(self):
        first = {"index": 1, "type": "2D", "images": ["2d.jpg"], "query": "q"}
        second = {"index": 1, "type": "3D", "images": ["3d.jpg"], "query": "q"}
        self.assertNotEqual(
            self.infer.make_sample_uid(first, "cv-bench"),
            self.infer.make_sample_uid(second, "cv-bench"),
        )

    def test_legacy_checkpoint_uid_is_recomputed(self):
        item = {
            "sample_uid": "cv-bench:index:1",
            "index": 1,
            "type": "2D",
            "images": ["2d.jpg"],
            "query": "q",
        }
        self.assertTrue(self.infer.make_sample_uid(item, "cv-bench").startswith("sha1:"))

    def test_explicit_sample_uid_is_preserved(self):
        self.assertEqual(self.infer.make_sample_uid({"sample_uid": "stable-id"}, "vstar"), "stable-id")

    def test_inference_error_records_are_counted_for_retry(self):
        records = [
            {"model_answer": "A"},
            {"model_answer": "[API_ERROR] timeout"},
            {"model_answer": "[FUTURE_ERROR] worker"},
        ]
        self.assertEqual(self.infer.count_failed_records(records), 2)

    def test_pope_yes_ratio_uses_predictions(self):
        items = [
            {"response": "yes", "judge": "Yes"},
            {"response": "yes", "judge": "No"},
            {"response": "no", "judge": "No"},
            {"response": "no", "judge": "No"},
        ]
        *_, yes_ratio, total = self.accuracy._pope_metrics(items)
        self.assertEqual(total, 4)
        self.assertEqual(yes_ratio, 0.75)

    def test_api_judge_failure_is_not_reported_as_incorrect_answer(self):
        openai = types.ModuleType("openai")

        class Client:
            def __init__(self, **kwargs):
                pass

            class chat:
                class completions:
                    @staticmethod
                    def create(**kwargs):
                        raise ConnectionError("offline")

        openai.OpenAI = Client
        with mock.patch.dict(sys.modules, {"openai": openai}), mock.patch.object(self.judge.time, "sleep"):
            self.assertEqual(
                self.judge.judge_via_api(["prompt"], "http://localhost", "secret", "judge", 8),
                ["[JUDGE_ERROR] ConnectionError"],
            )

    def test_benchmark_download_uses_pinned_revision(self):
        with mock.patch.object(self.prepare, "snapshot_download", return_value="/tmp/data") as download:
            self.prepare.download_benchmark("inclusionAI/ZoomBench", Path("/tmp/data"))
        download.assert_called_once_with(
            "inclusionAI/ZoomBench",
            repo_type="dataset",
            revision="b788097e57d30510c6877824833234a73bf80d25",
            local_dir="/tmp/data",
        )


if __name__ == "__main__":
    unittest.main()
