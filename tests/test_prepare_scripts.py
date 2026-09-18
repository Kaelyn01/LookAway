import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]


def load_script(name, relative_path, fake_modules):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules, fake_modules):
        spec.loader.exec_module(module)
    return module


class FakePool:
    def __init__(self, _, initializer, initargs):
        self.initializer = initializer
        self.initargs = initargs

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def imap_unordered(self, func, indices, chunksize):
        self.initializer(*self.initargs)
        return iter({"idx": i, "ok": True} for i in indices)


class PrepareDataTests(unittest.TestCase):
    def test_generation_uses_parent_row_count(self):
        fake_modules = {
            "cv2": types.ModuleType("cv2"),
            "datasets": types.ModuleType("datasets"),
        }
        module = load_script("prepare_data", "scripts/prepare_data.py", fake_modules)
        module.Pool = FakePool
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "train.jsonl"
            path.write_text('{"id": 1}\n{"id": 2}\n', encoding="utf-8")
            results = module.generate_negative_views(tmp, 1)
        self.assertEqual([row["idx"] for row in results], [0, 1])

    def test_missing_negative_view_keeps_parquet_column(self):
        fake_modules = {
            "cv2": types.ModuleType("cv2"),
            "datasets": types.ModuleType("datasets"),
        }
        module = load_script("prepare_data_record", "scripts/prepare_data.py", fake_modules)
        item = {
            "problem": "question",
            "images": ["student.png"],
            "teacher_images": ["teacher.png"],
        }
        self.assertEqual(module.build_record(item, "/data", None)["neg_bbox_images"], [])


class PreparePriorsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        transformers = types.ModuleType("transformers")
        transformers.AutoTokenizer = object
        transformers.AutoProcessor = object
        transformers.AutoModelForImageTextToText = object
        cls.module = load_script(
            "prepare_priors",
            "scripts/prepare_priors.py",
            {"transformers": transformers},
        )

    def test_generated_ids_are_not_retokenized(self):
        tokenizer = types.SimpleNamespace(all_special_ids=[0, 2])
        self.assertEqual(self.module.generated_answer_ids([11, 0, 12, 2], tokenizer), [11, 0, 12])

    def test_answer_alignment_is_explicit(self):
        self.assertEqual(self.module.find_answer_start([1, 7, 8, 9, 2], [7, 8, 9]), 1)
        with self.assertRaises(ValueError):
            self.module.find_answer_start([1, 7, 9, 2], [7, 8])

    def test_empty_dump_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            dump = Path(tmp) / "empty.jsonl"
            dump.touch()
            args = types.SimpleNamespace(data_dir=tmp)
            with self.assertRaisesRegex(ValueError, "empty"):
                self.module.run_priors(args, dump, object())

    def test_scoring_reuses_generation_prompt(self):
        import torch

        class Inputs(dict):
            def to(self, device):
                self["device"] = device
                return self

        class Processor:
            def apply_chat_template(self, messages, **kwargs):
                self.messages = messages
                self.kwargs = kwargs
                return "PROMPT:"

            def __call__(self, **kwargs):
                self.call_kwargs = kwargs
                return Inputs(
                    input_ids=torch.tensor([[1, 2]]),
                    attention_mask=torch.tensor([[1, 1]]),
                    mm_token_type_ids=torch.tensor([[1, 0]]),
                )

        proc = Processor()
        fake_image = mock.Mock()
        fake_image.convert.return_value = fake_image
        with mock.patch.object(self.module.Image, "open", return_value=fake_image):
            inputs = self.module.build_inputs(
                proc, "image.png", "<image>\nquestion", answer_ids=[11, 12], device="cpu"
            )
        self.assertTrue(proc.kwargs["add_generation_prompt"])
        self.assertEqual(len(proc.messages), 1)
        self.assertEqual(proc.call_kwargs["text"], ["PROMPT:"])
        self.assertEqual(inputs["input_ids"].tolist(), [[1, 2, 11, 12]])
        self.assertEqual(inputs["attention_mask"].tolist(), [[1, 1, 1, 1]])
        self.assertEqual(inputs["mm_token_type_ids"].tolist(), [[1, 0, 0, 0]])
        text_content = proc.messages[0]["content"][1]["text"]
        self.assertEqual(text_content, "\nquestion")


if __name__ == "__main__":
    unittest.main()
