import importlib.util
import sys
import tarfile
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
        indices = list(indices)
        output_dir = Path(self.initargs[2])
        for i in indices:
            (output_dir / f"{i:06d}.png").write_bytes(b"negative")
        return iter({"idx": i, "ok": True} for i in indices)


class PrepareDataTests(unittest.TestCase):
    def test_split_archive_extraction_rejects_path_traversal(self):
        fake_modules = {
            "cv2": types.ModuleType("cv2"),
            "datasets": types.ModuleType("datasets"),
        }
        module = load_script("prepare_data_archive", "scripts/prepare_data.py", fake_modules)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "archive.tar.gz"
            payload = root / "payload.txt"
            payload.write_text("payload", encoding="utf-8")
            with tarfile.open(archive, "w:gz") as stream:
                stream.add(payload, arcname="nested/payload.txt")
            content = archive.read_bytes()
            midpoint = len(content) // 2
            first, second = root / "part00", root / "part01"
            first.write_bytes(content[:midpoint])
            second.write_bytes(content[midpoint:])
            destination = root / "safe"
            destination.mkdir()
            module.extract_tar_parts([first, second], destination)
            self.assertEqual((destination / "nested/payload.txt").read_text(encoding="utf-8"), "payload")

            unsafe = root / "unsafe.tar.gz"
            with tarfile.open(unsafe, "w:gz") as stream:
                info = tarfile.TarInfo("../outside.txt")
                info.size = 1
                import io

                stream.addfile(info, io.BytesIO(b"x"))
            with self.assertRaisesRegex(ValueError, "Unsafe archive path"):
                module.extract_tar_parts([unsafe], destination)
            self.assertFalse((root / "outside.txt").exists())

    def test_dataset_paths_cannot_escape_data_directory(self):
        fake_modules = {
            "cv2": types.ModuleType("cv2"),
            "datasets": types.ModuleType("datasets"),
        }
        module = load_script("prepare_data_paths", "scripts/prepare_data.py", fake_modules)
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(
                module.safe_dataset_path(tmp, "images/image.png"),
                str((Path(tmp) / "images/image.png").resolve()),
            )
            with self.assertRaisesRegex(ValueError, "Unsafe dataset path"):
                module.safe_dataset_path(tmp, "../outside.png")

    def test_default_upstream_revision_is_pinned(self):
        fake_modules = {
            "cv2": types.ModuleType("cv2"),
            "datasets": types.ModuleType("datasets"),
        }
        module = load_script("prepare_data_revision", "scripts/prepare_data.py", fake_modules)
        self.assertEqual(module.DEFAULT_HF_REPO, "yuanqianhao/Vision-OPD-6K")
        self.assertEqual(module.DEFAULT_HF_REVISION, "eb5c1c2e7b9a7b6a619efe4161c7369c71bf8af4")
        self.assertEqual(module.resolve_dataset_revision(module.DEFAULT_HF_REPO, None), module.DEFAULT_HF_REVISION)
        with self.assertRaisesRegex(ValueError, "revision is required"):
            module.resolve_dataset_revision("org/other-dataset", None)

    def test_download_passes_pinned_revision(self):
        fake_cv2 = types.ModuleType("cv2")
        fake_datasets = types.ModuleType("datasets")
        fake_hub = types.ModuleType("huggingface_hub")
        calls = []

        def snapshot_download(*args, **kwargs):
            calls.append((args, kwargs))

        fake_hub.snapshot_download = snapshot_download
        module = load_script(
            "prepare_data_download",
            "scripts/prepare_data.py",
            {"cv2": fake_cv2, "datasets": fake_datasets, "huggingface_hub": fake_hub},
        )
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict(sys.modules, {"huggingface_hub": fake_hub}):
                module.download_dataset("org/dataset", "commit", tmp)
        self.assertEqual(calls, [(("org/dataset",), {"repo_type": "dataset", "revision": "commit", "local_dir": tmp})])

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

    def test_negative_view_count_ignores_empty_columns(self):
        module = load_script(
            "prepare_data_count",
            "scripts/prepare_data.py",
            {"cv2": types.ModuleType("cv2"), "datasets": types.ModuleType("datasets")},
        )
        records = [{"neg_bbox_images": []}, {"neg_bbox_images": [{"path": "negative.png"}]}]
        self.assertEqual(module.count_negative_views(records), 1)

    def test_generation_failure_blocks_incomplete_dataset(self):
        module = load_script(
            "prepare_data_failure",
            "scripts/prepare_data.py",
            {"cv2": types.ModuleType("cv2"), "datasets": types.ModuleType("datasets")},
        )

        class FailedPool(FakePool):
            def imap_unordered(self, func, indices, chunksize):
                return iter({"idx": i, "ok": False, "reason": "geometry mismatch"} for i in indices)

        module.Pool = FailedPool
        with tempfile.TemporaryDirectory() as tmp:
            old_negative = Path(tmp) / "teacher_neg" / "old.png"
            old_negative.parent.mkdir()
            old_negative.write_bytes(b"old-negative")
            old_train = Path(tmp) / "train.parquet"
            old_train.write_bytes(b"old-train")
            old_priors = Path(tmp) / "token_priors.json"
            old_priors.write_text("{}", encoding="utf-8")
            (Path(tmp) / "train.jsonl").write_text('{"id": 1}' + chr(10), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "Negative generation incomplete"):
                module.generate_negative_views(tmp, 1)
            self.assertTrue((Path(tmp) / "results.failed.json").exists())
            self.assertEqual(old_negative.read_bytes(), b"old-negative")
            self.assertEqual(old_train.read_bytes(), b"old-train")
            self.assertEqual(old_priors.read_text(encoding="utf-8"), "{}")

    def test_successful_regeneration_invalidates_training_artifacts(self):
        fake_modules = {
            "cv2": types.ModuleType("cv2"),
            "datasets": types.ModuleType("datasets"),
        }
        module = load_script("prepare_data_atomic", "scripts/prepare_data.py", fake_modules)
        module.Pool = FakePool
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "train.jsonl").write_text('{"id": 1}' + chr(10), encoding="utf-8")
            old_negative = root / "teacher_neg" / "old.png"
            old_negative.parent.mkdir()
            old_negative.write_bytes(b"old-negative")
            (root / "train.parquet").write_bytes(b"old-train")
            (root / "token_priors.json").write_text("old-priors", encoding="utf-8")
            (root / "results.json").write_text("old-results", encoding="utf-8")
            module.generate_negative_views(tmp, 1)
            self.assertFalse(old_negative.exists())
            self.assertEqual((root / "teacher_neg/000000.png").read_bytes(), b"negative")
            self.assertFalse((root / "train.parquet").exists())
            self.assertFalse((root / "token_priors.json").exists())
            self.assertEqual((root / "train.parquet.stale").read_bytes(), b"old-train")
            self.assertEqual((root / "token_priors.json.stale").read_text(encoding="utf-8"), "old-priors")
            self.assertEqual((root / "results.json.stale").read_text(encoding="utf-8"), "old-results")
            self.assertNotEqual((root / "results.json").read_text(encoding="utf-8"), "old-results")

    def test_missing_staged_image_blocks_publish(self):
        fake_modules = {
            "cv2": types.ModuleType("cv2"),
            "datasets": types.ModuleType("datasets"),
        }
        module = load_script("prepare_data_missing_file", "scripts/prepare_data.py", fake_modules)

        class MissingFilePool(FakePool):
            def imap_unordered(self, func, indices, chunksize):
                return iter({"idx": i, "ok": True} for i in indices)

        module.Pool = MissingFilePool
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "train.jsonl").write_text('{"id": 1}' + chr(10), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "Negative image set is incomplete"):
                module.generate_negative_views(tmp, 1)
            self.assertFalse((root / "teacher_neg").exists())

    def test_parquet_conversion_rejects_incomplete_results(self):
        fake_modules = {
            "cv2": types.ModuleType("cv2"),
            "datasets": types.ModuleType("datasets"),
        }
        module = load_script("prepare_data_parquet", "scripts/prepare_data.py", fake_modules)
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "train.jsonl").write_text('{"problem": "q"}' + chr(10), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "incomplete negative-view results"):
                module.convert_to_parquet(tmp, [])


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

    def test_prior_dump_requires_aligned_generation_results(self):
        args = types.SimpleNamespace(
            data_dir="", limit=0, chat_template=None, device="cpu",
            shard=0, num_shards=1, merge_shards=0,
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args.data_dir = tmp
            (root / "train.jsonl").write_text('{"problem": "q"}' + chr(10), encoding="utf-8")
            with self.assertRaisesRegex(FileNotFoundError, "results.json is required"):
                self.module.run_dump(args, object(), object(), object())
            (root / "results.json").write_text(
                '[{"idx": 0, "ok": true, "generation_version": "1.0.0"}]', encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "aligned v2 pipeline"):
                self.module.run_dump(args, object(), object(), object())

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
            inputs = self.module.build_inputs(proc, "image.png", "<image>\nquestion", answer_ids=[11, 12], device="cpu")
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
