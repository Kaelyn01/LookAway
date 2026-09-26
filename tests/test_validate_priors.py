import importlib.util
import json
import tempfile
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("validate_priors", ROOT / "scripts/validate_priors.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
RESOLVE_SPEC = importlib.util.spec_from_file_location("resolve_model", ROOT / "scripts/resolve_model.py")
RESOLVE = importlib.util.module_from_spec(RESOLVE_SPEC)
RESOLVE_SPEC.loader.exec_module(RESOLVE)


class ValidatePriorsTests(unittest.TestCase):
    def test_tokenizer_fingerprint_changes_with_vocabulary(self):
        class Tokenizer:
            special_tokens_map = {"eos_token": "</s>"}

            def __init__(self, vocab):
                self.vocab = vocab

            def get_vocab(self):
                return self.vocab

        first = MODULE.tokenizer_fingerprint(Tokenizer({"a": 0, "b": 1}))
        same_reordered = MODULE.tokenizer_fingerprint(Tokenizer({"b": 1, "a": 0}))
        changed = MODULE.tokenizer_fingerprint(Tokenizer({"a": 0, "b": 2}))
        self.assertEqual(first, same_reordered)
        self.assertNotEqual(first, changed)

    def test_local_model_path_is_not_downloaded(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(RESOLVE.resolve(tmp, "ignored-revision"), tmp)

    def test_remote_model_is_resolved_at_requested_revision(self):
        hub = types.ModuleType("huggingface_hub")
        calls = []

        def snapshot_download(model_path, revision):
            calls.append((model_path, revision))
            return "/cache/snapshot"

        hub.snapshot_download = snapshot_download
        from unittest import mock

        with mock.patch.dict("sys.modules", {"huggingface_hub": hub}):
            self.assertEqual(RESOLVE.resolve("org/model", "commit"), "/cache/snapshot")
        self.assertEqual(calls, [("org/model", "commit")])

    def test_remote_model_without_revision_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "revision is required"):
            RESOLVE.resolve("org/model", "")

    def test_matching_metadata_passes_and_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            template = root / "template.jinja"
            results = root / "dataset_generation_manifest.json"
            priors = root / "token_priors_frozen.json"
            train_file = root / "trainset_clean.parquet"
            template.write_text("template", encoding="utf-8")
            results.write_text("[]", encoding="utf-8")
            train_file.write_bytes(b"parquet")
            meta = {
                "model_path": "Qwen/Qwen3.5-4B",
                "model_revision": "revision",
                "chat_template_sha256": MODULE.sha256(template),
                "generation_results_sha256": MODULE.sha256(results),
                "train_parquet_sha256": MODULE.sha256(train_file),
                "n_samples": 1,
                "tokenizer_sha256": "tokenizer-hash",
            }
            priors.write_text(json.dumps({"meta": meta}), encoding="utf-8")
            MODULE.validate(priors, "Qwen/Qwen3.5-4B", "revision", template, results, train_file, 1, "tokenizer-hash")
            with self.assertRaisesRegex(ValueError, "incompatible"):
                MODULE.validate(
                    priors,
                    "Qwen/Qwen3.5-4B",
                    "different",
                    template,
                    results,
                    train_file,
                    1,
                    "tokenizer-hash",
                )


if __name__ == "__main__":
    unittest.main()
