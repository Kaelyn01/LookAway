#!/usr/bin/env python3
"""Reject token priors built for a different model, template, or negative dataset."""

import argparse
import hashlib
import json
from pathlib import Path

DEFAULT_MODEL_PATH = "Qwen/Qwen3.5-4B"
DEFAULT_MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"


def sha256(path):
    value = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def tokenizer_fingerprint(tokenizer):
    payload = {
        "vocab": sorted((str(token), int(token_id)) for token, token_id in tokenizer.get_vocab().items()),
        "special_tokens_map": {key: str(value) for key, value in sorted(tokenizer.special_tokens_map.items())},
    }
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate(
    prior_file,
    model_path,
    model_revision,
    chat_template,
    generation_results,
    train_file,
    train_num_rows,
    tokenizer_sha256,
):
    with open(prior_file, encoding="utf-8") as stream:
        meta = json.load(stream).get("meta", {})
    expected = {
        "model_path": model_path,
        "model_revision": model_revision or None,
        "chat_template_sha256": sha256(chat_template),
        "generation_results_sha256": sha256(generation_results),
        "train_parquet_sha256": sha256(train_file),
        "n_samples": train_num_rows,
        "tokenizer_sha256": tokenizer_sha256,
    }
    mismatches = {
        key: {"expected": value, "actual": meta.get(key)} for key, value in expected.items() if meta.get(key) != value
    }
    if mismatches:
        details = ", ".join(
            f"{key}: expected {item['expected']!r}, got {item['actual']!r}" for key, item in mismatches.items()
        )
        raise ValueError(f"token_priors.json is incompatible ({details}); rerun scripts/prepare_priors.py")


def main():
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior-file", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-revision", default="")
    parser.add_argument("--chat-template", required=True)
    parser.add_argument("--generation-results", required=True)
    parser.add_argument("--train-file", required=True)
    args = parser.parse_args()
    if Path(args.model_path).exists():
        revision = None
    elif args.model_revision:
        revision = args.model_revision
    elif args.model_path == DEFAULT_MODEL_PATH:
        revision = DEFAULT_MODEL_REVISION
    else:
        raise ValueError("--model-revision is required for a non-default remote model")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, revision=revision, trust_remote_code=True)
    validate(
        Path(args.prior_file),
        args.model_path,
        revision,
        Path(args.chat_template),
        Path(args.generation_results),
        Path(args.train_file),
        pq.ParquetFile(args.train_file).metadata.num_rows,
        tokenizer_fingerprint(tokenizer),
    )
    print("Token priors match the model, template, and generated negative views.")


if __name__ == "__main__":
    main()
