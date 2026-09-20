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


def validate_freq(freq_file, model_path, model_revision, tokenizer_sha256, token_dump, expected_kappa=50):
    """Reject a frequency table built for a different tokenizer or token dump."""
    with open(freq_file, encoding="utf-8") as stream:
        blob = json.load(stream)
    meta = blob.get("meta", {})
    problems = []
    if meta.get("schema_version") != 1:
        problems.append(f"schema_version: expected 1, got {meta.get('schema_version')}")
    if meta.get("model_path") != model_path:
        problems.append(f"model_path: expected {model_path!r}, got {meta.get('model_path')!r}")
    if meta.get("model_revision") != model_revision:
        problems.append(f"model_revision: expected {model_revision!r}, got {meta.get('model_revision')!r}")
    if meta.get("tokenizer_sha256") != tokenizer_sha256:
        problems.append("tokenizer_sha256: table was built for a different tokenizer")
    vocab_size = blob.get("vocab_size")
    if vocab_size != meta.get("vocab_size") or not isinstance(vocab_size, int) or vocab_size <= 0:
        problems.append("vocab_size: missing, inconsistent, or non-positive")
    if meta.get("token_dump_sha256") != sha256(token_dump):
        problems.append("token_dump_sha256: the token dump changed after the table was built")
    if meta.get("kappa") != expected_kappa:
        problems.append(f"kappa: expected {expected_kappa}, got {meta.get('kappa')}")
    counts = blob.get("counts")
    if not isinstance(counts, dict) or not counts:
        problems.append("counts: missing or empty")
    else:
        bad = [k for k in counts if not k.lstrip("-").isdigit()]
        out_of_range = [k for k in counts if k.lstrip("-").isdigit() and not (0 <= int(k) < vocab_size)]
        nonpos = [v for v in counts.values() if not isinstance(v, int) or v <= 0]
        if bad or out_of_range or nonpos:
            problems.append(
                f"counts: {len(bad)} non-integer ids, {len(out_of_range)} out-of-range ids, "
                f"{len(nonpos)} non-positive values"
            )
    if problems:
        details = "; ".join(problems)
        raise ValueError(f"token_freq.json is incompatible ({details}); rerun scripts/build_freq_table.py")


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
    parser.add_argument("--freq-file", default=None, help="token_freq.json to validate (frequency decay)")
    parser.add_argument("--token-dump", default=None, help="token dump the frequency table was built from")
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
    fingerprint = tokenizer_fingerprint(tokenizer)
    validate(
        Path(args.prior_file),
        args.model_path,
        revision,
        Path(args.chat_template),
        Path(args.generation_results),
        Path(args.train_file),
        pq.ParquetFile(args.train_file).metadata.num_rows,
        fingerprint,
    )
    print("Token priors match the model, template, and generated negative views.")
    if args.freq_file:
        if not args.token_dump:
            raise ValueError("--token-dump is required when validating a frequency table")
        validate_freq(Path(args.freq_file), args.model_path, revision, fingerprint, Path(args.token_dump))
        print("Frequency table matches the model, tokenizer, and token dump.")


if __name__ == "__main__":
    main()
