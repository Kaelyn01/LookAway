#!/usr/bin/env python3
"""Build the frozen frequency table used by the frequency-decay reallocation.

Usage:
    python scripts/build_freq_table.py --data-dir ./cache/lookaway \
        --model-path Qwen/Qwen3.5-4B \
        --model-revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a

Reads the three-view token dump produced by scripts/prepare_priors.py
(teacher_posneg_token_scores.jsonl) and counts occurrences of every generated token.
Tokens are mapped to ids through an exact vocabulary lookup (no lowercasing,
no BPE re-merging, no unk fallback: an unknown token string is an error).

The output carries full provenance (model, revision, tokenizer fingerprint,
dump hash, totals, kappa) so that scripts/validate_priors.py can reject a
stale table or one built for a different tokenizer. The file is published
atomically via a temporary file + os.replace.

Output: {data_dir}/token_freq_counts.json  with keys  meta / vocab_size / counts.
"""

import argparse
import json
import os
import tempfile

from transformers import AutoTokenizer
from validate_priors import sha256, tokenizer_fingerprint

DEFAULT_MODEL_PATH = "Qwen/Qwen3.5-4B"
DEFAULT_MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
KAPPA = 50  # must match redistribute_by_freq's default


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the LookAway token frequency table.")
    parser.add_argument("--data-dir", default="./cache/lookaway")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument(
        "--model-revision",
        default=None,
        help="Pinned model revision (required for non-default remote models)",
    )
    args = parser.parse_args()
    if args.model_revision is None:
        if args.model_path == DEFAULT_MODEL_PATH:
            args.model_revision = DEFAULT_MODEL_REVISION
        elif not os.path.exists(args.model_path):
            raise SystemExit("--model-revision is required for a non-default remote model")
    return args


def main() -> None:
    args = parse_args()
    dump_path = os.path.join(args.data_dir, "teacher_posneg_token_scores.jsonl")
    if not os.path.exists(dump_path):
        raise SystemExit(f"Missing {dump_path} -- run scripts/prepare_priors.py first.")

    revision = None if os.path.exists(args.model_path) else args.model_revision
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, revision=revision, trust_remote_code=True
    )
    vocab = tokenizer.get_vocab()
    counts = {}
    n_samples = 0
    n_instances = 0
    with open(dump_path, encoding="utf-8") as stream:  # strict UTF-8: a corrupt dump is an error
        for line in stream:
            record = json.loads(line)
            n_samples += 1
            for token in record["tokens"]:
                token_str = token["tok"]
                token_id = vocab.get(token_str)
                if token_id is None:
                    raise SystemExit(
                        f"Token {token_str!r} (row idx={record.get('idx')}) is not in the "
                        "tokenizer vocabulary; the dump and the tokenizer do not match."
                    )
                counts[token_id] = counts.get(token_id, 0) + 1
                n_instances += 1

    if n_instances == 0:
        raise SystemExit("Token dump is empty; cannot build the frequency table.")

    payload = {
        "meta": {
            "schema_version": 1,
            "model_path": args.model_path,
            "model_revision": revision,
            "tokenizer_sha256": tokenizer_fingerprint(tokenizer),
            "vocab_size": len(tokenizer),
            "token_dump_sha256": sha256(dump_path),
            "n_samples": n_samples,
            "n_token_instances": n_instances,
            "kappa": KAPPA,
        },
        "vocab_size": len(tokenizer),
        "counts": {str(k): v for k, v in counts.items()},
    }
    out_path = os.path.join(args.data_dir, "token_freq_counts.json")
    fd, tmp_path = tempfile.mkstemp(dir=args.data_dir, prefix=".token_freq.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream)
        os.replace(tmp_path, out_path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise
    print(
        f"Frequency table written: {out_path} "
        f"({n_samples} samples, {n_instances} token instances, {len(counts)} distinct ids)"
    )


if __name__ == "__main__":
    main()
