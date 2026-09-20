#!/usr/bin/env python3
"""Build the frozen frequency table used by the frequency-decay reallocation.

Usage:
    python scripts/build_freq_table.py --data-dir ./cache/lookaway \
        --model-path Qwen/Qwen3.5-4B \
        --model-revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a

Reads the three-view token dump produced by scripts/prepare_priors.py
(token_scores_full.jsonl) and counts occurrences of every generated token,
keyed by token id (exact vocab bijection on the raw token strings; no
lowercasing or BPE re-merging). Unseen runtime tokens fall back to n=0.

Output: {data_dir}/token_freq.json  with keys  vocab_size / counts.
"""

import argparse
import json
import os
from collections import Counter

from transformers import AutoTokenizer

DEFAULT_MODEL_PATH = "Qwen/Qwen3.5-4B"
DEFAULT_MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"


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
    dump_path = os.path.join(args.data_dir, "token_scores_full.jsonl")
    if not os.path.exists(dump_path):
        raise SystemExit(f"Missing {dump_path} -- run scripts/prepare_priors.py first.")

    revision = None if os.path.exists(args.model_path) else args.model_revision
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, revision=revision, trust_remote_code=True
    )
    counts = Counter()
    total = 0
    with open(dump_path, encoding="utf-8", errors="ignore") as stream:
        for line in stream:
            record = json.loads(line)
            for token in record["tokens"]:
                token_id = tokenizer.convert_tokens_to_ids(token["tok"])
                if token_id is not None and token_id >= 0:
                    counts[int(token_id)] += 1
                    total += 1

    if not counts:
        raise SystemExit("Token dump is empty; cannot build the frequency table.")

    out_path = os.path.join(args.data_dir, "token_freq.json")
    with open(out_path, "w", encoding="utf-8") as stream:
        json.dump(
            {"vocab_size": len(tokenizer), "counts": {str(k): v for k, v in counts.items()}},
            stream,
        )
    print(
        f"Frequency table written: {out_path} "
        f"({total} token instances, {len(counts)} distinct ids)"
    )


if __name__ == "__main__":
    main()
