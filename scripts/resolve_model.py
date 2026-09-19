#!/usr/bin/env python3
"""Resolve a Hugging Face model ID to a revision-pinned local snapshot."""

import argparse
from pathlib import Path


def resolve(model_path, revision):
    if Path(model_path).exists():
        return model_path
    if not revision:
        raise ValueError("A revision is required for a remote Hugging Face model")
    from huggingface_hub import snapshot_download

    return snapshot_download(model_path, revision=revision)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_path")
    parser.add_argument("--revision", default="")
    args = parser.parse_args()
    try:
        resolved = resolve(args.model_path, args.revision)
    except ValueError as exc:
        parser.error(str(exc))
    print(resolved)


if __name__ == "__main__":
    main()
