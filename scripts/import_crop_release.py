#!/usr/bin/env python3
"""Import the published CROP release into the LookAway training layout.

Usage:
    python scripts/import_crop_release.py \
        --data-dir ./cache/lookaway \
        --release-snapshot /path/to/hf-cache/.../snapshots/<revision> \
        --source-images /path/to/workspace-with-student/teacher_pos/teacher_neg \
        [--quality-labels healthy]

Why this exists: scripts/build_trainset.py regenerates negative views with an
unseeded sampler, which would fork the published dataset. This importer
consumes the published CROP release
(Kaelyn01/crop-paired-visual-evidence) directly:

1. loads the release shards and keeps rows whose legacy quality label is in
   --quality-labels (default: healthy only);
2. renumbers the kept rows 0..N-1 in sample_id order -- the negative-view
   path convention teacher_neg/{row_index:06d}.png requires it;
3. copies the three views byte-exactly: student images and positive crops
   from --source-images (verified byte-identical to the release positives),
   negative crops from --source-images/teacher_neg renumbered to the new
   row order; positive and negative SHA-256 are checked against the release
   during the copy;
4. writes trainset_clean.jsonl and dataset_generation_manifest.json in the exact format
   scripts/prepare_priors.py expects (generation_version 2.0.0 records);
5. reuses build_trainset.convert_to_parquet to build trainset_clean.parquet.
"""

import argparse
import glob
import hashlib
import json
import os
import shutil

DEFAULT_QUALITY_LABELS = "healthy"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import the published CROP release.")
    parser.add_argument("--data-dir", default="./cache/lookaway")
    parser.add_argument("--release-snapshot", required=True,
                        help="Path to the release snapshot (contains data/*.parquet)")
    parser.add_argument("--source-images", required=True,
                        help="Workspace with student/, teacher_pos/, teacher_neg/ image dirs")
    parser.add_argument("--quality-labels", default=DEFAULT_QUALITY_LABELS,
                        help="Comma-separated legacy quality labels to keep (default: healthy)")
    return parser.parse_args()


def load_release_rows(snapshot: str) -> list[dict]:
    import pyarrow.parquet as pq

    shards = sorted(glob.glob(os.path.join(snapshot, "data", "*.parquet")))
    if not shards:
        raise SystemExit(f"No parquet shards under {snapshot}/data -- not a release snapshot?")
    columns = [
        "sample_id", "question", "answer", "legacy_quality_label",
        "source_bbox", "negative_bbox", "negative_crop_window",
        "original_size", "output_size", "target_bbox_in_crop",
        "bbox_iou", "positive_sha256", "negative_sha256",
        "shared_frame_mask_sha256", "generation_version",
    ]
    rows = []
    for shard in shards:
        table = pq.read_table(shard, columns=columns)
        rows.extend(table.to_pylist())
    rows.sort(key=lambda row: row["sample_id"])
    return rows


def filter_rows(rows: list[dict], labels: str) -> list[dict]:
    allowed = {label.strip() for label in labels.split(",") if label.strip()}
    kept = [row for row in rows if row["legacy_quality_label"] in allowed]
    if not kept:
        raise SystemExit(f"No rows with legacy quality label in {sorted(allowed)}")
    return kept


def make_jsonl_row(row: dict, sample_id: int) -> dict:
    """trainset_clean.jsonl row in the upstream Vision-OPD-6K shape prepare_priors reads."""
    return {
        "problem": "<image>\n" + row["question"],
        "images": [f"images/{sample_id:06d}.png"],
        "teacher_images": [f"teacher_images/{sample_id:06d}.png"],
        "answer": row["answer"],
        # Non-empty struct: pyarrow cannot serialize an empty-struct column,
        # and the upstream rows carry answer/question here.
        "extra_info": {"answer": row["answer"], "question": row["question"]},
    }


def make_result_row(row: dict, new_idx: int) -> dict:
    """dataset_generation_manifest.json record: the v2 generation metadata prepare_priors validates."""
    return {
        "idx": new_idx,
        "ok": True,
        "gt_bbox": list(row["source_bbox"]),
        "neg_bbox": list(row["negative_bbox"]),
        "iou": row["bbox_iou"],
        "crop_size": list(row["output_size"]),
        "negative_crop_window": list(row["negative_crop_window"]),
        "original_size": list(row["original_size"]),
        "target_bbox_in_crop": list(row["target_bbox_in_crop"]),
        "positive_sha256": row["positive_sha256"],
        "negative_sha256": row["negative_sha256"],
        "shared_frame_mask_sha256": row["shared_frame_mask_sha256"],
        "generation_version": row["generation_version"],
        "alignment_verified": True,
    }


def copy_verified(src: str, dst: str, expected_sha256, label: str) -> None:
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copyfile(src, dst)
    if expected_sha256 is None:
        return
    with open(dst, "rb") as stream:
        actual = hashlib.sha256(stream.read()).hexdigest()
    if actual != expected_sha256:
        raise SystemExit(f"{label} hash mismatch for {src}: release says {expected_sha256}, file is {actual}")


def main() -> None:
    args = parse_args()
    rows = filter_rows(load_release_rows(args.release_snapshot), args.quality_labels)
    data_dir = args.data_dir
    os.makedirs(data_dir, exist_ok=True)

    jsonl_path = os.path.join(data_dir, "trainset_clean.jsonl")
    n_copied = 0
    with open(jsonl_path, "w", encoding="utf-8") as jsonl:
        results = []
        for new_idx, row in enumerate(rows):
            sample_id = row["sample_id"]
            jsonl.write(json.dumps(make_jsonl_row(row, sample_id), ensure_ascii=False) + "\n")
            results.append(make_result_row(row, new_idx))
            copy_verified(
                os.path.join(args.source_images, "student", f"{sample_id:06d}.png"),
                os.path.join(data_dir, "images", f"{sample_id:06d}.png"),
                None, "student image",
            )
            copy_verified(
                os.path.join(args.source_images, "teacher_pos", f"{sample_id:06d}.png"),
                os.path.join(data_dir, "teacher_images", f"{sample_id:06d}.png"),
                row["positive_sha256"], "positive crop",
            )
            copy_verified(
                os.path.join(args.source_images, "teacher_neg", f"{sample_id:06d}.png"),
                os.path.join(data_dir, "teacher_neg", f"{new_idx:06d}.png"),
                row["negative_sha256"], "negative crop",
            )
            n_copied += 3
    with open(os.path.join(data_dir, "dataset_generation_manifest.json"), "w", encoding="utf-8") as stream:
        json.dump(results, stream)
    print(
        f"Kept {len(rows)} rows ({args.quality_labels}); wrote trainset_clean.jsonl"
        f" + dataset_generation_manifest.json; copied {n_copied} images."
    )

    from build_trainset import convert_to_parquet

    convert_to_parquet(data_dir, results)


if __name__ == "__main__":
    main()
