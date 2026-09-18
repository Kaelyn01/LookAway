"""
Download and preprocess the LookAway training data.

Usage:
    python scripts/prepare_data.py --data-dir ./cache/lookaway

This script:
1. Downloads the Vision-OPD-6K dataset from HuggingFace
   (train.jsonl, student images, teacher crops)
2. Generates the negative teacher views: a red box drawn at a displaced
   position (IoU < 0.1 vs the ground-truth box), cropped with padding matched
   to the official positive crop and resized to its size, so both teacher
   views share the same visual format
3. Converts train.jsonl to train.parquet and attaches the `neg_bbox_images`
   column, producing the LookAway training file

Outputs (under --data-dir):
    images/           official student images (red-box full image)
    teacher_images/   official positive teacher crops
    teacher_neg/      generated negative teacher crops
    results.json      per-sample generation metadata (boxes, IoU)
    train.parquet     training file with the negative-view column
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from multiprocessing import Pool
from typing import Any

import cv2
import datasets
import numpy as np
from PIL import Image

PAD_RATIO = 0.105   # assumed official crop padding ratio
THICK = 3           # red box stroke width (measured from official crops)
NEG_IOU_LIMIT = 0.1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare LookAway training data.")
    parser.add_argument("--data-dir", default="./cache/lookaway", help="Output directory")
    parser.add_argument("--hf-repo", default="yuanqianhao/Vision-OPD-6K", help="HuggingFace dataset repo")
    parser.add_argument("--skip-download", action="store_true", help="Skip downloading, only preprocess")
    parser.add_argument("--nproc", type=int, default=16, help="Worker processes for negative-view generation")
    return parser.parse_args()


# ---------------------------------------------------------------- download --
def download_dataset(repo_id: str, data_dir: str) -> None:
    print(f"Downloading dataset from {repo_id} ...")
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id, repo_type="dataset", local_dir=data_dir)

    # Extract every shipped image bundle (multi-part tars are cat-joined).
    for sub, prefix in (("images", "images.tar.gz"),
                        ("teacher_images", "teacher_images.tar.gz"),
                        ("original_images", "original_images.tar.gz")):
        d = os.path.join(data_dir, sub)
        if not os.path.isdir(d):
            print(f"Warning: {d} not present in the download", file=sys.stderr)
            continue
        tars = sorted(f for f in os.listdir(d) if f.startswith(prefix))
        if not tars:
            continue
        print(f"Extracting {sub} ...")
        cat_process = subprocess.Popen(["cat", *tars], cwd=d, stdout=subprocess.PIPE)
        try:
            subprocess.run(["tar", "-xf", "-", "-C", "."], cwd=d, stdin=cat_process.stdout, check=True)
        finally:
            if cat_process.stdout is not None:
                cat_process.stdout.close()
            cat_returncode = cat_process.wait()
        if cat_returncode != 0:
            raise subprocess.CalledProcessError(cat_process.returncode, cat_process.args)
        for f in tars:
            os.remove(os.path.join(d, f))

    print("Image extraction complete.")


# ------------------------------------------------------- negative  views --
def iou(a, b):
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua else 0.0


def measure_expand(drawn_np, gt):
    """Measure the official box stroke expansion around the GT box on a
    red-boxed image; used to replicate the exact visual format."""
    x1, y1, x2, y2 = gt
    H, W = drawn_np.shape[:2]
    band = 30
    sx1, sy1 = max(0, x1 - band), max(0, y1 - band)
    sx2, sy2 = min(W, x2 + band), min(H, y2 + band)
    sub = drawn_np[sy1:sy2, sx1:sx2]
    r, g, b = sub[..., 0].astype(int), sub[..., 1].astype(int), sub[..., 2].astype(int)
    m = (r > 245) & (g < 25) & (b < 25)
    if m.sum() < 20:
        return None
    ys, xs = np.where(m)
    leftmost, rightmost = sx1 + xs.min(), sx1 + xs.max()
    topmost, bottommost = sy1 + ys.min(), sy1 + ys.max()
    return (max(0, x1 - leftmost), max(0, y1 - topmost), max(0, rightmost - x2), max(0, bottommost - y2))


_ROWS = None
_DATA_DIR = None


def _init_worker(rows, data_dir):
    """Pool initializer: safe under both fork and spawn start methods."""
    global _ROWS, _DATA_DIR
    _ROWS = rows
    _DATA_DIR = data_dir


def _worker(i):
    h = _ROWS[i]
    try:
        data_dir = _DATA_DIR
        src_student = os.path.join(data_dir, h["images"][0])
        src_tpos = os.path.join(data_dir, h["teacher_images"][0])
        orig = Image.open(os.path.join(data_dir, h["original_images"][0])).convert("RGB")
        W, H = orig.size
        gt = h["bbox"]
        bw, bh = gt[2] - gt[0], gt[3] - gt[1]
        dx, dy = max(int(bw * PAD_RATIO), 16), max(int(bh * PAD_RATIO), 16)

        drawn_np = np.array(Image.open(src_student).convert("RGB"))
        ex = measure_expand(drawn_np, gt)
        if ex is None:
            return {"idx": i, "ok": False, "reason": "expand_measure_fail"}
        el, et, er, eb = ex

        rng = random.Random(9000 + i)
        neg = None
        if W - bw - 2 * dx >= 0 and H - bh - 2 * dy >= 0:
            for _ in range(2000):
                nx, ny = rng.randint(dx, W - bw - dx), rng.randint(dy, H - bh - dy)
                c = (nx, ny, nx + bw, ny + bh)
                if iou(c, gt) < NEG_IOU_LIMIT:
                    neg = c
                    break
        if neg is None:
            for _ in range(2000):
                nx, ny = rng.randint(0, W - bw), rng.randint(0, H - bh)
                c = (nx, ny, nx + bw, ny + bh)
                if iou(c, gt) < NEG_IOU_LIMIT:
                    neg = c
                    break
        if neg is None:
            return {"idx": i, "ok": False, "reason": "no_neg"}

        orig_np = np.array(orig)
        nx1, ny1, nx2, ny2 = neg
        drawn_neg_np = orig_np.copy()
        cv2.rectangle(drawn_neg_np, (nx1 - el, ny1 - et), (nx2 + er, ny2 + eb), (255, 0, 0), THICK)
        drawn_neg = Image.fromarray(drawn_neg_np)
        official = Image.open(src_tpos).convert("RGB")
        window = (max(0, nx1 - dx), max(0, ny1 - dy), min(W, nx2 + dx), min(H, ny2 + dy))
        tneg = drawn_neg.crop(window).resize(official.size, Image.LANCZOS)

        tneg.save(os.path.join(data_dir, "teacher_neg", f"{i:06d}.png"), compress_level=1)
        return {"idx": i, "ok": True, "gt_bbox": [int(v) for v in gt], "neg_bbox": list(neg),
                "iou": round(iou(neg, gt), 4), "crop_size": [int(v) for v in official.size]}
    except Exception as e:
        return {"idx": i, "ok": False, "reason": f"exc:{type(e).__name__}:{str(e)[:80]}"}


def generate_negative_views(data_dir: str, nproc: int) -> list:
    with open(os.path.join(data_dir, "train.jsonl"), encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    os.makedirs(os.path.join(data_dir, "teacher_neg"), exist_ok=True)

    t0 = time.time()
    results = []
    with Pool(nproc, initializer=_init_worker, initargs=(rows, data_dir)) as pool:
        for n, r in enumerate(pool.imap_unordered(_worker, range(len(rows)), chunksize=16), 1):
            results.append(r)
            if n % 500 == 0:
                ok = sum(1 for x in results if x.get("ok"))
                print(f"[{n}/{len(rows)}] ok={ok} elapsed={time.time() - t0:.0f}s", flush=True)
    results.sort(key=lambda r: r["idx"])
    with open(os.path.join(data_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    ok = sum(1 for r in results if r.get("ok"))
    success_rate = 100 * ok / len(results) if results else 0.0
    print(f"Negative views: {ok}/{len(results)} ({success_rate:.1f}%) "
          f"in {(time.time() - t0) / 60:.1f} min")
    return results


# ---------------------------------------------------------------- parquet --
def clean_question(problem: str) -> str:
    text = (problem or "").replace("<image>", "").strip()
    hint = ("Only focus on the objects inside the red bounding box in the image "
            "to answer this question.")
    text = text.replace(f"\n\n{hint}", "").replace(hint, "")
    return text.strip()


def build_record(item: dict[str, Any], data_dir: str, neg_path: str | None) -> dict[str, Any]:
    record = {
        "data_source": "zwz_rl_vqa_bbox_teacher",
        "prompt": [{"role": "user", "content": item["problem"]}],
        "images": [{"path": os.path.join(data_dir, item["images"][0])}],
        "bbox_images": [{"path": os.path.join(data_dir, item["teacher_images"][0])}],
        "ability": "visual_question_answering",
        "reward_model": {"style": "none", "ground_truth": item.get("answer", "")},
        "extra_info": {
            "answer": item.get("answer", ""),
            "question": clean_question(item.get("problem", "")),
            "source_extra_info": item.get("extra_info", {}),
        },
    }
    record["neg_bbox_images"] = [{"path": neg_path}] if neg_path is not None else []
    return record


def convert_to_parquet(data_dir: str, results: list) -> None:
    jsonl_path = os.path.join(data_dir, "train.jsonl")
    if not os.path.exists(jsonl_path):
        print(f"Error: {jsonl_path} not found", file=sys.stderr)
        sys.exit(1)

    by_idx_ok = {r["idx"]: r.get("ok", False) for r in results}
    print("Converting train.jsonl to train.parquet ...")
    records = []
    with open(jsonl_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            item = json.loads(line)
            neg = os.path.join(data_dir, "teacher_neg", f"{i:06d}.png") if by_idx_ok.get(i) else None
            records.append(build_record(item, data_dir, neg))

    dataset = datasets.Dataset.from_list(records)
    output_path = os.path.join(data_dir, "train.parquet")
    dataset.to_parquet(output_path)
    n_neg = sum(1 for r in records if "neg_bbox_images" in r)
    print(f"Saved {len(records)} records ({n_neg} with negative views) to {output_path}")


def main() -> None:
    args = parse_args()
    data_dir = os.path.abspath(args.data_dir)
    os.makedirs(data_dir, exist_ok=True)

    if not args.skip_download:
        download_dataset(args.hf_repo, data_dir)

    results = generate_negative_views(data_dir, args.nproc)
    convert_to_parquet(data_dir, results)
    print(f"\nData preparation complete. Training data at: {data_dir}/train.parquet")


if __name__ == "__main__":
    main()
