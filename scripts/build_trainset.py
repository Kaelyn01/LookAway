"""
Download and preprocess the LookAway training data.

Usage:
    python scripts/build_trainset.py --data-dir ./cache/lookaway

This script:
1. Downloads the Vision-OPD-6K dataset from HuggingFace
   (trainset_clean.jsonl, student images, teacher crops)
2. Generates the negative teacher views: a red box drawn at a displaced
   position (IoU < 0.1 vs the ground-truth box), using the exact recovered
   positive crop window size, within-crop box, Pillow stroke, and 2x resize.
   The official positive is reconstructed pixel-exactly before generation
3. Converts trainset_clean.jsonl to trainset_clean.parquet and attaches the `neg_bbox_images`
   column, producing the LookAway training file

Outputs (under --data-dir):
    images/           official student images (red-box full image)
    teacher_images/   official positive teacher crops
    teacher_neg/      generated negative teacher crops
    dataset_generation_manifest.json      per-sample generation metadata (boxes, IoU)
    trainset_clean.parquet     training file with the negative-view column
"""

from __future__ import annotations

import argparse
import hashlib
import io
import itertools
import json
import os
import random
import shutil
import sys
import tarfile
import tempfile
import time
from multiprocessing import Pool
from typing import Any

import cv2
import datasets
import numpy as np
from PIL import Image, ImageDraw

FRAME_WIDTH = 5  # verified against every official teacher image
NEG_IOU_LIMIT = 0.1
DEFAULT_HF_REPO = "yuanqianhao/Vision-OPD-6K"
DEFAULT_HF_REVISION = "eb5c1c2e7b9a7b6a619efe4161c7369c71bf8af4"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare LookAway training data.")
    parser.add_argument("--data-dir", default="./cache/lookaway", help="Output directory")
    parser.add_argument("--hf-repo", default=DEFAULT_HF_REPO, help="Hugging Face dataset repository")
    parser.add_argument(
        "--hf-revision",
        default=None,
        help="Pinned Hugging Face dataset revision (required for non-default repositories)",
    )
    parser.add_argument("--skip-download", action="store_true", help="Skip downloading, only preprocess")
    parser.add_argument("--nproc", type=int, default=16, help="Worker processes for negative-view generation")
    return parser.parse_args()


# ---------------------------------------------------------------- download --
class ConcatenatedReader(io.RawIOBase):
    """Read split archive parts as one forward-only byte stream."""

    def __init__(self, paths):
        self.paths = iter(paths)
        self.current = None

    def readable(self):
        return True

    def read(self, size=-1):
        chunks = []
        remaining = size
        while remaining != 0:
            if self.current is None:
                try:
                    self.current = open(next(self.paths), "rb")
                except StopIteration:
                    break
            chunk = self.current.read(remaining)
            if chunk:
                chunks.append(chunk)
                if remaining > 0:
                    remaining -= len(chunk)
            else:
                self.current.close()
                self.current = None
        return b"".join(chunks)

    def close(self):
        if self.current is not None:
            self.current.close()
        super().close()


def extract_tar_parts(parts, destination):
    """Extract regular files/directories while rejecting links and traversal."""
    destination = os.path.realpath(destination)
    with ConcatenatedReader(parts) as stream, tarfile.open(fileobj=stream, mode="r|gz") as archive:
        for member in archive:
            target = os.path.realpath(os.path.join(destination, member.name))
            if os.path.commonpath([destination, target]) != destination:
                raise ValueError(f"Unsafe archive path: {member.name}")
            if member.isdir():
                os.makedirs(target, exist_ok=True)
                continue
            if not member.isfile():
                raise ValueError(f"Unsupported archive entry: {member.name}")
            os.makedirs(os.path.dirname(target), exist_ok=True)
            source = archive.extractfile(member)
            if source is None:
                raise ValueError(f"Unable to read archive entry: {member.name}")
            with source, open(target, "wb") as output:
                shutil.copyfileobj(source, output)


def safe_dataset_path(data_dir: str, relative_path: str) -> str:
    root = os.path.realpath(data_dir)
    target = os.path.realpath(os.path.join(root, relative_path))
    if os.path.commonpath([root, target]) != root:
        raise ValueError(f"Unsafe dataset path: {relative_path}")
    return target


def resolve_dataset_revision(repo_id: str, revision: str | None) -> str:
    if revision:
        return revision
    if repo_id == DEFAULT_HF_REPO:
        return DEFAULT_HF_REVISION
    raise ValueError("--hf-revision is required for a non-default dataset repository")


def download_dataset(repo_id: str, revision: str, data_dir: str) -> None:
    print(f"Downloading dataset from {repo_id}@{revision} ...")
    from huggingface_hub import snapshot_download

    snapshot_download(repo_id, repo_type="dataset", revision=revision, local_dir=data_dir)

    # Extract every shipped image bundle (multi-part tars are cat-joined).
    for sub, prefix in (
        ("images", "images.tar.gz"),
        ("teacher_images", "teacher_images.tar.gz"),
        ("original_images", "original_images.tar.gz"),
    ):
        d = os.path.join(data_dir, sub)
        if not os.path.isdir(d):
            print(f"Warning: {d} not present in the download", file=sys.stderr)
            continue
        tars = sorted(f for f in os.listdir(d) if f.startswith(prefix))
        if not tars:
            continue
        print(f"Extracting {sub} ...")
        extract_tar_parts([os.path.join(d, name) for name in tars], d)
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


def project_bands(values):
    indices = np.flatnonzero(values >= values.max() * 0.65)
    return [group for group in np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1) if len(group)]


def recover_geometry(original, official, gt):
    w, h = official.size
    if w % 2 or h % 2:
        raise ValueError("Official teacher dimensions must match the verified 2x pipeline")
    cw, ch = w // 2, h // 2
    pixels = np.array(official)
    red = (pixels[:, :, 0] > 245) & (pixels[:, :, 1] < 25) & (pixels[:, :, 2] < 25)
    xb, yb = project_bands(red.sum(0)), project_bands(red.sum(1))
    if len(xb) < 2 or len(yb) < 2:
        return recover_difficult(original, official, gt)
    box = [int(xb[0][0]) // 2, int(yb[0][0]) // 2, int(xb[-1][-1]) // 2, int(yb[-1][-1]) // 2]
    if xb[-1][-1] == w - 1:
        box[2] = cw
    if yb[-1][-1] == h - 1:
        box[3] = ch
    x = max(0, min(original.width - cw, int((gt[0] + gt[2] - cw) / 2)))
    y = max(0, min(original.height - ch, int((gt[1] + gt[3] - ch) / 2)))
    small = np.array(official.resize((cw, ch), Image.Resampling.BOX))
    strong = (
        (small[:, :, 0].astype(int) > small[:, :, 1] * 1.5)
        & (small[:, :, 0].astype(int) > small[:, :, 2] * 1.5)
        & (small[:, :, 0] > 150)
    )
    valid = 1 - cv2.dilate(strong.astype("uint8"), np.ones((7, 7), np.uint8))
    sx, sy = max(0, x - 4), max(0, y - 4)
    ex, ey = min(original.width, x + cw + 5), min(original.height, y + ch + 5)
    if valid.sum() == 0:
        return recover_difficult(original, official, gt)
    scores = cv2.matchTemplate(
        np.array(original.crop((sx, sy, ex, ey))).astype("float32"),
        small.astype("float32"),
        cv2.TM_SQDIFF,
        mask=np.repeat(valid[:, :, None], 3, axis=2),
    )
    _, _, loc, _ = cv2.minMaxLoc(scores)
    ox, oy = sx + loc[0], sy + loc[1]
    source_crop = original.crop((ox, oy, ox + cw, oy + ch))

    def compare(candidate):
        drawn = source_crop.copy()
        ImageDraw.Draw(drawn).rectangle(candidate, outline=(255, 0, 0), width=FRAME_WIDTH)
        return np.array(drawn.resize((w, h), Image.Resampling.LANCZOS))

    rebuilt = compare(box)
    if not np.array_equal(rebuilt, pixels):
        # Natural red content can move a threshold edge by one pixel.
        choices = [[v, v - 1, v + 1] for v in box]
        for candidate in itertools.product(*choices):
            if np.array_equal(compare(candidate), pixels):
                box = list(candidate)
                break
        else:
            return recover_difficult(original, official, gt)
    return {
        "positive_crop_window": [ox, oy, ox + cw, oy + ch],
        "frame_box_in_crop": box,
        "source_crop_size": [cw, ch],
        "output_size": [w, h],
        "resize_scale": [2, 2],
        "frame_width": FRAME_WIDTH,
        "reconstruction_exact": True,
    }


def recover_difficult(original, official, gt):
    w, h = official.size
    cw, ch = w // 2, h // 2
    pixels = np.array(official)
    x = max(0, min(original.width - cw, int((gt[0] + gt[2] - cw) / 2)))
    y = max(0, min(original.height - ch, int((gt[1] + gt[3] - ch) / 2)))
    offsets = [0, 1, -1, 2, -2]
    for ox, oy in itertools.product(
        [x + v for v in offsets if 0 <= x + v <= original.width - cw],
        [y + v for v in offsets if 0 <= y + v <= original.height - ch],
    ):
        crop = original.crop((ox, oy, ox + cw, oy + ch))
        bw, bh = gt[2] - gt[0], gt[3] - gt[1]
        estimate = [
            int(gt[0] - bw * 0.05) - ox,
            int(gt[1] - bh * 0.05) - oy,
            int(gt[2] + bw * 0.05) - ox,
            int(gt[3] + bh * 0.05) - oy,
        ]
        options = [[v + d for d in [0, 1, -1, 2, -2]] for v in estimate]
        for box in itertools.product(*options):
            if box[2] < box[0] or box[3] < box[1]:
                continue
            drawn = crop.copy()
            ImageDraw.Draw(drawn).rectangle(box, outline="red", width=FRAME_WIDTH)
            if np.array_equal(np.array(drawn.resize((w, h), Image.Resampling.LANCZOS)), pixels):
                return {
                    "positive_crop_window": [ox, oy, ox + cw, oy + ch],
                    "frame_box_in_crop": list(box),
                    "source_crop_size": [cw, ch],
                    "output_size": [w, h],
                    "resize_scale": [2, 2],
                    "frame_width": FRAME_WIDTH,
                    "reconstruction_exact": True,
                }
    raise ValueError("No exact transform found for difficult sample")


_ROWS = None
_DATA_DIR = None
_NEG_OUTPUT_DIR = None


def _init_worker(rows, data_dir, neg_output_dir=None):
    """Pool initializer: safe under both fork and spawn start methods."""
    global _ROWS, _DATA_DIR, _NEG_OUTPUT_DIR
    _ROWS = rows
    _DATA_DIR = data_dir
    _NEG_OUTPUT_DIR = neg_output_dir or os.path.join(data_dir, "teacher_neg")


def render_teacher(original, window, geometry):
    crop = original.crop(window)
    if list(crop.size) != geometry["source_crop_size"]:
        raise ValueError("Crop size differs from the official positive template")
    ImageDraw.Draw(crop).rectangle(geometry["frame_box_in_crop"], outline=(255, 0, 0), width=FRAME_WIDTH)
    return crop.resize(tuple(geometry["output_size"]), Image.Resampling.LANCZOS)


def sample_negative_window(gt, positive_window, image_size, rng):
    """Translate the whole official crop, preserving every within-crop coordinate."""
    W, H = image_size
    px, py, x2, y2 = positive_window
    cw, ch = x2 - px, y2 - py
    candidates = ((rng.randint(0, W - cw), rng.randint(0, H - ch)) for _ in range(2000))
    corners = itertools.product((0, W - cw), (0, H - ch))
    for x, y in itertools.chain(candidates, corners):
        negative = [gt[0] + x - px, gt[1] + y - py, gt[2] + x - px, gt[3] + y - py]
        if iou(negative, gt) < NEG_IOU_LIMIT:
            return [x, y, x + cw, y + ch], negative
    raise ValueError("No geometry-preserving negative window satisfies IoU < 0.1")


def _worker(i):
    h = _ROWS[i]
    try:
        data_dir = _DATA_DIR
        src_tpos = safe_dataset_path(data_dir, h["teacher_images"][0])
        with Image.open(safe_dataset_path(data_dir, h["original_images"][0])) as image:
            original = image.convert("RGB")
        with Image.open(src_tpos) as image:
            official = image.convert("RGB")
        gt = h["bbox"]
        geometry = recover_geometry(original, official, gt)
        positive_window = geometry["positive_crop_window"]
        reconstructed = render_teacher(original, positive_window, geometry)
        if not np.array_equal(np.asarray(reconstructed), np.asarray(official)):
            raise ValueError("Official positive reconstruction is not pixel-exact")
        negative_window, negative_bbox = sample_negative_window(
            gt, positive_window, original.size, random.Random(9000 + i)
        )
        negative = render_teacher(original, negative_window, geometry)
        path = os.path.join(_NEG_OUTPUT_DIR, f"{i:06d}.png")
        negative.save(path, compress_level=1)
        frame = Image.new("L", tuple(geometry["source_crop_size"]), 0)
        ImageDraw.Draw(frame).rectangle(geometry["frame_box_in_crop"], outline=255, width=FRAME_WIDTH)
        frame = frame.resize(official.size, Image.Resampling.LANCZOS)
        with open(src_tpos, "rb") as stream:
            positive_sha256 = hashlib.sha256(stream.read()).hexdigest()
        with open(path, "rb") as stream:
            negative_sha256 = hashlib.sha256(stream.read()).hexdigest()
        return {
            "idx": i,
            "ok": True,
            "gt_bbox": list(gt),
            "neg_bbox": negative_bbox,
            "iou": iou(negative_bbox, gt),
            "crop_size": list(official.size),
            "negative_crop_window": negative_window,
            "original_size": list(original.size),
            "target_bbox_in_crop": [
                gt[0] - positive_window[0],
                gt[1] - positive_window[1],
                gt[2] - positive_window[0],
                gt[3] - positive_window[1],
            ],
            "positive_sha256": positive_sha256,
            "negative_sha256": negative_sha256,
            "shared_frame_mask_sha256": hashlib.sha256(frame.tobytes()).hexdigest(),
            "generation_version": "2.0.0",
            "resampling": "PIL.LANCZOS",
            "renderer": "PIL.ImageDraw.rectangle",
            **geometry,
        }
    except Exception as e:
        return {"idx": i, "ok": False, "reason": f"exc:{type(e).__name__}:{str(e)[:180]}"}


def generate_negative_views(data_dir: str, nproc: int) -> list:
    with open(os.path.join(data_dir, "trainset_clean.jsonl"), encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    if not rows:
        raise ValueError("trainset_clean.jsonl is empty")
    target_dir = os.path.join(data_dir, "teacher_neg")
    staging_dir = tempfile.mkdtemp(prefix=".teacher_neg.", dir=data_dir)
    t0 = time.time()
    results = []
    try:
        with Pool(nproc, initializer=_init_worker, initargs=(rows, data_dir, staging_dir)) as pool:
            for n, r in enumerate(pool.imap_unordered(_worker, range(len(rows)), chunksize=16), 1):
                results.append(r)
                if n % 500 == 0:
                    ok = sum(1 for x in results if x.get("ok"))
                    print(f"[{n}/{len(rows)}] ok={ok} elapsed={time.time() - t0:.0f}s", flush=True)
        results.sort(key=lambda r: r["idx"])
        ok = sum(1 for r in results if r.get("ok"))
        success_rate = 100 * ok / len(results) if results else 0.0
        print(f"Negative views: {ok}/{len(results)} ({success_rate:.1f}%) in {(time.time() - t0) / 60:.1f} min")
        if ok != len(results) or [r.get("idx") for r in results] != list(range(len(rows))):
            write_json_atomic(os.path.join(data_dir, "results.failed.json"), results)
            raise RuntimeError("Negative generation incomplete; inspect results.failed.json before retrying")
        expected_files = {f"{i:06d}.png" for i in range(len(rows))}
        actual_files = {entry.name for entry in os.scandir(staging_dir) if entry.is_file()}
        if actual_files != expected_files:
            missing = sorted(expected_files - actual_files)[:5]
            extra = sorted(actual_files - expected_files)[:5]
            write_json_atomic(os.path.join(data_dir, "results.failed.json"), results)
            raise RuntimeError(f"Negative image set is incomplete: missing={missing}, extra={extra}")
        invalidate_training_artifacts(data_dir)
        publish_negative_views(staging_dir, target_dir)
        staging_dir = None
        write_json_atomic(os.path.join(data_dir, "dataset_generation_manifest.json"), results)
        failed_results = os.path.join(data_dir, "results.failed.json")
        if os.path.exists(failed_results):
            os.remove(failed_results)
        return results
    finally:
        if staging_dir is not None:
            shutil.rmtree(staging_dir, ignore_errors=True)


def write_json_atomic(path: str, value) -> None:
    temp_path = f"{path}.tmp.{os.getpid()}"
    try:
        with open(temp_path, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=1)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_path, path)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def publish_negative_views(staging_dir: str, target_dir: str) -> None:
    backup_dir = None
    if os.path.exists(target_dir):
        backup_dir = tempfile.mkdtemp(prefix=".teacher_neg.backup.", dir=os.path.dirname(target_dir))
        os.rmdir(backup_dir)
        os.replace(target_dir, backup_dir)
    try:
        os.replace(staging_dir, target_dir)
    except Exception:
        if backup_dir is not None:
            os.replace(backup_dir, target_dir)
        raise
    if backup_dir is not None:
        shutil.rmtree(backup_dir)


def invalidate_training_artifacts(data_dir: str) -> None:
    """Keep stale artifacts recoverable but impossible to launch accidentally."""
    for name in ("trainset_clean.parquet", "token_priors_frozen.json", "dataset_generation_manifest.json"):
        path = os.path.join(data_dir, name)
        if os.path.exists(path):
            os.replace(path, f"{path}.stale")


# ---------------------------------------------------------------- parquet --
def clean_question(problem: str) -> str:
    text = (problem or "").replace("<image>", "").strip()
    hint = "Only focus on the objects inside the red bounding box in the image to answer this question."
    text = text.replace(f"\n\n{hint}", "").replace(hint, "")
    return text.strip()


def build_record(item: dict[str, Any], data_dir: str, neg_path: str | None) -> dict[str, Any]:
    record = {
        "data_source": "zwz_rl_vqa_bbox_teacher",
        "prompt": [{"role": "user", "content": item["problem"]}],
        "images": [{"path": safe_dataset_path(data_dir, item["images"][0])}],
        "bbox_images": [{"path": safe_dataset_path(data_dir, item["teacher_images"][0])}],
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


def count_negative_views(records: list[dict[str, Any]]) -> int:
    return sum(bool(record["neg_bbox_images"]) for record in records)


def convert_to_parquet(data_dir: str, results: list) -> None:
    jsonl_path = os.path.join(data_dir, "trainset_clean.jsonl")
    if not os.path.exists(jsonl_path):
        print(f"Error: {jsonl_path} not found", file=sys.stderr)
        sys.exit(1)

    with open(jsonl_path, encoding="utf-8") as stream:
        source_rows = [line for line in stream if line.strip()]
    expected_ids = list(range(len(source_rows)))
    if [r.get("idx") for r in results] != expected_ids or not all(r.get("ok") for r in results):
        raise ValueError("Refusing to build trainset_clean.parquet from incomplete negative-view results")
    by_idx_ok = {r["idx"]: True for r in results}
    print("Converting trainset_clean.jsonl to trainset_clean.parquet ...")
    records = []
    for i, line in enumerate(source_rows):
        item = json.loads(line)
        neg = os.path.join(data_dir, "teacher_neg", f"{i:06d}.png") if by_idx_ok.get(i) else None
        records.append(build_record(item, data_dir, neg))

    dataset = datasets.Dataset.from_list(records)
    output_path = os.path.join(data_dir, "trainset_clean.parquet")
    temp_path = f"{output_path}.tmp.{os.getpid()}"
    try:
        dataset.to_parquet(temp_path)
        os.replace(temp_path, output_path)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
    n_neg = count_negative_views(records)
    print(f"Saved {len(records)} records ({n_neg} with negative views) to {output_path}")


def main() -> None:
    args = parse_args()
    data_dir = os.path.abspath(args.data_dir)
    os.makedirs(data_dir, exist_ok=True)

    if not args.skip_download:
        download_dataset(args.hf_repo, resolve_dataset_revision(args.hf_repo, args.hf_revision), data_dir)

    results = generate_negative_views(data_dir, args.nproc)
    convert_to_parquet(data_dir, results)
    # The training data just changed: derived artifacts are now stale.
    for stale in ("token_priors_frozen.json", "token_freq_counts.json"):
        stale_path = os.path.join(data_dir, stale)
        if os.path.exists(stale_path):
            os.replace(stale_path, stale_path + ".stale")
            print(f"Marked stale (training data changed): {stale} -> {stale}.stale")
    print(f"\nData preparation complete. Training data at: {data_dir}/trainset_clean.parquet")


if __name__ == "__main__":
    main()
