"""Unit tests for the CROP-release importer's pure mapping logic."""
import importlib.util
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "import_crop_release", ROOT / "scripts" / "import_crop_release.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def fake_row(sample_id=7, label="healthy"):
    return {
        "sample_id": sample_id,
        "question": "What color?\n\nOnly focus ...",
        "answer": "B",
        "legacy_quality_label": label,
        "source_bbox": [10, 20, 30, 40],
        "negative_bbox": [1, 2, 3, 4],
        "negative_crop_window": [0, 0, 50, 50],
        "original_size": [640, 480],
        "output_size": [200, 300],
        "target_bbox_in_crop": [5, 6, 7, 8],
        "bbox_iou": 0.02,
        "positive_sha256": "pos" * 16,
        "negative_sha256": "neg" * 16,
        "shared_frame_mask_sha256": "fm" * 16,
        "generation_version": "2.0.0",
    }


class FilterRowsTests(unittest.TestCase):
    def test_keeps_only_requested_labels_in_sample_order(self):
        rows = [fake_row(5, "annotation_suspect"), fake_row(2), fake_row(9, "healthy")]
        kept = MODULE.filter_rows(rows, "healthy")
        self.assertEqual([r["sample_id"] for r in kept], [2, 9])

    def test_empty_selection_is_an_error(self):
        with self.assertRaises(SystemExit):
            MODULE.filter_rows([fake_row(1, "common_sense")], "healthy")


class MakeJsonlRowTests(unittest.TestCase):
    def test_problem_carries_image_placeholder_and_release_question(self):
        row = MODULE.make_jsonl_row(fake_row(7), 7)
        self.assertTrue(row["problem"].startswith("<image>\n"))
        self.assertIn("What color?", row["problem"])
        self.assertEqual(row["images"], ["images/000007.png"])
        self.assertEqual(row["teacher_images"], ["teacher_images/000007.png"])
        self.assertEqual(row["answer"], "B")
        self.assertEqual(row["extra_info"]["answer"], "B")
        self.assertIn("What color?", row["extra_info"]["question"])
        # stays JSON-serializable with the exact upstream key set
        self.assertEqual(
            set(row), {"problem", "images", "teacher_images", "answer", "extra_info"}
        )
        json.dumps(row)


class MakeResultRowTests(unittest.TestCase):
    def test_maps_release_fields_to_v2_results_schema(self):
        record = MODULE.make_result_row(fake_row(7), new_idx=3)
        self.assertEqual(record["idx"], 3)
        self.assertTrue(record["ok"])
        self.assertEqual(record["generation_version"], "2.0.0")
        self.assertEqual(record["gt_bbox"], [10, 20, 30, 40])
        self.assertEqual(record["neg_bbox"], [1, 2, 3, 4])
        self.assertEqual(record["iou"], 0.02)
        self.assertEqual(record["crop_size"], [200, 300])
        self.assertEqual(record["negative_sha256"], "neg" * 16)
        json.dumps(record)


if __name__ == "__main__":
    unittest.main()
