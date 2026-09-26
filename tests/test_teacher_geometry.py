import importlib.util
import random
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image, ImageDraw

spec = importlib.util.spec_from_file_location(
    "teacher_geometry", Path(__file__).resolve().parents[1] / "scripts/build_trainset.py"
)
geometry = importlib.util.module_from_spec(spec)
with mock.patch.dict(sys.modules, {"datasets": types.ModuleType("datasets")}):
    spec.loader.exec_module(geometry)


class TeacherGeometryTests(unittest.TestCase):
    def make_scene(self, gt, window, frame, size=(300, 240), red_background=False):
        pixels = np.random.default_rng(42).integers(0, 180, (*size[::-1], 3), dtype=np.uint8)
        if red_background:
            pixels[:] = (255, 0, 0)
            pixels[80:130, 100:140] = (0, 100, 150)
        original = Image.fromarray(pixels)
        crop = original.crop(window)
        ImageDraw.Draw(crop).rectangle(frame, outline=(255, 0, 0), width=5)
        official = crop.resize((crop.width * 2, crop.height * 2), Image.Resampling.LANCZOS)
        recovered = geometry.recover_geometry(original, official, gt)
        rebuilt = geometry.render_teacher(original, recovered["positive_crop_window"], recovered)
        self.assertTrue(np.array_equal(official, rebuilt))
        return original, official, recovered

    def test_narrow_box_preserves_layout_and_uniform_scale(self):
        gt, window, frame = [100, 40, 142, 170], [96, 26, 146, 183], [2, 7, 47, 150]
        original, official, recovered = self.make_scene(gt, window, frame)
        self.assertEqual(recovered["positive_crop_window"], window)
        self.assertEqual(recovered["frame_box_in_crop"], frame)
        negative_window, negative_bbox = geometry.sample_negative_window(gt, window, original.size, random.Random(40))
        negative = geometry.render_teacher(original, negative_window, recovered)
        self.assertEqual(official.size, negative.size)
        self.assertEqual(recovered["resize_scale"], [2, 2])
        self.assertLess(geometry.iou(gt, negative_bbox), 0.1)
        self.assertEqual(
            [negative_bbox[0] - negative_window[0], negative_bbox[1] - negative_window[1]],
            [gt[0] - window[0], gt[1] - window[1]],
        )

    def test_edge_clipped_positive_keeps_its_asymmetric_frame(self):
        gt, window, frame = [3, 71, 98, 126], [0, 65, 108, 132], [0, 3, 102, 65]
        _, _, recovered = self.make_scene(gt, window, frame)
        self.assertEqual(recovered["positive_crop_window"], window)
        self.assertEqual(recovered["frame_box_in_crop"], frame)

    def test_large_frame_is_not_limited_to_30_pixels(self):
        gt, window, frame = [200, 300, 1200, 400], [95, 289, 1305, 410], [55, 6, 1155, 116]
        _, _, recovered = self.make_scene(gt, window, frame, size=(1600, 700))
        self.assertEqual(recovered["frame_box_in_crop"], frame)

    def test_red_scene_content_requires_exact_reconstruction(self):
        gt, window, frame = [96, 60, 166, 160], [89, 50, 173, 170], [3, 5, 80, 115]
        _, _, recovered = self.make_scene(gt, window, frame, red_background=True)
        self.assertTrue(recovered["reconstruction_exact"])

    def test_small_box_with_no_safe_registration_pixels(self):
        gt, window, frame = [100, 100, 111, 151], [99, 95, 112, 156], [0, 2, 12, 58]
        _, _, recovered = self.make_scene(gt, window, frame)
        self.assertEqual(recovered["positive_crop_window"], window)
        self.assertEqual(recovered["frame_box_in_crop"], frame)

    def test_impossible_negative_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "No geometry-preserving"):
            geometry.sample_negative_window([0, 0, 100, 100], [0, 0, 100, 100], (100, 100), random.Random(0))


if __name__ == "__main__":
    unittest.main()
