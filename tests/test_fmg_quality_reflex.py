from __future__ import annotations

import base64
import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

import fmg_image_generator as m
from fmg_quality_reflex import LightweightImageEvaluator


class QualityReflexTests(unittest.TestCase):
    def test_evaluator_reports_fixed_resolution(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "test.png"
            image = Image.new("RGB", (1024, 1024), "gray")
            draw = ImageDraw.Draw(image)
            for x in range(0, 1024, 32):
                draw.line((x, 0, x, 1024), fill="white", width=2)
            image.save(path)
            report = LightweightImageEvaluator().evaluate(path)
            self.assertTrue(report.resolution_ok)
            self.assertGreaterEqual(report.score, 0.0)
            self.assertLessEqual(report.score, 1.0)

    def test_mask_matches_1024_canvas(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "flat.png"
            Image.new("RGB", (1024, 1024), "gray").save(path)
            evaluator = LightweightImageEvaluator()
            report = evaluator.evaluate(path)
            mask = evaluator.build_mask(report)
            self.assertEqual(mask.size, (1024, 1024))

    def test_a1111_masked_repair_uses_img2img(self):
        png = b"\\x89PNG\\r\\n\\x1a\\n" + b"x" * 256
        with tempfile.TemporaryDirectory() as td:
            source = Path(td) / "source.png"
            Image.new("RGB", (1024, 1024), "gray").save(source)
            backend = m.A1111Backend("http://127.0.0.1:7860")
            calls = {}

            def fake_json(path, **kwargs):
                calls["path"] = path
                calls["payload"] = kwargs.get("payload")
                return {"images": [base64.b64encode(png).decode()]}

            backend._json = fake_json
            mask = Image.new("L", (1024, 1024), 0)
            ImageDraw.Draw(mask).rectangle((100, 100, 400, 400), fill=255)
            req = m.ImageRequest(prompt="test", backend="a1111")
            out = backend.repair_masked(req, source, mask, Path(td))
            self.assertEqual(calls["path"], "/sdapi/v1/img2img")
            self.assertIn("mask", calls["payload"])
            self.assertEqual(calls["payload"]["width"], 1024)
            self.assertEqual(calls["payload"]["height"], 1024)
            self.assertTrue(Path(out["path"]).is_file())


if __name__ == "__main__":
    unittest.main()
