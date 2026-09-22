from __future__ import annotations

import base64
import tempfile
import unittest
from pathlib import Path

import fmg_image_generator as m


class RequestTests(unittest.TestCase):
    def test_resolution_is_fixed_1024(self):
        r = m.ImageRequest.from_mapping({
            "prompt": "cat",
            "width": 777,
            "height": 1999,
            "backend": "connectome",
        })
        self.assertFalse(hasattr(r, "width"))
        self.assertFalse(hasattr(r, "height"))

    def test_quality_defaults(self):
        r = m.ImageRequest.from_mapping({"prompt": "cat"})
        self.assertEqual(r.steps, 50)
        self.assertEqual(r.guidance, 9.0)

    def test_prompt_required(self):
        with self.assertRaises(ValueError):
            m.ImageRequest.from_mapping({"prompt": ""})


class ControllerTests(unittest.TestCase):
    def test_connectome_controller_present(self):
        with tempfile.TemporaryDirectory() as td:
            g = m.FMGImageGenerator(td)
            self.assertEqual(
                g.connectome.status()["mode"],
                "fly-connectome-inspired",
            )

    def test_explicit_diffusers_organ(self):
        with tempfile.TemporaryDirectory() as td:
            g = m.FMGImageGenerator(td)
            self.assertIs(g._backend_for("diffusers"), g.diffusers)


class A1111PayloadTests(unittest.TestCase):
    def test_generation_writes_png(self):
        png = b"\x89PNG\r\n\x1a\n" + b"x" * 128
        with tempfile.TemporaryDirectory() as td:
            backend = m.A1111Backend("http://127.0.0.1:7860")
            backend._json = lambda *a, **k: {
                "images": [base64.b64encode(png).decode()]
            }
            req = m.ImageRequest(
                prompt="test",
                steps=2,
                guidance=1.0,
                seed=1,
                backend="a1111",
            )
            out = backend.generate(req, Path(td))
            self.assertTrue(out["ok"])
            self.assertTrue((Path(td) / out["name"]).is_file())


if __name__ == "__main__":
    unittest.main()
