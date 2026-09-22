from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from fmg_fca_vision_lite import FCAVisionLite, OrganEvidence, _prompt_intent


class FCAVisionLiteTests(unittest.TestCase):
    def fake_organs(self, fca):
        fca.organs.loaded["face"] = lambda *a: OrganEvidence(
            "face", False, True, 0.0, 0.0, detail="test unavailable"
        )
        fca.organs.loaded["hand"] = lambda *a: OrganEvidence(
            "hand", False, False, 0.0, 1.0
        )
        fca.organs.loaded["text"] = lambda *a: OrganEvidence(
            "text", False, False, 0.0, 1.0
        )

    def test_prompt_intent(self):
        p = _prompt_intent('portrait of a person holding a sign with "HELLO"')
        self.assertTrue(p["face"])
        self.assertTrue(p["text"])

    def test_top16_sparse_pattern(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "image.png"
            Image.new("RGB", (1024, 1024), "gray").save(path)
            fca = FCAVisionLite(Path(td) / "state.json")
            self.fake_organs(fca)
            decision, pattern = fca.assess(
                prompt="portrait of a person",
                image_path=path,
                technical_score=0.8,
                weak_regions=(),
            )
            self.assertEqual(len(pattern.active), 16)
            self.assertIn(decision.action, {"accept", "local_repair", "global_repair"})

    def test_unavailable_specialist_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "image.png"
            Image.new("RGB", (1024, 1024), "gray").save(path)
            fca = FCAVisionLite(Path(td) / "state.json")
            self.fake_organs(fca)
            decision, _ = fca.assess(
                prompt="portrait face",
                image_path=path,
                technical_score=0.9,
                weak_regions=(),
            )
            self.assertEqual(decision.semantic_defect, 0.0)

    def test_learning_persists(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "image.png"
            Image.new("RGB", (1024, 1024), "gray").save(path)
            state = Path(td) / "state.json"
            fca = FCAVisionLite(state)
            self.fake_organs(fca)
            decision, pattern = fca.assess(
                prompt="landscape",
                image_path=path,
                technical_score=0.7,
                weak_regions=(),
            )
            fca.learn(pattern, decision.action, 0.5)
            self.assertTrue(state.is_file())
            self.assertEqual(FCAVisionLite(state).policy.events, 1)

    def test_bootstrap_policy_loads_two_million_events(self):
        with tempfile.TemporaryDirectory() as td:
            fca = FCAVisionLite(Path(td) / "missing_runtime_state.json")
            self.assertEqual(fca.policy.events, 2_000_000)


if __name__ == "__main__":
    unittest.main()
