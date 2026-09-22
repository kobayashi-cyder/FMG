from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from fmg_fly_connectome import FlyConnectomeRouter


class FlyConnectomeTests(unittest.TestCase):
    def make_router(self, td):
        return FlyConnectomeRouter(Path(td) / "state.json", seed=123)

    def test_sparse_kc_activation(self):
        with tempfile.TemporaryDirectory() as td:
            r = self.make_router(td)
            request = {
                "prompt": "夕暮れの湖畔を走る犬",
                "steps": 28,
                "guidance": 9.0,
            }
            d = r.route(request, {"a1111": True, "diffusers": True})
            self.assertEqual(d.selected_mbon, "MBON::" + d.selected_backend)
            self.assertAlmostEqual(d.kc_sparsity, 0.05, places=2)
            self.assertGreater(len(d.kc_active), 0)

    def test_deterministic_before_learning(self):
        with tempfile.TemporaryDirectory() as td:
            r = self.make_router(td)
            req = {
                "prompt": "cat photo",
                "steps": 20,
                "guidance": 7.0,
            }
            d1 = r.route(req, {"a1111": True, "diffusers": True})
            d2 = r.route(req, {"a1111": True, "diffusers": True})
            self.assertEqual(d1.selected_backend, d2.selected_backend)
            self.assertEqual(d1.kc_active, d2.kc_active)

    def test_inhibition_selects_other_mbon(self):
        with tempfile.TemporaryDirectory() as td:
            r = self.make_router(td)
            req = {
                "prompt": "test image",
                "steps": 20,
                "guidance": 7.0,
            }
            d1 = r.route(req, {"a1111": True, "diffusers": True})
            d2 = r.route(
                req,
                {"a1111": True, "diffusers": True},
                inhibited={d1.selected_backend},
            )
            self.assertNotEqual(d1.selected_backend, d2.selected_backend)

    def test_reward_persists(self):
        with tempfile.TemporaryDirectory() as td:
            state = Path(td) / "state.json"
            r = FlyConnectomeRouter(state, seed=123)
            req = {
                "prompt": "bird",
                "steps": 20,
                "guidance": 7.0,
            }
            d = r.route(req, {"a1111": True, "diffusers": True})
            r.reward(d, 0.8, structural_verified=True, elapsed_s=1.0)
            self.assertTrue(state.is_file())
            r2 = FlyConnectomeRouter(state, seed=123)
            self.assertEqual(r2.events, 1)
            self.assertGreater(r2.rewards[d.selected_backend], 0.0)


if __name__ == "__main__":
    unittest.main()
