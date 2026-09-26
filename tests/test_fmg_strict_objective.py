from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

from fmg_strict_objective import StrictObjectiveEvaluator
from run_fmg_eval_loop import FailureMemory, JsonlLog, load_suite, refine_prompt


class FakeAlignment:
    def __init__(self, probabilities=None, available=True):
        self.probabilities = probabilities or [0.70, 0.10, 0.10, 0.10]
        self.available = available

    def score(self, image_path, texts):
        if not self.available:
            return {
                "available": False,
                "model": "fake",
                "probabilities": [],
                "cosines": [],
                "error": "offline",
            }
        probs = list(self.probabilities)
        if len(probs) < len(texts):
            remain = max(0.0, 1.0 - sum(probs))
            probs += [remain / max(1, len(texts) - len(probs))] * (len(texts) - len(probs))
        probs = probs[:len(texts)]
        total = sum(probs) or 1.0
        probs = [x / total for x in probs]
        return {
            "available": True,
            "model": "fake",
            "probabilities": probs,
            "cosines": probs,
        }


def make_image(path: Path):
    image = Image.new("RGB", (1024, 1024), "gray")
    draw = ImageDraw.Draw(image)
    for x in range(0, 1024, 32):
        draw.line((x, 0, x, 1024), fill="white", width=2)
    for y in range(0, 1024, 32):
        draw.line((0, y, 1024, y), fill="black", width=2)
    image.save(path)


class StrictObjectiveTests(unittest.TestCase):
    def test_alignment_failure_blocks_pass(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "image.png"
            make_image(path)
            evaluator = StrictObjectiveEvaluator(
                alignment=FakeAlignment([0.18, 0.72, 0.06, 0.04])
            )
            report = evaluator.evaluate(
                path,
                {
                    "prompt": "red mug",
                    "contrastive_negatives": ["blue mug", "two mugs", "empty frame"],
                    "min_technical": 0.0,
                    "min_positive_prob": 0.42,
                    "min_margin": 0.07,
                },
            )
            self.assertEqual(report.verdict, "fail")
            self.assertFalse(report.gates["alignment_ok"])

    def test_unavailable_alignment_is_unknown_not_pass(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "image.png"
            make_image(path)
            evaluator = StrictObjectiveEvaluator(
                alignment=FakeAlignment(available=False)
            )
            report = evaluator.evaluate(
                path,
                {
                    "prompt": "red mug",
                    "contrastive_negatives": ["blue mug"],
                    "min_technical": 0.0,
                },
            )
            self.assertEqual(report.verdict, "unknown")
            self.assertFalse(report.gates["evidence_complete"])

    def test_prompt_refinement_uses_failure_reason(self):
        prompt = refine_prompt(
            "studio photograph of a red mug",
            {
                "reasons": [
                    "prompt alignment below strict threshold (p=0.2000, margin=-0.5000)",
                    "technical score below threshold (0.4 < 0.5)",
                ]
            },
        )
        self.assertIn("satisfy every requested", prompt)
        self.assertIn("sharp focus", prompt)

    def test_log_usage_counts_jsonl_bytes(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "logs"
            mirror = Path(td) / "generated" / "LOG_USAGE.json"
            logger = JsonlLog(root, mirror)
            logger.append({"hello": "世界", "n": 1})
            logger.append({"hello": "again", "n": 2})
            state = json.loads(mirror.read_text(encoding="utf-8"))
            self.assertEqual(state["events"], 2)
            self.assertGreater(state["total_log_bytes"], 0)
            self.assertEqual(state["next_threshold_gb"], 50)

    def test_failure_memory_mirrors_metrics(self):
        with tempfile.TemporaryDirectory() as td:
            state = Path(td) / "runtime" / "failure.json"
            mirror = Path(td) / "generated" / "METRICS.json"
            memory = FailureMemory(state, mirror)
            memory.observe(
                {"id": "case-1", "category": "counting", "_effective_prompt": "exactly two cats"},
                {"verdict": "fail", "score": 0.25, "reasons": ["prompt alignment below strict threshold"]},
            )
            data = json.loads(mirror.read_text(encoding="utf-8"))
            self.assertEqual(data["categories"]["counting"]["fail"], 1)
            self.assertEqual(data["cases"]["case-1"]["attempts"], 1)

    def test_suite_has_train_and_holdout(self):
        suite_path = Path(__file__).resolve().parents[1] / "data" / "fmg_prompt_suite_v25.jsonl"
        train = load_suite(suite_path, "train")
        holdout = load_suite(suite_path, "holdout")
        self.assertGreaterEqual(len(train), 200)
        self.assertGreaterEqual(len(holdout), 50)
        self.assertTrue(all(x["split"] == "train" for x in train))
        self.assertTrue(all(x["split"] == "holdout" for x in holdout))


if __name__ == "__main__":
    unittest.main()
