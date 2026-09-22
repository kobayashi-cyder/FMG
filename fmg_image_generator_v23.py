from __future__ import annotations

import argparse
import base64
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

import fmg_image_generator as base
from fmg_fca_vision_lite import FCAVisionLite

VERSION = "FMG-IMG-CONNECTOME-2.3"
base.VERSION = VERSION
base.Handler.server_version = VERSION
base._UI = base._UI.replace(
    "masked repair",
    "FCA-Vision-Lite · local/global repair",
).replace(
    "FMG Image Generator — Fly Connectome V2",
    "FMG Image Generator — Fly Connectome V2.3",
)


class A1111BackendV23(base.A1111Backend):
    def repair_full(self, req, source_path: Path, output_dir: Path, *, reason="semantic consistency"):
        source_b64 = base64.b64encode(source_path.read_bytes()).decode("ascii")
        payload = {
            "init_images": [source_b64],
            "prompt": (
                req.prompt
                + ", preserve composition, improve "
                + reason
                + ", coherent anatomy, accurate visible details"
            ),
            "negative_prompt": req.negative_prompt,
            "width": base.IMAGE_WIDTH,
            "height": base.IMAGE_HEIGHT,
            "steps": req.steps,
            "cfg_scale": req.guidance,
            "seed": -1,
            "denoising_strength": 0.22,
            "batch_size": 1,
            "n_iter": 1,
        }
        started = time.perf_counter()
        value = self._json(
            "/sdapi/v1/img2img",
            method="POST",
            payload=payload,
            timeout=600,
        )
        images = value.get("images") if isinstance(value, dict) else None
        if not images:
            raise base.BackendError("A1111 full repair returned no image")
        raw = base64.b64decode(str(images[0]).split(",", 1)[-1], validate=False)
        if not (raw.startswith(b"\x89PNG\r\n\x1a\n") or raw.startswith(b"\xff\xd8")):
            raise base.BackendError("A1111 full repair returned unsupported image payload")
        ext = ".png" if raw.startswith(b"\x89PNG") else ".jpg"
        output_dir.mkdir(parents=True, exist_ok=True)
        name = f"fmg_fca_repair_{int(time.time()*1000)}{ext}"
        path = output_dir / name
        path.write_bytes(raw)
        return {
            "ok": True,
            "backend": "a1111",
            "operation": "fca_global_repair",
            "path": str(path),
            "name": name,
            "elapsed_s": round(time.perf_counter() - started, 3),
        }


def _mask(regions):
    image = Image.new("L", (base.IMAGE_WIDTH, base.IMAGE_HEIGHT), 0)
    draw = ImageDraw.Draw(image)
    for x0, y0, x1, y1 in regions:
        draw.rectangle(
            (
                max(0, x0 - 48),
                max(0, y0 - 48),
                min(base.IMAGE_WIDTH, x1 + 48),
                min(base.IMAGE_HEIGHT, y1 + 48),
            ),
            fill=255,
        )
    return image.filter(ImageFilter.GaussianBlur(18))


class FMGImageGeneratorV23(base.FMGImageGenerator):
    def __init__(self, output_dir=None):
        super().__init__(output_dir)
        self.a1111 = A1111BackendV23()
        self.fca_vision = FCAVisionLite(
            self.output_dir.parent / "fca_vision" / "state.json"
        )

    def status(self):
        value = super().status()
        value["version"] = VERSION
        value["quality_reflex"]["semantic_vision"] = self.fca_vision.status()
        value["quality_reflex"]["motor_outputs"] = [
            "MBON::evaluate",
            "MBON::repair-local",
            "MBON::repair-global",
            "MBON::accept",
        ]
        return value

    def _quality_reflex(self, req, result, availability):
        before = self.evaluator.evaluate(result["path"])
        decision, pattern = self.fca_vision.assess(
            prompt=req.prompt,
            image_path=result["path"],
            technical_score=before.score,
            weak_regions=before.weak_regions,
        )
        trace = {
            "evaluate_mbon": "MBON::evaluate",
            "before": before.as_dict(),
            "fca_vision_lite": decision.as_dict(),
            "repair_attempted": False,
            "repair_applied": False,
            "decision_mbon": "MBON::accept",
        }

        action = decision.action
        if action == "accept":
            reward = 0.20 if before.score >= base.QUALITY_REPAIR_THRESHOLD else 0.02
            self.fca_vision.learn(pattern, action, reward)
            trace["repair_skipped_reason"] = "FCA-Vision-Lite accepted artifact"
            return result, trace

        if not availability.get("a1111"):
            self.fca_vision.learn(pattern, action, -0.18)
            trace["repair_skipped_reason"] = "A1111/Forge repair organ unavailable"
            return result, trace

        trace["repair_attempted"] = True
        source = Path(result["path"])
        try:
            if action == "local_repair":
                trace["decision_mbon"] = "MBON::repair-local"
                regions = decision.repair_boxes or before.weak_regions
                repaired = self.a1111.repair_masked(
                    req,
                    source,
                    _mask(regions),
                    self.output_dir,
                )
            else:
                trace["decision_mbon"] = "MBON::repair-global"
                reasons = [
                    e.name for e in decision.evidence
                    if e.relevant and e.available and e.defect > 0
                ]
                repaired = self.a1111.repair_full(
                    req,
                    source,
                    self.output_dir,
                    reason=", ".join(reasons) or "semantic consistency",
                )

            after = self.evaluator.evaluate(repaired["path"])
            after_decision, _ = self.fca_vision.assess(
                prompt=req.prompt,
                image_path=repaired["path"],
                technical_score=after.score,
                weak_regions=after.weak_regions,
            )
            gain = (after.score - before.score) + 0.35 * (
                decision.semantic_defect - after_decision.semantic_defect
            )
            trace["after"] = after.as_dict()
            trace["fca_after"] = after_decision.as_dict()
            trace["repair_gain"] = round(gain, 4)

            if gain >= 0.012 and after.resolution_ok:
                trace["repair_applied"] = True
                repaired["seed"] = result.get("seed")
                repaired["source_artifact"] = result.get("name")
                self.fca_vision.learn(
                    pattern,
                    action,
                    min(1.0, 0.35 + 2.0 * gain),
                )
                return repaired, trace

            Path(repaired["path"]).unlink(missing_ok=True)
            trace["decision_mbon"] = "MBON::accept"
            trace["repair_rejected_reason"] = "repair did not improve verified score"
            self.fca_vision.learn(pattern, action, -0.28)
            return result, trace
        except Exception as exc:
            trace["decision_mbon"] = "MBON::accept"
            trace["repair_error"] = str(exc)
            self.fca_vision.learn(pattern, action, -0.42)
            return result, trace


def main():
    ap = argparse.ArgumentParser(description="FMG V2.3 with FCA-Vision-Lite")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=18765)
    ap.add_argument("--output", default="runtime/images")
    ap.add_argument("--status", action="store_true")
    args = ap.parse_args()

    base.GENERATOR = FMGImageGeneratorV23(args.output)
    if args.status:
        import json
        print(json.dumps(base.GENERATOR.status(), ensure_ascii=False, indent=2))
        return

    server = base.ThreadingHTTPServer((args.host, args.port), base.Handler)
    print(f"{VERSION} listening on http://{args.host}:{args.port}/")
    print("FCA-Vision-Lite: 128 sensory -> 256 KC -> top-16 -> MBON")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
