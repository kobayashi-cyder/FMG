from __future__ import annotations

import math
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

from fmg_fca_vision_lite import FaceOrgan, HandOrgan, TextOrgan
from fmg_quality_reflex import LightweightImageEvaluator

VERSION = "FMG-STRICT-OBJECTIVE-2.5"
DEFAULT_ALIGNMENT_MODEL = "openai/clip-vit-base-patch32"


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _norm_text(value: str) -> str:
    return re.sub(r"[^0-9A-Z]+", "", value.upper())


class AlignmentScorer(Protocol):
    def score(self, image_path: str | Path, texts: list[str]) -> dict[str, Any]:
        ...


class CLIPAlignmentScorer:
    """Lazy local contrastive image/text scorer.

    It is intentionally fail-closed. If the model cannot be loaded, strict
    evaluation returns UNKNOWN rather than silently treating alignment as good.
    """

    def __init__(self, model_id: str | None = None):
        self.model_id = model_id or os.environ.get(
            "FMG_ALIGNMENT_MODEL", DEFAULT_ALIGNMENT_MODEL
        )
        self.offline = _env_bool("FMG_IMAGE_OFFLINE", False)
        self._model = None
        self._processor = None
        self._torch = None
        self._device = "unloaded"
        self._load_error: str | None = None

    def _load(self) -> None:
        if self._model is not None or self._load_error is not None:
            return
        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor

            kwargs = {"local_files_only": True} if self.offline else {}
            self._model = CLIPModel.from_pretrained(self.model_id, **kwargs)
            self._processor = CLIPProcessor.from_pretrained(self.model_id, **kwargs)
            self._torch = torch
            if torch.cuda.is_available():
                self._device = "cuda"
            elif (
                getattr(torch.backends, "mps", None)
                and torch.backends.mps.is_available()
            ):
                self._device = "mps"
            else:
                self._device = "cpu"
            self._model.to(self._device)
            self._model.eval()
        except Exception as exc:
            self._load_error = str(exc)

    def status(self) -> dict[str, Any]:
        self._load()
        return {
            "model": self.model_id,
            "available": self._model is not None,
            "device": self._device,
            "offline": self.offline,
            "error": self._load_error,
        }

    def score(self, image_path: str | Path, texts: list[str]) -> dict[str, Any]:
        self._load()
        if self._model is None or self._processor is None or self._torch is None:
            return {
                "available": False,
                "model": self.model_id,
                "error": self._load_error or "alignment model unavailable",
                "probabilities": [],
                "cosines": [],
            }
        from PIL import Image

        with Image.open(Path(image_path)) as source:
            image = source.convert("RGB")
            inputs = self._processor(
                text=texts,
                images=image,
                return_tensors="pt",
                padding=True,
            )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with self._torch.inference_mode():
            out = self._model(**inputs)
            image_embeds = out.image_embeds
            text_embeds = out.text_embeds
            image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
            text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
            cosines = image_embeds @ text_embeds.T
            logits = cosines / 0.07
            probs = logits.softmax(dim=-1)[0]
        return {
            "available": True,
            "model": self.model_id,
            "device": self._device,
            "probabilities": [float(x) for x in probs.detach().cpu().tolist()],
            "cosines": [float(x) for x in cosines[0].detach().cpu().tolist()],
        }


@dataclass(frozen=True)
class StrictObjectiveReport:
    verdict: str
    score: float
    technical_score: float
    alignment_positive_probability: float | None
    alignment_margin: float | None
    exact_text_ok: bool | None
    specialist_checks: dict[str, Any]
    reasons: tuple[str, ...]
    gates: dict[str, Any]
    raw_alignment: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["version"] = VERSION
        value["reasons"] = list(self.reasons)
        return value


class StrictObjectiveEvaluator:
    def __init__(
        self,
        *,
        width: int = 1024,
        height: int = 1024,
        alignment: AlignmentScorer | None = None,
    ):
        self.technical = LightweightImageEvaluator(width=width, height=height)
        self.alignment = alignment or CLIPAlignmentScorer()
        self._organs = {
            "face": FaceOrgan(),
            "hand": HandOrgan(),
            "text": TextOrgan(),
        }

    def _specialists(
        self,
        image_path: str | Path,
        required: list[str],
        exact_text: str | None,
    ) -> tuple[dict[str, Any], list[str], bool]:
        checks: dict[str, Any] = {}
        reasons: list[str] = []
        unknown = False
        for name in required:
            organ = self._organs.get(name)
            if organ is None:
                checks[name] = {"available": False, "ok": None, "detail": "unknown specialist"}
                reasons.append(f"required specialist unavailable: {name}")
                unknown = True
                continue
            evidence = organ(image_path, True)
            ok: bool | None
            if not evidence.available or evidence.confidence <= 0:
                ok = None
                unknown = True
                reasons.append(f"required specialist unavailable: {name}")
            else:
                ok = evidence.defect < 0.45
                if not ok:
                    reasons.append(f"{name} specialist detected objective defect")
            checks[name] = {
                "available": evidence.available,
                "ok": ok,
                "defect": round(evidence.defect, 4),
                "confidence": round(evidence.confidence, 4),
                "detail": evidence.detail,
            }

        if exact_text:
            text_evidence = self._organs["text"](image_path, True)
            target = _norm_text(exact_text)
            observed = ""
            if text_evidence.detail.startswith("ocr="):
                observed = _norm_text(text_evidence.detail[4:])
            if not text_evidence.available or text_evidence.confidence <= 0:
                checks["exact_text"] = {
                    "available": False,
                    "ok": None,
                    "target": exact_text,
                    "observed": "",
                }
                reasons.append("exact-text OCR evidence unavailable")
                unknown = True
            else:
                ok = bool(target) and target in observed
                checks["exact_text"] = {
                    "available": True,
                    "ok": ok,
                    "target": exact_text,
                    "observed": text_evidence.detail[4:] if text_evidence.detail.startswith("ocr=") else "",
                }
                if not ok:
                    reasons.append("exact requested text was not reproduced")
        return checks, reasons, unknown

    def evaluate(self, image_path: str | Path, case: dict[str, Any]) -> StrictObjectiveReport:
        tech = self.technical.evaluate(image_path)
        min_technical = float(case.get("min_technical", 0.50))
        min_positive = float(case.get("min_positive_prob", 0.42))
        min_margin = float(case.get("min_margin", 0.07))

        negatives = [str(x) for x in case.get("contrastive_negatives", []) if str(x).strip()]
        texts = [str(case["prompt"])] + negatives
        alignment = self.alignment.score(image_path, texts)

        reasons: list[str] = []
        unknown = False
        positive_prob: float | None = None
        margin: float | None = None
        align_ok: bool | None = None

        if not alignment.get("available"):
            reasons.append("semantic alignment scorer unavailable")
            unknown = True
        else:
            probs = [float(x) for x in alignment.get("probabilities", [])]
            if not probs:
                reasons.append("semantic alignment scorer returned no probabilities")
                unknown = True
            else:
                positive_prob = probs[0]
                strongest_negative = max(probs[1:], default=0.0)
                margin = positive_prob - strongest_negative
                align_ok = positive_prob >= min_positive and margin >= min_margin
                if not align_ok:
                    reasons.append(
                        "prompt alignment below strict threshold "
                        f"(p={positive_prob:.4f}, margin={margin:.4f})"
                    )

        technical_ok = bool(tech.resolution_ok and tech.score >= min_technical)
        if not tech.resolution_ok:
            reasons.append("resolution is not the fixed 1024x1024 objective")
        if tech.score < min_technical:
            reasons.append(
                f"technical score below threshold ({tech.score:.4f} < {min_technical:.4f})"
            )

        required = [str(x) for x in case.get("required_specialists", [])]
        exact_text = case.get("exact_text")
        specialist_checks, specialist_reasons, specialist_unknown = self._specialists(
            image_path,
            required,
            str(exact_text) if exact_text else None,
        )
        reasons.extend(specialist_reasons)
        unknown = unknown or specialist_unknown

        specialist_fail = any(
            check.get("ok") is False for check in specialist_checks.values()
        )
        exact_text_ok = None
        if "exact_text" in specialist_checks:
            exact_text_ok = specialist_checks["exact_text"].get("ok")

        if unknown:
            verdict = "unknown"
        elif not technical_ok or align_ok is False or specialist_fail:
            verdict = "fail"
        else:
            verdict = "pass"

        align_component = max(0.0, min(1.0, positive_prob or 0.0))
        margin_component = max(0.0, min(1.0, ((margin or 0.0) + 0.15) / 0.55))
        specialist_component = 1.0
        if specialist_checks:
            vals = [
                1.0 if v.get("ok") is True else 0.0
                for v in specialist_checks.values()
                if v.get("ok") is not None
            ]
            if vals:
                specialist_component = sum(vals) / len(vals)
            elif unknown:
                specialist_component = 0.0

        # Geometric mean punishes any weak axis more strongly than a simple average.
        parts = [
            max(1e-6, tech.score),
            max(1e-6, align_component),
            max(1e-6, margin_component),
            max(1e-6, specialist_component),
        ]
        score = math.prod(parts) ** (1.0 / len(parts))
        if verdict == "unknown":
            score *= 0.35

        gates = {
            "technical_ok": technical_ok,
            "alignment_ok": align_ok,
            "specialist_fail": specialist_fail,
            "evidence_complete": not unknown,
            "thresholds": {
                "min_technical": min_technical,
                "min_positive_probability": min_positive,
                "min_margin": min_margin,
            },
        }
        return StrictObjectiveReport(
            verdict=verdict,
            score=round(score, 6),
            technical_score=round(float(tech.score), 6),
            alignment_positive_probability=None if positive_prob is None else round(positive_prob, 6),
            alignment_margin=None if margin is None else round(margin, 6),
            exact_text_ok=exact_text_ok,
            specialist_checks=specialist_checks,
            reasons=tuple(reasons),
            gates=gates,
            raw_alignment=alignment,
        )
