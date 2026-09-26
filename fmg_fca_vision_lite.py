from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

VERSION = "FMG-FCA-VISION-LITE-2.4"
ACTIONS = ("accept", "local_repair", "global_repair")


def _u64(text: str) -> int:
    return int.from_bytes(
        hashlib.blake2b(text.encode("utf-8", "ignore"), digest_size=8).digest(),
        "big",
    )


@dataclass(frozen=True)
class SparsePattern:
    active: tuple[int, ...]
    values: tuple[float, ...]


@dataclass(frozen=True)
class OrganEvidence:
    name: str
    available: bool
    relevant: bool
    defect: float
    confidence: float
    boxes: tuple[tuple[int, int, int, int], ...] = ()
    detail: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "available": self.available,
            "relevant": self.relevant,
            "defect": round(self.defect, 4),
            "confidence": round(self.confidence, 4),
            "boxes": [list(x) for x in self.boxes],
            "detail": self.detail,
        }


@dataclass(frozen=True)
class VisionDecision:
    action: str
    scores: dict[str, float]
    kc_active: tuple[int, ...]
    semantic_defect: float
    evidence_confidence: float
    repair_boxes: tuple[tuple[int, int, int, int], ...]
    evidence: tuple[OrganEvidence, ...]
    prompt_intent: dict[str, bool]

    def as_dict(self) -> dict[str, Any]:
        return {
            "version": VERSION,
            "action": self.action,
            "scores": {k: round(v, 5) for k, v in self.scores.items()},
            "kc_active": list(self.kc_active),
            "semantic_defect": round(self.semantic_defect, 4),
            "evidence_confidence": round(self.evidence_confidence, 4),
            "repair_boxes": [list(x) for x in self.repair_boxes],
            "evidence": [x.as_dict() for x in self.evidence],
            "prompt_intent": self.prompt_intent,
        }


class SensoryHash:
    def __init__(self, channels: int = 128):
        self.channels = channels

    def encode(self, text: str, features: dict[str, float]) -> list[float]:
        vec = [0.0] * self.channels
        tokens = text.casefold().split()
        if len(tokens) == 1 and tokens:
            s = tokens[0]
            tokens = [s[i:i + 2] for i in range(max(1, len(s) - 1))] or [s]
        for token in tokens:
            for salt in range(3):
                h = _u64(f"{salt}:{token}")
                vec[h % self.channels] += 1.0 if ((h >> 8) & 1) else -1.0
        for key, value in sorted(features.items()):
            clipped = max(-1.0, min(1.0, float(value)))
            for salt in range(2):
                h = _u64(f"feature:{salt}:{key}")
                vec[h % self.channels] += clipped
        scale = max(1.0, max(abs(x) for x in vec))
        return [x / scale for x in vec]


class KenyonLayer:
    def __init__(self, inputs=128, kcs=256, fan_in=6, winners=16, seed="FCA-VISION-LITE"):
        self.inputs = inputs
        self.kcs = kcs
        self.fan_in = fan_in
        self.winners = winners
        self.seed = seed
        self.wiring = tuple(self._fan_in_for(kc) for kc in range(kcs))

    def _fan_in_for(self, kc: int) -> tuple[int, ...]:
        seen: set[int] = set()
        i = 0
        while len(seen) < self.fan_in:
            seen.add(_u64(f"{self.seed}:{kc}:{i}") % self.inputs)
            i += 1
        return tuple(sorted(seen))

    def activate(self, sensory: list[float], trace: dict[int, float]) -> SparsePattern:
        scored: list[tuple[float, int]] = []
        for kc, inputs in enumerate(self.wiring):
            drive = sum(sensory[i] for i in inputs) / len(inputs)
            drive += 0.10 * trace.get(kc, 0.0)
            scored.append((math.tanh(drive), kc))
        scored.sort(key=lambda p: (p[0], -p[1]), reverse=True)
        winners = scored[:self.winners]
        return SparsePattern(
            active=tuple(kc for _, kc in winners),
            values=tuple(value for value, _ in winners),
        )


class TemporalTrace:
    def __init__(self, decay=0.82):
        self.decay = decay
        self.state: dict[int, float] = {}

    def update(self, pattern: SparsePattern) -> None:
        nxt = {
            k: v * self.decay
            for k, v in self.state.items()
            if abs(v * self.decay) >= 1e-4
        }
        for kc, value in zip(pattern.active, pattern.values):
            nxt[kc] = max(nxt.get(kc, 0.0), value)
        self.state = nxt


class MBONPolicy:
    def __init__(self, state_path: str | Path, learning_rate=0.06, bootstrap_path: str | Path | None = None):
        self.state_path = Path(state_path)
        self.learning_rate = learning_rate
        self.weights: dict[str, dict[int, float]] = {a: {} for a in ACTIONS}
        self.baseline = 0.0
        self.events = 0
        self.bootstrap_path = Path(bootstrap_path) if bootstrap_path else None
        self._load()

    def learned_scores(self, pattern: SparsePattern) -> dict[str, float]:
        return {
            action: sum(
                self.weights[action].get(kc, 0.0) * value
                for kc, value in zip(pattern.active, pattern.values)
            )
            for action in ACTIONS
        }

    def learn(self, pattern: SparsePattern, action: str, reward: float) -> float:
        prediction = self.learned_scores(pattern)[action]
        rpe = float(reward) - prediction - self.baseline
        for kc, value in zip(pattern.active, pattern.values):
            old = self.weights[action].get(kc, 0.0)
            self.weights[action][kc] = max(
                -1.5,
                min(1.5, old + self.learning_rate * rpe * value),
            )
        self.baseline = 0.98 * self.baseline + 0.02 * float(reward)
        self.events += 1
        self._save()
        return rpe

    def _save(self) -> None:
        payload = {
            "schema": "fmg.fca-vision-lite.v2.4",
            "baseline": self.baseline,
            "events": self.events,
            "weights": {
                a: {str(k): v for k, v in row.items()}
                for a, row in self.weights.items()
            },
        }
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(
            prefix=self.state_path.name + ".",
            dir=str(self.state_path.parent),
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"))
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, self.state_path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    def _load(self) -> None:
        source = self.state_path if self.state_path.is_file() else self.bootstrap_path
        if source is None or not source.is_file():
            return
        try:
            raw = json.loads(source.read_text(encoding="utf-8"))
            if raw.get("schema") not in {
                "fmg.fca-vision-lite.v2.3",
                "fmg.fca-vision-lite.v2.4",
            }:
                return
            self.baseline = float(raw.get("baseline", 0.0))
            self.events = int(raw.get("events", 0))
            loaded = raw.get("weights") or {}
            for action in ACTIONS:
                self.weights[action] = {
                    int(k): max(-1.5, min(1.5, float(v)))
                    for k, v in (loaded.get(action) or {}).items()
                }
        except Exception:
            return


class LazyOrganRegistry:
    def __init__(self):
        self.factories: dict[str, Callable[[], Callable[..., OrganEvidence]]] = {}
        self.loaded: dict[str, Callable[..., OrganEvidence]] = {}

    def register(self, name: str, factory) -> None:
        self.factories[name] = factory

    def run(self, name: str, *args) -> OrganEvidence:
        if name not in self.loaded:
            self.loaded[name] = self.factories[name]()
        return self.loaded[name](*args)


def _prompt_intent(prompt: str) -> dict[str, bool]:
    p = prompt.casefold()
    has = lambda words: any(word in p for word in words)
    return {
        "face": has(("face", "portrait", "person", "woman", "man", "girl", "boy", "人物", "人間", "顔", "ポートレート")),
        "hand": has(("hand", "hands", "finger", "fingers", "手", "指")),
        "text": bool(re.search(r'["“”「」『』].+?["“”「」『』]', prompt))
        or has(("text", "lettering", "signboard", "caption", "文字", "看板", "ロゴ", "logo")),
    }


class FaceOrgan:
    def __init__(self):
        self.cv2 = None
        self.detector = None
        try:
            import cv2
            cascade = str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml")
            detector = cv2.CascadeClassifier(cascade)
            if not detector.empty():
                self.cv2 = cv2
                self.detector = detector
        except Exception:
            pass

    def __call__(self, image_path, relevant: bool) -> OrganEvidence:
        if not relevant:
            return OrganEvidence("face", self.detector is not None, False, 0.0, 1.0)
        if self.detector is None or self.cv2 is None:
            return OrganEvidence("face", False, True, 0.0, 0.0, detail="opencv unavailable")
        image = self.cv2.imread(str(image_path))
        if image is None:
            return OrganEvidence("face", True, True, 0.0, 0.0, detail="read failed")
        gray = self.cv2.cvtColor(image, self.cv2.COLOR_BGR2GRAY)
        faces = self.detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(64, 64))
        boxes = tuple((int(x), int(y), int(x+w), int(y+h)) for x, y, w, h in faces)
        return OrganEvidence(
            "face", True, True,
            0.72 if not boxes else 0.0,
            0.64 if not boxes else min(0.92, 0.70 + 0.05 * len(boxes)),
            boxes,
            f"faces={len(boxes)}",
        )


class HandOrgan:
    def __init__(self):
        self.mp = None
        self.hands = None
        try:
            import mediapipe as mp
            self.mp = mp
            self.hands = mp.solutions.hands.Hands(
                static_image_mode=True,
                max_num_hands=4,
                min_detection_confidence=0.45,
            )
        except Exception:
            pass

    def __call__(self, image_path, relevant: bool) -> OrganEvidence:
        if not relevant:
            return OrganEvidence("hand", self.hands is not None, False, 0.0, 1.0)
        if self.hands is None:
            return OrganEvidence("hand", False, True, 0.0, 0.0, detail="mediapipe unavailable")
        try:
            import numpy as np
            from PIL import Image
            image = Image.open(image_path).convert("RGB")
            result = self.hands.process(np.asarray(image))
            landmarks = result.multi_hand_landmarks or []
            boxes = []
            w, h = image.size
            for hand in landmarks:
                xs = [lm.x for lm in hand.landmark]
                ys = [lm.y for lm in hand.landmark]
                pad = 0.04
                boxes.append((
                    max(0, int((min(xs)-pad)*w)),
                    max(0, int((min(ys)-pad)*h)),
                    min(w, int((max(xs)+pad)*w)),
                    min(h, int((max(ys)+pad)*h)),
                ))
            return OrganEvidence(
                "hand", True, True,
                0.70 if not boxes else 0.0,
                0.62 if not boxes else 0.86,
                tuple(boxes),
                f"hands={len(boxes)}",
            )
        except Exception as exc:
            return OrganEvidence("hand", True, True, 0.0, 0.0, detail=str(exc))


class TextOrgan:
    def __init__(self):
        self.pytesseract = None
        try:
            import pytesseract
            _ = pytesseract.get_tesseract_version()
            self.pytesseract = pytesseract
        except Exception:
            pass

    def __call__(self, image_path, relevant: bool) -> OrganEvidence:
        if not relevant:
            return OrganEvidence("text", self.pytesseract is not None, False, 0.0, 1.0)
        if self.pytesseract is None:
            return OrganEvidence("text", False, True, 0.0, 0.0, detail="tesseract unavailable")
        try:
            from PIL import Image
            data = self.pytesseract.image_to_data(
                Image.open(image_path).convert("RGB"),
                output_type=self.pytesseract.Output.DICT,
            )
            boxes, words = [], []
            for i, raw in enumerate(data.get("text", [])):
                word = str(raw or "").strip()
                try:
                    conf = float(data["conf"][i])
                except Exception:
                    conf = -1.0
                if word and conf >= 25:
                    words.append(word)
                    x, y = int(data["left"][i]), int(data["top"][i])
                    w, h = int(data["width"][i]), int(data["height"][i])
                    boxes.append((x, y, x+w, y+h))
            return OrganEvidence(
                "text", True, True,
                0.74 if not words else 0.0,
                0.66 if not words else min(0.95, 0.68 + 0.02 * len(words)),
                tuple(boxes[:8]),
                "ocr=" + " ".join(words[:12]),
            )
        except Exception as exc:
            return OrganEvidence("text", True, True, 0.0, 0.0, detail=str(exc))


class FCAVisionLite:
    def __init__(self, state_path: str | Path):
        self.sensory = SensoryHash(128)
        self.kc = KenyonLayer()
        self.trace = TemporalTrace()
        bootstrap = Path(__file__).resolve().parent / "data" / "fca_vision_bootstrap_v24.json"
        self.policy = MBONPolicy(
            state_path,
            bootstrap_path=bootstrap,
        )
        self.organs = LazyOrganRegistry()
        self.organs.register("face", FaceOrgan)
        self.organs.register("hand", HandOrgan)
        self.organs.register("text", TextOrgan)

    def assess(self, *, prompt, image_path, technical_score, weak_regions):
        intent = _prompt_intent(prompt)
        evidence = (
            self.organs.run("face", image_path, intent["face"]),
            self.organs.run("hand", image_path, intent["hand"]),
            self.organs.run("text", image_path, intent["text"]),
        )
        useful = [e for e in evidence if e.relevant and e.available and e.confidence > 0]
        if useful:
            denom = sum(e.confidence for e in useful) or 1.0
            semantic_defect = max(0.0, min(1.0, sum(e.defect * e.confidence for e in useful) / denom))
            evidence_confidence = max(e.confidence for e in useful)
        else:
            semantic_defect = 0.0
            evidence_confidence = 0.0

        boxes = list(weak_regions)
        for e in evidence:
            if e.relevant and e.available and e.defect > 0:
                boxes.extend(e.boxes)
        boxes = list(dict.fromkeys(boxes))[:4]

        technical_defect = max(0.0, min(1.0, 1.0 - float(technical_score)))
        features = {
            "technical_defect": technical_defect,
            "semantic_defect": semantic_defect,
            "evidence_confidence": evidence_confidence,
            "has_weak_regions": 1.0 if weak_regions else 0.0,
            "intent_face": 1.0 if intent["face"] else 0.0,
            "intent_hand": 1.0 if intent["hand"] else 0.0,
            "intent_text": 1.0 if intent["text"] else 0.0,
        }
        observation = (
            f"prompt={prompt} technical={technical_score:.3f} "
            f"semantic_defect={semantic_defect:.3f} confidence={evidence_confidence:.3f}"
        )
        sensory = self.sensory.encode(observation, features)
        pattern = self.kc.activate(sensory, self.trace.state)
        self.trace.update(pattern)
        learned = self.policy.learned_scores(pattern)
        priors = {
            "accept": 0.34 + 0.72 * technical_score - 0.82 * semantic_defect,
            "local_repair": 0.10 + 0.76 * technical_defect
            + (0.24 if weak_regions else -0.18) + 0.18 * semantic_defect,
            "global_repair": -0.05 + 1.05 * semantic_defect * evidence_confidence
            + (0.10 if not boxes else 0.0),
        }
        scores = {a: priors[a] + 0.18 * learned[a] for a in ACTIONS}
        action = max(ACTIONS, key=lambda a: (scores[a], -ACTIONS.index(a)))
        return VisionDecision(
            action=action,
            scores=scores,
            kc_active=pattern.active,
            semantic_defect=semantic_defect,
            evidence_confidence=evidence_confidence,
            repair_boxes=tuple(boxes),
            evidence=evidence,
            prompt_intent=intent,
        ), pattern

    def learn(self, pattern: SparsePattern, action: str, reward: float) -> float:
        return self.policy.learn(pattern, action, reward)

    def status(self) -> dict[str, Any]:
        return {
            "version": VERSION,
            "sensory_channels": 128,
            "kc_units": 256,
            "kc_winners": 16,
            "fan_in": 6,
            "actions": list(ACTIONS),
            "lazy_organs": ["face", "hand", "text"],
            "loaded_organs": list(self.organs.loaded),
            "learning_events": self.policy.events,
        }
