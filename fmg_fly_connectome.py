from __future__ import annotations

import hashlib
import json
import math
import os
import random
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


CONNECTOME_VERSION = "FMG-FLY-CONNECTOME-2.0"


@dataclass(frozen=True)
class RouteDecision:
    selected_backend: str
    selected_mbon: str
    pn_active: tuple[int, ...]
    kc_active: tuple[int, ...]
    kc_sparsity: float
    scores: dict[str, float]
    available_backends: tuple[str, ...]
    retry_order: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "connectome_version": CONNECTOME_VERSION,
            "selected_backend": self.selected_backend,
            "selected_mbon": self.selected_mbon,
            "pn_active": list(self.pn_active),
            "kc_active": list(self.kc_active),
            "kc_sparsity": round(self.kc_sparsity, 6),
            "scores": {k: round(v, 6) for k, v in self.scores.items()},
            "available_backends": list(self.available_backends),
            "retry_order": list(self.retry_order),
        }


class FlyConnectomeRouter:
    """Fly-connectome-inspired sparse controller for FMG.

    Architecture:
        prompt/request -> PN-like sensory channels
        -> KC-like sparse expansion
        -> MBON-like competing action populations
        -> selected lazy image organ
        -> DAN-like reward update

    This is an engineering abstraction inspired by fly mushroom-body routing.
    It is not a literal neuron-by-neuron reconstruction of a biological fly.
    """

    def __init__(
        self,
        state_path: str | Path,
        *,
        pn_count: int = 64,
        kc_count: int = 1024,
        kc_fan_in: int = 6,
        active_fraction: float = 0.05,
        seed: int = 8756,
        learning_rate: float = 0.035,
    ):
        self.state_path = Path(state_path)
        self.state_path.parent.mkdir(parents=True, exist_ok=True)

        self.pn_count = int(pn_count)
        self.kc_count = int(kc_count)
        self.kc_fan_in = int(kc_fan_in)
        self.active_fraction = float(active_fraction)
        self.seed = int(seed)
        self.learning_rate = float(learning_rate)
        self._lock = threading.Lock()

        if self.pn_count < 16:
            raise ValueError("pn_count too small")
        if self.kc_count < 128:
            raise ValueError("kc_count too small")
        if not 1 <= self.kc_fan_in <= self.pn_count:
            raise ValueError("invalid kc_fan_in")
        if not 0.005 <= self.active_fraction <= 0.25:
            raise ValueError("invalid active_fraction")

        rng = random.Random(self.seed)
        self.kc_inputs: tuple[tuple[int, ...], ...] = tuple(
            tuple(sorted(rng.sample(range(self.pn_count), self.kc_fan_in)))
            for _ in range(self.kc_count)
        )

        self.actions = ("a1111", "diffusers")
        self.weights: dict[str, list[float]] = {
            action: [0.0] * self.kc_count for action in self.actions
        }
        self.bias = {"a1111": 0.0, "diffusers": 0.0}
        self.events = 0
        self.rewards = {action: 0.0 for action in self.actions}
        self._load_state()

    @staticmethod
    def _norm_prompt(text: str) -> str:
        return " ".join(str(text or "").lower().split())

    def _hash_index(self, token: str) -> int:
        raw = hashlib.blake2b(
            token.encode("utf-8", "ignore"),
            digest_size=8,
            person=b"FMG-PN2",
        ).digest()
        return int.from_bytes(raw, "big") % self.pn_count

    def _sensory_pn(self, request: Mapping[str, Any]) -> list[float]:
        prompt = self._norm_prompt(str(request.get("prompt") or ""))
        vec = [0.0] * self.pn_count

        compact = "".join(ch for ch in prompt if not ch.isspace())
        for n, gain in ((2, 0.55), (3, 0.80), (4, 1.0)):
            if len(compact) < n:
                continue
            for i in range(len(compact) - n + 1):
                gram = compact[i : i + n]
                vec[self._hash_index(f"g{n}:{gram}")] += gain

        for token in prompt.split():
            vec[self._hash_index("w:" + token)] += 1.2

        width = max(1, int(request.get("width", 768)))
        height = max(1, int(request.get("height", 768)))
        steps = max(1, int(request.get("steps", 28)))
        guidance = float(request.get("guidance", 9.0))
        shape_tokens = [
            f"ratio:{'wide' if width > height else 'tall' if height > width else 'square'}",
            f"pixels:{min(7, (width * height) // (256 * 256))}",
            f"steps:{min(7, steps // 8)}",
            f"guidance:{min(7, int(guidance // 2))}",
        ]
        for token in shape_tokens:
            vec[self._hash_index("ctl:" + token)] += 2.0

        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def _kc_expand(self, pn: list[float]) -> tuple[list[float], tuple[int, ...]]:
        activations = []
        for inputs in self.kc_inputs:
            value = sum(pn[idx] for idx in inputs) / math.sqrt(len(inputs))
            activations.append(value)

        keep = max(1, int(round(self.kc_count * self.active_fraction)))
        ranked = sorted(
            range(self.kc_count),
            key=lambda i: (activations[i], -i),
            reverse=True,
        )
        active = tuple(sorted(ranked[:keep]))

        sparse = [0.0] * self.kc_count
        if active:
            peak = max(activations[i] for i in active) or 1.0
            floor = min(activations[i] for i in active)
            span = max(1e-9, peak - floor)
            for i in active:
                sparse[i] = 0.15 + 0.85 * ((activations[i] - floor) / span)
        return sparse, active

    def _score(
        self,
        sparse: list[float],
        availability: Mapping[str, bool],
        inhibited: set[str] | None = None,
    ) -> dict[str, float]:
        inhibited = inhibited or set()
        scores: dict[str, float] = {}
        active_total = sum(sparse) or 1.0
        for action in self.actions:
            if action in inhibited or not availability.get(action, False):
                scores[action] = float("-inf")
                continue
            syn = (
                sum(w * a for w, a in zip(self.weights[action], sparse))
                / active_total
            )
            innate = 0.035 if action == "a1111" else 0.0
            scores[action] = self.bias[action] + syn + innate
        return scores

    def route(
        self,
        request: Mapping[str, Any],
        availability: Mapping[str, bool],
        *,
        inhibited: set[str] | None = None,
    ) -> RouteDecision:
        pn = self._sensory_pn(request)
        sparse, active = self._kc_expand(pn)
        scores = self._score(sparse, availability, inhibited)

        finite = [
            (score, action)
            for action, score in scores.items()
            if math.isfinite(score)
        ]
        if not finite:
            raise RuntimeError("connectome found no available image organ")

        finite.sort(key=lambda x: (x[0], x[1]), reverse=True)
        selected = finite[0][1]
        retry = tuple(action for _, action in finite)
        pn_active = tuple(i for i, v in enumerate(pn) if v > 0.0)

        return RouteDecision(
            selected_backend=selected,
            selected_mbon=f"MBON::{selected}",
            pn_active=pn_active[:32],
            kc_active=active[:96],
            kc_sparsity=len(active) / self.kc_count,
            scores=scores,
            available_backends=tuple(
                a for a in self.actions if availability.get(a, False)
            ),
            retry_order=retry,
        )

    def reward(
        self,
        decision: RouteDecision,
        reward: float,
        *,
        structural_verified: bool,
        elapsed_s: float | None = None,
    ) -> None:
        value = max(-1.0, min(1.0, float(reward)))
        if structural_verified:
            value = min(1.0, value + 0.08)
        if elapsed_s is not None and elapsed_s > 0:
            value += max(
                -0.05,
                min(0.05, 0.02 * (15.0 - elapsed_s) / 15.0),
            )
            value = max(-1.0, min(1.0, value))

        with self._lock:
            action = decision.selected_backend
            if action not in self.weights:
                return
            active = decision.kc_active
            if active:
                delta = (
                    self.learning_rate
                    * value
                    / max(1.0, math.sqrt(len(active)))
                )
                for idx in active:
                    old = self.weights[action][idx]
                    self.weights[action][idx] = max(
                        -2.0, min(2.0, old + delta)
                    )
            self.bias[action] = max(
                -0.5,
                min(0.5, self.bias[action] + 0.004 * value),
            )
            self.events += 1
            self.rewards[action] += value
            self._save_state()

    def _state_payload(self) -> dict[str, Any]:
        return {
            "schema": "fmg.fly-connectome-state.v2",
            "connectome_version": CONNECTOME_VERSION,
            "seed": self.seed,
            "pn_count": self.pn_count,
            "kc_count": self.kc_count,
            "kc_fan_in": self.kc_fan_in,
            "active_fraction": self.active_fraction,
            "events": self.events,
            "bias": self.bias,
            "rewards": self.rewards,
            "weights": self.weights,
        }

    def _save_state(self) -> None:
        payload = self._state_payload()
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(
            prefix=self.state_path.name + ".",
            dir=str(self.state_path.parent),
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(
                    payload,
                    handle,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, self.state_path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    def _load_state(self) -> None:
        if not self.state_path.is_file():
            return
        try:
            raw = json.loads(self.state_path.read_text(encoding="utf-8"))
            if raw.get("schema") != "fmg.fly-connectome-state.v2":
                return
            if int(raw.get("seed")) != self.seed:
                return
            if (
                int(raw.get("pn_count")) != self.pn_count
                or int(raw.get("kc_count")) != self.kc_count
            ):
                return
            loaded = raw.get("weights") or {}
            for action in self.actions:
                row = loaded.get(action)
                if isinstance(row, list) and len(row) == self.kc_count:
                    self.weights[action] = [
                        max(-2.0, min(2.0, float(x)))
                        for x in row
                    ]
            for action in self.actions:
                self.bias[action] = float(
                    (raw.get("bias") or {}).get(
                        action,
                        self.bias[action],
                    )
                )
                self.rewards[action] = float(
                    (raw.get("rewards") or {}).get(action, 0.0)
                )
            self.events = int(raw.get("events", 0))
        except Exception:
            return

    def status(self) -> dict[str, Any]:
        return {
            "version": CONNECTOME_VERSION,
            "mode": "fly-connectome-inspired",
            "topology": {
                "sensory_layer": f"{self.pn_count} PN-like channels",
                "sparse_expansion": f"{self.kc_count} KC-like units",
                "fan_in": self.kc_fan_in,
                "target_sparsity": self.active_fraction,
                "motor_outputs": [f"MBON::{x}" for x in self.actions],
                "plasticity": (
                    "DAN-like reward modulation on KC->MBON weights"
                ),
            },
            "adaptation_events": self.events,
            "cumulative_reward": {
                k: round(v, 6)
                for k, v in self.rewards.items()
            },
            "state_path": str(self.state_path),
        }
