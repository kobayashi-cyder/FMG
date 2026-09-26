from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import fmg_image_generator as base
from fmg_image_generator_v23 import FMGImageGeneratorV23
from fmg_strict_objective import StrictObjectiveEvaluator

VERSION = "FMG-EVAL-LOOP-2.5"
DEFAULT_SUITE = "data/fmg_prompt_suite_v25.jsonl"
GB = 1000 ** 3


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class JsonlLog:
    def __init__(self, root: Path, mirror_path: Path | None = None):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "evaluations.jsonl"
        self.state_path = self.root / "log_usage_state.json"
        self.mirror_path = mirror_path
        self.total_bytes = 0
        self.events = 0
        self._load()

    def _load(self) -> None:
        if self.state_path.is_file():
            try:
                raw = json.loads(self.state_path.read_text(encoding="utf-8"))
                self.total_bytes = int(raw.get("total_log_bytes", 0))
                self.events = int(raw.get("events", 0))
            except Exception:
                pass

    def append(self, record: dict[str, Any]) -> None:
        payload = (json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
        with self.path.open("ab") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        self.total_bytes += len(payload)
        self.events += 1
        self._write_state()

    def _write_state(self) -> None:
        state = {
            "schema": "fmg.eval-log-usage.v1",
            "updated_at": utc_now(),
            "total_log_bytes": self.total_bytes,
            "total_log_gb_decimal": round(self.total_bytes / GB, 6),
            "events": self.events,
            "last_completed_50gb_band": self.total_bytes // (50 * GB),
            "next_threshold_gb": (self.total_bytes // (50 * GB) + 1) * 50,
            "note": "Counts evaluation JSONL/log payload bytes, not generated image bytes.",
        }
        text = json.dumps(state, ensure_ascii=False, indent=2) + "\n"
        self.state_path.write_text(text, encoding="utf-8")
        if self.mirror_path is not None:
            self.mirror_path.parent.mkdir(parents=True, exist_ok=True)
            self.mirror_path.write_text(text, encoding="utf-8")


class FailureMemory:
    def __init__(self, path: Path):
        self.path = path
        self.data: dict[str, Any] = {"schema": "fmg.failure-memory.v1", "categories": {}, "cases": {}}
        if path.is_file():
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    self.data = raw
            except Exception:
                pass

    def observe(self, case: dict[str, Any], report: dict[str, Any]) -> None:
        category = str(case.get("category", "unknown"))
        case_id = str(case.get("id", "unknown"))
        cat = self.data.setdefault("categories", {}).setdefault(
            category, {"pass": 0, "fail": 0, "unknown": 0, "reason_counts": {}}
        )
        verdict = str(report.get("verdict", "unknown"))
        cat[verdict] = int(cat.get(verdict, 0)) + 1
        for reason in report.get("reasons", []):
            key = str(reason).split("(", 1)[0].strip()
            rc = cat.setdefault("reason_counts", {})
            rc[key] = int(rc.get(key, 0)) + 1
        item = self.data.setdefault("cases", {}).setdefault(
            case_id, {"attempts": 0, "best_score": 0.0, "best_prompt": case.get("prompt", "")}
        )
        item["attempts"] = int(item.get("attempts", 0)) + 1
        score = float(report.get("score", 0.0))
        if score >= float(item.get("best_score", 0.0)):
            item["best_score"] = score
            item["best_prompt"] = case.get("_effective_prompt", case.get("prompt", ""))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


class GitPublisher:
    def __init__(self, repo_root: Path, tracked_root: Path, push_every: int):
        self.repo_root = repo_root
        self.tracked_root = tracked_root
        self.push_every = max(1, int(push_every))
        self.pending = 0

    def copy_image(self, source: str | Path, case: dict[str, Any], attempt: int, verdict: str) -> Path:
        src = Path(source)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        target = (
            self.tracked_root
            / str(case.get("split", "unknown"))
            / str(case.get("category", "unknown"))
            / str(case.get("id", "case"))
            / f"{stamp}_a{attempt}_{verdict}{src.suffix.lower()}"
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, target)
        self.pending += 1
        return target

    def write_record(self, target_image: Path, record: dict[str, Any]) -> None:
        meta = target_image.with_suffix(target_image.suffix + ".json")
        meta.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def maybe_push(self, force: bool = False) -> bool:
        if not force and self.pending < self.push_every:
            return False
        if not (self.repo_root / ".git").exists():
            return False
        rel = os.path.relpath(self.tracked_root, self.repo_root)
        subprocess.run(["git", "add", "--", rel], cwd=self.repo_root, check=True)
        usage = self.repo_root / "generated" / "fmg_eval" / "LOG_USAGE.json"
        if usage.exists():
            subprocess.run(
                ["git", "add", "--", os.path.relpath(usage, self.repo_root)],
                cwd=self.repo_root,
                check=True,
            )
        staged = subprocess.run(
            ["git", "diff", "--cached", "--quiet"],
            cwd=self.repo_root,
        )
        if staged.returncode == 0:
            self.pending = 0
            return False
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        subprocess.run(
            ["git", "commit", "-m", f"Add FMG evaluated image batch {stamp}"],
            cwd=self.repo_root,
            check=True,
        )
        subprocess.run(["git", "push", "origin", "HEAD"], cwd=self.repo_root, check=True)
        self.pending = 0
        return True


def load_suite(path: Path, split: str) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if split != "all" and row.get("split") != split:
            continue
        rows.append(row)
    return rows


def refine_prompt(prompt: str, report: dict[str, Any], exact_text: str | None = None) -> str:
    reasons = " | ".join(str(x) for x in report.get("reasons", [])).casefold()
    additions: list[str] = []
    if "alignment" in reasons:
        additions.append(
            "satisfy every requested subject, count, color, spatial relation, material and camera condition exactly"
        )
    if "technical score" in reasons:
        additions.append(
            "sharp focus, clean edges, controlled exposure, high local detail, no clipping"
        )
    if "face specialist" in reasons:
        additions.append("single unobscured face, both eyes visible, anatomically coherent features")
    if "hand specialist" in reasons:
        additions.append("hands fully visible, five fingers per hand, anatomically coherent joints")
    if exact_text and ("exact requested text" in reasons or "exact-text" in reasons):
        additions.append(
            f'render the literal text "{exact_text}" exactly, uppercase/lowercase and punctuation preserved, no other text'
        )
    if not additions:
        additions.append("literal prompt compliance, no substitutions, no omitted requested elements")
    return prompt.rstrip(" ,.") + ", " + ", ".join(dict.fromkeys(additions))


def build_request(case: dict[str, Any], prompt: str, backend: str) -> base.ImageRequest:
    return base.ImageRequest(
        prompt=prompt,
        negative_prompt=str(case.get("negative_prompt") or base.DEFAULT_NEGATIVE),
        steps=int(case.get("steps", base.DEFAULT_STEPS)),
        guidance=float(case.get("guidance", base.DEFAULT_GUIDANCE)),
        seed=-1,
        backend=backend,
    )


def run_once(
    generator: FMGImageGeneratorV23,
    evaluator: StrictObjectiveEvaluator,
    case: dict[str, Any],
    retries: int,
    logger: JsonlLog,
    memory: FailureMemory,
    publisher: GitPublisher | None,
    backend: str,
) -> dict[str, Any]:
    prompt = str(case["prompt"])
    max_attempts = 1 if case.get("split") == "holdout" else max(1, retries + 1)
    last: dict[str, Any] = {}

    for attempt in range(max_attempts):
        effective = dict(case)
        effective["_effective_prompt"] = prompt
        started = time.perf_counter()
        try:
            req = build_request(case, prompt, backend)
            result = generator.generate(req)
            report = evaluator.evaluate(result["path"], effective).as_dict()
            elapsed = round(time.perf_counter() - started, 3)
            record = {
                "schema": "fmg.eval-event.v2.5",
                "timestamp": utc_now(),
                "version": VERSION,
                "case_id": case.get("id"),
                "split": case.get("split"),
                "category": case.get("category"),
                "attempt": attempt,
                "original_prompt": case.get("prompt"),
                "effective_prompt": prompt,
                "backend": result.get("backend"),
                "seed": result.get("seed"),
                "generator_version": result.get("version"),
                "objective": report,
                "generation_elapsed_s": result.get("elapsed_s"),
                "total_elapsed_s": elapsed,
                "artifact_name": result.get("name"),
            }
            logger.append(record)
            memory.observe(effective, report)
            if publisher is not None:
                target = publisher.copy_image(result["path"], case, attempt, report["verdict"])
                record["published_artifact"] = str(target)
                publisher.write_record(target, record)
                publisher.maybe_push()
            last = record
            if report["verdict"] == "pass":
                return record
            if report["verdict"] == "unknown":
                # More diffusion generations cannot repair missing evaluator evidence.
                return record
            prompt = refine_prompt(prompt, report, case.get("exact_text"))
        except Exception as exc:
            record = {
                "schema": "fmg.eval-event.v2.5",
                "timestamp": utc_now(),
                "version": VERSION,
                "case_id": case.get("id"),
                "split": case.get("split"),
                "category": case.get("category"),
                "attempt": attempt,
                "original_prompt": case.get("prompt"),
                "effective_prompt": prompt,
                "error": str(exc),
                "objective": {"verdict": "error", "score": 0.0, "reasons": [str(exc)]},
            }
            logger.append(record)
            last = record
            return record
    return last


def main() -> None:
    ap = argparse.ArgumentParser(description="Run strict FMG prompt/objective evaluation loop")
    ap.add_argument("--suite", default=DEFAULT_SUITE)
    ap.add_argument("--split", choices=("train", "holdout", "all"), default="train")
    ap.add_argument("--backend", choices=("connectome", "a1111", "diffusers"), default="connectome")
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shuffle-seed", type=int, default=250927)
    ap.add_argument("--forever", action="store_true")
    ap.add_argument("--sleep", type=float, default=0.0)
    ap.add_argument("--publish-all", action="store_true")
    ap.add_argument("--git-push-every", type=int, default=32)
    ap.add_argument("--output", default="runtime/images")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parent
    suite = load_suite((repo_root / args.suite).resolve(), args.split)
    if not suite:
        raise SystemExit("prompt suite is empty")
    rng = random.Random(args.shuffle_seed)
    generator = FMGImageGeneratorV23(args.output)
    evaluator = StrictObjectiveEvaluator()
    runtime = repo_root / "runtime" / "fmg_eval"
    tracked = repo_root / "generated" / "fmg_eval"
    logger = JsonlLog(runtime / "logs", tracked / "LOG_USAGE.json")
    memory = FailureMemory(runtime / "failure_memory.json")
    publisher = (
        GitPublisher(repo_root, tracked / "images", args.git_push_every)
        if args.publish_all
        else None
    )

    processed = 0
    try:
        while True:
            order = list(suite)
            rng.shuffle(order)
            for case in order:
                run_once(
                    generator,
                    evaluator,
                    case,
                    args.retries,
                    logger,
                    memory,
                    publisher,
                    args.backend,
                )
                processed += 1
                if args.limit and processed >= args.limit:
                    if publisher is not None:
                        publisher.maybe_push(force=True)
                    return
                if args.sleep > 0:
                    time.sleep(args.sleep)
            if not args.forever:
                break
    finally:
        if publisher is not None:
            publisher.maybe_push(force=True)


if __name__ == "__main__":
    main()
