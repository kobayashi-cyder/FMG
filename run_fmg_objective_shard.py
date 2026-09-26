from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import fmg_image_generator as base
from fmg_image_generator_v23 import FMGImageGeneratorV23
from fmg_strict_objective import StrictObjectiveEvaluator


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def load_checkpoint(path: Path) -> int:
    if not path.is_file():
        return 0
    try:
        return max(0, int(json.loads(path.read_text(encoding="utf-8")).get("next_index", 0)))
    except Exception:
        return 0


def save_checkpoint(path: Path, next_index: int, processed: int, shard: int, shards: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema": "fmg.objective-shard-checkpoint.v1",
                "updated_at": utc_now(),
                "shard": shard,
                "shards": shards,
                "next_index": next_index,
                "processed": processed,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def candidate_from_suite(row: dict, trial_id: int, shard: int) -> dict:
    return {
        "trial_id": trial_id,
        "shard": shard,
        "case_id": row.get("id", f"suite-{trial_id}"),
        "split": row.get("split", "train"),
        "category": row.get("category", "general"),
        "prompt": row.get("prompt", "high quality image"),
        "negative_prompt": row.get("negative_prompt") or base.DEFAULT_NEGATIVE,
        "steps": int(row.get("steps", base.DEFAULT_STEPS)),
        "guidance": float(row.get("guidance", base.DEFAULT_GUIDANCE)),
        "seed": int(row.get("seed", trial_id % (2**31 - 1))),
        "exact_text": row.get("exact_text"),
        "strategy": "suite-bootstrap",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Checkpointed FMG objective evaluator shard")
    ap.add_argument("--candidates", default="exchange/fmg_candidates.jsonl")
    ap.add_argument("--suite", default="data/fmg_prompt_suite_v25.jsonl")
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--shards", type=int, default=16)
    ap.add_argument("--limit", type=int, default=125000)
    ap.add_argument("--max-wall-seconds", type=int, default=20700)
    ap.add_argument("--backend", choices=("connectome", "a1111", "diffusers"), default="diffusers")
    ap.add_argument("--output-root", default="runtime/objective_2m")
    ap.add_argument("--steps-cap", type=int, default=50)
    args = ap.parse_args()

    if not (0 <= args.shard < args.shards):
        raise SystemExit("invalid shard")
    if args.limit < 1 or args.shards < 1:
        raise SystemExit("limit and shards must be positive")

    root = Path(args.output_root) / f"shard-{args.shard:02d}"
    root.mkdir(parents=True, exist_ok=True)
    checkpoint = root / "checkpoint.json"
    log_path = root / "feedback.jsonl"
    start_index = load_checkpoint(checkpoint)

    candidates = load_jsonl(Path(args.candidates))
    suite = load_jsonl(Path(args.suite))
    if not candidates and not suite:
        raise SystemExit("no candidates and no prompt suite")

    generator = FMGImageGeneratorV23(root / "images")
    evaluator = StrictObjectiveEvaluator()
    started_wall = time.monotonic()
    processed = 0
    local_index = start_index

    with log_path.open("a", encoding="utf-8") as log:
        while processed < args.limit and (time.monotonic() - started_wall) < args.max_wall_seconds:
            trial_id = args.shard + local_index * args.shards
            if candidates:
                source = candidates[trial_id % len(candidates)]
                cand = dict(source)
                cand.setdefault("trial_id", trial_id)
                cand.setdefault("shard", args.shard)
                cand.setdefault("case_id", cand.get("parent_case_id") or f"candidate-{trial_id}")
                cand.setdefault("split", "train")
                cand.setdefault("category", "objective-search")
                cand.setdefault("negative_prompt", base.DEFAULT_NEGATIVE)
                cand.setdefault("steps", base.DEFAULT_STEPS)
                cand.setdefault("guidance", base.DEFAULT_GUIDANCE)
                cand.setdefault("seed", trial_id % (2**31 - 1))
            else:
                cand = candidate_from_suite(suite[trial_id % len(suite)], trial_id, args.shard)

            request = base.ImageRequest(
                prompt=str(cand.get("prompt") or "high quality image"),
                negative_prompt=str(cand.get("negative_prompt") or base.DEFAULT_NEGATIVE),
                steps=max(1, min(args.steps_cap, int(cand.get("steps", base.DEFAULT_STEPS)))),
                guidance=float(cand.get("guidance", base.DEFAULT_GUIDANCE)),
                seed=int(cand.get("seed", -1)),
                backend=args.backend,
            )
            t0 = time.perf_counter()
            try:
                result = generator.generate(request)
                case = {
                    "id": cand.get("case_id"),
                    "prompt": request.prompt,
                    "split": cand.get("split", "train"),
                    "category": cand.get("category", "objective-search"),
                    "exact_text": cand.get("exact_text"),
                }
                objective = evaluator.evaluate(result["path"], case).as_dict()
                record = {
                    "schema": "fmg.objective-feedback.v1",
                    "timestamp": utc_now(),
                    "trial_id": trial_id,
                    "shard": args.shard,
                    "case_id": cand.get("case_id"),
                    "parent_case_id": cand.get("parent_case_id"),
                    "strategy": cand.get("strategy"),
                    "prompt": request.prompt,
                    "negative_prompt": request.negative_prompt,
                    "seed": result.get("seed", request.seed),
                    "steps": request.steps,
                    "guidance": request.guidance,
                    "backend": result.get("backend"),
                    "artifact_name": result.get("name"),
                    "generation_elapsed_s": result.get("elapsed_s"),
                    "total_elapsed_s": round(time.perf_counter() - t0, 3),
                    "objective": objective,
                }
            except Exception as exc:
                record = {
                    "schema": "fmg.objective-feedback.v1",
                    "timestamp": utc_now(),
                    "trial_id": trial_id,
                    "shard": args.shard,
                    "case_id": cand.get("case_id"),
                    "strategy": cand.get("strategy"),
                    "prompt": request.prompt,
                    "seed": request.seed,
                    "steps": request.steps,
                    "guidance": request.guidance,
                    "backend": args.backend,
                    "total_elapsed_s": round(time.perf_counter() - t0, 3),
                    "objective": {"verdict": "error", "score": 0.0, "reasons": [str(exc)]},
                    "error": str(exc),
                }

            log.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            log.flush()
            processed += 1
            local_index += 1
            save_checkpoint(checkpoint, local_index, processed, args.shard, args.shards)

    summary = {
        "schema": "fmg.objective-shard-summary.v1",
        "finished_at": utc_now(),
        "shard": args.shard,
        "shards": args.shards,
        "processed_this_run": processed,
        "next_index": local_index,
        "wall_seconds": round(time.monotonic() - started_wall, 3),
        "limit": args.limit,
        "max_wall_seconds": args.max_wall_seconds,
    }
    (root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
