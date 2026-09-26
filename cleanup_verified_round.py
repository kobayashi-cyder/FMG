from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise SystemExit(f"verification file missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SystemExit(f"invalid verification file: {exc}") from exc
    if not isinstance(value, dict):
        raise SystemExit("verification payload must be an object")
    return value


def main() -> None:
    ap = argparse.ArgumentParser(description="Delete FMG round raw data only after verified improvement")
    ap.add_argument("--root", default="runtime/objective_2m")
    ap.add_argument("--verification", default="runtime/objective_2m/verified_improvement.json")
    ap.add_argument("--manifest", default="runtime/objective_2m/cleanup_manifest.json")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    verification_path = Path(args.verification).resolve()
    manifest_path = Path(args.manifest).resolve()
    verification = load_json(verification_path)

    if verification.get("verified") is not True:
        raise SystemExit("cleanup refused: improvement is not verified")
    if verification.get("status") not in {None, "improved", "verified"}:
        raise SystemExit(f"cleanup refused: unexpected verification status {verification.get('status')!r}")

    preserved_names = {
        verification_path.name,
        manifest_path.name,
        "round_summary.json",
        "representative_cases.jsonl",
        "final_tuning.json",
    }
    deleted: list[str] = []
    preserved: list[str] = []

    if not root.exists():
        raise SystemExit(f"round root missing: {root}")

    # Raw per-trial state is expendable only after a verified improvement.
    # Keep compact final evidence and configuration files.
    for path in sorted(root.rglob("*"), key=lambda p: len(p.parts), reverse=True):
        if path.is_file():
            if path.name in preserved_names:
                preserved.append(str(path.relative_to(root)))
                continue
            rel = str(path.relative_to(root))
            deleted.append(rel)
            if not args.dry_run:
                path.unlink(missing_ok=True)
        elif path.is_dir():
            if not args.dry_run:
                try:
                    path.rmdir()
                except OSError:
                    pass

    manifest = {
        "schema": "fmg.verified-cleanup.v1",
        "cleaned_at": utc_now(),
        "dry_run": bool(args.dry_run),
        "verification": verification,
        "deleted_files": deleted,
        "preserved_files": sorted(set(preserved_names) | set(preserved)),
        "deleted_count": len(deleted),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"deleted_count": len(deleted), "dry_run": args.dry_run}, ensure_ascii=False))


if __name__ == "__main__":
    main()
