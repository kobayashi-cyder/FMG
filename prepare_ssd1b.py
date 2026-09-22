from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

MODEL_REPO = "segmind/SSD-1B"
FILENAME = "SSD-1B-A1111.safetensors"
EXPECTED_SHA256 = "1895a00bfc769a00b0c0c43a95e433e79e9db8a85402b45a33e8448785bde94d"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(8 * 1024 * 1024)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(
        description="Download and verify SSD-1B A1111 checkpoint"
    )
    ap.add_argument("--dir", default="models/SSD-1B")
    args = ap.parse_args()

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise SystemExit(
            "Install first: pip install huggingface_hub"
        ) from exc

    target_dir = Path(args.dir).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    downloaded = Path(
        hf_hub_download(
            repo_id=MODEL_REPO,
            filename=FILENAME,
            local_dir=str(target_dir),
        )
    )
    digest = sha256_file(downloaded)
    print("SHA256:", digest)
    if digest.lower() != EXPECTED_SHA256:
        raise SystemExit(
            "Checksum mismatch; refusing to mark the checkpoint ready."
        )
    print("[OK] Verified:", downloaded)


if __name__ == "__main__":
    main()
