#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
exec python3 run_fmg_eval_loop.py --split train --forever --publish-all --git-push-every 32 "$@"
