#!/usr/bin/env bash
set -u
cd "$(dirname "$0")"

stop=0
trap 'stop=1' INT TERM

while [ "$stop" -eq 0 ]; do
  echo "[FMG watchdog] starting strict evaluation loop..."
  ./RUN_FMG_EVAL_LOOP.sh "$@"
  rc=$?
  if [ "$stop" -ne 0 ]; then
    break
  fi
  echo "[FMG watchdog] evaluator exited with code $rc; restarting in 10 seconds..."
  sleep 10
done
