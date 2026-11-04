#!/usr/bin/env bash
# Runs ON the pod (via run.sh). Fine-tunes on the synced dataset; default: silent takes, both strategies, 30 epochs.
set -euo pipefail
export MINDREADER_DATA=/workspace/mindreader-data SILENTDICT_DATA=/workspace/mindreader-data PYTHONUNBUFFERED=1
cd /workspace/mindreader
[[ $# -gt 0 ]] || set -- --condition silent --strategy both --epochs 30
mkdir -p "$MINDREADER_DATA/personal/reports"
LOG="$MINDREADER_DATA/personal/reports/train-pod-$(date +%Y%m%d-%H%M%S).log"
echo "mindreader train --device cuda $*" | tee "$LOG"
PYTHONPATH=third_party/silent-dictation python3 -m mindreader train --device cuda "$@" 2>&1 | tee -a "$LOG"
