#!/usr/bin/env bash
# Bring the trained personal model and reports back from the pod into the local data folder.
# usage: scripts/remote/sync-down.sh root@HOST [PORT]
set -euo pipefail
HOST="${1:?usage: sync-down.sh root@HOST [PORT]}"; PORT="${2:-22}"
DATA="${MINDREADER_DATA:-$HOME/Library/Application Support/Mindreader/data}"
SSH="ssh -p $PORT -o StrictHostKeyChecking=accept-new"
P="$DATA/personal"
if [[ -f "$P/flare_personal.pth" ]]; then
  cp "$P/flare_personal.pth" "$P/flare_personal.pth.bak-$(date +%Y%m%d-%H%M%S)"
  echo "kept a copy of the previous personal model"
fi
rsync -az -e "$SSH" \
  --include 'flare_personal.pth' --include 'flare_decode.json' --include 'reports/***' --exclude '*' \
  "$HOST:/workspace/mindreader-data/personal/" "$P/"
echo "personal model and reports are in $P"
ls -la "$P/flare_personal.pth" "$P/reports" | tail -5
