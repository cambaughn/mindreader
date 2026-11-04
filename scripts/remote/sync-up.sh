#!/usr/bin/env bash
# Copy the code and your personal dataset to a GPU pod. The pod downloads the FLARE weights itself
# (setup-pod.sh): a home uplink can be 40 KB/s while the pod pulls at 50 MB/s (learnings 2026-11-03).
# SSH key: ~/.ssh/runpod_ed25519 if present, else ssh's default; override with SSH_KEY=path.
# usage: scripts/remote/sync-up.sh root@HOST [PORT]      (RunPod: the "SSH over exposed TCP" host and port)
set -euo pipefail
HOST="${1:?usage: sync-up.sh root@HOST [PORT]}"; PORT="${2:-22}"
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
DATA="${MINDREADER_DATA:-$HOME/Library/Application Support/Mindreader/data}"
KEY="${SSH_KEY:-$HOME/.ssh/runpod_ed25519}"; KEYOPT=""; [[ -f "$KEY" ]] && KEYOPT="-i $KEY"
SSH="ssh -p $PORT $KEYOPT -o StrictHostKeyChecking=accept-new"
R="/workspace/mindreader"

$SSH "$HOST" "mkdir -p $R /workspace/mindreader-data/personal"
echo "code"
rsync -az --delete -e "$SSH" \
  --exclude .git --exclude .venv --exclude models --exclude __pycache__ --exclude '*.pyc' \
  --exclude docs/results --exclude .pytest_cache \
  "$REPO/" "$HOST:$R/"
echo "dataset (clips + manifest; the pod does not need the local personal model)"
rsync -a --partial --timeout=120 -e "$SSH" --exclude history --exclude reports \
  --exclude 'flare_personal.pth*' --exclude flare_decode.json \
  "$DATA/personal/" "$HOST:/workspace/mindreader-data/personal/"
echo "up to date on $HOST"
