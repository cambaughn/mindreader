#!/usr/bin/env bash
# Copy the code, the FLARE weights and your personal dataset to a GPU pod.
# SSH key: ~/.ssh/runpod_ed25519 if present, else ssh's default; override with SSH_KEY=path.
# usage: scripts/remote/sync-up.sh root@HOST [PORT]      (RunPod: the "SSH over exposed TCP" host and port)
set -euo pipefail
HOST="${1:?usage: sync-up.sh root@HOST [PORT]}"; PORT="${2:-22}"
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
DATA="${MINDREADER_DATA:-$HOME/Library/Application Support/Mindreader/data}"
ENGINE="$REPO/third_party/silent-dictation"
KEY="${SSH_KEY:-$HOME/.ssh/runpod_ed25519}"; KEYOPT=""; [[ -f "$KEY" ]] && KEYOPT="-i $KEY"
SSH="ssh -p $PORT $KEYOPT -o StrictHostKeyChecking=accept-new"
R="/workspace/mindreader"

$SSH "$HOST" "mkdir -p $R/third_party/silent-dictation/models/flare_large /workspace/mindreader-data/personal"
echo "code"
rsync -az --delete -e "$SSH" \
  --exclude .git --exclude .venv --exclude models --exclude __pycache__ --exclude '*.pyc' \
  --exclude docs/results --exclude .pytest_cache \
  "$REPO/" "$HOST:$R/"
echo "FLARE weights + tokenizer (764 MB once; rsync resumes)"
rsync -az -e "$SSH" \
  "$ENGINE/models/flare_large/model.pt" "$HOST:$R/third_party/silent-dictation/models/flare_large/"
rsync -az -e "$SSH" \
  "$ENGINE/models/unigram5000.model" "$ENGINE/models/unigram5000_units.txt" \
  "$HOST:$R/third_party/silent-dictation/models/"
echo "dataset (clips + manifest + current personal model)"
rsync -az --delete -e "$SSH" --exclude history --exclude reports \
  "$DATA/personal/" "$HOST:/workspace/mindreader-data/personal/"
echo "up to date on $HOST"
