#!/usr/bin/env bash
# One shot: sync up, set up, train on the GPU, sync the model back.
# usage: scripts/remote/run.sh root@HOST [PORT] [-- train args]
#   e.g. scripts/remote/run.sh root@1.2.3.4 22222
#        scripts/remote/run.sh root@1.2.3.4 22222 -- --condition silent --strategy full --epochs 40
set -euo pipefail
HOST="${1:?usage: run.sh root@HOST [PORT] [-- train args]}"; shift
PORT=22
if [[ "${1:-}" =~ ^[0-9]+$ ]]; then PORT="$1"; shift; fi
[[ "${1:-}" == "--" ]] && shift
HERE="$(cd "$(dirname "$0")" && pwd)"
SSH="ssh -p $PORT -o StrictHostKeyChecking=accept-new"
"$HERE/sync-up.sh" "$HOST" "$PORT"
$SSH "$HOST" "bash /workspace/mindreader/scripts/remote/setup-pod.sh"
$SSH "$HOST" "bash /workspace/mindreader/scripts/remote/train-pod.sh $*"
"$HERE/sync-down.sh" "$HOST" "$PORT"
echo "done. Tick 'personal model' in the Decode tab, or run: scripts/mindreader eval"
