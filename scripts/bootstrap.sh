#!/usr/bin/env bash
# One-time setup: engine submodule, uv, Python env, models (~7 GB download), Mindreader extras.
# Usage: scripts/bootstrap.sh [--no-llm]
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$PWD"
ENGINE="$ROOT/third_party/silent-dictation"

WITH_LLM=--with-llm
for arg in "$@"; do
  case "$arg" in
    --no-llm) WITH_LLM="" ;;
    *) echo "unknown option: $arg" >&2; exit 1 ;;
  esac
done

git submodule update --init --recursive
command -v uv >/dev/null || brew install uv

# The engine's own setup: uv venv (Python 3.12), pinned deps, checksum-verified model downloads.
# Model weights are non-commercial (FLARE: CC BY-NC 4.0; Auto-AVSR: research use); see README.
(cd "$ENGINE" && ./setup.sh $WITH_LLM --accept-model-license)

# Mindreader's extras go into the same environment.
uv pip install --python "$ENGINE/.venv/bin/python" -r "$ROOT/requirements.txt"

echo
echo "Done. Next (from the macOS Terminal app, so the camera permission attaches to Terminal):"
echo "  scripts/mindreader doctor      # models, permissions, camera"
echo "  scripts/mindreader             # the app"
