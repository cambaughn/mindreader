"""Wires the vendored silent-dictation engine into this process.

Import this module before any `silentdict` import. It puts the engine on sys.path and points the
engine's personal-data folder at Mindreader's own folder (the engine reads SILENTDICT_DATA once,
when `silentdict.paths` is first imported).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ENGINE = REPO / "third_party" / "silent-dictation"
SUPPORT = Path.home() / "Library" / "Application Support" / "Mindreader"
DATA = Path(os.environ.get("MINDREADER_DATA") or SUPPORT / "data")
RAW = DATA / "raw"          # full-frame mp4 of every recorded take (the engine keeps only 96x96 crops)
RESULTS = REPO / "docs" / "results"

os.environ.setdefault("SILENTDICT_DATA", str(DATA))
if str(ENGINE) not in sys.path:
    sys.path.insert(0, str(ENGINE))


def engine_present() -> bool:
    return (ENGINE / "silentdict" / "__init__.py").exists()


def checks() -> list[tuple[str, bool, str]]:
    """(name, ok, detail) for the status strip. Cheap: no model is loaded here."""
    out: list[tuple[str, bool, str]] = []
    out.append(("engine", engine_present(), str(ENGINE)))
    if not engine_present():
        return out
    from silentdict import paths  # noqa: E402  (after sys.path / env setup)
    out.append(("venv", (ENGINE / ".venv" / "bin" / "python").exists(), "third_party/silent-dictation/.venv"))
    out.append(("face model", paths.FACE_MODEL.exists(), paths.FACE_MODEL.name))
    out.append(("auto-avsr", (paths.VSR_DIR / "model.pth").exists(), "models/vsr_lrs3"))
    out.append(("flare", paths.FLARE_WEIGHTS.exists(), "models/flare_large/model.pt"))
    out.append(("llm", paths.LLM_DIR.exists(), "Qwen3-4B 4-bit (optional)"))
    out.append(("personal model", paths.PERSONAL_FLARE_WEIGHTS.exists(), "trained on your takes"))
    try:
        import torch
        mps = bool(torch.backends.mps.is_available())
        out.append(("apple gpu", mps, "torch mps"))
    except Exception as exc:  # torch missing or broken
        out.append(("apple gpu", False, f"torch: {exc}"))
    out.append(("data", True, str(DATA)))
    return out


def models_ready() -> bool:
    if not engine_present():
        return False
    from silentdict import paths
    return paths.FACE_MODEL.exists() and ((paths.VSR_DIR / "model.pth").exists() or paths.FLARE_WEIGHTS.exists())
