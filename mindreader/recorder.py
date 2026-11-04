"""Hold-Space recording sessions: a prompt list under one speaking condition, quality gates,
save to the engine's personal dataset (tagged with the condition), instant read-back.

The engine's CalibrationSession does the same for one condition ("calibration"). Mindreader's
version adds subvox's three conditions so the same sentences can be compared voiced vs whispered
vs silent, keeps the raw camera video of every take, and is UI-free and injectable for tests.
"""
from __future__ import annotations

import threading
import time
from collections import Counter
from pathlib import Path

from . import engine  # noqa: F401  (must come before any silentdict import)
from silentdict.activity import motion_score
from silentdict.text import align, sentence_case

CONDITIONS = ("silent", "whisper", "voiced")
SOURCE = "mindreader"
MIN_SECONDS = 0.8   # same gates as the engine's calibration
MIN_MOTION = 0.05   # spread of mouth opening; real mouthing is ~0.2-0.4


def gate(frames) -> str | None:
    """Why a take is unusable, or None when it passes the cheap checks (face loss is caught at crop time)."""
    if not frames or frames[-1].ts - frames[0].ts < MIN_SECONDS:
        return "Too short. Hold Space for the whole sentence."
    if motion_score([f.obs for f in frames]) < MIN_MOTION:
        return "Your lips barely moved. Mouth the words a little more clearly."
    return None


def condition_of(item) -> str:
    """Condition label of a dataset item; engine-made calibration takes count as their own condition."""
    return item.meta.get("condition") or ("calibration" if item.source == "calibration" else "other")


class DatasetStore:
    """Default store: the engine's personal dataset (96x96 lip + face crops, manifest.jsonl)."""

    def __init__(self):
        from silentdict.roi import MouthCropper
        self.cropper = MouthCropper()

    def crop(self, frames):
        from silentdict.roi import frames_to_clip
        return frames_to_clip(frames, self.cropper)   # raises ValueError when the face is lost

    def add(self, clip, text: str, meta: dict) -> str:
        from silentdict.dataset import add_item
        if meta.get("take") is None:
            meta["take"] = next_take(text, meta.get("condition", ""))
        return add_item(clip, text, SOURCE, meta=meta).id

    def delete(self, item_id: str):
        from silentdict.dataset import delete_item
        delete_item(item_id)


def raw_path(item) -> Path | None:
    """Full-frame mp4 of a recorded take, if it was kept (meta.raw names the folder, the file is <id>.mp4)."""
    folder = (getattr(item, "meta", None) or {}).get("raw")
    if not folder:
        return None
    p = Path(folder) / f"{item.id}.mp4"
    return p if p.exists() else None


class RawVideoWriter:
    """Keeps the full (640 px wide) camera frames of a take as mp4, so a future model with a
    different crop can reuse every recording. Off by default in tests; on by default in the app."""

    def __init__(self, folder: Path | None = None, enabled: bool = True):
        self.folder = Path(folder) if folder else engine.RAW
        self.enabled = enabled

    def write(self, frames, item_id: str) -> Path | None:
        if not self.enabled or not frames:
            return None
        import cv2
        self.folder.mkdir(parents=True, exist_ok=True)
        h, w = frames[0].bgr.shape[:2]
        span = frames[-1].ts - frames[0].ts
        fps = (len(frames) - 1) / span if span > 0 and len(frames) > 1 else 25.0
        path = self.folder / f"{item_id}.mp4"
        out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (w, h))
        try:
            for f in frames:
                out.write(f.bgr)
        finally:
            out.release()
        return path


class RecordSession:
    """One pass through `prompts` under one `condition`. `handle(frames)` runs on a worker thread."""

    def __init__(self, prompts: list[str], condition: str, store=None, recognizer=None,
                 raw: RawVideoWriter | None = None, camera: str = "", take: int | None = None):
        if condition not in CONDITIONS:
            raise ValueError(f"condition must be one of {CONDITIONS}, not {condition!r}")
        self.prompts = list(prompts)
        self.condition = condition
        self.store = store or DatasetStore()
        self.recognizer = recognizer
        self.raw = raw
        self.camera = camera
        self.take = take
        self.idx = 0
        self.busy = False
        self.saved: list[tuple[str, int]] = []
        self.scores: list[tuple[int, int]] = []
        self.status = "ready"       # ready | saving | saved | retry
        self.message = ""
        self.last: dict | None = None
        self.lock = threading.Lock()

    @property
    def done(self) -> bool:
        return self.idx >= len(self.prompts)

    @property
    def prompt(self) -> str | None:
        return None if self.done else self.prompts[self.idx]

    def accepting(self) -> bool:
        return not self.busy and not self.done

    def _retry(self, message: str) -> dict:
        self.status, self.message = "retry", message
        return self.snapshot()

    def handle(self, frames) -> dict:
        if self.done:
            return self.snapshot()
        self.busy, self.status = True, "saving"
        try:
            reason = gate(frames)
            if reason:
                return self._retry(reason)
            try:
                clip = self.store.crop(frames)
            except ValueError:
                return self._retry("Your face left the frame. Stay centred and try again.")
            prompt = self.prompts[self.idx]
            span = frames[-1].ts - frames[0].ts
            meta = {"condition": self.condition, "take": self.take, "camera": self.camera,
                    "seconds": round(span, 3), "fps": round((len(frames) - 1) / span, 2) if span > 0 else None,
                    "raw": str(self.raw.folder) if self.raw and self.raw.enabled else None}
            item_id = self.store.add(clip, prompt, meta)
            if self.raw:
                self.raw.write(frames, item_id)
            self.saved.append((item_id, self.idx))
            self.idx += 1
            self.status, self.message = "saved", "Saved"
        finally:
            self.busy = False
        self._read_back(clip, prompt)
        return self.snapshot()

    def _read_back(self, clip, prompt: str):
        if self.recognizer is None:
            return
        try:
            res = self.recognizer.recognize_clip(clip, rescore=False)
            read = res.raw.lower()
        except Exception as exc:  # a read-back failure must not lose the saved take
            self.last = {"prompt": prompt, "read": f"(read-back failed: {exc})", "align": [], "errors": 0, "words": 0}
            return
        ops = align(prompt, read)
        errors = sum(op["op"] != "ok" for op in ops)
        words = max(1, sum(op["op"] != "ins" for op in ops))
        self.scores.append((min(errors, words), words))
        self.last = {"prompt": prompt, "read": read, "align": ops, "errors": errors, "words": words,
                     "timings": getattr(res, "timings", {})}

    def redo(self):
        with self.lock:
            if self.saved and not self.busy:
                item_id, idx = self.saved.pop()
                self.store.delete(item_id)
                self.idx = idx
                if self.scores:
                    self.scores.pop()
                self.last = None
                self.status, self.message = "ready", "Removed. Record it again."

    def skip(self):
        if not self.done and not self.busy:
            self.idx += 1
            self.status, self.message = "ready", ""

    def accuracy(self) -> float | None:
        if not self.scores:
            return None
        e = sum(x for x, _ in self.scores)
        n = sum(y for _, y in self.scores)
        return max(0.0, 1 - e / n)

    @staticmethod
    def display(prompt: str | None) -> str | None:
        if prompt is None:
            return None
        return sentence_case(prompt) if prompt == prompt.lower() else prompt

    def snapshot(self) -> dict:
        return {
            "condition": self.condition, "idx": self.idx, "total": len(self.prompts),
            "prompt": self.display(self.prompt),
            "next": self.display(self.prompts[self.idx + 1]) if self.idx + 1 < len(self.prompts) else None,
            "status": self.status, "message": self.message, "busy": self.busy,
            "saved": len(self.saved), "done": self.done, "last": self.last, "accuracy": self.accuracy(),
        }


def load_prompts(count: int, skip_recorded: bool = True) -> list[str]:
    from silentdict.dataset import load_prompts as _lp
    return _lp(count, skip_recorded=skip_recorded)


def shuffled(prompts: list[str], seed: int) -> list[str]:
    """The engine's prompt list alternates 3 everyday sentences then 2 Harvard ones, and the engine
    holds out every 5th take, so in list order the held-out slot always lands on a Harvard sentence
    (learnings 2026-11-03). A seeded shuffle per session makes the held-out set a fair mix."""
    import random
    out = list(prompts)
    random.Random(seed).shuffle(out)
    return out


def prompts_for_condition(count: int, condition: str) -> list[str]:
    """Sentences to record next under `condition`.

    silent: fresh sentences (engine prompt lists, minus anything already recorded).
    whisper / voiced: the same sentences already recorded silently, so conditions are comparable;
    falls back to fresh sentences when nothing silent exists yet.
    """
    from silentdict.dataset import load_items
    if condition == "silent":
        return shuffled(load_prompts(count), seed=len(load_items()))
    done = {it.text for it in load_items() if condition_of(it) == condition}
    silent = []
    for it in sorted(load_items(), key=lambda i: i.created):
        if condition_of(it) == "silent" and it.text not in done and it.text not in silent:
            silent.append(it.text)
    return (silent or load_prompts(count))[:count]


def counts() -> dict[str, dict[str, int]]:
    """{condition: {"train": n, "val": n}} over the personal dataset."""
    from silentdict.dataset import load_items
    out: dict[str, Counter] = {}
    for it in load_items():
        out.setdefault(condition_of(it), Counter())[it.split] += 1
    return {k: dict(v) for k, v in out.items()}


def next_take(text: str, condition: str) -> int:
    from silentdict.dataset import load_items
    from silentdict.text import normalize
    t = normalize(text)
    return 1 + sum(1 for it in load_items() if it.text == t and condition_of(it) == condition)
