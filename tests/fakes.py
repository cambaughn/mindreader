"""Camera-free, model-free stand-ins for the engine's Frame / FaceObs / Recognizer."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Obs:
    mouth_open: float
    bbox: tuple = (100, 100, 300, 300)


@dataclass
class Frame:
    ts: float
    obs: Obs | None
    bgr: np.ndarray = field(default_factory=lambda: np.zeros((36, 64, 3), np.uint8))
    gray: np.ndarray = field(default_factory=lambda: np.zeros((36, 64), np.uint8))


def frames(seconds: float = 2.0, fps: int = 30, motion: float = 0.3, face: bool = True) -> list[Frame]:
    """A take whose mouth opens and closes with the given spread."""
    n = max(2, int(seconds * fps))
    out = []
    for i in range(n):
        t = i / fps
        mo = 0.1 + motion / 2 * (1 + np.sin(2 * np.pi * 2.5 * t))  # 2.5 syllables per second
        out.append(Frame(ts=t, obs=Obs(mouth_open=float(mo)) if face else None))
    return out


class Store:
    def __init__(self, lose_face: bool = False):
        self.lose_face = lose_face
        self.added: list[tuple[str, dict]] = []
        self.deleted: list[str] = []

    def crop(self, frames):
        if self.lose_face:
            raise ValueError("no face found in this clip")
        return np.zeros((len(frames), 96, 96), np.uint8)

    def add(self, clip, text, meta) -> str:
        self.added.append((text, dict(meta)))
        return f"item{len(self.added)}"

    def delete(self, item_id):
        self.deleted.append(item_id)


@dataclass
class Result:
    raw: str
    timings: dict = field(default_factory=dict)


class Recognizer:
    def __init__(self, reads: list[str]):
        self.reads = list(reads)
        self.calls = 0

    def recognize_clip(self, clip, rescore=True):
        self.calls += 1
        return Result(self.reads.pop(0) if self.reads else "")


class RawWriter:
    def __init__(self):
        self.written: list[str] = []

    def write(self, frames, item_id):
        self.written.append(item_id)
        return f"/raw/{item_id}.mp4"
