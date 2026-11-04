"""LLM clean-up: rewrite the lip reader's guesses into the sentence the person most likely said.

The engine's `Rescorer` can only choose among the lip reader's candidates. This step lets the local
language model repair them as well: fix look-alike words (pull/moon, parked/mark), restore phrases,
and add capitalisation and punctuation, the way a dictation app's "clean up" pass does. It may also
invent fluent nonsense, so it is measured on held-out takes like everything else (see evaluate.py).

Measured 2026-11-03 with Qwen3-4B (learnings): rewriting from the n-best list is neutral to slightly
positive (silent held-out 48.4% -> 46.8%, voiced 24.2% -> 23.6%); rewriting from the single best
guess is neutral or worse. Putting the personal context words in the prompt made it worse (the 4B
model forces them into sentences), so the rewrite prompt leaves them out by default and the
engine's Rescorer, which only chooses among candidates, remains the place for personal vocabulary.
"""
from __future__ import annotations

import re
import threading

from . import engine  # noqa: F401  (must come before any silentdict import)

SYSTEM = (
    "You clean up the output of a lip-reading system. The system watched a person silently mouth one "
    "short English sentence and produced its best guesses, most likely first. Guesses often contain "
    "wrong words that look the same on the lips (b/p/m, f/v, d/t/n/l, s/z, similar vowels), and "
    "sometimes a wrong phrase where the right one is a common expression. Reply with the single "
    "sentence the person most likely said, in plain English with normal capitalisation and "
    "punctuation. Keep words that are plausible; change only what is clearly wrong; keep the same "
    "length and meaning; never add information. Reply with the sentence only."
)


def load_context() -> list[str]:
    """Words and names the person uses often (data/personal/context.txt), as the engine reads them."""
    from silentdict import paths
    if not paths.PERSONAL_CONTEXT.exists():
        return []
    words = [w.strip() for w in paths.PERSONAL_CONTEXT.read_text().splitlines()]
    return [w for w in words if w and not w.startswith("#")][:100]


def build_messages(candidates: list[str], context: list[str] | None = None, mode: str = "nbest") -> list[dict]:
    """Chat messages for the rewrite. mode: "best" shows only the top guess, "nbest" the whole list."""
    from silentdict.text import sentence_case
    shown = candidates[:1] if mode == "best" else candidates[:8]
    lines = []
    if context:
        lines.append("Words and names this person uses often: " + ", ".join(context))
    if len(shown) == 1:
        lines.append("Lip reader's guess: " + sentence_case(shown[0]))
    else:
        lines.append("Lip reader's guesses, most likely first:")
        lines += [f"{i}. {sentence_case(c)}" for i, c in enumerate(shown, 1)]
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "\n".join(lines)}]


def first_sentence(reply: str) -> str:
    """Just the sentence: first non-empty line, quotes and list markers stripped."""
    for line in reply.strip().splitlines():
        line = line.strip().strip('"“”').strip()
        line = re.sub(r"^(\d+[.)]\s*|[-*]\s*)", "", line)
        if line:
            return line
    return ""


class Cleaner:
    """Rewrites candidates with the local MLX language model (shares weights with the engine's Rescorer)."""

    def __init__(self, rescorer=None, model_path=None, max_tokens: int = 60, use_context: bool = False):
        if rescorer is not None:
            self.model, self.tok = rescorer.model, rescorer.tok
        else:
            from mlx_lm import load
            from silentdict.rewrite import DEFAULT_LLM
            self.model, self.tok = load(str(model_path or DEFAULT_LLM))
        self.max_tokens = max_tokens
        self.lock = threading.Lock()
        self.context = load_context() if use_context else []

    def generate(self, messages: list[dict]) -> str:
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler
        prompt = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        with self.lock:
            return generate(self.model, self.tok, prompt=prompt, max_tokens=self.max_tokens,
                            sampler=make_sampler(temp=0.0), verbose=False)

    def rewrite(self, candidates: list[str], mode: str = "nbest") -> str:
        """The cleaned sentence; falls back to the top candidate if the model returns nothing."""
        candidates = [c for c in candidates if c]
        if not candidates:
            return ""
        reply = first_sentence(self.generate(build_messages(candidates, self.context, mode)))
        return reply or candidates[0]
