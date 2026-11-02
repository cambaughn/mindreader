"""Evaluation per speaking condition, a chance baseline, and condition-filtered training.

Wraps the engine's own machinery: `silentdict.train.backend_for` (FLARE or Auto-AVSR),
`decode_items` and `corpus_wer`. Adds what subvox's Evaluate screen adds: a per-condition
breakdown and an explicit "better than chance" number.
"""
from __future__ import annotations

import json
import random
import time
from collections import defaultdict

from . import engine
from .recorder import condition_of
from silentdict.text import corpus_wer, normalize


def chance_wer(pairs: list[tuple[str, str]], trials: int = 50, seed: int = 0) -> float | None:
    """WER when every hypothesis is assigned to a random other sentence (mean over shuffles).

    If the real WER is not clearly below this, the model is not actually reading these lips."""
    if len(pairs) < 2:
        return None
    rng = random.Random(seed)
    refs = [r for r, _ in pairs]
    hyps = [h for _, h in pairs]
    total = 0.0
    for _ in range(trials):
        order = list(range(len(hyps)))
        rng.shuffle(order)
        total += corpus_wer(zip(refs, (hyps[i] for i in order)))
    return total / trials


def exact_match(pairs) -> float:
    pairs = list(pairs)
    if not pairs:
        return 0.0
    return sum(normalize(r) == normalize(h) for r, h in pairs) / len(pairs)


def summarize(pairs_by_model: dict[str, list[tuple[str, str]]], conditions: list[str]) -> dict:
    """pairs_by_model: {"base": [(ref, hyp), ...], "personal": [...]} aligned with `conditions`."""
    report: dict = {"conditions": {}, "n": len(conditions)}
    groups: dict[str, list[int]] = defaultdict(list)
    for i, c in enumerate(conditions):
        groups[c].append(i)
    groups["all"] = list(range(len(conditions)))
    for cond, idx in groups.items():
        entry: dict = {"n": len(idx), "models": {}}
        for model, pairs in pairs_by_model.items():
            sub = [pairs[i] for i in idx]
            entry["models"][model] = {"wer": corpus_wer(sub), "exact": exact_match(sub), "chance": chance_wer(sub)}
        report["conditions"][cond] = entry
    return report


def format_table(report: dict) -> str:
    models = []
    for entry in report["conditions"].values():
        for m in entry["models"]:
            if m not in models:
                models.append(m)
    head = "| condition | n | " + " | ".join(f"{m} WER | {m} exact | {m} chance" for m in models) + " |"
    sep = "|" + "---|" * (2 + 3 * len(models))
    rows = [head, sep]
    order = [c for c in ("silent", "whisper", "voiced", "calibration", "other") if c in report["conditions"]]
    order += [c for c in report["conditions"] if c not in order and c != "all"] + ["all"]
    for cond in order:
        entry = report["conditions"][cond]
        cells = []
        for m in models:
            r = entry["models"].get(m)
            if r is None:
                cells += ["-", "-", "-"]
            else:
                cells += [f"{r['wer']:.1%}", f"{r['exact']:.0%}", "-" if r["chance"] is None else f"{r['chance']:.1%}"]
        rows.append(f"| {cond} | {entry['n']} | " + " | ".join(cells) + " |")
    return "\n".join(rows)


def evaluate(device: str = "auto", model: str | None = None, log=print, use_llm: bool = False,
             on_progress=None) -> dict:
    """Base vs personal WER on the held-out split, per condition, with the chance baseline.
    Writes docs/results/eval-<stamp>.md in the repo and eval-<stamp>.json in the data folder."""
    from silentdict import paths
    from silentdict.dataset import NotEnoughData, load_items
    from silentdict.train import backend_for, decode_items

    backend = backend_for(model)
    val = load_items("val")
    if not val:
        raise NotEnoughData("No held-out sentences yet. Record at least 5 takes first (every 5th is held out).")
    conditions = [condition_of(it) for it in val]
    log(f"Model: {backend.name}. Held-out clips: {len(val)} "
        f"({', '.join(f'{c}: {conditions.count(c)}' for c in dict.fromkeys(conditions))})")
    rescorer = None
    if use_llm and paths.LLM_DIR.exists():
        from silentdict.rewrite import Rescorer  # pragma: no cover
        rescorer = Rescorer()
    variants = [("base", False)]
    if backend.weights_path.exists():
        variants.append(("personal", True))
    pairs_by_model: dict[str, list[tuple[str, str]]] = {}
    for k, (name, personal) in enumerate(variants):
        t0 = time.time()
        reader = backend.reader(device, personal)
        pairs = decode_items(reader, val, rescorer=rescorer,
                             on_item=(lambda f, k=k: on_progress((k + f) / len(variants))) if on_progress else None)
        del reader
        label = name + ("+llm" if rescorer else "")
        pairs_by_model[label] = pairs
        log(f"{label:13s} WER {corpus_wer(pairs):.1%}  ({time.time() - t0:.0f}s)")
    report = summarize(pairs_by_model, conditions)
    report.update({"time": time.strftime("%Y-%m-%d %H:%M"), "model": backend.name,
                   "examples": [{"ref": r, **{m: pairs_by_model[m][i][1] for m in pairs_by_model}}
                                for i, (r, _) in enumerate(pairs_by_model[next(iter(pairs_by_model))])]})
    table = format_table(report)
    log("\n" + table)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    paths.REPORTS.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(paths.REPORTS / f"eval-{stamp}.json", "w"), indent=2)
    engine.RESULTS.mkdir(parents=True, exist_ok=True)
    md = engine.RESULTS / f"eval-{stamp}.md"
    examples = "\n".join(f"- **{e['ref']}**  \n  " + "  \n  ".join(f"{m}: {e[m]}" for m in pairs_by_model)
                         for e in report["examples"][:15])
    md.write_text(f"# Eval {report['time']} ({backend.name})\n\n{table}\n\n"
                  f"Chance = WER with hypotheses shuffled across sentences (mean of 50 shuffles).\n\n"
                  f"## Held-out examples\n\n{examples}\n")
    log(f"Wrote {md.relative_to(engine.REPO)}")
    report["markdown"] = str(md)
    return report


def train(condition: str | None = None, log=print, progress=None, **kwargs) -> dict:
    """The engine's personal fine-tune (`silentdict.train.run`), optionally restricted to takes of one
    condition (e.g. only your silent takes). Held-out clips are filtered the same way, so the reported
    WER is for that condition."""
    from silentdict import train as T

    original = T.load_items
    if condition:
        def filtered(split: str | None = None):
            return [it for it in original(split) if condition_of(it) == condition]
        T.load_items = filtered
        log(f"Training on condition={condition} only")
    try:
        return T.run(log=log, progress=progress, **kwargs)
    finally:
        T.load_items = original
