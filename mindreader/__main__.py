"""python -m mindreader [app|doctor|train|eval|dictate|counts]"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

from . import __version__, engine


def cmd_app(args):
    from .ui.app import main
    return main(camera=args.camera)


def cmd_doctor(args):
    from silentdict.cli import main as cli_main
    sys.argv = ["silentdict", "doctor", *args.rest]
    return cli_main()


def cmd_train(args):
    from .evaluate import train
    strategies = ("light", "full") if args.strategy == "both" else (args.strategy,)
    extra = {k: v for k, v in (("epochs", args.epochs), ("lr", args.lr)) if v is not None}
    train(condition=args.condition, strategies=strategies, model=args.model,
          min_items=args.min_items, device=args.device, **extra)
    return 0


def cmd_eval(args):
    from .evaluate import evaluate
    evaluate(model=args.model, use_llm=args.llm, device=args.device)
    return 0


def cmd_counts(args):
    from .recorder import counts
    for cond, n in sorted(counts().items()):
        print(f"{cond:12s} train {n.get('train', 0):3d}  val {n.get('val', 0):3d}")
    return 0


def dictate_command() -> list[str]:
    return [str(engine.ENGINE / ".venv" / "bin" / "python"), "-m", "silentdict.mac"]


def dictate_env() -> dict:
    env = dict(os.environ)
    env["SILENTDICT_DATA"] = str(engine.DATA)
    env["PYTHONPATH"] = str(engine.ENGINE) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    return env


def cmd_dictate(args):
    """Phase 3, zero-code path: the engine's hotkey app (hold fn, mouth, release -> pasted at the cursor),
    pointed at Mindreader's data so it uses the personal model trained here."""
    print("Starting the dictation app with Mindreader's data folder. Hold fn to dictate into any app.")
    print("Needs Camera, Input Monitoring and Accessibility permissions for the launching app (Terminal).")
    return subprocess.call(dictate_command(), env=dictate_env(), cwd=str(engine.ENGINE))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="mindreader", description=__doc__)
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="cmd")
    a = sub.add_parser("app", help="the Mindreader window (default)")
    a.add_argument("--camera", default="0", help="camera index, or a video file to replay")
    d = sub.add_parser("doctor", help="engine self-check (models, permissions, camera)")
    d.add_argument("rest", nargs=argparse.REMAINDER)
    t = sub.add_parser("train", help="fine-tune the personal model on your takes")
    t.add_argument("--condition", choices=["silent", "whisper", "voiced"], default=None,
                   help="train/evaluate on one condition only (default: all takes)")
    t.add_argument("--strategy", choices=["light", "full", "both"], default="light",
                   help="light (default): front-end, norms and biases at lr 3e-4 for 30 epochs; full swaps on 24 GB")
    t.add_argument("--epochs", type=int, default=None)
    t.add_argument("--lr", type=float, default=None)
    t.add_argument("--model", choices=["auto", "flare", "auto_avsr"], default="auto")
    t.add_argument("--min-items", type=int, default=20)
    t.add_argument("--device", default="auto")
    e = sub.add_parser("eval", help="base vs personal WER per condition, with chance baseline")
    e.add_argument("--model", choices=["auto", "flare", "auto_avsr"], default="auto")
    e.add_argument("--llm", action="store_true", help="also rescore with the local language model")
    e.add_argument("--device", default="auto")
    sub.add_parser("counts", help="takes recorded per condition")
    sub.add_parser("dictate", help="hotkey dictation into any app (the engine's app, using your personal model)")
    raw = list(sys.argv[1:] if argv is None else argv)
    if raw[:1] == ["doctor"]:  # pass the engine's own flags straight through
        return cmd_doctor(argparse.Namespace(rest=raw[1:])) or 0
    args = p.parse_args(raw)
    cmd = args.cmd or "app"
    if cmd == "app" and args.cmd is None:
        args.camera = "0"
    return {"app": cmd_app, "doctor": cmd_doctor, "train": cmd_train, "eval": cmd_eval,
            "counts": cmd_counts, "dictate": cmd_dictate}[cmd](args) or 0


if __name__ == "__main__":
    sys.exit(main())
