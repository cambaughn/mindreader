# Mindreader

A local macOS research tool for a single question:

> **Can the MacBook's camera read words I only mouth, well enough to type them for me?**

It is the camera-and-lips sibling of [subvox](https://github.com/joeblau/subvox), which asked the
same question of AirPods head-motion. Where subvox learns per-word motion templates from scratch,
Mindreader starts from a pretrained open-vocabulary lip-reading model and fine-tunes it on *your*
silent mouthing. The experiment is: how badly does a model trained on people talking out loud read
one person mouthing silently at a laptop, and how much does personal adaptation close the gap?

Mindreader is a **research project, not a product**. The running observations log in
[docs/learnings.md](docs/learnings.md) is a primary deliverable, on equal footing with the code.

> ⚠️ Experimental. Lip-reading is lossy (b/p/m look identical), and the models here are licensed for
> **non-commercial** use only. Use the camera only on yourself or with consent.

---

## What it does

You hold **Space**, mouth a sentence, release. Mindreader:

1. tracks your face (MediaPipe) and cuts a stabilised 96×96 mouth crop per frame at 25 fps,
2. runs a visual speech recognition model (FLARE-Large, or Auto-AVSR) to get the 10 best transcripts,
3. optionally re-ranks them with a small local language model (Qwen3-4B on MLX),
4. shows the text, the candidates and the timings.

In the **Record** tab the same gesture saves labelled takes of a prompted sentence under one of three
conditions, **silent / whisper / voiced** (subvox's protocol), plus the raw camera video. **Train**
fine-tunes a personal model on those takes and keeps it only if it beats the base model on held-out
sentences. **Evaluate** reports word error rate per condition against a chance baseline. **Dictate
anywhere** launches a global-hotkey mode (hold `fn`, text is pasted at the cursor in any app).

## The engine

All of the heavy lifting is [silent-dictation](https://github.com/KeWang0622/silent-dictation)
(Apache-2.0), vendored as a git submodule in `third_party/silent-dictation`. It provides the camera
pipeline, the FLARE port, the personal dataset format, the fine-tuner and the hotkey app. Mindreader
adds the experiment around it: conditions, raw-video retention, per-condition evaluation, a chance
baseline, condition-filtered training, and a plain window to do it all in without the CLI.

Models (downloaded by bootstrap, ~7 GB, kept in the submodule's `models/`):

| Model | Role | License |
|---|---|---|
| [FLARE-Large](https://huggingface.co/Anony94/FLARE-checkpoints) (Zhang et al., arXiv 2610.19559) | lip + face reader, 15.9% WER on LRS3 | CC BY-NC 4.0 |
| [Auto-AVSR](https://github.com/mpc001/auto_avsr) LRS3 VSR | fallback lip-only reader | non-commercial research |
| MediaPipe Face Landmarker | face tracking | Apache-2.0 |
| Qwen3-4B-Instruct 4-bit (MLX) | optional candidate re-ranking | Apache-2.0 |

A personal model fine-tuned from FLARE inherits CC BY-NC 4.0.

## Requirements

- Apple Silicon Mac, macOS 14 or later (developed on an M2 / 24 GB / macOS 26)
- Homebrew, ~9 GB free disk during setup (5.5 GB after)
- A webcam (the built-in FaceTime camera is fine; an iPhone works via Continuity Camera)

## Setup

```bash
git clone --recurse-submodules <repo> mindreader
cd mindreader
scripts/bootstrap.sh            # installs uv, a Python 3.12 env, the models; add --no-llm to skip the 2 GB LLM
```

## Launch

Run from the macOS **Terminal** app, because the camera permission is granted to the hosting app:

```bash
scripts/mindreader doctor --camera-test   # models, permissions, camera
scripts/mindreader                        # the window
```

Other commands: `scripts/mindreader counts | train [--condition silent] | eval [--llm] | dictate`.

## Protocol (the subvox analog)

1. **Decode tab first.** Hold Space, mouth anything, release. Does the base model read you at all?
2. **Record 60 silent sentences** (Record tab, condition *silent*). Every 5th take is held out.
3. **Train** on all takes (or silent-only). ~10 min per 60 sentences on an M5 Max; slower on an M2.
4. **Record the same 20 sentences voiced and whispered** so the conditions can be compared.
5. **Evaluate**: WER per condition, base vs personal, against chance (hypotheses shuffled across
   sentences). Results land in `docs/results/` and are summarised in `docs/learnings.md`.
6. **Dictate anywhere** to try it as a tool.

## Tests

No camera, models or network required (fake frames and a fake recognizer):

```bash
third_party/silent-dictation/.venv/bin/python -m pytest tests/
```

## What is stored

`~/Library/Application Support/Mindreader/data/`: the personal dataset (96×96 mouth and face crops
as `.npz` + `manifest.jsonl`, each take tagged with its condition), raw take videos in `raw/`, the
personal model, training and evaluation reports. Delete the folder to erase everything.

## Results

See [docs/learnings.md](docs/learnings.md) and [docs/results/](docs/results/).

## License

Mindreader's original source is MIT (see [LICENSE](LICENSE)). The engine is Apache-2.0; the model
weights keep their own (non-commercial) terms, listed above.
