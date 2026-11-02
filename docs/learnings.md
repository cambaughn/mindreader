# Learnings

## Question

Can the MacBook's camera read words I only mouth, well enough to type them for me?

## Answer

_Pending._

## Log

### 2026-11-01 — Setup

- Chose to build on a pretrained open-vocabulary lip reader rather than per-word templates
  (subvox's approach), because one already existed end to end:
  [silent-dictation](https://github.com/KeWang0622/silent-dictation) ports FLARE-Large to plain
  PyTorch and already has calibration, fine-tuning and a hotkey app. Its author reports 32.7% WER
  on voiced VidTIMIT sentences and says silent mouthing is unmeasured. That gap is this project.
- Mindreader wraps the engine with subvox's three conditions (silent / whisper / voiced), keeps the raw
  video of every take, evaluates per condition against a chance baseline, and can fine-tune on one
  condition only.
- Hardware: Apple M2, 24 GB. The engine's author used an M5 Max; expect slower training here.

### Open questions to answer

1. Base-model WER on my silent mouthing vs voiced speech (same sentences).
2. How much 60 silent sentences of fine-tuning move that number.
3. Whether training on silent-only beats training on all conditions.
4. Whether the LLM re-ranker helps or hurts for silent input.
5. Decode latency on the M2 (the author: ~0.2 s on an M5 Max).

### 2026-11-01 — First run on the M2

- Bootstrap: ~4 GB of models on disk after conversion (FLARE 764 MB, Auto-AVSR 961 MB + LM 208 MB,
  Qwen3-4B 2.1 GB). `doctor` reports models, FLARE, LLM and MPS all OK.
- Headless timing with the base FLARE model: load 2.3 s, warm-up 1.9 s, decoding a 60-frame clip 0.4 s.
  Noise frames decode to "Thank you." — the decoder has a strong prior and will always say *something*,
  which is why the chance baseline in Evaluate matters.
- Engine bug found: `silentdict.permissions.ensure_camera_access` passes a lambda returning a value as
  the AVFoundation completion handler; PyObjC aborts the process ("expecting void return value") the
  moment the camera prompt resolves. The engine's own app avoids the path by asking asynchronously.
  Mindreader carries its own void-returning copy in `mindreader/ui/app.py`.
