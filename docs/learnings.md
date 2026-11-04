# Learnings

## Question

Can the MacBook's camera read words I only mouth, well enough to type them for me?

## Answer

_As of 2026-11-02:_ not out of the box, and the reason is silent mouthing itself. The public FLARE
model reads the same 20 sentences at 25% word error rate when I say them aloud and 72% when I only
mouth them, from the same camera on the same evening. Fine-tuning on 17 silent sentences brought
silent held-out error from 94% to 72% and showed the model can learn the silent dialect; the open
question is now how many sentences it takes.

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

### 2026-11-01 — First live decode (base model, no adaptation)

- FaceTime HD camera at 30 fps, face tracked, mouth centre locked. Decode tab, hold Space.
- **The sentence was mouthed silently, no voice.** So the first silent-speech result is a clean hit
  from the public model, before any personal training. The question is no longer "is there signal"
  but "how consistent is it across sentences, and how much does adaptation add".
- "This is an experiment to transcribe my voice." came back exactly right, top candidate at -0.09
  vs -0.49 for the runner-up. The rest of the n-best were lip-confusions (transcribe/describe, my/by,
  an/a), which is what real lip-reading looks like, not a language prior guessing.
- Earlier takes with a name in them fell apart ("my name is amir amir amir…", "kinbury I'm a kinf"):
  proper nouns are out of the model's vocabulary. Expect fine-tuning to help pronunciation habits,
  not names; a custom prompts file with the user's own names is the lever there.
- Timing on the M2: vsr 1.37 s, crop 0.02 s for a 118-frame (~4 s) take. Usable for dictation.

### 2026-11-01 — Base-model silent baseline (21 takes)

- Record tab, condition silent, 21 takes in one sitting (17 train / 4 held-out, every 5th). Each
  take 2.6–5.6 s, 65–137 frames, raw video kept. Nothing rejected by the gates.
- Base FLARE, no LLM, decoded headlessly from the saved clips (same crops the app's read-back sees):

  | set      | n  | WER   | exact | chance WER |
  |----------|----|-------|-------|------------|
  | all      | 21 | 72.2% | 1/21  | 101.8%     |
  | held-out | 4  | 93.8% | 0/4   | 102.1%     |

  Decoding 21 clips took 12.4 s on the M2.
- So the word-perfect first decode earlier tonight was the good tail, not the average. Everyday
  dictation prompts do better than the Harvard sentences ("please review the pull request when you
  have time" lost 3/9; "the hogs were fed chopped corn and garbage" lost 8/8). Likely a language-prior
  effect: the decoder's beam favours phrases that look like dictation.
- Error shapes: dropped first words (i/let's/can you/thanks → nothing) and lip-confusable swaps
  (truck→trunk, parked→martin, lemons→in). Opening words vanishing suggests the activity trim or the
  model's start is cutting the first syllable; worth checking with the raw video.
- Observation from recording: facial expression varies with the sentence, not just the lips. FLARE
  has a separate face stream (96×96 face crop, stored with every take), so the fine-tune can use it.
  Expression tied to meaning will not transfer to unseen sentences; expression tied to sounds might.
  The held-out WER after training is the test.

### 2026-11-02 — First personal fine-tune (silent only, 17 train / 4 held-out)

- `mindreader train --condition silent --strategy both` on the M2, app closed. 84 minutes wall clock.
  Light epochs took 36–58 s each. Full epochs ranged 70 s to 23 min with 19 min of system time
  over the run: the full fine-tune was swapping on 24 GB. Light is the only practical strategy here
  unless the frame budget comes down.
- Result on the 4 held-out sentences (32 words):

  | model                    | held-out WER |
  |--------------------------|--------------|
  | base                     | 93.8%        |
  | personal (light won)     | 90.6%        |
  | personal + LLM rescoring | 78.1%        |

  One word out of 32. Noise.
- The telling number is the training set: base 66.9% → personal 60.0% WER on the 17 sentences it
  trained on. Loss only fell 2.3 → 1.8 across 20 epochs. The model did not fit its own training data,
  so this is underfitting, not a generalisation failure. The light strategy (front-end, norms,
  biases, lr 1e-4) at 17 sentences × 10 epochs is too little signal to move a model this size.
- LLM rescoring was worth more than the fine-tune (90.6 → 78.1 on held-out). For silent input the
  n-best is full of near-misses, and a language prior picks among them better than the decoder's
  own scores do.
- Unchanged error shape after training: opening words still vanish ("let's grab lunch on friday" →
  "describe ninja frank"; "can you share the link" → "there's nothing"). Needs a look at the raw video
  to see whether mouthing starts before Space is pressed or the activity trim is eating the onset.
- Next experiments, cheapest first: (1) light at lr 3e-4 for 30 epochs, to see if the model can fit
  the training set at all; (2) the same 20 sentences voiced, to separate "silent mouthing is hard"
  from "my recording pipeline is off" (the engine's author saw 32.7% WER voiced); (3) more silent
  takes: the engine's own guidance is 60, and 17 is well under its stated minimum of 20.
- Checked the dropped-opening-words theory against the raw video (8 takes, MediaPipe over the full
  frames): face tracked 100% of frames, lips closed at frame 0, first lip movement 0.43–0.87 s into
  the recording, last movement 0.1–0.6 s before the end, activity trim keeps a margin either side.
  The capture is not clipping onsets. The missing first words are the model's confusion, not ours.
- Bug found and fixed on the way: `meta.raw` was null in every manifest row because the raw path
  was filled in after the item was saved. It now records the raw folder (file is `<id>.mp4`);
  `recorder.raw_path(item)` resolves it. The 21 existing rows were backfilled.

### 2026-11-02 — Second fine-tune: same data, 3× learning rate, 30 epochs

- `light`, lr 3e-4, 30 epochs, silent only, app closed. 21 minutes on the M2, 31–62 s per epoch,
  no swapping. Loss 2.32 → 1.25; best held-out epoch was 14–16.

  | model                               | train WER (17) | held-out WER (4) | exact (of 21) |
  |-------------------------------------|----------------|------------------|---------------|
  | base                                | 66.9%          | 93.8%            | 1             |
  | personal, engine defaults (1e-4×10) | 60.0%          | 90.6%            | 1             |
  | personal, 3e-4 × 30                 | 3.1%           | 71.9%            | 14            |

- The mechanism works once it is pushed: it memorised the training sentences and moved held-out by
  22 points. The remaining gap is data, not optimisation. Next lever is more silent sentences.
- Confound: all 4 held-out takes are Harvard sentences (every 5th take landed on one), the hardest
  kind, while training is mostly everyday dictation. Held-out 71.9% is pessimistic for real use.
  More takes will dilute this; stratifying the split would fix it properly.
- LLM rescoring hurt this time (71.9% → 78.1%) where it helped the weaker model. With 4 sentences
  that is one or two words either way; not a conclusion yet.
- Mindreader's defaults now follow this: light at lr 3e-4 for 30 epochs unless overridden.

### 2026-11-02 — Voiced vs silent: the gap is the mouthing, not the pipeline

- Same 20 sentences, condition voiced, recorded ~1 h after the silent set. 16 train / 4 held-out.
- Base FLARE, no LLM:

  | condition | n  | WER all | WER held-out | exact | chance |
  |-----------|----|---------|--------------|-------|--------|
  | voiced    | 20 | 25.5%   | 21.9%        | 4/20  | 99%    |
  | silent    | 21 | 72.2%   | 93.8%        | 1/21  | 102%   |

  Voiced 25.5% sits right where the engine's author measured it (32.7% on VidTIMIT), so camera,
  lighting, cropping and timing are fine. The 47-point gap is what silent mouthing costs a model
  that only ever saw people speaking.
- Per sentence, voiced beats silent on 19 of 20. The one exception ("I am running a little late
  today" → "I'm wearing legal entity" voiced) is a reminder that 4-second clips are noisy.
- Voiced errors are the familiar lip-reading confusions (build/bill, parked/mark, punch/bunch,
  share/sell): the model is close and a language model can fix them. Silent errors are whole-phrase
  misses, which a language model cannot.
- The silent-trained personal model on voiced takes: 24.2% all / 31.2% held-out, so adapting to
  silent did not break voiced reading (one or two words on 4 sentences either way).
- This answers open question 1. Silent mouthing is a different visual dialect, and the project's
  job is to teach it. The lever is silent data volume; the next recording session should be silent
  only and as large as patience allows.

### 2026-11-03 — Third fine-tune: 81 silent takes (65 train / 16 held-out)

- 60 more silent takes recorded in one sitting, with natural variation in voicing and facial
  emphasis ("all subvocal"). Same recipe as before (light, lr 3e-4, 30 epochs). 70 minutes on the
  M2, 100–180 s per epoch, Mac hot but usable. Still improving at epoch 30 (best 50.8% at epoch 28).
- Learning curve so far, held-out WER on the same 16 sentences:

  | trained on          | held-out WER | + LLM rescoring |
  |---------------------|--------------|-----------------|
  | nothing (base)      | 72.2%        |                 |
  | 17 silent takes     | 61.9%        |                 |
  | 65 silent takes     | 54.8%        | 49.2%           |

  Training-set WER 1.4%, 59/81 sentences exact overall, 0/16 held-out exact. The model memorises
  what it sees and generalises partially; the curve is still bending down with data.
- **Confound, now understood:** every one of the 16 held-out sentences is a Harvard sentence
  (hogs, chopped corn, colts, stockings). The engine's prompt list alternates 3 everyday dictation
  sentences then 2 Harvard, and every 5th take is held out, so the held-out slot always lands on
  Harvard. The model is graded only on the hardest category (rare words, no everyday phrasing for
  the decoder's language prior to lean on) while training is 60% everyday phrases. The held-out
  numbers above are a floor for real dictation, not an estimate of it. Fixed going forward by
  shuffling the prompt order per session (seeded), so future held-out takes are a fair mix.
  Existing splits stay as they are; mixing them now would leak training sentences into held-out.
- Error shape on held-out: function words and common nouns mostly survive ("the fish twisted and
  turned on the ___", "read a verse out loud for the pleasure"); rare content words go
  (salmon→examiner, cushion→tool chip, colt→golden). That is the language prior filling in, which is
  what LLM rescoring with personal context is for.
