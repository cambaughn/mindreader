# Can a laptop camera read words you only mouth?

*A two-day experiment in silent-speech dictation. November 2026.*

I spent two evenings finding out whether the webcam on a MacBook can transcribe sentences that I
mouth without making a sound, and how much of my own data it takes to make that work. Where it
landed: a public lip-reading model gets about half the words of silently mouthed everyday
sentences, because silent mouthing looks different from speech with the sound off and no model has
been trained on it. After fine-tuning on 113 sentences of my own silent mouthing (about seven
minutes of video), it gets 87% of the words on everyday sentences it has never seen, and reads half
of them perfectly. But only when I mouth visibly. In casual use it is hit or miss, and the smaller
the mouthing, the less there is to read. The thing I actually wanted, barely-moving subvocal
speech, is out of a camera's reach: that is a sensor limit, not a data limit. So: yes for visible
mouthing with a few minutes of your own data, no for the real thing, and worth publishing because
it draws that line with numbers.

## Why

[subvox](https://github.com/joeblau/subvox) asked whether AirPods head-motion carries enough
signal to recognise mouthed words. I wanted the same question for the camera, and the same
endgame: hold a key, mouth a sentence, have the text appear wherever the cursor is. Lip reading
from video has real models behind it, so the question was less "is there signal" and more "does a
model trained on people speaking aloud transfer to someone deliberately making no sound".

## Setup

- **Engine:** [silent-dictation](https://github.com/KeWang0622/silent-dictation), which wraps
  FLARE-Large (a 400M-parameter lip-reading model trained on thousands of hours of TED and YouTube
  speech) with MediaPipe face tracking, stabilised 96×96 lip and face crops, beam search, an
  optional local LLM re-ranker, and a personal fine-tuner. I built the experiment harness around
  it rather than rebuilding any of it: a recorder with subvox's three conditions (silent, whisper,
  voiced), held-out evaluation with a chance baseline, condition-filtered training, raw video
  archiving, and a RunPod path for training.
- **Hardware:** MacBook Air M2, 24 GB, FaceTime HD camera at 30 fps, in a normally lit room. One
  RunPod RTX 4090 for the last training run.
- **Data:** 141 silently mouthed sentences and 20 of the same sentences spoken aloud, 2.6 to 5.6
  seconds each. Prompts are the engine's everyday dictation sentences ("can you send me the slides
  before the meeting") interleaved with phonetically balanced Harvard sentences ("the hogs were
  fed chopped corn and garbage"). Every fifth take is held out permanently and never trained on.
- **Metric:** word error rate (WER) on the held-out takes, with case, punctuation and digits
  normalised. Lower is better; 0% is perfect; the chance baseline (hypotheses shuffled across
  sentences) sits around 100%.

## Finding 1: silent mouthing is not speech minus audio

Same 20 sentences, same camera, same evening, public model, no adaptation:

| condition | word error rate | word-perfect sentences |
|-----------|-----------------|------------------------|
| spoken aloud | 25.5% | 4 of 20 |
| mouthed silently | 72.2% | 1 of 21 |

Spoken aloud, the model lands where its authors measured it on public data, so the camera, the
lighting and the crop are fine. Mouthed silently, it loses three words in four. The errors differ
in kind, too: voiced mistakes are near-misses in the right slot (build/bill, parked/mark), silent
mistakes are whole phrases gone. When you deliberately make no sound you move your mouth, jaw and
face differently from someone talking, and a model that has only ever watched people talk has
never seen it.

## Finding 2: it learns the silent dialect from a small amount of your own data

I fine-tuned the model on my silent takes three times as the dataset grew, scoring every model on
the same 28 held-out sentences (10 everyday, 18 Harvard):

| trained on | all 28 | everyday 10 | Harvard 18 | word-perfect |
|------------|--------|-------------|------------|--------------|
| nothing (public model) | 64.7% | 49.3% | 72.9% | 2 of 28 |
| 65 silent takes | 42.8% | 24.0% | 52.9% | 3 of 28 |
| 113 silent takes | **36.3%** | **13.3%** | 48.6% | 6 of 28 |

The curve is still bending down. On everyday sentences, 113 takes took word accuracy from 51% to
87%, and 5 of the 10 held-out everyday sentences came back exactly right: "please write down the
main action items", "let's start the meeting without him", "can you book a table for four
people". The remaining misses are the lip-reading classics: "first draft" read as "first graph",
"read lips" as "ring limbs".

The Harvard sentences stay hard. Rare words with no everyday phrasing to lean on ("the colt
reared and threw the tall rider") are where seven minutes of one person's video runs out. More
takes would keep moving this line; the real ceiling is in the next finding.

## Finding 3: live use is still hit or miss

After all that, casual use at the Decode tab is still hit or miss. Sentences land word-perfect one
moment and fall apart the next, and exaggerating the mouthing helps noticeably: bigger, clearer lip
shapes give the model more to read. The held-out numbers above come from takes recorded while
reading a prompt at a steady pace with the face centred, which is a kinder setting than muttering a
thought at the screen. And the smaller you mouth, the worse it gets, which is the wrong direction
for the thing I wanted. That is the gap between a measured 87% and a tool you would trust, and the
next section is why I don't think more data closes it.

## What did not help

- **Timid fine-tuning.** The engine's default (small learning rate, 10 epochs) could not even
  fit 17 training sentences. Three times the learning rate for 30 epochs fit them and moved
  held-out error by 22 points on the same data. The mechanism was fine; it was under-driven.
- **Unfreezing the whole network.** The "silent is a different dialect" hypothesis suggested the
  layers that interpret mouth shapes need to move, not just the front end. On a 4090, the full
  fine-tune scored 38.1% against the light one's 36.3%. At this data size the front-end adaptation
  already captures what there is to learn.
- **An LLM clean-up pass.** The local 4B-parameter language model can re-rank the lip reader's
  ten candidates (worth a few points) and can rewrite them into clean English with capitals and
  punctuation. The rewrite was neutral on words: it fixed "read it first out loud" to "read a
  verse out loud" and broke "the build is failing" into "the bill is failing" in equal measure,
  and on bad input it produced fluent, confident nonsense. Giving it my personal vocabulary made
  it worse: it forced those words into sentences. It stays as an opt-in formatter. The language
  model is the last mile; the lip reader has to deliver the words.

## Practicalities worth knowing

- The M2 can do the light fine-tune (70 minutes for 81 takes) but swaps hopelessly on the full
  one. The 4090 did 30 epochs in 8 minutes.
- Moving files was slower than training. My Wi-Fi uplink measured tens of KB/s and SSH into the
  pod's exposed port crawled at 5 KB/s; RunPod's own relay transfer moved the same files at 1.6
  MB/s. The pod should download the 4.8 GB model checkpoint itself (48 MB/s) rather than receive
  it from the laptop.
- Proper names are out of the model's vocabulary and come out garbled; a custom prompt file with
  your own names is the lever there.
- FLARE's weights are licensed for non-commercial use. A product would need its own backbone.

## Where this leaves it

As a tool, not there, and I don't think video gets there. The model reads lip shape, and lip shape is
exactly what disappears as you mouth smaller. Real subvocal speech is jaw, cheek and tongue muscle
with almost no visible movement, and a camera looking at a 96-pixel mouth cannot see tension in a
jaw. More data moves the curve for visible mouthing; it cannot recover a signal that never reached
the sensor. That is a sensor problem, not a training problem, which is why subvox reached for
head motion and why the serious attempts at silent speech put electrodes on the jaw and throat.

So this is the camera's answer to subvox's question: yes for visible mouthing, with a few minutes of
your own data; no for the barely-moving kind you would actually want at a desk. Worth doing because
it draws that line with numbers.

## What I would do next

If I kept going: another few hundred silent takes, deliberately including lazy, fast and casual
mouthing, to measure how small the mouthing can get before the curve stops moving; a stratified
held-out split from the start (my first held-out set was accidentally all Harvard sentences,
which made early numbers look worse than real use); the full fine-tune at a higher learning rate;
and then the dictation hotkey, which the engine already provides, with the LLM pass restored as a
pure formatter on top of a lip reader that is mostly right.

The bigger idea, for visible mouthing, is the one nobody has built: a model trained on silent
mouthing from many people,
so that a new user needs a few minutes of enrollment instead of an evening of recording. The data
pipeline for that is exactly what this experiment produced in miniature.

## Reproduce it

Code, protocol and the full lab notebook are at
[github.com/cambaughn/mindreader](https://github.com/cambaughn/mindreader). Recordings stay on
the laptop; nothing personal is in the repo.
