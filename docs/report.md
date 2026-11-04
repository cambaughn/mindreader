# Silent speech from a laptop camera

*Two evenings in November 2026.*

I wanted to know whether the webcam on my MacBook could read sentences I mouth without making a
sound. [subvox](https://github.com/joeblau/subvox) tried the same idea with AirPods head motion. I
tried it with the camera, with the same end goal: hold a key, mouth a sentence, and the text shows
up wherever the cursor is.

Short answer: sort of. A public lip-reading model gets about half the words when I mouth everyday
sentences silently. After I fine-tuned it on 113 of my own silent takes (about seven minutes of
video) it gets 87% of the words on everyday sentences it had never seen, and about half of those
sentences come back perfect. But I have to mouth the words big. The smaller I move my mouth, the
worse it gets, and the thing I actually wanted, subvocal speech where your mouth barely moves, is
not something a camera can see. I don't think more training fixes that. The signal isn't in the
pixels.

## Why I tried it

subvox asked whether head motion carries enough signal to recognise mouthed words. Lip reading
from video already has real models behind it, so for the camera the question was different: does
a model trained on people talking out loud work on someone who is deliberately making no sound?

## Setup

- **Engine.** [silent-dictation](https://github.com/KeWang0622/silent-dictation), which wraps
  FLARE-Large, a 400M-parameter lip-reading model trained on thousands of hours of TED and YouTube
  speech, with face tracking, 96×96 lip and face crops, beam search, an optional local LLM
  re-ranker and a personal fine-tuner. I didn't rebuild any of that. I built the experiment around
  it: a recorder with subvox's three conditions (silent, whisper, voiced), held-out evaluation with
  a chance baseline, training filtered by condition, raw video archiving, and a way to train on a
  rented GPU.
- **Hardware.** MacBook Air M2 with 24 GB, the built-in FaceTime camera at 30 fps, a normally lit
  room. One rented RTX 4090 for the last training run.
- **Data.** 141 silently mouthed sentences and 20 of the same sentences spoken aloud, 2.6 to 5.6
  seconds each. The prompts are everyday dictation sentences ("can you send me the slides before
  the meeting") mixed with Harvard sentences, which are built to cover the sounds of English ("the
  hogs were fed chopped corn and garbage"). Every fifth take is held out and never trained on.
- **Metric.** Word error rate on the held-out takes, ignoring case, punctuation and whether a
  number is written as digits. 0% is perfect. Shuffling the model's outputs across sentences gives
  about 100%, so that's chance.

## 1. Silent mouthing doesn't look like speech with the sound off

Same 20 sentences, same camera, same evening, public model, no training:

| condition | word error rate | sentences exactly right |
|-----------|-----------------|-------------------------|
| spoken aloud | 25.5% | 4 of 20 |
| mouthed silently | 72.2% | 1 of 21 |

Spoken aloud, the model did about as well as its authors report, so the camera, the lighting and
the cropping were fine. Mouthed silently, it lost three words in four. The mistakes were a
different kind, too. Spoken, it mixed up words that look alike on the lips (build/bill,
parked/mark). Silent, whole phrases were gone. When you make no sound you move your mouth, jaw and
face differently from someone talking, and this model had never seen anyone do that.

## 2. It learns from a small amount of my own data

I fine-tuned on my silent takes as I recorded more, and scored each model on the same 28 held-out
sentences (10 everyday, 18 Harvard) that none of them had trained on:

| trained on | all 28 | everyday 10 | Harvard 18 | sentences exactly right |
|------------|--------|-------------|------------|-------------------------|
| nothing (public model) | 64.7% | 49.3% | 72.9% | 2 of 28 |
| 65 silent takes | 42.8% | 24.0% | 52.9% | 3 of 28 |
| 113 silent takes | **36.3%** | **13.3%** | 48.6% | 6 of 28 |

It kept improving as I added data. On the everyday sentences, 113 takes took word accuracy from
51% to 87%, and 5 of the 10 came back exactly right: "please write down the main action items",
"let's start the meeting without him", "can you book a table for four people". The misses were
the usual lip-reading mix-ups. "First draft" came out as "first graph", "read lips" as "ring
limbs".

The Harvard sentences stayed hard. They use rare words with no everyday phrasing to lean on ("the
colt reared and threw the tall rider"), and seven minutes of one person's video doesn't cover
them. More takes would keep helping here. The real limit is the next one.

## 3. Live, it's hit or miss

Those numbers come from takes where I read a sentence off the screen at a steady pace with my face
centred. Mouthing casually at the app, sentences come back perfect one moment and garbled the next.
Mouthing bigger helps a lot. Mouthing smaller makes it worse, and smaller is the direction I wanted
to go.

## What didn't help

- **Gentle fine-tuning.** The engine's default settings (a small learning rate, 10 epochs) couldn't
  even fit 17 training sentences. Three times the learning rate for 30 epochs fit them and improved
  held-out error by 22 points on the same data. Nothing was broken, it just needed to be pushed
  harder.
- **Training the whole network.** If silent mouthing is a different visual language, maybe the
  layers that interpret mouth shapes needed to change, not just the front end. On the 4090 the full
  fine-tune scored 38.1% against 36.3% for the light one. With this much data, adapting the front
  end already gets what there is to get.
- **An LLM clean-up pass.** The local 4B language model can re-rank the lip reader's ten guesses,
  which is worth a few points, and it can rewrite the winner into clean English with capitals and
  punctuation. On words, the rewrite was a wash. It fixed "read it first out loud" to "read a verse
  out loud" and broke "the build is failing" into "the bill is failing" about equally often, and
  when the input was bad it produced fluent nonsense. Giving it my personal vocabulary made things
  worse, because it started forcing those words into sentences. I left it in as an optional
  formatter. The language model can tidy up, but the lip reader has to get the words.

## Things I'd want to know before doing this again

- The M2 can do the light fine-tune (70 minutes for 81 takes) but swaps hopelessly on the full one.
  The 4090 did 30 epochs in 8 minutes.
- Moving files took longer than training. My Wi-Fi uplink measured tens of KB/s, and SSH into the
  pod crawled at 5 KB/s. RunPod's own relay transfer moved the same files at 1.6 MB/s. The pod
  should download the 4.8 GB model checkpoint itself (it did that at 48 MB/s) rather than get it
  from the laptop.
- Proper names aren't in the model's vocabulary and come out garbled. A custom prompt file with
  your own names is the lever there.
- FLARE's weights are licensed for non-commercial use. Anything beyond a demo would need its own
  model.

## Where I landed

The model reads lip shape, and lip shape is exactly what goes away as you mouth smaller. Real
subvocal speech is jaw, cheek and tongue muscle with almost no visible movement. A camera looking at
a 96-pixel crop of a mouth can't see a jaw tense. More data moves the curve for visible mouthing;
it can't recover a signal that never reached the sensor. subvox went to head motion for the same
reason, and the serious silent-speech work puts electrodes on the jaw and throat.

So a camera can read silent mouthing after a few minutes of your own data, as long as you mouth
visibly. For speech you can't see, it can't, and I don't think training changes that. I still think
it was worth doing. Now there's a number on where the line is.

## If I kept going

Another few hundred silent takes, deliberately including lazy, fast and casual mouthing, to measure
how small the mouthing can get before the curve stops moving. A held-out split that mixes sentence
types from the start (my first held-out set was accidentally all Harvard sentences, which made the
early numbers look worse than real use). The full fine-tune at a higher learning rate. Then the
dictation hotkey, which the engine already provides, with the LLM pass as a formatter on top of a
lip reader that is mostly right.

The bigger version, for visible mouthing, is a model trained on silent mouthing from many people,
so a new user needs a few minutes of enrollment instead of an evening of recording. Nobody has
built that. This experiment is the data pipeline for it, in miniature.

## Code

Code, protocol and the full lab notebook are at
[github.com/cambaughn/mindreader](https://github.com/cambaughn/mindreader). The recordings stay on
my laptop; nothing personal is in the repo. Built on
[silent-dictation](https://github.com/KeWang0622/silent-dictation) by Ke Wang.
