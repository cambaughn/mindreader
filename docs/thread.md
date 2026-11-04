# Thread draft

Suggested media: tweet 1 gets a 15-second screen recording of the Decode tab reading a silently
mouthed sentence (record it with the personal model ticked). Tweet 4 gets the learning-curve chart
from the shareable page. Numbers below are from docs/report.md.

---

**1/**
I spent two evenings teaching my MacBook's webcam to read words I only mouth. No sound, no mic.
After ~7 minutes of my own recordings it gets 87% of the words on everyday sentences it has never
seen. Live use is still hit or miss. Here's what I learned.

**2/**
Starting point: a public lip-reading model (FLARE, trained on thousands of hours of TED/YouTube
speech). Spoken aloud it gets 3 of 4 words right. Mouthed silently, same sentences, same camera,
same evening: 1 of 4.

**3/**
That gap is the finding. Silent mouthing isn't speech with the audio removed. You move your
mouth, jaw and face differently when you're deliberately making no sound, and no public model
has ever seen anyone do it.

**4/**
So I recorded my own. Fine-tuned on my silent takes, scored on held-out sentences it never saw:
0 takes → 49% word error on everyday sentences
65 takes → 24%
113 takes → 13%
Still bending down.

**5/**
Half the held-out everyday sentences came back word-perfect: "please write down the main action
items", "let's start the meeting without him", "can you book a table for four people". The misses
are the lip-reading classics: "first draft" → "first graph".

**6/**
What didn't help: unfreezing the whole network (no better than adapting the front end), and an
LLM clean-up pass (fixed as many sentences as it broke, and confidently rewrote garbage into
fluent garbage). The language model is the last mile. The lip reader has to deliver the words.

**7/**
Honest caveat: those numbers are on recorded takes, read from a prompt. Mouthing casually at the
screen it's still hit or miss, and exaggerating the mouthing helps a lot. The gap between a
measured 87% and a tool you'd trust is consistency, and training data that looks like real use.

**8/**
Practical bits: an M2 can do the light fine-tune in ~an hour; a rented 4090 did it in 8 minutes.
Moving files took longer than training. Rare vocabulary is still hard: 7 minutes of one person's
video runs out at "the colt reared and threw the tall rider".

**9/**
Everything's open: code, protocol, and the full lab notebook with the dead ends.
github.com/cambaughn/mindreader
Built on @KeWang0622's silent-dictation engine. Inspired by @joeblau's subvox.
