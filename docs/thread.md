# Thread draft

Media: tweet 1 gets a short screen recording of the app reading a sentence you mouth silently
(personal model on). Tweet 4 gets the error-rate chart. Numbers are from docs/report.md.

---

**1/**
Spent two evenings picking up where @joeblau's subvox left off: can a MacBook webcam read words
you only mouth? With ~7 minutes of my own recordings it gets 87% of the words on everyday
sentences it had never seen. As long as I mouth them big. Thread.

**2/**
Starting point was a public lip-reading model (FLARE, trained on thousands of hours of TED and
YouTube speech). When I spoke out loud it got 3 of 4 words right. When I mouthed the same
sentences silently, same camera, same evening: 1 of 4.

**3/**
That gap is the interesting part. Silent mouthing isn't speech with the audio removed. When you're
deliberately making no sound you move your mouth and jaw differently, and no public model has seen
anyone do it.

**4/**
So I recorded my own. Trained on my silent takes, scored on held-out sentences it never saw:
0 takes: 49% of words wrong on everyday sentences
65 takes: 24%
113 takes: 13%
Still going down when I stopped.

**5/**
Half the held-out everyday sentences came back word for word: "please write down the main action
items", "let's start the meeting without him", "can you book a table for four people". The misses
look like lip reading: "first draft" came out as "first graph".

**6/**
Things that didn't help: training the whole network instead of just the front end (no better),
and an LLM clean-up pass (fixed as many sentences as it broke, and turned garbage into fluent
garbage). The LLM can tidy up. The lip reader has to get the words.

**7/**
The catch: those numbers are from takes where I read a sentence off the screen. Mouthing casually
at the app it's hit or miss, and mouthing bigger helps a lot. Mouthing smaller makes it worse.

**8/**
And smaller was the whole point. Real subvocal speech is jaw and cheek muscle with almost no
visible movement. A camera can't see that, and I don't think more training changes it. The signal
isn't in the pixels. That's why the serious attempts put electrodes on the jaw.

**9/**
Practical bits: an M2 does the light fine-tune in about an hour; a rented 4090 did it in 8 minutes.
Moving files took longer than training. Rare words are still rough: 7 minutes of one person's video
runs out at "the colt reared and threw the tall rider".

**10/**
Code, protocol and the full lab notebook, dead ends included:
github.com/cambaughn/mindreader
Built on @KeWang0622's silent-dictation engine.
