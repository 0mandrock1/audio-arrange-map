# audio-arrange-map

A Claude Agent Skill that turns a reference track into a bar-numbered arrangement map — so you can block out your own track against an energy curve that already works.

```
tempo  120.0 BPM   bars 49 (4/4)   length 2:36.0

 #       bars  len   start   energy      Δ  bright
 1     1-16     16  0:00.0   -12.1dB           0.3k
 2    17-32     16  0:30.1    -0.9dB  +11.2    4.5k
 3    33-49     17  1:02.6    -7.6dB   -6.8    0.6k

cues (bar.beat → name; rename after listening)
  1.1  S1
  17.1  S2
  33.1  S3
```

## What it does

1. Detects tempo and beats, estimates the downbeat from low-band onsets, builds a bar grid.
2. Computes per-bar MFCC + chroma, a bar-level self-similarity matrix and a checkerboard novelty curve; boundaries snap to the 4-bar phrase grid.
3. Lays a regular bar grid from the tempo and the estimated downbeat (beat trackers drop beats in sparse passages, so raw beat counting would merge bars).
4. Reports each section's length in bars, start time, RMS energy relative to the loudest section, its delta vs the previous one, and spectral brightness.
5. The skill then reads the table (lift / breakdown / build-by-orchestration) and transfers the shape to a blank arrangement at your tempo.

`--json` additionally writes per-bar energy and brightness, the novelty curve, the bar-level self-similarity matrix and the sections — enough to draw your own charts.

## Install

- **Claude apps:** download `audio-arrange-map.skill` from [mandrock.me/skills](https://mandrock.me/skills/audio-arrange-map/audio-arrange-map.skill) and add it as a skill, or copy this folder into your skills directory.
- **Script alone:**

```sh
pip install librosa soundfile
python3 scripts/armap.py track.wav [--bpm 120] [--sections 6] [--json map.json]
```

## Limits

Energy is RMS of the full mix, so a heavily mastered reference shows small deltas. Section detection is timbre+harmony novelty: a key change with the same instrumentation can be missed, a fill can be caught. With no kick (pure orchestral), bar 1 may be off by a beat — check the first cue by ear, and pass `--bpm` when tempo detection halves or doubles.

## License

MIT
