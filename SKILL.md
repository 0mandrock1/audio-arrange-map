---
name: audio-arrange-map
description: "Turn a reference track (audio file) into a bar-numbered arrangement map: tempo, bar grid, section boundaries snapped to the 4-bar phrase grid, per-section energy and brightness deltas, and a cue-marker list for a DAW — so a new track can be blocked out against a proven energy curve. Runs scripts/armap.py (librosa). Trigger on: \"/audio-arrange-map\", \"розклади референс\", \"структура референсу\", \"арейнж-мапа\", \"по тактах розпиши трек\", \"де дропи і білди в цьому треку\", \"arrangement map\", \"map this reference\", \"how is this track arranged\", or when an audio reference is attached with a question about its structure. NOT trigger: mixing, device chains, sound design — answer directly or use a DAW helper skill; delay/reverb ms from BPM — bpm; stem separation or audio-to-MIDI — answer directly."
metadata:
  version: '1.0.0'
---

# audio-arrange-map

Runtime: any environment with Python 3.9+ and a shell. Stateless.

```sh
pip install librosa soundfile        # add --break-system-packages on a system Python
```

## Step 1 — Get the audio

An attached or local file (wav, flac, mp3, ogg). A streaming link only → say the skill needs the file; do not rip streams.

## Step 2 — Run

```sh
python3 scripts/armap.py <file> [--bpm N] [--sections K] [--beats-per-bar 4]
```

- Tempo off by a ratio (half/double), or a few BPM off on rubato material → rerun with `--bpm` from the user or the track's metadata. A wrong tempo shifts every bar number; check this first.
- Auto boundaries over-split ambient intros/outros → rerun with `--sections K` (K from a listen, usually 5–8).

## Step 3 — Read, don't just paste

Under the table, name each section by its energy/brightness move, not by genre cliché:
`Δ ≥ +6 dB` = lift/drop arrival; `Δ ≤ −6 dB` = breakdown; brightness rising with flat energy = build by orchestration (layering), not by loudness.

Then give the **transfer**: the same bar lengths and Δ shape as a blank arrangement for the user's track — section lengths in bars at their tempo, where the biggest lift lands as % of track length, and the length of the longest build in bars.

## Output

```
<armap table as printed>

read     S1 intro 16 → S2 lift +11 dB … (one line per section)
transfer <bars per section at target BPM> · peak at <N>% · longest build <N> bars
cues     <bar.beat list, ready to place as markers>
```

## Limits (say them when relevant)

Energy is RMS of the full mix — a mastered reference flattens dynamics, so a small Δ on a loud master still matters. Section detection is timbre+harmony novelty: a key change with the same instrumentation may be missed; a fill may be caught. The downbeat is estimated from low-band onsets; with no kick (pure orchestral) bar 1 can be off by a beat — check the first cue by ear.
