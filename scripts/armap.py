#!/usr/bin/env python3
"""Bar-numbered arrangement map of a reference track.

Usage: armap.py <audio> [--bpm N] [--sections K] [--beats-per-bar 4] [--json OUT]
Prints: tempo, bar count, a section table (bars, start time, energy, brightness,
delta vs previous section) and a DAW cue list (bar.beat + name).
--json OUT also writes per-bar energy/brightness, the novelty curve, the bar-level
self-similarity matrix and the sections, for charts or further analysis.
"""
import argparse
import json
import sys
import warnings

import librosa
import numpy as np


def main():
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("--bpm", type=float, default=None, help="override detected tempo")
    ap.add_argument("--sections", type=int, default=0, help="force K sections (0 = auto)")
    ap.add_argument("--beats-per-bar", type=int, default=4)
    ap.add_argument("--json", default=None, help="also write analysis data to this file")
    a = ap.parse_args()

    y, sr = librosa.load(a.audio, sr=22050, mono=True)
    dur = len(y) / sr
    hop = 512

    if a.bpm:
        tempo = a.bpm
        _, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop, bpm=tempo, tightness=400)
    else:
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop)
        tempo = float(np.atleast_1d(tempo)[0])
    beat_t = librosa.frames_to_time(beats, sr=sr, hop_length=hop)
    if len(beat_t) < a.beats_per_bar * 4:
        sys.exit("too few beats detected; pass --bpm")

    # Bar grid from detected downbeat phase = phase with the strongest low-band onsets.
    onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop, fmax=200)
    strength = onset[np.clip(beats, 0, len(onset) - 1)]
    bpb = a.beats_per_bar
    phase = int(np.argmax([strength[p::bpb].mean() for p in range(bpb)]))
    # Regular grid from tempo + downbeat anchor: the beat tracker drops beats in
    # sparse passages (pads, no percussion), and counting raw beats would merge bars.
    period = bpb * 60.0 / tempo
    t0 = float(np.median((beat_t[phase::bpb] - beat_t[phase]) % period + beat_t[phase]) % period)
    if t0 > period - 0.05:
        t0 = 0.0
    bar_t = np.arange(t0, dur, period)
    if bar_t[0] > 0.05:
        bar_t = np.concatenate([[0.0], bar_t])  # pickup counts as bar 1
    bar_t = np.append(bar_t[bar_t < dur - 0.25 * period], dur)
    nbars = len(bar_t) - 1

    # Per-bar features: RMS energy (dB), spectral centroid (brightness), chroma+mfcc for similarity.
    rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    cen = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop)[0]
    mf = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop, n_mfcc=13)
    ch = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop)
    fr = librosa.time_to_frames(bar_t, sr=sr, hop_length=hop)
    feats, e_db, bright = [], [], []
    for i in range(nbars):
        s, t = fr[i], max(fr[i + 1], fr[i] + 1)
        feats.append(np.concatenate([mf[:, s:t].mean(1), ch[:, s:t].mean(1) * 10]))
        e_db.append(20 * np.log10(rms[s:t].mean() + 1e-9))
        bright.append(cen[s:t].mean())
    X = np.array(feats)
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)

    # Novelty on a bar-level self-similarity matrix (checkerboard kernel, 4-bar half width).
    S = X @ X.T / X.shape[1]
    w = 4
    k = np.kron(np.array([[1, -1], [-1, 1]]), np.ones((w, w)))
    Sp = np.pad(S, w, mode="edge")
    nov = np.array([(Sp[i:i + 2 * w, i:i + 2 * w] * k).sum() for i in range(nbars)])
    nov = np.maximum(nov, 0)

    if a.sections:
        cand = np.argsort(nov)[::-1]
        bounds = []
        for c in cand:
            if 4 <= c <= nbars - 4 and all(abs(c - b) >= 4 for b in bounds):
                bounds.append(int(c))
            if len(bounds) == a.sections - 1:
                break
    else:
        thr = nov.mean() + 0.5 * nov.std()
        bounds = [i for i in range(4, nbars - 3)
                  if nov[i] > thr and nov[i] == nov[max(0, i - 3):i + 4].max()]
    # Snap to phrase grid (multiples of 4 bars) when within one bar of it.
    snapped = []
    for b in sorted(bounds):
        r = round(b / 4) * 4
        b = r if abs(r - b) <= 1 else b
        if 0 < b < nbars and b not in snapped:
            snapped.append(b)
    edges = [0] + snapped + [nbars]

    e_db = np.array(e_db)
    peak = e_db.max()
    print(f"file   {a.audio}")
    print(f"tempo  {tempo:.1f} BPM   bars {nbars} ({bpb}/4)   length {int(dur // 60)}:{dur % 60:04.1f}")
    print()
    print(f"{'#':>2}  {'bars':>9}  {'len':>3}  {'start':>6}  {'energy':>7}  {'Δ':>5}  {'bright':>6}")
    prev = None
    rows = []
    for n in range(len(edges) - 1):
        s, t = edges[n], edges[n + 1]
        en = float(e_db[s:t].mean() - peak)
        br = float(np.mean(bright[s:t]))
        d = "" if prev is None else f"{en - prev:+.1f}"
        st = bar_t[s]
        print(f"{n+1:>2}  {s+1:>4}-{t:<4}  {t-s:>3}  {int(st//60)}:{st%60:04.1f}  {en:>6.1f}dB  {d:>5}  {br/1000:>5.1f}k")
        rows.append((s, en))
        prev = en
    print()
    print("cues (bar.beat → name; rename after listening)")
    for n, (s, en) in enumerate(rows):
        print(f"  {s+1}.1  S{n+1}")

    if a.json:
        out = {
            "tempo": round(float(tempo), 2), "beats_per_bar": bpb, "duration": round(dur, 2),
            "bar_starts": [round(float(x), 3) for x in bar_t[:-1]],
            "energy_db": [round(float(x - peak), 2) for x in e_db],
            "brightness_hz": [round(float(x)) for x in bright],
            "novelty": [round(float(x), 3) for x in nov],
            "ssm": [[round(float(v), 2) for v in row] for row in S],
            "sections": [{"first_bar": edges[n] + 1, "last_bar": edges[n + 1],
                          "energy_db": round(rows[n][1], 2)} for n in range(len(rows))],
        }
        with open(a.json, "w") as f:
            json.dump(out, f, separators=(",", ":"))


if __name__ == "__main__":
    main()
