#!/usr/bin/env python3
"""V2.1 timestamps for one Ang (argv[1], default 1).

Method (each step answers a failure seen in review):
  1. Per-line CTC forced alignment from cached emissions, run TWICE per
     line with two windows:
       MID  - line span clamped to CTC neighbour midpoints (the regime
              where the aligner is trustworthy for normal lines);
       WIDE - floor = previous line's chosen V2 end, ceiling = next
              line's CTC start + 1.0s (rescues short lines whose CTC
              anchor was simply wrong: Japu CTC 30.84 vs true 29.05,
              Aad CTC start 31.66 vs true onset 30.86).
     Choice per line: WIDE only for <=4-word lines, or when MID leaves
     a >1.6s intra-line gap that WIDE does not; MID whenever WIDE's
     last word overruns the next line's CTC start by >0.2s, or WIDE
     leaves a >1.6s gap MID does not. (First wide-everywhere attempt
     compressed Kartarpurakh to end 8.59 leaving 2.9s of real speech
     unassigned, and stretched Neesaanu into the next line -- hence
     the guards.)
  2. Energy snap: Sehaj Paath audio is bimodal (silence RMS < ~100,
     speech > ~2000 at 10ms), so each start moves to the nearest
     silence->speech crossing within +/-0.30s, each end likewise.
  3. Burst-chain clamp: an end stretched >0.35s past the end of the
     burst chain containing its start (chain = bursts separated by
     <0.40s dips) is aligner stretch, not speech (Japu 31.70 -> 30.24).
  4. Min duration 0.14s; monotonic order enforced.
  5. Every word whose start or end moved >0.15s vs CTC (or whose CTC
     duration was <0.12s) is recorded in diagnostics.fixed_words --
     the Checker underlines exactly these in V2 mode.
"""
import array
import json
import math
import os
import sys
import wave

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import timestamper
import chunk_align
from ctc_align import FRAME_S, MODEL_PATH

BASE = os.path.dirname(os.path.abspath(__file__))
ANG = int(sys.argv[1]) if len(sys.argv) > 1 else 1
_npy = os.path.join(BASE, "output", f"ang{ANG}.ctc.emissions.npy")
EMISSIONS = _npy if os.path.exists(_npy) else os.path.join(BASE, "output", f"ang{ANG}.ctc.emissions.npz")
CTC_JSON = os.path.join(BASE, "app_data", f"ang{ANG}.json")
WAV = f"/tmp/ang{ANG}.wav"
OUT = os.path.join(BASE, "output", f"ang{ANG}.v2.json")
TEXT = os.path.join(BASE, "text", f"ang{ANG}.txt")

THRESH = 300.0
FRAME_E = 0.01
MIN_DUR = 0.14
SNAP_WIN = 0.30


def load_energy():
    w = wave.open(WAV)
    n = w.getnframes()
    a = array.array("h")
    a.frombytes(w.readframes(n))
    step = 160
    rms = []
    for i in range(0, len(a) - step, step):
        s = a[i:i + step]
        rms.append(math.sqrt(sum(x * x for x in s) / len(s)))
    return rms


def find_crossing(rms, t, rising, win=SNAP_WIN):
    lo = max(1, int((t - win) / FRAME_E))
    hi = min(len(rms) - 6, int((t + win) / FRAME_E))
    best, bestd = None, 1e9
    for i in range(lo, hi):
        if rising:
            hit = rms[i - 1] < THRESH <= rms[i] and \
                sum(1 for v in rms[i:i + 5] if v >= THRESH) >= 4
        else:
            hit = rms[i - 1] >= THRESH > rms[i] and \
                sum(1 for v in rms[i:i + 5] if v < THRESH) >= 4
        if hit:
            d = abs(i * FRAME_E - t)
            if d < bestd:
                best, bestd = i * FRAME_E, d
    return best


def main():
    chunk_align.VOCAB = json.load(open(
        os.path.join(MODEL_PATH, "vocab.json"), encoding="utf-8"))
    chunk_align.IDX2TOK = {v: k for k, v in chunk_align.VOCAB.items()}
    from ctc_align import BLANK_ID
    chunk_align.IDX2TOK[BLANK_ID] = "<blank>"

    _raw = np.load(EMISSIONS)
    emissions = (_raw["emissions"] if hasattr(_raw, "files") else _raw).astype(np.float32)
    ctc = json.load(open(CTC_JSON, encoding="utf-8"))
    lines = timestamper.load_lines(TEXT)
    ref = []
    for li, line in enumerate(lines):
        for tok in line.split():
            nw = timestamper.norm(tok)
            if nw:
                ref.append((tok, nw, li))
    line_words = {}
    for i, (_, _, li) in enumerate(ref):
        line_words.setdefault(li, []).append(i)

    rms = load_energy()
    segs, inseg, st = [], False, 0
    for i, v in enumerate(rms):
        sp = v >= THRESH
        if sp and not inseg:
            st = i; inseg = True
        if inseg and not sp and all(x < THRESH for x in rms[i:i + 8]):
            segs.append([st * FRAME_E, i * FRAME_E]); inseg = False
    if inseg:
        segs.append([st * FRAME_E, len(rms) * FRAME_E])
    bursts = []
    for sg in segs:
        if bursts and sg[0] - bursts[-1][1] < 0.20:
            bursts[-1][1] = sg[1]
        else:
            bursts.append(list(sg))

    def chain_end(t):
        for i, (bs, be) in enumerate(bursts):
            if bs - 0.10 <= t <= be + 0.05:
                end, j = be, i
                while j + 1 < len(bursts) and \
                        bursts[j + 1][0] - bursts[j][1] < 0.40:
                    j += 1; end = bursts[j][1]
                return end
            if bs > t + 0.5:
                break
        return None

    def align_line(li, win_lo_t, win_hi_t):
        idxs = line_words[li]
        w0, w1 = idxs[0], idxs[-1] + 1
        old = ctc["lines"][li]
        try:
            frames, *_ = chunk_align.align_words(
                emissions, ref, w0, w1,
                int(win_lo_t / FRAME_S), int(win_hi_t / FRAME_S))
        except Exception:
            return None
        words = []
        for j, gi in enumerate(idxs):
            tok = ref[gi][0]
            if gi in frames:
                s, e = frames[gi]
                words.append({"word": tok, "start": round(s * FRAME_S, 2),
                              "end": round((e + 1) * FRAME_S, 2)})
            else:
                ow = old["words"][j]
                words.append({"word": tok, "start": ow["start"],
                              "end": ow["end"]})
        return words

    def max_gap(words):
        return max((words[i + 1]["start"] - words[i]["end"]
                    for i in range(len(words) - 1)), default=0.0)

    out_lines = []
    prev_end_v2 = 0.0
    fixed_words = []
    stats = {"lines_mid": 0, "lines_wide": 0, "lines_fallback": 0,
             "words_snapped": 0, "short_before": 0, "short_after": 0}
    for li, line_text in enumerate(lines):
        old = ctc["lines"][li]
        idxs = line_words[li]
        prev_ctc_end = ctc["lines"][li - 1]["end"] if li > 0 else 0.0
        next_ctc_start = (ctc["lines"][li + 1]["start"]
                          if li + 1 < len(lines) else ctc["duration"])
        next_first_word = (ctc["lines"][li + 1]["words"][0]["start"]
                           if li + 1 < len(lines) else ctc["duration"])
        mid_lo = max(0.0, (prev_ctc_end + old["start"]) / 2 - 0.25) \
            if li > 0 else max(0.0, old["start"] - 1.2)
        mid_hi = min(ctc["duration"],
                     (old["end"] + next_ctc_start) / 2 + 0.25) \
            if li + 1 < len(lines) else min(ctc["duration"],
                                            old["end"] + 1.2)
        wide_lo = max(0.0, prev_end_v2 - 0.25) if li > 0 \
            else max(0.0, old["start"] - 1.2)
        wide_hi = min(ctc["duration"], next_ctc_start + 1.0) \
            if li + 1 < len(lines) else min(ctc["duration"],
                                            old["end"] + 1.2)
        mid = align_line(li, mid_lo, mid_hi)
        wide = align_line(li, wide_lo, wide_hi)
        if mid is None and wide is None:
            words = [dict(w) for w in old["words"]]
            stats["lines_fallback"] += 1
            choice = "fallback"
        elif wide is None:
            words, choice = mid, "mid"
        elif mid is None:
            words, choice = wide, "wide"
        else:
            wide_overruns = wide[-1]["end"] > next_first_word + 0.2
            if len(idxs) <= 4:
                words, choice = wide, "wide"
            elif wide_overruns and not (mid[-1]["end"] > next_first_word + 0.2):
                words, choice = mid, "mid"
            elif max_gap(mid) > 1.6 and max_gap(wide) <= 1.6:
                words, choice = wide, "wide"
            elif max_gap(wide) > 1.6 and max_gap(mid) <= 1.6:
                words, choice = mid, "mid"
            else:
                words, choice = mid, "mid"
        stats["lines_wide" if choice == "wide" else
              "lines_mid" if choice == "mid" else "lines_fallback"] += \
            1 if choice != "fallback" else 0
        if choice == "wide":
            print(f"line {li+1}: WIDE ({len(idxs)} words)")
        # --- snap + clamp + floors ---
        for j, w in enumerate(words):
            ow = old["words"][j]
            if ow["end"] - ow["start"] < 0.12:
                stats["short_before"] += 1
            s0, e0 = w["start"], w["end"]
            on = find_crossing(rms, s0, True)
            off = find_crossing(rms, e0, False)
            s1 = on if on is not None else s0
            e1 = off if off is not None else e0
            if on is not None or off is not None:
                stats["words_snapped"] += 1
            if j == 0 and out_lines:
                s1 = max(s1, prev_end_v2 + 0.02)
            elif j > 0:
                s1 = max(s1, words[j - 1]["end"] + 0.02)
            be = chain_end(s1)
            if be is not None and e1 > be + 0.35 and be >= s1 + MIN_DUR:
                f0 = int(be / FRAME_E); f1 = int(e1 / FRAME_E)
                run = 0; true_sil = False
                for v in rms[f0:f1]:
                    run = run + 1 if v < 120 else 0
                    if run >= 50:
                        true_sil = True; break
                if true_sil:
                    e1 = be
            if e1 <= s1 + MIN_DUR:
                e1 = s1 + MIN_DUR
            if j + 1 < len(words) and e1 > words[j + 1]["start"] - 0.02:
                e1 = max(s1 + MIN_DUR, words[j + 1]["start"] - 0.02)
            if (abs(s1 - ow["start"]) > 0.15 or abs(e1 - ow["end"]) > 0.15
                    or ow["end"] - ow["start"] < 0.12):
                fixed_words.append({"word": ow["word"],
                                    "ctc": [ow["start"], ow["end"]],
                                    "v2": [round(s1, 2), round(e1, 2)]})
            w["start"], w["end"] = round(s1, 2), round(e1, 2)
            if w["end"] - w["start"] < 0.12:
                stats["short_after"] += 1
        prev_end_v2 = words[-1]["end"]
        out_lines.append({"line": li + 1, "start": words[0]["start"],
                          "end": words[-1]["end"], "text": line_text,
                          "words": [{"word": w["word"], "start": w["start"],
                                     "end": w["end"]} for w in words]})
    out = {"audio": "audio/ang1.mp3", "text": "text/ang1.txt",
           "duration": ctc["duration"],
           "model": "V2.1: per-line CTC (mid/wide windows, guarded) "
                    "+ energy snap + burst-chain clamp",
           "lines": out_lines,
           "diagnostics": {"ref_words": len(ref),
                           "words_direct_timing": len(ref),
                           "words_interpolated": 0,
                           "anchored_lines": len(lines),
                           "total_lines": len(lines),
                           **stats,
                           "fixed_words": fixed_words,
                           "fixed_count": len(fixed_words)}}
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in stats.items()}, indent=1))
    print(f"fixed_count {len(fixed_words)}; wrote {OUT}")
    for li in range(3):
        print(f"Line {li+1}: " + " | ".join(
            f"{w['word']} {w['start']:.2f}-{w['end']:.2f}"
            for w in out_lines[li]["words"]))


if __name__ == "__main__":
    main()
