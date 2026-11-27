#!/usr/bin/env python3
"""
Gurbani subtitle timestamper.

Aligns line-by-line Gurmukhi text to a recording:
  1. faster-whisper -> word-level timestamps (Punjabi)
  2. monotonic fuzzy word alignment (Needleman-Wunsch) of the known
     text against Whisper's noisy words
  3. per-LINE cues from first/last matched words (+ interpolation)
  4. per-WORD timestamps: matched words take Whisper's timings directly,
     unmatched words are interpolated between neighbours

Outputs: <out>.srt  (line subtitles)
         <out>.json (lines + per-word timings + diagnostics)

Usage:
    python timestamper.py --audio audio/ang1.mp3 --text text/ang1.txt --out output/ang1
"""

import argparse
import json
import re
import time
import unicodedata
from difflib import SequenceMatcher

from faster_whisper import WhisperModel


# ------------------------------------------------------------ normalization

PUNCT_RE = re.compile(r"[।॥.,;:!?\-–—\"'()\[\]…]")
WS_RE = re.compile(r"\s+")
GURMUKHI_DIGITS = str.maketrans("੦੧੨੩੪੫੬੭੮੯", "0123456789")


def norm(t: str) -> str:
    t = unicodedata.normalize("NFC", t)
    t = t.translate(GURMUKHI_DIGITS)
    t = PUNCT_RE.sub("", t)
    return WS_RE.sub(" ", t).strip()


def load_lines(path):
    with open(path, encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]


# ------------------------------------------------------------ transcription

def transcribe(model, audio_path):
    segments, _info = model.transcribe(
        audio_path, language="pa", word_timestamps=True, vad_filter=True
    )
    words = []
    for seg in segments:
        for w in seg.words or []:
            t = norm(w.word)
            if t:
                words.append({"word": t, "start": w.start, "end": w.end})
    return words


# ------------------------------------------------------------ alignment

def word_score(a: str, b: str) -> float:
    if a == b:
        return 2.0
    # near-miss: nukta / spelling variants Whisper introduces
    if SequenceMatcher(None, a, b, autojunk=False).ratio() >= 0.75:
        return 1.0
    return -1.0


def needleman_wunsch(ref, hyp, gap=-1.0):
    """ref/hyp: lists of normalized words. Returns monotonic (i, j) index pairs."""
    n, m = len(ref), len(hyp)
    S = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        S[i][0] = i * gap
    for j in range(1, m + 1):
        S[0][j] = j * gap
    for i in range(1, n + 1):
        a = ref[i - 1]
        Si, Sp = S[i], S[i - 1]
        for j in range(1, m + 1):
            Si[j] = max(
                Sp[j] + gap,
                Si[j - 1] + gap,
                Sp[j - 1] + word_score(a, hyp[j - 1]),
            )
    pairs = []
    i, j = n, m
    while i > 0 and j > 0:
        s = word_score(ref[i - 1], hyp[j - 1])
        if S[i][j] == S[i - 1][j - 1] + s:
            pairs.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif S[i][j] == S[i - 1][j] + gap:
            i -= 1
        else:
            j -= 1
    pairs.reverse()
    return pairs


# ------------------------------------------------------------ line cues

START_PAD = 0.15
END_PAD = 0.30


def build_cues(lines, ref_line, hyp_words, pairs, audio_dur):
    """ref_line: norm-word index -> line index. Returns (cues, hits)."""
    hits = {}  # line_idx -> list of hyp word indices
    for i, j in pairs:
        hits.setdefault(ref_line[i], []).append(j)

    bounds = {}
    for li, js in hits.items():
        s = hyp_words[min(js)]["start"] - START_PAD
        e = hyp_words[max(js)]["end"] + END_PAD
        bounds[li] = (max(0.0, s), e)

    # interpolate lines Whisper missed entirely
    anchored = sorted(bounds)
    cues = []
    for li in range(len(lines)):
        if li in bounds:
            cues.append(bounds[li])
            continue
        prev = max([a for a in anchored if a < li], default=None)
        nxt = min([a for a in anchored if a > li], default=None)
        if prev is None and nxt is None:
            cues.append((0.0, audio_dur))
        elif prev is None:
            cues.append((0.0, bounds[nxt][0]))
        elif nxt is None:
            cues.append((bounds[prev][1], audio_dur))
        else:
            lo, hi = bounds[prev][1], bounds[nxt][0]
            wc = [len(norm(lines[k]).split()) for k in range(prev + 1, nxt)]
            total = sum(wc) or 1
            before = sum(wc[: li - prev - 1])
            frac0 = before / total
            frac1 = (before + wc[li - prev - 1]) / total
            cues.append((lo + frac0 * (hi - lo), lo + frac1 * (hi - lo)))

    # sanitize: ordered, non-overlapping, inside [0, audio_dur]
    fixed = []
    for s, e in cues:
        s = max(0.0, min(s, audio_dur))
        e = max(0.0, min(e, audio_dur))
        if e <= s:
            e = min(s + 0.5, audio_dur)
        fixed.append([s, e])
    for k in range(1, len(fixed)):
        if fixed[k][0] < fixed[k - 1][1]:
            mid = (fixed[k][0] + fixed[k - 1][1]) / 2
            fixed[k - 1][1] = mid
            fixed[k][0] = mid
    return fixed, hits


# ------------------------------------------------------------ word timings

def build_word_timings(lines, ref, hyp_words, pairs, line_cues):
    """ref: list of (orig_tok, norm_w, line_idx).
    Returns per-line lists of {word, start, end} with original tokens."""
    match = {}  # ref idx -> (start, end) from Whisper
    for i, j in pairs:
        hw = hyp_words[j]
        match[i] = (hw["start"], hw["end"])

    by_line = {}
    for i, (_o, _n, li) in enumerate(ref):
        by_line.setdefault(li, []).append(i)

    out = []
    for li, _line in enumerate(lines):
        idxs = by_line.get(li, [])
        n = len(idxs)
        cue_s, cue_e = line_cues[li]
        spans = [None] * n
        for pos, i in enumerate(idxs):
            if i in match:
                spans[pos] = [match[i][0], match[i][1]]
        # boundary points: (pos, start, end); pos=-1 -> cue start, pos=n -> cue end
        bounds = [(-1, cue_s, cue_s)]
        for pos, se in enumerate(spans):
            if se is not None:
                bounds.append((pos, se[0], se[1]))
        bounds.append((n, cue_e, cue_e))
        # fill gaps between consecutive boundary points, weighted by word length
        for b in range(len(bounds) - 1):
            p0, _, e0 = bounds[b]
            p1, s1, _ = bounds[b + 1]
            lo, hi = e0, max(s1, e0)
            between = list(range(p0 + 1, p1))
            if not between:
                continue
            weights = [max(len(ref[idxs[p]][1]), 1) for p in between]
            tot = sum(weights)
            t = lo
            for p, wgt in zip(between, weights):
                d = (hi - lo) * wgt / tot
                spans[p] = [t, t + d]
                t += d
        # sanitize: inside cue, start<end, monotonic
        for p in range(n):
            s, e = spans[p]
            s = max(s, cue_s)
            e = min(e, cue_e)
            if e <= s:
                e = s + 0.05
            spans[p] = [round(s, 2), round(e, 2)]
        for p in range(1, n):
            if spans[p][0] < spans[p - 1][1]:
                spans[p][0] = spans[p - 1][1]
                if spans[p][1] <= spans[p][0]:
                    spans[p][1] = round(spans[p][0] + 0.05, 2)
        out.append([
            {"word": ref[idxs[p]][0], "start": spans[p][0], "end": spans[p][1]}
            for p in range(n)
        ])
    return out


# ------------------------------------------------------------ output

def ts(t):
    h, r = divmod(t, 3600)
    m, s = divmod(r, 60)
    ms = int(round((s % 1) * 1000))
    if ms == 1000:
        s += 1
        ms = 0
    return f"{int(h):02d}:{int(m):02d}:{int(s):02d},{ms:03d}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True)
    ap.add_argument("--text", required=True)
    ap.add_argument("--out", required=True, help="output prefix, e.g. output/ang1")
    ap.add_argument("--model", default="large-v3-turbo")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    t0 = time.time()
    print(f"[1/5] loading model {args.model} ...", flush=True)
    model = WhisperModel(
        args.model,
        device=args.device,
        compute_type="int8" if args.device == "cpu" else "float16",
        cpu_threads=2,
    )

    print(f"[2/5] transcribing {args.audio} ...", flush=True)
    hyp_words = transcribe(model, args.audio)
    hyp = [w["word"] for w in hyp_words]
    print(f"      got {len(hyp_words)} timestamped words", flush=True)

    lines = load_lines(args.text)
    ref = []  # (orig_tok, norm_w, line_idx)
    for li, line in enumerate(lines):
        for tok in line.split():
            nw = norm(tok)
            if nw:
                ref.append((tok, nw, li))
    ref_norm = [nw for (_, nw, _) in ref]
    ref_line = [li for (_, _, li) in ref]

    print(f"[3/5] aligning {len(ref)} text words vs {len(hyp)} asr words ...",
          flush=True)
    pairs = needleman_wunsch(ref_norm, hyp)

    audio_dur = hyp_words[-1]["end"] if hyp_words else 0.0
    print("[4/5] building line cues + word timings ...", flush=True)
    line_cues, hits = build_cues(lines, ref_line, hyp_words, pairs, audio_dur)
    word_timings = build_word_timings(lines, ref, hyp_words, pairs, line_cues)

    exact = sum(1 for i, j in pairs if ref_norm[i] == hyp[j])
    n_direct = len({i for i, _ in pairs})
    n_interp = len(ref) - n_direct
    print(f"      matched pairs: {len(pairs)} ({exact} exact, "
          f"{len(pairs) - exact} fuzzy)", flush=True)
    print(f"      lines with direct hits: {len(hits)}/{len(lines)}; "
          f"words direct: {n_direct}, interpolated: {n_interp}", flush=True)

    srt_path = args.out + ".srt"
    with open(srt_path, "w", encoding="utf-8") as f:
        for num, ((s, e), line) in enumerate(zip(line_cues, lines), 1):
            f.write(f"{num}\n{ts(s)} --> {ts(e)}\n{line}\n\n")

    json_path = args.out + ".json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "audio": args.audio,
                "text": args.text,
                "duration": round(audio_dur, 2),
                "lines": [
                    {
                        "line": num,
                        "start": round(s, 2),
                        "end": round(e, 2),
                        "text": line,
                        "words": wt,
                    }
                    for num, ((s, e), line, wt) in
                    enumerate(zip(line_cues, lines, word_timings), 1)
                ],
                "diagnostics": {
                    "ref_words": len(ref),
                    "asr_words": len(hyp),
                    "matched_pairs": len(pairs),
                    "exact_matches": exact,
                    "anchored_lines": len(hits),
                    "total_lines": len(lines),
                    "words_direct_timing": n_direct,
                    "words_interpolated": n_interp,
                },
            },
            f, ensure_ascii=False, indent=1,
        )
    print(f"[5/5] wrote {srt_path} + {json_path} in {time.time() - t0:.0f}s",
          flush=True)


if __name__ == "__main__":
    main()
