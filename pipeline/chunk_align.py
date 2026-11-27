#!/usr/bin/env python3
"""
Anchor-segmented CTC forced alignment for LONG single-track banis
(Jaap Sahib, Rehraas Sahib) whose full trellis OOMs the aligner.

Why anchors (learned the hard way, 2026-11-10): a CTC forced aligner
given a window much larger than its target lets the target's chars
wander to their individual likelihood maxima across the window — the
path stretches to fill any oversized window and scores BETTER than the
true placement, so window-size search by score fails. The aligner is
only trustworthy when the text roughly FILLS its window (the regime of
all the successful full-file alignments). This tool therefore:

  1. hunts distinctive ~36-word anchor phrases (line-start positions,
     spaced <= ~200 words apart) across the emissions with small
     sliding windows scored by token-only mean logprob, sequentially
     restricted to after the previous anchor, with a separation check
     against the best competing location;
  2. aligns each inter-anchor segment with BOTH ends pinned by hunted
     anchor spans, so the segment text fills its window;
  3. merges with boundary clamping and reports diagnostics.

Emissions reuse <out>.emissions.npz when present (ctc_align.py format);
otherwise they are computed and cached exactly as there.

Usage:
    python chunk_align.py --audio jaap.mp3 --text jaap.txt --out jaap
"""

import argparse
import json
import os
import sys
import time
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import timestamper  # noqa: E402
from ctc_align import (BLANK_ID, DIGIT_RE, FRAME_S, IK_OANKAR_EXPANSION,
                       MODEL_PATH, SEP_ID, compute_emissions, ctc_form)
from ctc_forced_aligner import forced_align, merge_repeats  # noqa: E402


def build_targets(ref_slice):
    tgt_ids, char_word, valid = [], [], []
    for i, (tok, nw, _li) in enumerate(ref_slice):
        if DIGIT_RE.fullmatch(nw):
            valid.append(False)
            continue
        pieces = IK_OANKAR_EXPANSION if nw == "ੴ" else [ctc_form(nw)]
        for piece in pieces:
            if tgt_ids:
                tgt_ids.append(SEP_ID)
                char_word.append(None)
            for ch in piece:
                cid = VOCAB.get(ch)
                if cid is None:
                    raise RuntimeError(f"char {ch!r} not in model vocab")
                tgt_ids.append(cid)
                char_word.append(i)
        valid.append(True)
    return tgt_ids, char_word, valid


def walk(emissions_slice, tgt_ids, char_word):
    """Verbatim walk from ctc_align.py."""
    targets = np.asarray([tgt_ids], dtype=np.int64)
    path, _scores = forced_align(np.expand_dims(emissions_slice, 0),
                                 targets, blank=BLANK_ID)
    path = path.squeeze()
    segments = merge_repeats(path.tolist(), IDX2TOK)
    exp_chars = [(char_word[pos], IDX2TOK[cid])
                 for pos, cid in enumerate(tgt_ids) if cid != SEP_ID]
    MAX_SKIP = 4
    word_frames = {}
    n_skipped = 0
    skipped_words = set()
    last_matched_label = None
    ei = 0
    for si, seg in enumerate(segments):
        if ei >= len(exp_chars):
            break
        if seg.label == "<blank>":
            continue
        if seg.label == "|":
            continue
        L = seg.label
        run_ahead = 0
        j = si + 1
        while j < len(segments):
            lj = segments[j].label
            if lj == "<blank>":
                j += 1
                continue
            if lj == L:
                run_ahead += 1
                j += 1
                continue
            break
        start = ei
        while (start < len(exp_chars) and start - ei <= MAX_SKIP
               and exp_chars[start][1] != L):
            start += 1
        if start >= len(exp_chars) or exp_chars[start][1] != L:
            if last_matched_label == L:
                continue
            raise RuntimeError(
                f"walk desync at segment {seg} (expected "
                f"{exp_chars[ei] if ei < len(exp_chars) else 'END'})")
        for k in range(ei, start):
            n_skipped += 1
            skipped_words.add(exp_chars[k][0])
        ei = start
        wi_seg = None
        while ei < len(exp_chars) and exp_chars[ei][1] == L:
            wi, _ch = exp_chars[ei]
            if wi_seg is None:
                wi_seg = wi
            elif wi != wi_seg:
                break
            if wi not in word_frames:
                word_frames[wi] = [seg.start, seg.end]
            else:
                word_frames[wi][1] = seg.end
            last_matched_label = L
            ei += 1
            if run_ahead > 0:
                break
    assert ei == len(exp_chars), \
        f"walk consumed {ei}/{len(exp_chars)} chars ({n_skipped} skipped)"
    return word_frames, n_skipped, skipped_words, path


def score_words(emissions_slice, path, wf, w0, scores_g, agreements_g):
    am = emissions_slice.argmax(axis=-1)
    for li_, (s, e) in wf.items():
        gi = w0 + li_
        if e < s:
            continue
        fr = path[s:e + 1]
        scores_g[gi] = float(emissions_slice[s:e + 1, fr].mean())
        char_mask = fr != BLANK_ID
        if char_mask.sum():
            agreements_g[gi] = float(
                (am[s:e + 1][char_mask] == fr[char_mask]).mean())


def align_words(emissions, ref, w0, w1, win0, win1):
    tgt_ids, char_word, valid = build_targets(ref[w0:w1])
    wf, n_sk, skw, path = walk(emissions[win0:win1], tgt_ids, char_word)
    frames = {w0 + li_: [s + win0, e + win0] for li_, (s, e) in wf.items()}
    return frames, valid, n_sk, {w0 + i for i in skw}, \
        emissions[win0:win1], path, win0


def hunt_anchor(emissions, tgt_ids, lo_f, hi_f):
    """Slide a 45s window over frames [lo_f, hi_f]; return
    (score, start_f, end_f, sep) for the best token-only placement,
    where sep = margin over the best placement >=40s away."""
    T = emissions.shape[0]
    win = int(45 / FRAME_S)
    step = int(10 / FRAME_S)
    hi_f = min(hi_f, T - win)
    if hi_f < lo_f:
        return None
    cands = []
    for w0 in range(lo_f, hi_f + 1, step):
        sl = emissions[w0:w0 + win]
        path, _ = forced_align(np.expand_dims(sl, 0),
                               np.asarray([tgt_ids], dtype=np.int64),
                               blank=BLANK_ID)
        p = path.squeeze()
        mask = p != BLANK_ID
        nz = np.nonzero(mask)[0]
        if len(nz) < len(tgt_ids) * 0.45:
            continue
        sc = float(sl[mask, p[mask]].mean())
        cands.append((sc, w0 + nz[0], w0 + nz[-1]))
    if not cands:
        return None
    cands.sort(reverse=True)
    best = cands[0]
    rivals = [c for c in cands if abs(c[1] - best[1]) > int(40 / FRAME_S)]
    sep = best[0] - (rivals[0][0] if rivals else -9.9)
    return best[0], best[1], best[2], sep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True)
    ap.add_argument("--text", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--anchor-words", type=int, default=36)
    ap.add_argument("--anchor-every", type=int, default=200)
    args = ap.parse_args()

    global VOCAB, IDX2TOK
    VOCAB = json.load(open(os.path.join(MODEL_PATH, "vocab.json"),
                           encoding="utf-8"))
    IDX2TOK = {v: k for k, v in VOCAB.items()}
    IDX2TOK[BLANK_ID] = "<blank>"

    import librosa
    print(f"[2/5] loading audio {args.audio} ...", flush=True)
    wav, _ = librosa.load(args.audio, sr=16000, mono=True)
    print(f"      {len(wav) / 16000:.1f}s audio", flush=True)

    print("[3/5] emissions ...", flush=True)
    t0 = time.time()
    em_cache = args.out + ".emissions.npz"
    if os.path.exists(em_cache):
        z = np.load(em_cache)
        if int(z["n_samples"]) != len(wav):
            raise RuntimeError("cached emissions are for a different audio")
        emissions = z["emissions"].astype(np.float32)
        print(f"      loaded cached emissions <- {em_cache}", flush=True)
    else:
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
        print("[1/5] loading model ...", flush=True)
        processor = Wav2Vec2Processor.from_pretrained(MODEL_PATH)
        model = Wav2Vec2ForCTC.from_pretrained(MODEL_PATH).eval()
        emissions = compute_emissions(model, processor, wav)
        print(f"      computed in {time.time() - t0:.0f}s", flush=True)
        np.savez(em_cache, emissions=emissions,
                 n_samples=np.int64(len(wav)))
        print(f"      cached emissions -> {em_cache}", flush=True)
    n_frames = emissions.shape[0]
    print(f"      {n_frames} frames", flush=True)

    lines = timestamper.load_lines(args.text)
    ref = []
    for li, line in enumerate(lines):
        for tok in line.split():
            nw = timestamper.norm(tok)
            if nw:
                ref.append((tok, nw, li))
    ref_line = [li for (_, _, li) in ref]
    n_words = len(ref)
    print(f"      {n_words} canonical words", flush=True)

    # valid (spoken) word indices + line-start word indices
    valid_idx = [i for i, (_, nw, _) in enumerate(ref)
                 if not DIGIT_RE.fullmatch(nw)]
    line_starts = []
    seen_lines = set()
    for i, (_, _, li) in enumerate(ref):
        if li not in seen_lines:
            seen_lines.add(li)
            line_starts.append(i)

    # anchor word positions: word 0, then line starts spaced
    # anchor_every..anchor_every*1.6 words apart
    anchor_pos = [0]
    nxt = args.anchor_every
    for i in line_starts:
        if i >= nxt and i < n_words - 40:
            anchor_pos.append(i)
            nxt = i + args.anchor_every
    print(f"      anchor positions (words): {anchor_pos}", flush=True)

    print("[4/5] hunting anchors ...", flush=True)
    anchors = []  # (word_idx, span_start_f, span_end_f, score, sep)
    prev_end_f = 0
    for ai, p in enumerate(anchor_pos):
        # phrase = next anchor_words valid words starting at p
        vi = [i for i in valid_idx if i >= p][:args.anchor_words]
        if len(vi) < args.anchor_words * 0.7:
            continue
        tgt_ids, char_word, _v = build_targets(
            [ref[i] for i in range(p, vi[-1] + 1)])
        lo = 0 if ai == 0 else max(0, prev_end_f - int(5 / FRAME_S))
        hi = int(120 / FRAME_S) if ai == 0 else n_frames
        got = hunt_anchor(emissions, tgt_ids, lo, hi)
        if got is None:
            print(f"      anchor @{p}: NOT FOUND", flush=True)
            continue
        sc, s_f, e_f, sep = got
        ok = sc >= -2.4 and (ai == 0 or sep >= 0.15)
        print(f"      anchor @{p}: {s_f * FRAME_S:.1f}-{e_f * FRAME_S:.1f}s "
              f"score {sc:.2f} sep {sep:.2f} {'OK' if ok else 'REJECTED'}",
              flush=True)
        if ok:
            anchors.append((p, s_f, e_f, sc, sep))
            prev_end_f = e_f
    if len(anchors) < 2:
        raise RuntimeError(f"only {len(anchors)} anchors accepted; aborting")
    print(f"      {len(anchors)} anchors accepted", flush=True)

    # ---- segment alignment: text fills each window ----
    print("[4/5] aligning segments ...", flush=True)
    word_frames = {}
    scores_g = [float("nan")] * n_words
    agreements_g = [float("nan")] * n_words
    valid_global = [False] * n_words
    tot_skipped = 0

    seg_bounds = []  # (text_lo_word, text_hi_word_exclusive, win0, win1)
    for k, (p, s_f, e_f, _sc, _sep) in enumerate(anchors):
        if k == 0:
            t_lo = 0
            w_lo = 0
        else:
            t_lo = p
            w_lo = max(0, s_f - int(3 / FRAME_S))
        if k == len(anchors) - 1:
            t_hi = n_words
            w_hi = n_frames
        else:
            np_, ns_f, ne_f = anchors[k + 1][0], anchors[k + 1][1], \
                anchors[k + 1][2]
            # segment text runs through the NEXT anchor's phrase
            vi = [i for i in valid_idx if i >= np_][:args.anchor_words]
            t_hi = vi[-1] + 1
            w_hi = min(n_frames, ne_f + int(3 / FRAME_S))
        seg_bounds.append((t_lo, t_hi, w_lo, w_hi))

    for si, (t_lo, t_hi, w_lo, w_hi) in enumerate(seg_bounds):
        print(f"      segment {si + 1}/{len(seg_bounds)}: words "
              f"{t_lo}-{t_hi - 1}, window {w_lo * FRAME_S:.0f}-"
              f"{w_hi * FRAME_S:.0f}s", flush=True)
        frames, valid, n_sk, skw, esl, path, off = align_words(
            emissions, ref, t_lo, t_hi, w_lo, w_hi)
        for i, v in enumerate(valid):
            valid_global[t_lo + i] = v
        keep_hi = seg_bounds[si + 1][0] if si + 1 < len(seg_bounds) \
            else n_words
        kept = 0
        for gi, se in frames.items():
            if gi >= keep_hi:
                continue
            if gi in word_frames:
                continue
            word_frames[gi] = se
            kept += 1
        score_words(esl, path,
                    {gi - t_lo: [se[0] - off, se[1] - off]
                     for gi, se in frames.items() if gi < keep_hi},
                    t_lo, scores_g, agreements_g)
        tot_skipped += n_sk
        fw = sorted(frames)
        dur = (frames[fw[-1]][1] - frames[fw[0]][0]) * FRAME_S
        print(f"        kept {kept} spans, {n_sk} chars skipped, "
              f"rate {kept / dur:.2f} w/s over kept region", flush=True)

    valid = valid_global
    n_clamped = 0
    prev_end = -1
    for i in sorted(word_frames):
        s, e = word_frames[i]
        assert 0 <= s <= e < n_frames, f"word {i} span out of bounds"
        if s <= prev_end:
            n_clamped += 1
            s = prev_end + 1
            if e < s:
                e = s
            word_frames[i] = [s, e]
        prev_end = word_frames[i][1]
    print(f"      total: {len(word_frames)} word spans, "
          f"{tot_skipped} chars skipped, {n_clamped} clamped", flush=True)

    hyp_words, scores, agreements = [], [], []
    for i, (tok, nw, _li) in enumerate(ref):
        if valid[i] and i in word_frames:
            s, e = word_frames[i]
            hyp_words.append({"word": tok,
                              "start": round(s * FRAME_S, 2),
                              "end": round((e + 1) * FRAME_S, 2)})
            scores.append(scores_g[i])
            agreements.append(agreements_g[i])
        else:
            hyp_words.append({"word": tok, "start": 0.0, "end": 0.0})
            scores.append(float("nan"))
            agreements.append(float("nan"))
    n_direct = sum(1 for i in range(n_words)
                   if valid[i] and i in word_frames)
    n_interp = n_words - n_direct
    sc_arr = np.array([s for s in scores if not np.isnan(s)])
    mean_sc = float(sc_arr.mean()) if len(sc_arr) else float("nan")
    ag_arr = np.array([a for a in agreements if not np.isnan(a)])
    mean_ag = float(ag_arr.mean()) if len(ag_arr) else float("nan")
    print(f"      aligned words: {n_direct}/{n_words} "
          f"(interpolated: {n_interp}), mean word logprob: {mean_sc:.2f}, "
          f"mean argmax agreement: {mean_ag:.3f}", flush=True)

    print("[5/5] building line cues + word timings ...", flush=True)
    pairs = [(i, i) for i in range(n_words)
             if valid[i] and i in word_frames]
    audio_dur = n_frames * FRAME_S
    line_cues, hits = timestamper.build_cues(lines, ref_line, hyp_words,
                                             pairs, audio_dur)
    word_timings = timestamper.build_word_timings(lines, ref, hyp_words,
                                                  pairs, line_cues)
    json_path = args.out + ".json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "audio": args.audio,
                "text": args.text,
                "duration": round(audio_dur, 2),
                "model": "manandey/wav2vec2-large-xlsr-punjabi (CTC, "
                         "anchor-segmented forced alignment)",
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
                    "ref_words": n_words,
                    "words_direct_timing": n_direct,
                    "words_interpolated": n_interp,
                    "anchored_lines": len(hits),
                    "total_lines": len(lines),
                    "mean_word_logprob": round(mean_sc, 3),
                    "mean_argmax_agreement": round(mean_ag, 3),
                    "chars_skipped_by_path": tot_skipped,
                    "anchors": len(anchors),
                },
            },
            f, ensure_ascii=False, indent=1,
        )
    print(f"wrote {json_path}", flush=True)


VOCAB = None
IDX2TOK = None

if __name__ == "__main__":
    main()
