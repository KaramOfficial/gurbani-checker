#!/usr/bin/env python3
"""
CTC forced alignment of canonical Gurmukhi text to audio.

Uses manandey/wav2vec2-large-xlsr-punjabi directly (no transcription step):
  1. wav2vec2 -> frame-level CTC posteriors (20ms frames)
  2. Viterbi forced alignment of the canonical char sequence
     (nukta chars precomposed to the model's vocab forms; ੴ expanded to
     "ਇਕ ਓਅੰਕਾਰ"; unspoken digit markers dropped from targets)
  3. char spans -> word timestamps -> same JSON schema as timestamper.py,
     so approaches are directly comparable in the checker.

Unaligned/dropped words are interpolated between neighbours by the shared
line/timing builders.

Usage:
    python ctc_align.py --audio audio/ang1.mp3 --text text/ang1.txt \
        --out output/ang1.ctc
"""

import argparse
import json
import os
import re
import sys
import time

import numpy as np
import torch
import librosa

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import timestamper  # noqa: E402
from ctc_forced_aligner import forced_align, merge_repeats  # noqa: E402

MODEL_PATH = os.path.expanduser(
    "~/.cache/huggingface/hub/models--manandey--wav2vec2-large-xlsr-punjabi"
    "/snapshots/7d311d05f70e7e2c97f42e7e831499e5995ee0ee")
FRAME_S = 0.02  # wav2vec2 stride 320 @ 16kHz
BLANK_ID = 62
SEP_ID = 22  # '|' word separator

DIGIT_RE = re.compile(r"[0-9੦-੯]+$")
# decomposed (NFC) -> precomposed (model vocab) nukta forms
NUKTA = [("\u0a17\u0a3c", "\u0a5a"), ("\u0a38\u0a3c", "\u0a36"),
         ("\u0a1c\u0a3c", "\u0a5b"), ("\u0a16\u0a3c", "\u0a59"),
         ("\u0a2b\u0a3c", "\u0a5e"), ("\u0a32\u0a3c", "\u0a33")]
IK_OANKAR_EXPANSION = ["ਇਕ", "ਓਅੰਕਾਰ"]
# chars missing from the wav2vec2 Punjabi model vocab -> closest available
VOCAB_FIXUPS = [
    ("ਞ", "ਨ"),   # U+0A1E NYA -> NA (closest nasal)
    ("ਙ", "ਨ"),   # U+0A19 NGA -> NA (closest nasal)
    ("ਃ", "ਹ"),   # U+0A03 VISARGA -> HA (breathy release)
    ("ੵ", "ਯ"),   # U+0A75 YAKASH -> YA
    ("ੑ", ""),    # U+0A51 UDAAT (tone mark, no consonantal value) -> drop
    ("ੲ", "ਇ"),   # U+0A72 IRI: vowel base pronounced "i" (e.g. ਪਾੲਂੀ "ਪਾਈਂ")
    # subscript/superscript digits: unspoken footnote markers -> drop
    ("₀", ""), ("₁", ""), ("₂", ""), ("₃", ""), ("₄", ""),
    ("₅", ""), ("₆", ""), ("₇", ""), ("₈", ""), ("₉", ""),
    ("⁰", ""), ("¹", ""), ("²", ""), ("³", ""), ("⁴", ""),
    ("⁵", ""), ("⁶", ""), ("⁷", ""), ("⁸", ""), ("⁹", ""),
    # ASCII digits: unspoken markers glued to words (e.g. ਪਉੜੀ੫॥, ਗਉੜੀ੯॥) -> drop
    ("0", ""), ("1", ""), ("2", ""), ("3", ""), ("4", ""),
    ("5", ""), ("6", ""), ("7", ""), ("8", ""), ("9", ""),
]


def ctc_form(nw: str) -> str:
    """timestamper-norm word -> model-vocab form (precompose nukta)."""
    t = nw
    for d, p in NUKTA:
        t = t.replace(d, p)
    # chars missing from the wav2vec2 Punjabi model vocab, mapped to the
    # closest available char (found by scanning SGGS text 2026-09-26)
    for d, p in VOCAB_FIXUPS:
        t = t.replace(d, p)
    return t


def compute_emissions(model, processor, wav):
    """Frame-level log-posteriors, chunked to bound memory.

    The wav is normalized once over the full audio (same as the
    processor's default), then split into overlapping chunks; overlap
    frames are trimmed so the concatenation is seamless.
    """
    w = wav - wav.mean()
    std = w.std()
    if std > 1e-7:
        w = w / std
    CHUNK_S, OVERLAP_S = 90, 2
    sr = 16000
    n = len(w)
    cs, ov = int(CHUNK_S * sr), int(OVERLAP_S * sr)
    of = int(OVERLAP_S / FRAME_S)
    parts = []
    pos = 0
    while pos < n:
        s = max(0, pos - (ov if pos > 0 else 0))
        e = min(n, pos + cs + (ov if pos + cs < n else 0))
        inputs = processor(w[s:e], sampling_rate=sr, return_tensors="pt")
        with torch.no_grad():
            logits = model(inputs.input_values).logits
        em_c = torch.log_softmax(logits, dim=-1).numpy().astype(np.float32)[0]
        f_s = 0 if pos == 0 else of
        f_e = em_c.shape[0] if e == n else em_c.shape[0] - of
        parts.append(em_c[f_s:f_e])
        pos += cs
    return np.concatenate(parts, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True)
    ap.add_argument("--text", required=True)
    ap.add_argument("--out", required=True,
                    help="output prefix, e.g. output/ang1.ctc")
    args = ap.parse_args()

    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

    print("[1/5] loading model ...", flush=True)
    t0 = time.time()
    processor = Wav2Vec2Processor.from_pretrained(MODEL_PATH)
    model = Wav2Vec2ForCTC.from_pretrained(MODEL_PATH).eval()
    vocab = json.load(open(os.path.join(MODEL_PATH, "vocab.json"),
                           encoding="utf-8"))
    print(f"      loaded in {time.time() - t0:.0f}s", flush=True)

    print(f"[2/5] loading audio {args.audio} ...", flush=True)
    wav, _ = librosa.load(args.audio, sr=16000, mono=True)
    print(f"      {len(wav) / 16000:.1f}s audio", flush=True)

    print("[3/5] computing CTC emissions ...", flush=True)
    t0 = time.time()
    # Cache stores the audio's sample count alongside the emissions, so a
    # cache is only reused for the exact audio that produced it. (An older
    # revision validated the frame count against len(wav)/320, but chunked
    # convolution edge effects make that formula approximate; the stored
    # emissions are self-consistent, so identity-by-sample-count is the
    # correct check.)
    em_cache = args.out + ".emissions.npz"
    if os.path.exists(em_cache):
        z = np.load(em_cache)
        if int(z["n_samples"]) != len(wav):
            raise RuntimeError(
                f"cached emissions {em_cache} are for {int(z['n_samples'])} "
                f"samples but the audio has {len(wav)}; delete the cache")
        emissions = z["emissions"].astype(np.float32)
        print(f"      loaded cached emissions <- {em_cache} "
              f"in {time.time() - t0:.0f}s", flush=True)
    else:
        emissions = compute_emissions(model, processor, wav)
        print(f"      computed in {time.time() - t0:.0f}s", flush=True)
        np.savez(em_cache, emissions=emissions,
                 n_samples=np.int64(len(wav)))
        print(f"      cached emissions -> {em_cache}", flush=True)
    n_frames = emissions.shape[0]
    print(f"      {n_frames} frames", flush=True)

    lines = timestamper.load_lines(args.text)
    # canonical ref: SAME tokenizer as timestamper.py (drops empty-norm
    # tokens), so word lists are directly comparable across approaches
    ref = []  # (orig_tok, norm_w, line_idx)
    for li, line in enumerate(lines):
        for tok in line.split():
            nw = timestamper.norm(tok)
            if nw:
                ref.append((tok, nw, li))
    ref_line = [li for (_, _, li) in ref]
    print(f"      {len(ref)} canonical words", flush=True)

    print("[4/5] building targets + forced alignment ...", flush=True)
    # targets: char ids joined by SEP_ID; char_word parallels tgt_ids
    # (None at separators) so every char maps back to its output word
    tgt_ids = []
    char_word = []
    valid = []             # output word idx -> bool (in targets)
    for i, (tok, nw, _li) in enumerate(ref):
        if DIGIT_RE.fullmatch(nw):
            valid.append(False)  # unspoken verse markers -> interpolated
            continue
        # expand ੴ to its spoken form; keep single output word
        pieces = IK_OANKAR_EXPANSION if nw == "ੴ" else [ctc_form(nw)]
        for piece in pieces:
            if tgt_ids:
                tgt_ids.append(SEP_ID)
                char_word.append(None)
            for ch in piece:
                cid = vocab.get(ch)
                if cid is None:
                    raise RuntimeError(f"char {ch!r} not in model vocab")
                tgt_ids.append(cid)
                char_word.append(i)
        valid.append(True)

    targets = np.asarray([tgt_ids], dtype=np.int64)
    path, _scores = forced_align(np.expand_dims(emissions, 0), targets,
                                 blank=BLANK_ID)
    path = path.squeeze()
    idx_to_token = {v: k for k, v in vocab.items()}
    idx_to_token[BLANK_ID] = "<blank>"
    segments = merge_repeats(path.tolist(), idx_to_token)

    # walk segments against expected char sequence (skip blanks).
    # The Viterbi path may SKIP target chars the model is confident are
    # absent (e.g. reduced/elided vowel signs like ਿ in fast recitation).
    # Tolerate short skips: scan ahead for the segment's label, record the
    # skipped chars, and keep the word's span from its emitted chars. A
    # word whose every char was skipped gets no span -> interpolated,
    # same as the unspoken digit markers.
    #
    # Consumption rule: the path may emit the same label in several
    # blank-separated runs (e.g. ਹ, blank, ਹ for target "ਹਹ"); each run
    # aligns to ONE target char. Only a single merged run with no
    # same-label run following it may cover a whole run of identical
    # target chars. (Greedily consuming all identical chars per segment
    # desyncs the walk on words like ਸੋਹਹਿ.)
    exp_chars = [(char_word[pos], idx_to_token[cid])
                 for pos, cid in enumerate(tgt_ids) if cid != SEP_ID]
    n_sep_expected = sum(1 for cid in tgt_ids if cid == SEP_ID)
    MAX_SKIP = 4  # max consecutive expected chars the path may skip
    word_frames = {}  # out_word_idx -> [first_frame, last_frame]
    n_sep_seen = 0
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
            n_sep_seen += 1
            continue
        L = seg.label
        # count same-label runs ahead (ignoring blanks) before any other
        # label/separator: each such run gets exactly one target char
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
        # find seg.label at/after ei, tolerating skipped expected chars
        start = ei
        while (start < len(exp_chars) and start - ei <= MAX_SKIP
               and exp_chars[start][1] != L):
            start += 1
        if start >= len(exp_chars) or exp_chars[start][1] != L:
            # model stutter: re-emitted the just-matched label with no
            # target char left for it -> ignore the extra segment
            if last_matched_label == L:
                continue
            raise RuntimeError(
                f"walk desync at segment {seg} (expected "
                f"{exp_chars[ei] if ei < len(exp_chars) else 'END'}, "
                f"skipped {n_skipped} chars so far)")
        for k in range(ei, start):
            n_skipped += 1
            skipped_words.add(exp_chars[k][0])
        ei = start
        # consume matching target chars, but never across a word boundary
        # (a '|'/blank segment always separates words in the path); when
        # more same-label runs follow, take only one char and leave the
        # rest for them
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

    # --- structural assertions: the walk must consume the full target ---
    # (ei already jumps over skipped chars, so it counts every expected
    # char exactly once; n_skipped is tracked separately for diagnostics)
    n_exp = len(exp_chars)
    assert ei == n_exp, \
        f"walk consumed {ei}/{n_exp} expected chars ({n_skipped} skipped)"
    # separators may be skipped when words run together without a pause
    assert n_sep_seen <= n_sep_expected, \
        f"{n_sep_seen} separators seen, expected at most {n_sep_expected}"
    prev_end = -1
    for i in sorted(word_frames):
        s, e = word_frames[i]
        assert 0 <= s <= e < n_frames, \
            f"word {i} span [{s},{e}] out of bounds (T={n_frames})"
        assert s > prev_end or i == 0, \
            f"word {i} starts at {s}, not after prev end {prev_end}"
        prev_end = e
    print(f"      walk consumed {ei}/{n_exp} chars ({n_skipped} skipped, "
          f"{len(skipped_words)} words affected), "
          f"{n_sep_seen}/{n_sep_expected} separators, "
          f"{len(word_frames)} word spans OK",
          flush=True)

    hyp_words = []
    scores = []
    agreements = []
    argmax = emissions.argmax(axis=-1)  # model's free decode per frame
    for i, (tok, nw, _li) in enumerate(ref):
        if valid[i] and i in word_frames:
            s, e = word_frames[i]
            st, en = round(s * FRAME_S, 2), round((e + 1) * FRAME_S, 2)
            fr = path[s:e + 1]
            sc = float(emissions[s:e + 1, fr].mean())
            # agreement: on frames where the forced path emits a char
            # (not blank), how often does the model's free argmax agree?
            # Robust to peaky posteriors; measures alignment quality.
            char_mask = fr != BLANK_ID
            if char_mask.sum():
                agr = float((argmax[s:e + 1][char_mask]
                             == fr[char_mask]).mean())
            else:
                agr = float("nan")
            agreements.append(agr)
            hyp_words.append({"word": tok, "start": st, "end": en})
            scores.append(sc)
        else:
            hyp_words.append({"word": tok, "start": 0.0, "end": 0.0})
            scores.append(float("nan"))
            agreements.append(float("nan"))

    n_direct = sum(1 for i in range(len(ref)) if valid[i] and i in word_frames)
    n_interp = len(ref) - n_direct
    sc_arr = np.array([s for s in scores if not np.isnan(s)])
    mean_sc = float(sc_arr.mean()) if len(sc_arr) else float("nan")
    if len(sc_arr):
        qs = np.quantile(sc_arr, [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
        print(f"      word logprob quantiles (0/10/25/50/75/90/100): "
              + " ".join(f"{q:.2f}" for q in qs), flush=True)
    ag_arr = np.array([a for a in agreements if not np.isnan(a)])
    mean_ag = float(ag_arr.mean()) if len(ag_arr) else float("nan")
    print(f"      aligned words: {n_direct}/{len(ref)} "
          f"(interpolated: {n_interp}), mean word logprob: {mean_sc:.2f}, "
          f"mean argmax agreement: {mean_ag:.3f}", flush=True)

    print("[5/5] building line cues + word timings ...", flush=True)
    pairs = [(i, i) for i in range(len(ref))
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
                "model": "manandey/wav2vec2-large-xlsr-punjabi (CTC)",
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
                    "words_direct_timing": n_direct,
                    "words_interpolated": n_interp,
                    "anchored_lines": len(hits),
                    "total_lines": len(lines),
                    "mean_word_logprob": round(mean_sc, 3),
                    "mean_argmax_agreement": round(mean_ag, 3),
                    "chars_skipped_by_path": n_skipped,
                    "words_with_skipped_chars": len(skipped_words),
                },
            },
            f, ensure_ascii=False, indent=1,
        )
    print(f"wrote {json_path}", flush=True)


if __name__ == "__main__":
    main()
