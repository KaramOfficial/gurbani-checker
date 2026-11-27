#!/usr/bin/env python3
"""Compute + cache CTC emissions for an audio file, then exit.
Used for long banis whose forced alignment runs separately
(chunk_align.py) — the full-file walk would OOM, but the emissions
themselves are computed in bounded 90s chunks by compute_emissions.
Writes <out>.emissions.npz in ctc_align.py's cache format."""
import argparse
import os
import sys

import librosa
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ctc_align import MODEL_PATH, compute_emissions  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
    print("loading model ...", flush=True)
    processor = Wav2Vec2Processor.from_pretrained(MODEL_PATH)
    model = Wav2Vec2ForCTC.from_pretrained(MODEL_PATH).eval()
    wav, _ = librosa.load(args.audio, sr=16000, mono=True)
    print(f"{len(wav) / 16000:.1f}s audio; computing emissions ...",
          flush=True)
    em = compute_emissions(model, processor, wav)
    cache = args.out + ".emissions.npz"
    np.savez(cache, emissions=em, n_samples=np.int64(len(wav)))
    print(f"cached {em.shape[0]} frames -> {cache}", flush=True)


if __name__ == "__main__":
    main()
