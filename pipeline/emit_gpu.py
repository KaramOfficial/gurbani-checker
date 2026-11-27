#!/usr/bin/env python3
"""GPU variant of emit_only.py for the Colab batch run.

Identical chunking/normalization to ctc_align.compute_emissions (copied
verbatim), except the model and inputs live on CUDA when available, so
emissions run at GPU speed. Falls back to CPU unchanged. Writes
<out>.emissions.npz in ctc_align.py's cache format.
"""
import argparse
import os
import sys

import librosa
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ctc_align import FRAME_S, MODEL_PATH  # noqa: E402


def compute_emissions_gpu(model, processor, wav, device):
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
            logits = model(inputs.input_values.to(device)).logits
        em_c = torch.log_softmax(logits, dim=-1).cpu().numpy().astype(np.float32)[0]
        f_s = 0 if pos == 0 else of
        f_e = em_c.shape[0] if e == n else em_c.shape[0] - of
        parts.append(em_c[f_s:f_e])
        pos += cs
    return np.concatenate(parts, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"loading model on {device} ...", flush=True)
    processor = Wav2Vec2Processor.from_pretrained(MODEL_PATH)
    model = Wav2Vec2ForCTC.from_pretrained(MODEL_PATH).eval().to(device)
    wav, _ = librosa.load(args.audio, sr=16000, mono=True)
    print(f"{len(wav) / 16000:.1f}s audio; computing emissions ...", flush=True)
    em = compute_emissions_gpu(model, processor, wav, device)
    cache = args.out + ".emissions.npz"
    np.savez(cache, emissions=em, n_samples=np.int64(len(wav)))
    print(f"cached {em.shape[0]} frames -> {cache}", flush=True)


if __name__ == "__main__":
    main()
