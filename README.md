# Gurbani Subtitle Timestamper

Word-level Gurbani subtitle alignment checker + Colab batch alignment notebook (Sehaj Paath, Bhagat Jaswant Singh Ji).

- CTC forced alignment (manandey/wav2vec2-large-xlsr-punjabi), 20 ms frames
- Angs 1–1150 aligned so far (target: all 1430); the checker app currently covers 1–1050, 1150 coming next
- Static page — open `index.html`, no build step; audio streams from media.gursevak.com; alignment data loads on demand from `chunks/`

Live checker: https://karamofficial.github.io/gurbani-checker/

## Data downloads (`data/`)

Each batch ships one cumulative ZIP with per-Ang per-word JSON + line-level SRT:

- `data/gurbani_ang1-1150_ctc.zip` — all aligned Angs so far (1150 JSON + 1150 SRT)

Only the latest ZIP is kept; it is replaced as new batches land.
Individual `data/ang<N>.json` files (Angs 1–64) were an early upload experiment and
are superseded by the cumulative ZIPs.

## Notebook

`Gurbani_Batch_Align.ipynb` — the batch alignment notebook used on Google Colab (T4 GPU).
