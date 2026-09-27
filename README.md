# Gurbani Subtitle Timestamper

Word-level Gurbani subtitle alignment checker + Colab batch alignment notebook (Sehaj Paath, Bhagat Jaswant Singh Ji).

- CTC forced alignment (manandey/wav2vec2-large-xlsr-punjabi), 20 ms frames
- **All 1430 Angs aligned** — complete
- Static page — open `index.html`, no build step; audio streams from media.gursevak.com; alignment data loads on demand from `chunks/`

Live checker: https://karamofficial.github.io/gurbani-checker/

## Data downloads (`data/`)

- `data/gurbani_ang1-1430_ctc.zip` — the complete set: per-Ang per-word JSON + line-level SRT (1430 JSON + 1430 SRT)

## Notebook

`Gurbani_Batch_Align.ipynb` — the batch alignment notebook used on Google Colab (T4 GPU).
