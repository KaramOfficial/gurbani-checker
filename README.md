# Gurbani Subtitle Timestamper

Word-level timestamped Gurbani subtitles for Sehaj Paath (Bhagat Jaswant Singh Ji) — an alignment checker web app plus the Colab batch-alignment notebook that produces the data.

## Checker web app

Live: https://karamofficial.github.io/gurbani-checker/

- Per-word JSON timestamps with line-anchored SRT subtitles, Angs 1–550
- CTC forced alignment (manandey/wav2vec2-large-xlsr-punjabi), 20 ms frames
- Audio streams from media.gursevak.com; static page, no build step

## Alignment notebook

`Gurbani_Batch_Align.ipynb` — the batch pipeline run on Google Colab (GPU):

- 100 Angs per batch, 2 workers, per-word JSON + SRT output
- Change only `ANG_START` / `ANG_END` per batch; re-upload only when pipeline code changes

## Pipeline notes

- ੴ expands to ਇਕ ਓਅੰਕਾਰ; vocab fixups: ਞ→ਨ, ਙ→ਨ, ਃ→ਹ, ੵ→ਯ, ੑ dropped, ੲ→ਇ
- Per-word JSON is authoritative; SRT is secondary
