# V2 timestamp pipeline (Colab batch)

`v2_colab.ipynb` recomputes V2 fixed timestamps for all 1430 Angs on a
Colab GPU: per Ang it builds the CTC baseline from this repo's published
chunks, downloads the Sehaj Paath audio + Gursevak text, computes CTC
emissions (GPU), runs the V2.1 alignment, validates the result, and saves
`angN.v2.json` to Google Drive (MyDrive/gurbani-v2/v2). Resumable —
finished Angs are skipped on re-run.

Open it in Colab:
https://colab.research.google.com/github/KaramOfficial/gurbani-checker/blob/main/pipeline/v2_colab.ipynb
