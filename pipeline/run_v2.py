#!/usr/bin/env python3
"""Batch V2.1 timestamps for a range of Angs (built for Google Colab GPU,
runs anywhere the timestamper venv works).

Per Ang: build app_data/angN.json from the published Checker chunk (the
CTC baseline), download the Sehaj Paath audio, compute CTC emissions
(emit_only.py), run the V2.1 alignment (align_v2.py), validate the result
(text identical to CTC, line count, no collapsed words), and copy the
finished angN.v2.json to --out-dir.

Resumable: an Ang whose v2.json already exists in --out-dir is skipped,
so a dead Colab session can simply be re-run.

Usage: python run_v2.py --start 1 --end 1430 --out-dir /path/to/results
"""
import argparse
import csv
import io
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request

BASE = os.path.dirname(os.path.abspath(__file__))
CHUNK_URL = "https://karamofficial.github.io/gurbani-checker/chunks/chunk-{lo}-{hi}.json"
AUDIO_URL = "https://media.gursevak.com/media/Sehaj_Paath_Pagewise/Sehaj_Paath_16kbps/{ang:04d}.mp3"
TEXT_URL = "https://readgurbani.com/data/sggs/gurbani/{ang}.csv"
UA = {"User-Agent": "Mozilla/5.0 (compatible; gurbani-v2-batch)"}


def fetch(url, dest=None, retries=4):
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=120) as r:
                data = r.read()
            if dest:
                with open(dest, "wb") as f:
                    f.write(data)
            return data
        except Exception as e:
            last = e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"GET failed after {retries} tries: {url} ({last})")


class Chunks:
    def __init__(self):
        self.cache = {}

    def ang(self, ang):
        lo = (ang - 1) // 100 * 100 + 1
        hi = min(lo + 99, 1430)
        if lo not in self.cache:
            raw = fetch(CHUNK_URL.format(lo=lo, hi=hi))
            self.cache = {lo: json.loads(raw)}
        return self.cache[lo]["angs"][str(ang)]


def run(cmd):
    print("  $", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def validate(ctc, v2):
    if len(ctc["lines"]) != len(v2["lines"]):
        return f"line count {len(v2['lines'])} != CTC {len(ctc['lines'])}"
    cw = [w["word"] for l in ctc["lines"] for w in l["words"]]
    vw = [w["word"] for l in v2["lines"] for w in l["words"]]
    if cw != vw:
        return "word text differs from CTC"
    vwords = [x for l in v2["lines"] for x in l["words"]]
    short = [w for w in vwords if w["end"] - w["start"] < 0.12]
    if short:
        return f"{len(short)} collapsed words remain"
    prev = -1.0
    for l in v2["lines"]:
        for w in l["words"]:
            if w["start"] < prev - 1e-9:
                return "non-monotonic word order"
            prev = max(prev, w["end"])
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, required=True)
    ap.add_argument("--end", type=int, required=True)
    ap.add_argument("--out-dir", required=True,
                    help="finished angN.v2.json files land here (Drive on Colab)")
    ap.add_argument("--keep-emissions", action="store_true")
    ap.add_argument("--emitter", default="emit_only.py",
                    help="emissions script (emit_gpu.py on Colab)")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    for d in ("app_data", "audio", "output"):
        os.makedirs(os.path.join(BASE, d), exist_ok=True)
    for d in ("text",):
        os.makedirs(os.path.join(BASE, d), exist_ok=True)
    chunks = Chunks()
    py = sys.executable
    done = skipped = failed = 0
    t0 = time.time()
    todo = [a for a in range(args.start, args.end + 1)]
    for i, ang in enumerate(todo):
        dest = os.path.join(args.out_dir, f"ang{ang}.v2.json")
        if os.path.exists(dest):
            skipped += 1
            continue
        print(f"[{i + 1}/{len(todo)}] Ang {ang} ...", flush=True)
        ta = time.time()
        try:
            ctc = chunks.ang(ang)
            with open(os.path.join(BASE, "app_data", f"ang{ang}.json"), "w",
                      encoding="utf-8") as f:
                json.dump(ctc, f, ensure_ascii=False)
            txt = os.path.join(BASE, "text", f"ang{ang}.txt")
            if not os.path.exists(txt):
                raw = fetch(TEXT_URL.format(ang=ang)).decode("utf-8")
                if not raw.lstrip("﻿ \t\r\n").startswith("id\t"):
                    raise RuntimeError(f"ang {ang}: text response is not the expected TSV")
                rows = list(csv.DictReader(io.StringIO(raw), delimiter="\t"))
                with open(txt, "w", encoding="utf-8") as f:
                    for r in rows:
                        line = (r.get("gurmukhi") or "").strip()
                        if line:
                            f.write(line + "\n")
            mp3 = os.path.join(BASE, "audio", f"ang{ang}.mp3")
            if not os.path.exists(mp3):
                fetch(AUDIO_URL.format(ang=ang), mp3)
            run(["ffmpeg", "-y", "-v", "error", "-i", mp3,
                 "-ar", "16000", "-ac", "1", f"/tmp/ang{ang}.wav"])
            run([py, os.path.join(BASE, args.emitter), "--audio", mp3,
                 "--out", os.path.join(BASE, "output", f"ang{ang}.ctc")])
            run([py, os.path.join(BASE, "align_v2.py"), str(ang)])
            out = os.path.join(BASE, "output", f"ang{ang}.v2.json")
            with open(out, encoding="utf-8") as f:
                v2 = json.load(f)
            err = validate(ctc, v2)
            if err:
                print(f"  VALIDATION FAILED: {err} — not publishing", flush=True)
                failed += 1
                continue
            v2["audio"] = f"audio/ang{ang}.mp3"   # align_v2 writes template labels
            v2["text"] = f"text/ang{ang}.txt"
            with open(dest, "w", encoding="utf-8") as f:
                json.dump(v2, f, ensure_ascii=False)
            done += 1
            el = time.time() - t0
            left = len(todo) - i - 1
            eta = el / max(1, done + failed) * left / 60 if (done + failed) else 0
            print(f"  OK in {time.time() - ta:.0f}s · done {done}, skipped {skipped}, "
                  f"failed {failed} · elapsed {el / 60:.1f}m · ETA ~{eta:.0f}m",
                  flush=True)
        except Exception as e:  # keep the batch alive on a single bad Ang
            failed += 1
            print(f"  FAILED: {type(e).__name__}: {e}", flush=True)
        finally:
            if not args.keep_emissions:
                for p in (os.path.join(BASE, "output", f"ang{ang}.ctc.emissions.npz"),
                          os.path.join(BASE, "output", f"ang{ang}.ctc.emissions.npy")):
                    if os.path.exists(p):
                        os.remove(p)
    print(f"FINISHED: done {done}, skipped {skipped}, failed {failed}, "
          f"total {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
