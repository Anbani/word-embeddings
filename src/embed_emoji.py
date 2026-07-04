#!/usr/bin/env python3
"""Stage 2c (optional) — build the emoji dataset (its own space + layout).

Emoji carry curated Georgian keywords (apps/emoji-static, vendored at
src/data/emoji/emoji-keywords.csv). Because those keywords ARE Georgian words,
an emoji's meaning vector can be derived WITHOUT the model: it is the
rank-weighted average of its keyword words' EmbeddingGemma 128-d vectors (from a
source word dataset). We then run the standard pipeline over those emoji vectors
— exact neighbours + UMAP — producing a first-class dataset `emoji-eg` that sits
in the dataset switcher next to the words and the poem. Emoji are close to each
other by shared keyword meaning; the web renders them as glyphs, not dots.

Neighbours are exact cosine among emoji (a real emoji-to-emoji space) — unlike a
raw word kNN off an emoji centroid, this is clean because both sides are emoji.

Needs no GPU/gated model (runs in the CPU build image). Output: dist/emoji-eg/
(standard schema) + emoji-kw.json sidecar (keywords per emoji, for tooltips).
"""
import argparse
import csv
import hashlib
import json
import os
import sys
import time
import unicodedata

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib  # noqa: E402
from reduce_layout import l2norm, exact_neighbors, run_umap, normalize_box  # noqa: E402
from build_dist import quantize_i8, pack_neighbors  # noqa: E402

DEFAULT_CSV = lib.path("src", "data", "emoji", "emoji-keywords.csv")
UMAP_CFG = {"n_neighbors": 30, "min_dist": 0.10, "n_components": 2, "metric": "cosine", "seed": 20260704}
DECAY = 0.65
TOKEN_RATIO = 0.5
KW_SHOW = 3


def nfc(s):
    return unicodedata.normalize("NFC", s)


def load_source(source_id):
    cfg = lib.load_config(source_id)
    corpus = cfg["corpus"]
    lay = np.load(lib.path("work", source_id, "layout.npz"), allow_pickle=True)
    vec128 = lay["vec128"].astype(np.float32)  # count × 128, L2-normed
    rows = sorted(lib.read_jsonl(lib.vocab_file(corpus)), key=lambda r: r["i"])
    words = [r["word"] for r in rows]
    freqs = [int(r.get("freq", 0)) for r in rows]
    word_index = {}
    for i, w in enumerate(words):
        word_index.setdefault(nfc(w), i)
    return vec128, freqs, word_index


def match_emoji(kws, word_index):
    weights = {}
    for rank, k in enumerate(kws):
        base = DECAY ** rank
        i = word_index.get(k)
        if i is not None:
            weights[i] = max(weights.get(i, 0.0), base)
            continue
        for tok in k.split():
            j = word_index.get(tok)
            if j is not None:
                weights[j] = max(weights.get(j, 0.0), base * TOKEN_RATIO)
    return weights


def main():
    ap = argparse.ArgumentParser(description="Build the emoji dataset")
    ap.add_argument("--source", default="kawiki-eg", help="word dataset providing the vectors")
    ap.add_argument("--id", default="emoji-eg")
    ap.add_argument("--name-ka", default="ემოჯი")
    ap.add_argument("--name-en", default="Emoji")
    ap.add_argument("--emoji-csv", default=DEFAULT_CSV)
    ap.add_argument("--min-coverage", type=float, default=0.5)
    args = ap.parse_args()

    vec128, freqs, word_index = load_source(args.source)
    rows = list(csv.DictReader(open(args.emoji_csv, encoding="utf-8")))
    lib.log(f"[{args.id}] {len(rows)} emoji, source '{args.source}' vocab {len(freqs)}")

    glyphs, vecs, gfreq, kws_show = [], [], [], []
    unplaced = 0
    for r in rows:
        raw_kws = [k.strip() for k in (r.get("keywords_ka") or "").split("|") if k.strip()]
        weights = match_emoji([nfc(k) for k in raw_kws], word_index)
        if not weights:
            unplaced += 1
            continue
        idx = np.fromiter(weights.keys(), dtype=np.int64)
        w = np.fromiter(weights.values(), dtype=np.float64)
        v = (vec128[idx] * (w / w.sum())[:, None]).sum(axis=0).astype(np.float32)
        nrm = float(np.linalg.norm(v))
        if nrm == 0:
            unplaced += 1
            continue
        glyphs.append(r["emoji"])
        vecs.append(v / nrm)
        gfreq.append(max(int(freqs[i]) for i in weights))
        kws_show.append(raw_kws[:KW_SHOW])

    n = len(glyphs)
    frac = n / len(rows)
    lib.log(f"[{args.id}] placed {n}/{len(rows)} ({frac*100:.1f}%), unplaced {unplaced}")
    if frac < args.min_coverage:
        raise SystemExit(f"coverage {frac*100:.1f}% < {args.min_coverage*100:.0f}% — aborting")

    E = np.stack(vecs).astype(np.float32)  # n × 128, L2-normed
    lib.log("exact emoji↔emoji neighbours…")
    nidx, nsim = exact_neighbors(E, 15)
    lib.log("UMAP over emoji vectors…")
    raw = run_umap(E, UMAP_CFG)
    points, _, _ = normalize_box(raw)

    out_dir = lib.path("dist", args.id)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "labels.tsv"), "w", encoding="utf-8") as f:
        for g, fr in zip(glyphs, gfreq):
            f.write(f"{g}\t{fr}\n")
    points.astype("<f4").reshape(-1).tofile(os.path.join(out_dir, "points.f32"))
    with open(os.path.join(out_dir, "neighbors.bin"), "wb") as f:
        f.write(pack_neighbors(nidx, nsim))
    q, scale = quantize_i8(E)
    q.tofile(os.path.join(out_dir, "vectors.i8"))
    scale.astype("<f4").tofile(os.path.join(out_dir, "vscales.f32"))
    with open(os.path.join(out_dir, "emoji-kw.json"), "w", encoding="utf-8") as f:
        json.dump({"v": 1, "kw": kws_show}, f, ensure_ascii=False, separators=(",", ":"))

    vhash = hashlib.sha256("".join(glyphs).encode("utf-8")).hexdigest()
    meta = {
        "v": 2, "id": args.id, "corpus": "emoji", "emoji": True,
        "model": {"id": f"keyword-avg({args.source})", "revision": "main"},
        "name_ka": args.name_ka, "name_en": args.name_en,
        "count": n, "dims_full": 128, "dims_i8": 128,
        "vocab_hash": vhash, "aligned_group": None, "reference": False,
        "source": args.source, "umap": UMAP_CFG,
        "displacement": {"median": 0.0, "p95": 0.0},
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # index.json: add/replace this dataset; scrub the old kawiki emoji-layer flag.
    idx_path = lib.path("dist", "index.json")
    index = json.load(open(idx_path, encoding="utf-8")) if os.path.exists(idx_path) else []
    index = [e for e in index if e["id"] != args.id]
    for e in index:
        e.pop("has_emoji", None)
        if isinstance(e.get("files"), dict):
            e["files"].pop("emoji.json", None)
    files = {fn: os.path.getsize(os.path.join(out_dir, fn))
             for fn in ("labels.tsv", "points.f32", "neighbors.bin", "vectors.i8", "vscales.f32")}
    index.append({
        "id": args.id, "corpus": "emoji", "model": args.name_en,
        "name_ka": args.name_ka, "name_en": args.name_en, "emoji": True,
        "count": n, "dims_full": 128, "dims_i8": 128,
        "aligned_group": None, "reference": False, "files": files,
    })
    index.sort(key=lambda e: e["id"])
    with open(idx_path, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)

    total = sum(files.values())
    lib.log(f"[{args.id}] dataset packed: {n} emoji, {total/1e3:.0f} KB -> {out_dir}")


if __name__ == "__main__":
    main()
