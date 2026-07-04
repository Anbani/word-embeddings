#!/usr/bin/env python3
"""Stage 2c (optional) — project the emoji set into an existing word layout.

Emoji carry curated Georgian keywords (apps/emoji-static, vendored here at
src/data/emoji/emoji-keywords.csv). Because those keywords ARE Georgian words,
most already live in a word dataset's vocab — so an emoji can be placed WITHOUT
re-running the model or UMAP: look its keywords up in the vocab and set its 2D
position to a rank-weighted blend of the matched words' coordinates. The same
matched words, most-relevant first, are the emoji's neighbour words.

Neighbours are the curated keyword words on purpose, NOT cosine-nearest words in
the 128-d space: EmbeddingGemma's token vectors carry a strong orthographic
signal, so raw kNN off an emoji centroid returns look-alikes (😀→გილგამეში) not
meanings. The curated keywords are clean by construction.

This "keywords" method needs no GPU and no gated model, so it runs in the CPU
build image alongside reduce/dist. It is meaningful only where the vocab covers
the emoji keyword space (kawiki: ~97% of emoji match; the archaic vef poem does
not — do not build an emoji layer there).

  method=embed (future): embed each emoji's keyword phrase with the dataset's
  EmbeddingGemma, MRL-truncate to 128-d, kNN in 128-d. Higher fidelity, needs
  the generate image / GPU node. Not implemented yet — keywords is the shipping
  path.

Output: dist/<ds>/emoji.json  (docs/dist-schema.md). Sets has_emoji in index.json.
"""
import argparse
import csv
import hashlib
import json
import os
import sys
import unicodedata

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib  # noqa: E402

DEFAULT_CSV = lib.path("src", "data", "emoji", "emoji-keywords.csv")

# placement weights: keyword rank r contributes DECAY**r; a within-keyword token
# match (multi-word keyword) counts for less than a whole-keyword hit.
DECAY = 0.65
TOKEN_RATIO = 0.5
NN_K = 15          # neighbour words shipped per emoji (its matched keywords)
KW_SHOW = 3        # ka keywords kept for the tooltip
JITTER = 0.006     # unstack co-located emoji (weighted centroid can collapse to
                   # one word); spread grows with the pile size, deterministic.


def nfc(s):
    return unicodedata.normalize("NFC", s)


def load_vocab_words(corpus):
    rows = sorted(lib.read_jsonl(lib.vocab_file(corpus)), key=lambda r: r["i"])
    return [r["word"] for r in rows]


def match_emoji(kws, word_index):
    """kws = ordered ka keywords. Return {word_i: weight} (max over occurrences)."""
    weights = {}
    for rank, k in enumerate(kws):
        base = DECAY ** rank
        i = word_index.get(k)
        if i is not None:
            weights[i] = max(weights.get(i, 0.0), base)
            continue
        # multi-word keyword: fall back to its individual tokens
        for tok in k.split():
            j = word_index.get(tok)
            if j is not None:
                weights[j] = max(weights.get(j, 0.0), base * TOKEN_RATIO)
    return weights


def main():
    ap = argparse.ArgumentParser(description="Project emoji into a word layout")
    ap.add_argument("--config", required=True, help="dataset id, e.g. kawiki-eg")
    ap.add_argument("--method", default="keywords", choices=["keywords", "embed"])
    ap.add_argument("--emoji-csv", default=DEFAULT_CSV)
    ap.add_argument("--min-coverage", type=float, default=0.5,
                    help="abort if fewer than this fraction of emoji get placed")
    args = ap.parse_args()

    if args.method == "embed":
        raise SystemExit("method=embed not implemented yet — use the default "
                         "keywords method (needs no model/GPU).")

    cfg = lib.load_config(args.config)
    ds = cfg["id"]
    corpus = cfg["corpus"]

    lay = np.load(lib.path("work", ds, "layout.npz"), allow_pickle=True)
    points = lay["points"].astype(np.float32)         # count × 2, in [-1,1]^2
    count = points.shape[0]

    words = load_vocab_words(corpus)
    if len(words) != count:
        raise SystemExit(f"vocab/layout mismatch: {len(words)} vs {count}")
    word_index = {}
    for i, w in enumerate(words):
        word_index.setdefault(nfc(w), i)              # first (most frequent) wins

    rows = list(csv.DictReader(open(args.emoji_csv, encoding="utf-8")))
    lib.log(f"[{ds}] {len(rows)} emoji, vocab {count}")

    placed = []          # (emoji, x, y, kw_show, nn)
    unplaced = 0
    for r in rows:
        raw_kws = [k.strip() for k in (r.get("keywords_ka") or "").split("|") if k.strip()]
        weights = match_emoji([nfc(k) for k in raw_kws], word_index)
        if not weights:
            unplaced += 1
            continue
        idx = np.fromiter(weights.keys(), dtype=np.int64)
        w = np.fromiter(weights.values(), dtype=np.float64)
        wn = w / w.sum()
        xy = (points[idx] * wn[:, None]).sum(axis=0)
        # neighbours = matched keyword words, most-relevant first (weight desc)
        ranked = sorted(weights.items(), key=lambda kv: -kv[1])[:NN_K]
        nn = [[int(i), round(float(wt), 3)] for i, wt in ranked]
        placed.append((r["emoji"], float(xy[0]), float(xy[1]), raw_kws[:KW_SHOW], nn))

    n = len(placed)
    frac = n / len(rows)
    lib.log(f"[{ds}] placed {n}/{len(rows)} ({frac*100:.1f}%), unplaced {unplaced}")
    if frac < args.min_coverage:
        raise SystemExit(f"coverage {frac*100:.1f}% < {args.min_coverage*100:.0f}% "
                         f"— emoji layer not meaningful for '{ds}', skipping")

    # de-stack: emoji whose centroid collapsed onto the same word get a small,
    # deterministic jitter scaled by how many share the spot (bigger pile → wider).
    from collections import defaultdict
    groups = defaultdict(list)
    for i, p in enumerate(placed):
        groups[(round(p[1], 3), round(p[2], 3))].append(i)
    xs = [p[1] for p in placed]
    ys = [p[2] for p in placed]
    for members in groups.values():
        if len(members) < 2:
            continue
        radius = min(0.05, JITTER * (len(members) ** 0.5))
        for i in members:
            h = int(hashlib.md5(placed[i][0].encode("utf-8")).hexdigest()[:8], 16)
            ang = (h & 0xFFFF) / 0xFFFF * 2 * np.pi
            rad = ((h >> 16) & 0xFFFF) / 0xFFFF * radius
            xs[i] = float(np.clip(placed[i][1] + rad * np.cos(ang), -1.0, 1.0))
            ys[i] = float(np.clip(placed[i][2] + rad * np.sin(ang), -1.0, 1.0))

    out = {
        "v": 1, "ds": ds, "method": "keywords", "count": n,
        "emoji": [p[0] for p in placed],
        "x": [round(v, 4) for v in xs],
        "y": [round(v, 4) for v in ys],
        "kw": [p[3] for p in placed],
        "nn": [p[4] for p in placed],
    }
    out_path = lib.path("dist", ds, "emoji.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, separators=(",", ":"))
    size = os.path.getsize(out_path)
    lib.log(f"[{ds}] emoji.json -> {out_path} ({size/1e3:.0f} KB)")

    # flag it in the catalog so the web app can show the layer toggle
    idx_path = lib.path("dist", "index.json")
    if os.path.exists(idx_path):
        index = json.load(open(idx_path, encoding="utf-8"))
        for e in index:
            if e["id"] == ds:
                e["has_emoji"] = True
                e.setdefault("files", {})["emoji.json"] = size
        with open(idx_path, "w", encoding="utf-8") as f:
            json.dump(index, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
