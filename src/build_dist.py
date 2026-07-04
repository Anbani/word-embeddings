#!/usr/bin/env python3
"""Stage 2b — pack the committed dist/<ds>/ artifact (schema v2).

Reads work/<ds>/layout.npz + work/<corpus>/vocab.jsonl + the dataset config.
Writes the little-endian binary files consumed directly by the browser
(docs/dist-schema.md). Never touches the large raw vectors.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib  # noqa: E402


def quantize_i8(vec128):
    """Per-row symmetric int8: scale = max(|v|)/127, q = round(v/scale)."""
    scale = np.abs(vec128).max(axis=1) / 127.0
    scale[scale == 0] = 1.0
    q = np.rint(vec128 / scale[:, None]).clip(-127, 127).astype(np.int8)
    return q, scale.astype(np.float32)


def pack_neighbors(nidx, nsim):
    dt = np.dtype([("idx", "<u2"), ("sim", "u1")])   # packed 3 bytes, no padding
    rec = np.zeros(nidx.shape, dtype=dt)
    rec["idx"] = nidx
    rec["sim"] = nsim
    assert rec.dtype.itemsize == 3, "neighbor record must be 3 bytes"
    return rec.tobytes(order="C")


def main():
    ap = argparse.ArgumentParser(description="Pack dist/<ds>/ (schema v2)")
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = lib.load_config(args.config)
    ds = cfg["id"]
    corpus = cfg["corpus"]

    lay = np.load(lib.path("work", ds, "layout.npz"), allow_pickle=True)
    points = lay["points"].astype("<f4")
    nidx = lay["nidx"]
    nsim = lay["nsim"]
    vec128 = lay["vec128"].astype(np.float32)
    d_model = int(lay["d_model"])
    count = points.shape[0]

    vocab = sorted(lib.read_jsonl(lib.vocab_file(corpus)), key=lambda r: r["i"])
    if len(vocab) != count:
        raise SystemExit(f"vocab/layout count mismatch: {len(vocab)} vs {count}")
    vmeta = json.load(open(lib.path("work", corpus, "vocab_meta.json"), encoding="utf-8"))

    out_dir = lib.path("dist", ds)
    os.makedirs(out_dir, exist_ok=True)

    # labels.tsv
    with open(os.path.join(out_dir, "labels.tsv"), "w", encoding="utf-8") as f:
        for r in vocab:
            f.write(f"{r['word']}\t{r['freq']}\n")

    # points.f32 (x,y interleaved)
    points.reshape(-1).tofile(os.path.join(out_dir, "points.f32"))

    # neighbors.bin
    with open(os.path.join(out_dir, "neighbors.bin"), "wb") as f:
        f.write(pack_neighbors(nidx, nsim))

    # vectors.i8 + vscales.f32
    q, scale = quantize_i8(vec128)
    q.tofile(os.path.join(out_dir, "vectors.i8"))
    scale.astype("<f4").tofile(os.path.join(out_dir, "vscales.f32"))

    embed_cfg = cfg["embed"]
    rcfg = cfg["reduce"]
    layer = embed_cfg.get("layer", embed_cfg.get("layer_frac"))
    meta = {
        "v": 2,
        "id": ds,
        "corpus": corpus,
        "model": {"id": cfg["model"]["id"], "revision": cfg["model"].get("revision", "main")},
        "name_ka": cfg.get("name_ka", ds),
        "name_en": cfg.get("name_en", ds),
        "prompt": embed_cfg.get("prompt_template", "{text}"),
        "layer": layer,
        "N": embed_cfg.get("N", 64),
        "per_article_cap": embed_cfg.get("per_article_cap", 2),
        "window_tokens": embed_cfg.get("window_tokens", 96),
        "seeds": {"sample": embed_cfg.get("seed"), "umap": rcfg["umap"].get("seed")},
        "dim128_method": rcfg.get("dim128_method", "mrl"),
        "umap": rcfg["umap"],
        "vocab_hash": vmeta["vocab_hash"],
        "config_hash": lib.config_hash(cfg),
        "count": count,
        "dims_full": d_model,
        "dims_i8": 128,
        "aligned_group": rcfg.get("aligned_group"),
        "reference": bool(rcfg.get("reference") == ds),
        "displacement": {"median": float(lay["disp_median"]), "p95": float(lay["disp_p95"])},
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # update dist/index.json
    idx_path = lib.path("dist", "index.json")
    index = []
    if os.path.exists(idx_path):
        index = json.load(open(idx_path, encoding="utf-8"))
    index = [e for e in index if e["id"] != ds]
    files = {fn: os.path.getsize(os.path.join(out_dir, fn))
             for fn in ("labels.tsv", "points.f32", "neighbors.bin", "vectors.i8", "vscales.f32")}
    index.append({
        "id": ds, "corpus": corpus, "model": meta["name_en"],
        "name_ka": meta["name_ka"], "name_en": meta["name_en"],
        "count": count, "dims_full": d_model, "dims_i8": 128,
        "aligned_group": meta["aligned_group"], "reference": meta["reference"],
        "files": files,
    })
    index.sort(key=lambda e: e["id"])
    with open(idx_path, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)

    total = sum(files.values()) + os.path.getsize(os.path.join(out_dir, "meta.json"))
    lib.log(f"[{ds}] dist packed: {count} words, {total/1e6:.2f} MB -> {out_dir}")


if __name__ == "__main__":
    main()
