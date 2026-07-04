#!/usr/bin/env python3
"""Stage 1b.5 — collapse inflectional variants to one representative per root.

The Hunspell form→lemma map under-collapses Georgian morphology, leaving many
inflectional variants (დედა, დედამ, დედათა…) as separate vocab points. They
crowd the semantic neighbor lists (token-span vectors share subwords) and split
the map. This groups vocab entries by `georgian_stem.stem`, keeps the
highest-frequency surface form as the representative label, sums frequencies,
and merges occurrences — producing the FROZEN vocab that every model on this
corpus shares.

Pipeline position: corpus → vocab → **collapse** → embed → reduce → dist.
For the already-embedded EmbeddingGemma run we pass `--vectors` so the existing
30K vectors are subselected to the representative rows (no re-embed). For a fresh
model (Gemma 4), collapse runs before embed and `--vectors` is omitted.

Overwrites work/<corpus>/vocab.jsonl (backup: vocab.pre_collapse.jsonl).
"""
import argparse
import hashlib
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib  # noqa: E402
import georgian_stem as gs  # noqa: E402

OCC_CAP = 400
SEED = 20260704


def main():
    ap = argparse.ArgumentParser(description="Collapse vocab by Georgian root")
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--vectors", default="", help="comma-sep .npy paths to subselect to reps")
    ap.add_argument("--target", type=int, default=0, help="optional cap on collapsed size")
    args = ap.parse_args()

    vpath = lib.path("work", args.corpus, "vocab.jsonl")   # raw vocab (source)
    rows = sorted(lib.read_jsonl(vpath), key=lambda r: r["i"])
    n_before = len(rows)

    groups = {}
    for r in rows:
        groups.setdefault(gs.stem(r["word"]), []).append(r)

    reps = []
    for root in sorted(groups):
        members = sorted(groups[root], key=lambda r: (-r["freq"], r["word"]))
        rep = members[0]
        freq = sum(m["freq"] for m in members)
        occ = []
        for m in members:
            occ.extend(m["occ"])
        if len(occ) > OCC_CAP:
            seed = SEED ^ int(hashlib.sha256(root.encode("utf-8")).hexdigest()[:8], 16)
            random.Random(seed).shuffle(occ)
            occ = occ[:OCC_CAP]
        reps.append({"root": root, "word": rep["word"], "freq": freq, "occ": occ, "src_i": rep["i"]})

    reps.sort(key=lambda r: (-r["freq"], r["word"]))
    if args.target and len(reps) > args.target:
        reps = reps[:args.target]

    labels = [r["word"] for r in reps]
    vocab_hash = lib.sha256_text("\n".join(labels))
    keep_rows = [r["src_i"] for r in reps]

    # Write the frozen collapsed vocab to a SEPARATE file (idempotent — always
    # regenerated from the raw vocab.jsonl; never destroys the source).
    out = lib.path("work", args.corpus, "vocab.collapsed.jsonl")
    with open(out, "w", encoding="utf-8") as f:
        for i, r in enumerate(reps):
            f.write(json.dumps({"i": i, "word": r["word"], "lemma": r["root"],
                                "freq": r["freq"], "occ": r["occ"]}, ensure_ascii=False) + "\n")

    meta_path = lib.path("work", args.corpus, "vocab_meta.json")
    meta = json.load(open(meta_path, encoding="utf-8")) if os.path.exists(meta_path) else {}
    meta.update({"corpus": args.corpus, "count": len(reps), "vocab_hash": vocab_hash,
                 "collapsed_from": n_before})
    json.dump(meta, open(meta_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    with open(lib.path("work", args.corpus, "collapse_rows.json"), "w") as f:
        json.dump(keep_rows, f)

    lib.log(f"[{args.corpus}] collapsed {n_before} -> {len(reps)} roots; vocab_hash {vocab_hash[:12]}…")

    # Subselect already-computed vectors to the representative rows (no re-embed).
    # Source is always the full pre-collapse backup, so this is idempotent.
    if args.vectors:
        import numpy as np
        for vp in [v.strip() for v in args.vectors.split(",") if v.strip()]:
            bak = vp.replace(".npy", ".pre_collapse.npy")
            if not os.path.exists(bak):
                if not os.path.exists(vp):
                    continue
                os.rename(vp, bak)
            V = np.load(bak)
            if V.shape[0] != n_before:
                lib.log(f"  WARN {vp}: rows {V.shape[0]} != pre-collapse {n_before}; skipping")
                continue
            np.save(vp, V[keep_rows])
            lib.log(f"  subselected {vp}: {n_before} -> {len(keep_rows)}")


if __name__ == "__main__":
    main()
