#!/usr/bin/env python3
"""Stage 3 — the gate. Validates the committed dist/ and runs the probe eval.

Runs in CI from the commit alone: probe metrics read dist/neighbors.bin, so no
raw vectors are needed. Exits non-zero on any failure (blocks the dist commit).
If dist/ is absent (fresh repo before the first generate), it skips cleanly.

Checks (SDD §7, §9): schema/sizes, vocab_hash equality across kawiki datasets,
neighbor indices in range, no NaN/Inf, no blocklisted labels, alignment
displacement bound, total size <= 30 MB, probe hit-rate + zero forbidden hits.
"""
import json
import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib  # noqa: E402
from vendor import normalize as norm  # noqa: E402

SIZE_BUDGET_MB = 30
DISP_BOUND = 0.35
FAILURES = []


def fail(msg):
    FAILURES.append(msg)
    lib.log(f"  FAIL: {msg}")


def read_labels(d):
    words, freqs = [], []
    for line in open(os.path.join(d, "labels.tsv"), encoding="utf-8"):
        line = line.rstrip("\n")
        if not line:
            continue
        w, _, fr = line.partition("\t")
        words.append(w)
        freqs.append(int(fr) if fr else 0)
    return words, freqs


def read_neighbors(d, count):
    raw = open(os.path.join(d, "neighbors.bin"), "rb").read()
    rec = 3
    if count and len(raw) % (rec * count) != 0:
        return None, None
    k = len(raw) // (rec * count) if count else 0
    idx = [[0] * k for _ in range(count)]
    sim = [[0] * k for _ in range(count)]
    for i in range(count):
        base = i * k * rec
        for j in range(k):
            o = base + j * rec
            idx[i][j] = struct.unpack_from("<H", raw, o)[0]
            sim[i][j] = raw[o + 2]
    return idx, k


def has_bad_floats(path_):
    import array
    a = array.array("f")
    a.frombytes(open(path_, "rb").read())
    for x in a:
        if x != x or x in (float("inf"), float("-inf")):
            return True
    return False


def load_blocklist():
    p = lib.path("src", "data", "vocab_blocklist.txt")
    out = set()
    if os.path.exists(p):
        for line in open(p, encoding="utf-8"):
            line = line.strip()
            if line and not line.startswith("#"):
                out.add(norm.fold(line))
    return out


def check_dataset(ds, blocklist):
    d = lib.path("dist", ds)
    meta = json.load(open(os.path.join(d, "meta.json"), encoding="utf-8"))
    count = meta["count"]
    words, _ = read_labels(d)
    if len(words) != count:
        fail(f"[{ds}] labels {len(words)} != count {count}")

    exp = {"points.f32": 8 * count, "vectors.i8": meta["dims_i8"] * count,
           "vscales.f32": 4 * count}
    for fn, want in exp.items():
        got = os.path.getsize(os.path.join(d, fn))
        if got != want:
            fail(f"[{ds}] {fn} size {got} != {want}")

    if has_bad_floats(os.path.join(d, "points.f32")):
        fail(f"[{ds}] NaN/Inf in points.f32")
    if has_bad_floats(os.path.join(d, "vscales.f32")):
        fail(f"[{ds}] NaN/Inf in vscales.f32")

    nidx, k = read_neighbors(d, count)
    if nidx is None:
        fail(f"[{ds}] neighbors.bin not a multiple of 3*count")
    else:
        for i in range(count):
            for j in range(k):
                if nidx[i][j] >= count:
                    fail(f"[{ds}] neighbor index {nidx[i][j]} >= count at word {i}")
                    break

    bad = [w for w in words if norm.fold(w) in blocklist]
    if bad:
        fail(f"[{ds}] blocklisted labels present: {bad[:5]}…")

    if meta.get("aligned_group") and not meta.get("reference"):
        dm = meta.get("displacement", {}).get("median", 0.0)
        if dm >= DISP_BOUND:
            fail(f"[{ds}] alignment displacement median {dm:.3f} >= {DISP_BOUND}")

    return meta, words, nidx, k


def run_probes(probes, datasets):
    """datasets: {corpus: [(ds, words, nidx, k), ...]}"""
    for corpus, groups in probes.items():
        if corpus in ("v", "note", "thresholds"):
            continue
        thr = probes.get("thresholds", {}).get(corpus, {}).get("hit_rate", 0.70)
        for ds, words, nidx, k in datasets.get(corpus, []):
            index = {w: i for i, w in enumerate(words)}
            present = hits = forbidden = 0
            for probe in groups:
                w = probe["word"]
                if w not in index:
                    continue
                present += 1
                nb = {words[nidx[index[w]][j]] for j in range(k)}
                if set(probe.get("expected", [])) & nb:
                    hits += 1
                if set(probe.get("forbidden", [])) & nb:
                    forbidden += 1
            rate = hits / present if present else 0.0
            lib.log(f"  [{ds}] probes: {hits}/{present} hit ({rate:.0%}), forbidden={forbidden}")
            if present and rate < thr:
                fail(f"[{ds}] probe hit-rate {rate:.0%} < {thr:.0%}")
            if forbidden:
                fail(f"[{ds}] {forbidden} forbidden neighbor(s) present")


def main():
    idx_path = lib.path("dist", "index.json")
    if not os.path.exists(idx_path):
        lib.log("no dist/index.json yet — skipping eval (nothing generated).")
        return
    index = json.load(open(idx_path, encoding="utf-8"))
    blocklist = load_blocklist()

    total = 0
    for root, _, files in os.walk(lib.path("dist")):
        for fn in files:
            total += os.path.getsize(os.path.join(root, fn))
    if total > SIZE_BUDGET_MB * 1e6:
        fail(f"dist total {total/1e6:.1f}MB > {SIZE_BUDGET_MB}MB budget")

    datasets = {}
    metas = {}
    for entry in index:
        ds = entry["id"]
        lib.log(f"checking {ds}…")
        meta, words, nidx, k = check_dataset(ds, blocklist)
        metas[ds] = meta
        datasets.setdefault(meta["corpus"], []).append((ds, words, nidx, k))

    # vocab_hash equality within each aligned group
    groups = {}
    for ds, m in metas.items():
        g = m.get("aligned_group")
        if g:
            groups.setdefault(g, []).append((ds, m["vocab_hash"]))
    for g, items in groups.items():
        hashes = {h for _, h in items}
        if len(hashes) > 1:
            fail(f"aligned group '{g}' has mismatched vocab_hash: {[d for d, _ in items]}")

    probes = json.load(open(lib.path("eval", "probes.json"), encoding="utf-8"))
    run_probes(probes, datasets)

    if FAILURES:
        lib.log(f"\n{len(FAILURES)} FAILURE(S). dist is NOT valid.")
        sys.exit(1)
    lib.log("\nAll checks passed.")


if __name__ == "__main__":
    main()
