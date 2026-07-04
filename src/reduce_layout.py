#!/usr/bin/env python3
"""Stage 2a — dimensionality reduction, exact neighbors, and 2D layout.

Consumes work/<ds>/vectors.f32.npy (full-dim, L2-normalized). Produces
work/<ds>/layout.npz with everything build_dist.py needs (so build_dist never
touches the large raw vectors):
  points   count×2  float32   final [-1,1]^2 layout
  nidx     count×k  uint16    top-k neighbor indices (full-dim cosine)
  nsim     count×k  uint8     quantized similarities
  vec128   count×128 float32  L2-normalized shipping vectors (int8-quantized later)

Landmines honored:
  * #3  128d = MRL truncation for EmbeddingGemma, PCA->128 for Gemma 4.
  * #4  aligned datasets: UMAP init=<reference coords> THEN orthogonal Procrustes.
  * neighbors are computed in FULL original dim, before any reduction (SDD §7).
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib  # noqa: E402


def l2norm(M):
    n = np.linalg.norm(M, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return M / n


# --------------------------------------------------------------------------- 128d

def reduce_128(vectors, method):
    if method == "mrl":                     # EmbeddingGemma: Matryoshka prefix
        v = vectors[:, :128].astype(np.float32)
        return l2norm(v)
    if method == "pca":                     # Gemma 4 hidden states
        from sklearn.decomposition import PCA
        p = PCA(n_components=128, whiten=False, svd_solver="full")
        v = p.fit_transform(vectors.astype(np.float64)).astype(np.float32)
        return l2norm(v)
    raise SystemExit(f"unknown dim128_method: {method}")


# --------------------------------------------------------------------------- neighbors

def exact_neighbors(vectors, k, block=2048):
    """Top-k cosine neighbors in FULL dim (vectors are L2-normalized -> dot=cos)."""
    count = vectors.shape[0]
    nidx = np.zeros((count, k), dtype=np.uint16)
    nsim = np.zeros((count, k), dtype=np.uint8)
    V = vectors.astype(np.float32)
    for start in range(0, count, block):
        B = V[start:start + block]
        S = B @ V.T                          # (b × count) cosine
        rows = np.arange(B.shape[0])
        S[rows, start + rows] = -np.inf      # exclude self
        top = np.argpartition(-S, k, axis=1)[:, :k]
        # sort the k by similarity desc
        part = np.take_along_axis(S, top, axis=1)
        order = np.argsort(-part, axis=1)
        top = np.take_along_axis(top, order, axis=1)
        sims = np.take_along_axis(part, order, axis=1)
        nidx[start:start + B.shape[0]] = top.astype(np.uint16)
        q = np.clip(sims, 0.0, 1.0) * 255.0
        nsim[start:start + B.shape[0]] = np.rint(q).astype(np.uint8)
        lib.log(f"  neighbors {min(start + block, count)}/{count}")
    return nidx, nsim


# --------------------------------------------------------------------------- layout

def run_umap(vectors, umap_cfg, init=None):
    import umap
    kwargs = dict(
        n_components=umap_cfg.get("n_components", 2),
        n_neighbors=umap_cfg.get("n_neighbors", 30),
        min_dist=umap_cfg.get("min_dist", 0.08),
        metric=umap_cfg.get("metric", "cosine"),
        random_state=umap_cfg.get("seed", 20260704),
        verbose=True,
    )
    if init is not None:
        kwargs["init"] = init.astype(np.float32)
    reducer = umap.UMAP(**kwargs)
    return reducer.fit_transform(vectors).astype(np.float64)


def procrustes_onto(X, Y):
    """Map X onto reference Y with rotation+reflection+uniform scale."""
    from scipy.linalg import orthogonal_procrustes
    Xc = X - X.mean(0)
    Yc = Y - Y.mean(0)
    R, sca = orthogonal_procrustes(Xc, Yc)
    s = sca / (Xc ** 2).sum()
    return s * (Xc @ R) + Y.mean(0)


def normalize_box(P, center=None, scale=None):
    """Fit P into [-1,1]^2 preserving aspect. Reuse (center,scale) if given."""
    if center is None:
        center = (P.max(0) + P.min(0)) / 2.0
        half = (P.max(0) - P.min(0)) / 2.0
        scale = 1.0 / max(half.max(), 1e-9)
    return ((P - center) * scale).astype(np.float32), center, scale


# --------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description="Reduce + neighbors + 2D layout")
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = lib.load_config(args.config)
    ds = cfg["id"]
    rcfg = cfg["reduce"]
    k = cfg.get("neighbors", {}).get("k", 15)

    vpath = lib.path("work", ds, "vectors.f32.npy")
    if not os.path.exists(vpath):
        raise SystemExit(f"missing {vpath} — run `make embed DS={ds}` first")
    vectors = np.load(vpath)
    vectors = l2norm(vectors.astype(np.float32))
    count, d_model = vectors.shape
    lib.log(f"[{ds}] {count}×{d_model} vectors")

    lib.log("computing 128d shipping vectors…")
    vec128 = reduce_128(vectors, rcfg.get("dim128_method", "mrl"))

    lib.log("computing exact full-dim neighbors…")
    nidx, nsim = exact_neighbors(vectors, k)

    lib.log("running UMAP…")
    ref_id = rcfg.get("reference")
    group = rcfg.get("aligned_group")
    disp_median = disp_p95 = 0.0

    if group and ref_id and ref_id != ds:
        # aligned member: init from reference coords, then Procrustes snap
        ref = np.load(lib.path("work", ref_id, "layout.npz"), allow_pickle=True)
        ref_raw = ref["raw"]                 # reference pre-normalization UMAP coords
        raw = run_umap(vectors, rcfg["umap"], init=ref_raw)
        raw = procrustes_onto(raw, ref_raw)
        center = ref["norm_center"]
        scale = float(ref["norm_scale"])
        points, center, scale = normalize_box(raw, center, scale)
        d = np.linalg.norm(points - normalize_box(ref_raw, center, scale)[0], axis=1)
        disp_median, disp_p95 = float(np.median(d)), float(np.percentile(d, 95))
        lib.log(f"alignment displacement: median={disp_median:.3f} p95={disp_p95:.3f}")
    else:
        # reference or independent: plain UMAP, own box
        raw = run_umap(vectors, rcfg["umap"])
        points, center, scale = normalize_box(raw)

    out = lib.path("work", ds, "layout.npz")
    np.savez(out,
             points=points, nidx=nidx, nsim=nsim, vec128=vec128,
             raw=raw.astype(np.float32),
             norm_center=np.asarray(center, dtype=np.float64),
             norm_scale=np.float64(scale),
             d_model=np.int64(d_model),
             disp_median=np.float64(disp_median),
             disp_p95=np.float64(disp_p95))
    lib.log(f"[{ds}] layout -> {out}")


if __name__ == "__main__":
    main()
