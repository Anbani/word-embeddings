# dist/ schema — `"v": 2`

The committed `dist/` is the entire contract between the pipeline (this repo) and
the web viewer (`Anbani.Web.Main`, vendored to `public/embeddings/`). Everything
is little-endian and read in the browser via typed arrays over `fetch`
ArrayBuffers — **no parquet / DuckDB / Mosaic** (we use the bare `EmbeddingView`
component, not the full Atlas app — SDD landmine #13).

## Layout

```
dist/
  index.json                 # dataset catalog (below)
  kawiki-eg/                  # one dir per dataset id
    meta.json
    labels.tsv
    points.f32
    neighbors.bin
    vectors.i8
    vscales.f32
  vef-eg/ …
  kawiki-g4e4b/ …            # added in the next pass
  kawiki-g4-12b/ …
```

## `index.json`

Array of dataset descriptors:

```json
[
  {
    "id": "kawiki-eg",
    "corpus": "kawiki",
    "model": "EmbeddingGemma-300m",
    "name_ka": "ემბედინგ-ჯემა",
    "name_en": "EmbeddingGemma-300m",
    "count": 30000,
    "dims_full": 768,
    "dims_i8": 128,
    "aligned_group": "kawiki",     // null for vef-eg (independent layout)
    "reference": true,             // is this the morph reference layout
    "files": { "labels.tsv": 412345, "points.f32": 240000, "neighbors.bin": 1350000,
               "vectors.i8": 3840000, "vscales.f32": 120000 }
  }
]
```

`"v": 2` and a top-level `generated_at` live in each dataset's `meta.json`; the
index is a convenience catalog for the loader.

## Per-dataset files (index order is authoritative and identical for one `count`)

- **`meta.json`** — reproducibility + viewer params:
  `{ v, id, corpus, model:{id,revision}, prompt, layer, N, per_article_cap,
     window_tokens, seeds:{sample,umap}, dim128_method, umap:{…}, vocab_hash,
     config_hash, count, dims_full, dims_i8, aligned_group, reference,
     displacement:{median,p95}, generated_at }`.
  - `vocab_hash` = sha256 of the newline-joined label list. **Identical across
    all kawiki datasets** — `evaluate.py` asserts equality (landmine #5).
  - `prompt` = the EXACT EmbeddingGemma template string. The browser live-query
    path (P3) must reproduce it byte-for-byte (landmine #2).

- **`labels.tsv`** — `count` lines, index order, `word \t freq` (freq = lemma
  corpus frequency). UTF-8, `\n`-terminated. No header.

- **`points.f32`** — `count × 2` float32, x,y interleaved (`x0,y0,x1,y1,…`).
  Aligned datasets share the `[-1,1]²` box; `vef-eg` has its own box.

- **`neighbors.bin`** — `count × k` records, `k = 15`. Each record is
  `u16 index` + `u8 sim` (3 bytes), so `3·k = 45` bytes per word, in index
  order. `sim_u8 = round(clamp(cosine, 0, 1) × 255)`. Neighbors are computed in
  **FULL original dim** before any reduction (landmine, SDD §7). Indices are
  `< count`.

- **`vectors.i8`** — `count × 128` int8, row-major. Per-row symmetric quant:
  `scale = max(|v|)/127`, `q = round(v/scale)`, where `v` is the L2-normalized
  128-d vector (MRL-truncated for EmbeddingGemma, PCA-projected for Gemma 4).
  Browser cosine: `dot(qA,qB) · scaleA · scaleB` (≈ cosine to ~1e-2).

- **`vscales.f32`** — `count` float32, per-row dequant `scale` aligned to
  `vectors.i8` rows.

## Reader (JS)

```js
const meta   = await (await fetch(`${base}/meta.json`)).json()
const labels = (await (await fetch(`${base}/labels.tsv`)).text()).split('\n')
const pts    = new Float32Array(await (await fetch(`${base}/points.f32`)).arrayBuffer())
const nb     = new DataView(await (await fetch(`${base}/neighbors.bin`)).arrayBuffer())
// neighbor j of word i: idx = nb.getUint16((i*15 + j)*3, true); sim = nb.getUint8((i*15 + j)*3 + 2)/255
const vi8    = new Int8Array(await (await fetch(`${base}/vectors.i8`)).arrayBuffer())     // lazy
const vsc    = new Float32Array(await (await fetch(`${base}/vscales.f32`)).arrayBuffer()) // lazy
```

## CI validation (`evaluate.py`, stage 3)

- shapes match `meta` (`points.f32` = `8·count` bytes, `neighbors.bin` = `45·count`,
  `vectors.i8` = `128·count`, `vscales.f32` = `4·count`, `labels.tsv` = `count` lines);
- `vocab_hash` equal across kawiki datasets;
- every neighbor index `< count`; no NaN/Inf in `points.f32`/`vscales.f32`;
- no blocklisted token in `labels.tsv`;
- total `dist/` size ≤ 30 MB;
- probe hit-rate ≥ 70 % for `kawiki-eg`, forbidden-hit count 0 (probes read from
  `neighbors.bin` — no raw vectors needed, so this runs in CI from the commit).
