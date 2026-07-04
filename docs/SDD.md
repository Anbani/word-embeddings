# SDD — Georgian LLM Word Embeddings v2

Implementation spec for [PRD.md](PRD.md). Sections marked ⚠ are landmines: decisions that look substitutable but aren't. Read §12 (landmine index) before starting.

## 1. Repo layout (this repo, branch `master`)

```
word-embeddings/
  docker/Dockerfile.generate     # CUDA 12.x + torch + transformers + bitsandbytes + libzim + sentencepiece
  docker/Dockerfile.build        # slim python: numpy, umap-learn, scipy, pyarrow (no torch)
  src/
    extract_corpus.py            # stage 1a: ZIM → sentences.jsonl
    build_vocab.py               # stage 1b: vocab + form→lemma map + occurrence index
    embed_contextual.py          # stage 1c: model forward passes → raw vectors (.npy)
    reduce_layout.py             # stage 2a: PCA/MRL → UMAP-2D → alignment
    build_dist.py                # stage 2b: quantize, neighbors, pack dist/
    evaluate.py                  # stage 3: probe-set eval, schema checks
  eval/probes.json               # curated probe words + expected neighbors
  configs/*.yaml                 # one per dataset (model id, layer, sampling params, seeds)
  work/                          # stage-1 artifacts, gitignored, lives on GPU node
  dist/                          # committed output, consumed by web repo
  docs/{PRD.md,SDD.md,dist-schema.md}
```

Stage 1 runs ONLY in Docker on the GPU node (`ssh george@100.88.125.127`, host `exce`) — never install anything on the host (org rule; conda on host is off-limits). `docker run --gpus all -v ~/we-work:/work ...`. ⚠ GPU had ~9.7GB/24GB already allocated at spec time — check `nvidia-smi` and free VRAM (or pick a time) before the 12B run; don't assume 24GB free.

Stage 2+3 run in CI (CPU) from stage-1 artifacts. Stage-1 raw vectors (~30K×{768,2048,3840}×f32 ≈ 90–460MB each) are too big for plain git: keep them in `work/` on the node, rsync a copy to the Mac as backup, commit only `dist/` (~≤30MB). Record exact model revisions + config hash in `dist/*/meta.json` so regeneration is reproducible.

⚠ Gemma weights are gated on HuggingFace — accept licenses for `google/embeddinggemma-300m` and the Gemma 4 models on the account first, pass `HF_TOKEN` into the container. Pin exact model revisions in configs.

## 2. Corpus extraction (`extract_corpus.py`)

Input: `wikipedia_ka_all_nopic_2026-04.zim` (already at `~/.local/share/kiwix-desktop/` on node — mount read-only; do NOT re-download dumps).

- Use `libzim` (python-libzim) to iterate entries; keep only article namespace (`C`/`A` content entries whose mimetype is text/html), skip redirects (`entry.is_redirect`).
- HTML → text with selectolax/lxml: drop `table, infobox, .references, sup, style, script, math`, navboxes, image captions. Keep paragraph text only.
- ⚠ Sentence splitting in Georgian cannot use capitalization (Mkhedruli has no case). Split on `[.!?…]` + whitespace, with an abbreviation blocklist (წ., ე.წ., ა.შ., სთ., გვ., მაგ., ე.ი., ლათ., ბერძ., inits like "ი."). Keep sentences 4–60 words; drop sentences <60% Georgian chars (U+10D0–U+10FF).
- Deduplicate sentences globally (hash set) — wiki boilerplate repeats massively and would bias contextual averages.
- Output `work/kawiki/sentences.jsonl`: `{"id": int, "article": str, "text": str}`.

ვეფხისტყაოსანი: raw text NOT in repo. Acquire public-domain full text (e.g. Georgian Wikisource — it's inside `wikisource`… not on node; fetch from ka.wikisource.org or another public-domain source; commit the cleaned plaintext to `data/vefkhistkaosani.txt` since it's small and stable). Split by stanza lines; treat each 4-line stanza as one "sentence" context. ⚠ Preserve archaic letters ჱ ჲ ჳ ჴ ჵ ჶ — extend all Georgian-char regexes to U+10D0–U+10FF full block, never the modern-33 subset.

## 3. Vocabulary (`build_vocab.py`)

kawiki vocab (shared verbatim by all three kawiki datasets):

- Lemma source: intersection strategy — start from Anbani.Spellcheck's lemma list (the Hunspell `ka_GE.dic` stems + wordlist used by that pipeline) ∪ wordnet lexicon headwords; rank by corpus frequency; cut at **30,000**. ⚠ vocab MUST be ≤65,535 (u16 neighbor indices).
- Form→lemma map: reuse Anbani.Spellcheck's stdlib affix expander (Hunspell aff rules) to expand each candidate lemma into its inflected forms, invert to `form → lemma`. Ambiguous forms (map to >1 lemma) are dropped from the map (counted toward no lemma) — cheaper and safer than guessing.
- Frequency of a lemma = Σ counts of its surviving forms in the corpus.
- Hard blocklist: wiki-namespace words (კატეგორია, თარგი, ფაილი, ვიკიპედია, რესურსები, სქოლიო…), tokens with Latin/digits/punct, single letters. Blocklist is a checked-in text file, extend during QA.
- ⚠ Vocab index order is frozen after this step and identical across `kawiki-eg`/`kawiki-g4e4b`/`kawiki-g4-12b` — same index ⇒ same word. The morph animation and cross-model neighbor comparison silently break if any dataset reorders or refilters. `meta.json.vocab_hash` (sha256 of the label list) must match across the three; `evaluate.py` asserts it.
- Occurrence index: for each lemma, list of sentence ids containing any of its forms, matched with Georgian word boundaries: `(?<![ა-ჿ])form(?![ა-ჿ])`. Lemmas with <8 occurrences are dropped BEFORE the 30K cut (don't ship words whose vector is 3 noisy samples).

vef vocab: from the poem itself — all word forms (no lemmatization; archaic morphology defeats the modern affix rules) with freq ≥3, expect ~6–12K. Own index space.

## 4. Contextual embedding (`embed_contextual.py`)

Per (dataset config, lemma): sample up to **N=64** occurrence sentences — ⚠ uniformly at random over the whole occurrence list, seeded; NOT the first N (first-N biases toward alphabetically-early / intro-paragraph contexts). Cap ≤2 sentences per article to avoid topical capture (one long article about a niche topic shouldn't own a word).

Context window: truncate each sentence to ≤96 tokens around the target word (window centered on it) — keeps batches fast, semantics local.

### 4.1 EmbeddingGemma-300m (`kawiki-eg`, `vef-eg`)

- Load via `sentence-transformers` (or transformers directly), bf16.
- ⚠ Do NOT use the pooled sentence vector — that's a sentence embedding, not a word vector. Run the encoder, take **token-level hidden states from the final layer**, mean-pool ONLY the subword tokens of the target word occurrence (use tokenizer offset mapping to find the span; the form may be inflected — pool the actual matched form's span).
- ⚠ Prompt template: EmbeddingGemma is prompt-conditioned. Use the documented document template (`title: none | text: {sentence}`, per model card) — and record the EXACT template string in `meta.json.prompt`, because the browser live-query path (§8) must reproduce it byte-for-byte or its vectors land in a different subspace and every neighbor is garbage. Target-word offsets must be computed AFTER the template is prepended.
- Average the ≤64 occurrence vectors in **fp32** (bf16 accumulation loses precision over 64 sums), then L2-normalize. Output `work/<ds>/vectors.f32.npy` (30000×768).

### 4.2 Gemma 4 contextual (`kawiki-g4e4b`, `kawiki-g4-12b`)

- E4B: bf16, fits 24GB natively. 12B: ⚠ does not fit bf16 on the 3090 — load 4-bit NF4 (bitsandbytes) or int8; hidden states still come out bf16 — cast to fp32 before accumulating.
- Plain forward pass with `output_hidden_states=True`, no generation. Batch sentences (pad to bucket lengths); this is a full corpus pass ×64 samples ×30K words ≈ ~1.2M short sequences per model — budget hours, checkpoint progress (resume by lemma-id ranges) so an OOM/preemption doesn't restart from zero.
- Layer: ⚠ do not use the last layer (decoder last layers specialize for next-token prediction, poor as representations). Default extract layer = `round(0.65 × n_layers)`; make it a config knob and run the §9 eval over layers {0.5, 0.65, 0.8} on a 2K-word subsample first; lock best layer per model in the config before the full run.
- Pooling: mean over the target word's subword token positions (offset mapping again). Decoder is causal, so these tokens only see left context — that's expected and fine at N=64 averaging; do NOT try bidirectional hacks.
- Average fp32, L2-normalize. Output shapes: E4B 30000×d_model(E4B), 12B 30000×d_model(12B) (read d_model from config at runtime, don't hardcode).

## 5. Dimensionality for shipping (`reduce_layout.py` / `build_dist.py`)

Two consumers: (a) full-dim vectors → exact neighbors + UMAP input; (b) 128d int8 vectors → browser playground.

- ⚠ **Matryoshka truncation (first 128 dims + renormalize) is valid ONLY for EmbeddingGemma** (trained with MRL). For Gemma 4 hidden states it is meaningless — use PCA→128 (fit per dataset, fp64, whiten=False) then renormalize. Getting this wrong produces a playground where similarities are noise for two of three models.
- Neighbors (§7) are computed in FULL original dim, before any reduction.

## 6. 2D layout + cross-model alignment (`reduce_layout.py`)

- UMAP per dataset: `metric='cosine', n_neighbors=30, min_dist=0.08, n_components=2`, fixed `random_state` from config. Input: full-dim L2-normalized vectors (UMAP handles 3840d fine at 30K points).
- ⚠ Alignment strategy (the morph feature depends on this): compute `kawiki-eg` layout first — it is the reference. For `kawiki-g4e4b` and `kawiki-g4-12b`, run UMAP with `init=<reference 2D coords>` (identical vocab order makes this a per-point init). This produces layouts that are already locally consistent. Then apply orthogonal Procrustes (scipy, rotation+reflection+uniform scale) of each layout onto the reference as a final snap. Plain independent UMAP + Procrustes alone is NOT enough — UMAP solutions differ by more than a rigid transform and the morph becomes spaghetti.
- After alignment, standardize all layouts to a shared bounding box ([-1,1]², preserving aspect) so the viewer needs no per-dataset camera logic.
- `vef-eg`: independent UMAP, no alignment, own bounding box.
- Sanity metric emitted to meta: median per-word displacement between aligned layouts (expect ≪ bbox diagonal; evaluate.py asserts < 0.35).

## 7. dist/ schema (document as `docs/dist-schema.md`, versioned `"v": 2`)

```
dist/
  index.json                     # [{id, corpus, model, name_ka, name_en, count, dims_full, aligned_group, files:{...sizes}}]
  kawiki-eg/
    meta.json                    # model id+revision, prompt template, layer, N, seeds, umap params,
                                 # vocab_hash, config hash, generated_at, displacement stats
    labels.tsv                   # index-ordered: word \t lemma_freq   (~400KB)
    points.f32                   # little-endian float32 x,y interleaved, 2×4×count (~240KB)
    neighbors.bin                # per word: 15 × (u16 index + u8 similarity), sim quantized
                                 # sim_u8 = round(clamp(cos,0,1)×255); 45B×30K ≈ 1.35MB
    vectors.i8                   # count × 128 int8, row-major, symmetric per-vector scale
    vscales.f32                  # count × float32 per-row dequant scale
  kawiki-g4e4b/ …                # same shape
  kawiki-g4-12b/ …
  vef-eg/ …
```

- int8 quantization: per-row symmetric: `scale = max(|v|)/127`, `q = round(v/scale)`. Browser cosine: dot(int8,int8)×scaleA×scaleB (vectors were L2-normed pre-quantization so dot≈cosine; document ~1e-2 error).
- Everything little-endian; JS reads via typed arrays over `fetch` ArrayBuffers. No parquet/DuckDB — we use bare `EmbeddingView`, not the full Atlas app (which wants Mosaic/DuckDB; ⚠ don't pull that stack in).
- CI (stage 2+3) validates: shapes match meta, vocab_hash equality across kawiki datasets, neighbor indices < count, no NaN/Inf, total size ≤ 30MB.
- ⚠ wordnet synonym CSV is watermarked-private: it may inform the eval probes (§9) but NO upstream ids/scores may appear in dist/ or eval/probes.json.

## 8. Web integration (`Anbani.Web.Main`)

- `npm i embedding-atlas`; use the React `EmbeddingView` component only. ⚠ Static export + SSR: import with `next/dynamic` `{ ssr: false }`; the component touches `navigator.gpu`/WebGL at mount. Verify it renders under `output: export` in CI build before building features on top.
- Data loading: small loader lib fetches `index.json` then lazy-fetches per-dataset files as typed arrays; `points.f32` + `labels.tsv` + `neighbors.bin` on dataset activation; `vectors.i8` deferred until playground/live-query first use.
- `EmbeddingView` receives `{x: Float32Array, y: Float32Array, category?, text: labels}`. Selection/hover wired to the word panel. If the shipped version lacks a needed hook (e.g., programmatic fly-to on search), wrap don't fork: overlay pins/labels in an absolutely-positioned layer using the view's coordinate transform props.
- **Morph**: on model switch, interpolate `x/y` arrays over ~800ms (rAF, easeInOutCubic) feeding updated arrays to `EmbeddingView` each frame (typed-array lerp of 30K points is trivial). ⚠ If the component re-ingests data too slowly for 60fps (measure first!), fallback: crossfade two views, or render morph in a lightweight custom canvas scatter shown only during transition. Budget risk here — do the measurement spike early in P2.
- Word panel neighbors: read from `neighbors.bin` of active dataset — zero computation.
- Playground: cosine/analogy over `vectors.i8` in a Web Worker (30K×128 int8 dot ≈ ms-fast; still keep off main thread).
- **Live query (P3)**: transformers.js + ONNX EmbeddingGemma (e.g. `onnx-community/embeddinggemma-300m-ONNX`, q4). Gate: feature-detect WebGPU (`navigator.gpu`), otherwise hide the button entirely (wasm fallback too slow — don't offer). Pipeline in worker: template (byte-identical to `meta.json.prompt`) → embed → mean-pool per model card → L2-norm → truncate to first 128 dims → renorm (MRL — same as offline §5) → int8-compatible f32 dot vs `vectors.i8` → top-k. Pin position = similarity-softmax(τ=0.05) weighted average of top-8 neighbors' 2D coords. ⚠ Never attempt UMAP transform in JS — it doesn't exist; kNN interpolation is the design, not a shortcut. Live query enabled only on `kawiki-eg`/`vef-eg` (browser model must equal dataset model).
- Cleanup: delete `public/emb/` entirely (~100MB incl. projector.html), remove iframe from `components/miniapps/Embeddings.js`, keep routes `pages/embeddings.js` (redirect) + `pages/[locale]/embeddings.js`. Vendor dist as `public/embeddings/`. i18n keys under `common:embeddings.*` in ka/en locale files.
- Verify via CI + beta.anbani.ge only (no local run of the site — org rule; heavy-ML-in-Docker exception covers the pipeline, not the app).

## 9. Evaluation (`evaluate.py`, stage 3, blocks dist commit)

- `eval/probes.json`: ~60 probe words, each with `expected` (words that should appear in top-15) and `forbidden` (junk that must not). Seed examples to extend (use wordnet synonym graph to curate, but hand-verify; do not copy scores):
  - დედა → {მამა, მშობელი, შვილი}; წითელი → {ლურჯი, ყვითელი, მწვანე, ფერი}; მეფე → {დედოფალი, მმართველი, ხელმწიფე}; ზღვა → {ოკეანე, ტბა, სანაპირო}; სიყვარული → {გრძნობა, სიძულვილი, ვნება}; თბილისი → {ქუთაისი, ბათუმი, ქალაქი}; ღვინო → {ყურძენი, ჭაჭა, სასმელი}; წიგნი → {ავტორი, გამომცემლობა, რომანი}; ომი → {ბრძოლა, მშვიდობა, ჯარი}; ექიმი → {პაციენტი, საავადმყოფო, მკურნალობა}.
- Metrics per dataset: probe hit-rate (≥1 expected in top-15), mean expected-rank, forbidden-hit count (must be 0). Thresholds: `kawiki-eg` ≥70% hit-rate; Gemma 4 datasets expected ≥ that (if a "smarter" model scores materially worse, suspect layer choice or span pooling before shipping).
- Also asserts: schema checks (§7), vocab_hash equality, alignment displacement bound (§6), no blocklisted tokens in labels.
- Layer-ablation mode (§4.2) reuses the same probes on the 2K subsample.

## 10. Configs

One YAML per dataset: model id + revision, quantization, layer fraction, N=64, min_occ=8, per-article cap=2, window=96, seeds (sampling / UMAP), UMAP params, MRL-vs-PCA flag, prompt template. Everything §4–6 tunable lives here; code reads config, no magic numbers inline.

## 11. Execution order for the implementer

1. Repo scaffold, Dockerfiles, configs; corpus extraction + vocab on node (CPU-only steps, can run while GPU busy). QA vocab top-500 by eye — this is where the current page failed.
2. EmbeddingGemma kawiki run (fast, ~small hours) → full stage-2 → eval → dist for `kawiki-eg`. Prove the whole pipe end-to-end on the cheap model first.
3. Layer ablation (2K words) for E4B and 12B → lock layers → full runs (long; checkpointed).
4. `vef-eg` (needs poem text acquisition).
5. Alignment + morph-displacement validation across the three kawiki layouts.
6. Web P2 (start with the EmbeddingView + static-export spike, §8 ⚠), then P3.

## 12. Landmine index (each expands above)

1. Token-span pooling, never sentence pooling (§4.1) — the single most likely silent-failure point.
2. EmbeddingGemma prompt template byte-identical offline vs browser (§4.1, §8).
3. MRL truncation only for EmbeddingGemma; PCA for Gemma 4 (§5).
4. UMAP `init=reference` + Procrustes, not Procrustes alone (§6).
5. Frozen shared vocab order + vocab_hash assertion (§3).
6. Random occurrence sampling with per-article cap, not first-N (§4).
7. fp32 accumulation of bf16 states (§4).
8. Not the last decoder layer; ablate 0.5/0.65/0.8 (§4.2).
9. 12B needs quantized load on 24GB; check free VRAM first — node had ~10GB in use (§1).
10. Georgian: no case for sentence splitting; full U+10D0–10FF incl. archaic letters; word-boundary regex (§2, §3).
11. Form→lemma via Spellcheck affix expander; drop ambiguous forms (§3).
12. No UMAP transform in JS — kNN-interpolated pin (§8).
13. `EmbeddingView` bare component, no Mosaic/DuckDB stack; dynamic import ssr:false; verify under static export early (§8).
14. Morph perf spike before committing to implementation path (§8).
15. HF gated models: license + token; pin revisions (§1).
16. Watermarked wordnet data stays out of anything public (§7, §9).
17. Vocab ≤65,535 for u16 indices (§3).
18. Delete public/emb/ only in the same PR-equivalent commit that ships the replacement page (§8).
19. Stage-1 artifacts stay out of git; dist-only commits; meta carries reproducibility info (§1).
20. Repo default branch is `master`, not `main` (§ PRD 10).
