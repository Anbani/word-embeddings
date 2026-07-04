# Implementation Plan — Georgian LLM Word Embeddings v2

## Context

`anbani.ge/embeddings` currently iframes a stock TensorBoard projector (`public/emb/`, ~100MB) over two 2019-era word2vec datasets whose vocab is polluted with wiki-markup and scattered inflected forms. The approved [PRD](PRD.md) + [SDD](SDD.md) replace it with a site-native explorer over **2026 LLM contextual word embeddings** (EmbeddingGemma-300m + Gemma 4), rendered via Apple's `embedding-atlas` `EmbeddingView`, with a signature "model-morph" animation across aligned UMAP layouts. The SDD is the load-bearing spec (20-item landmine index); this plan grounds it in the actual repo and sequences execution.

**First pass scope (confirmed): cheap-first.** Build the pipeline and prove it end-to-end on the two fast EmbeddingGemma datasets (`kawiki-eg`, `vef-eg`) — eval passing, `dist/` committed — then **halt for review before the long Gemma-4 runs**. Gemma-4 access is ready (licenses + `HF_TOKEN` on node) but deferred to the next pass. Web (P2) and live-query (P3) follow after.

## Ground truth discovered (deltas from the SDD's assumptions)

The SDD assumed several inputs must be acquired; they already exist. Reuse in place:

- **ვეფხისტყაოსანი text is in-repo** — `src/data/rustaveli/raw/vef.csv` (6,676 lines, columns `id,line,chapter,chapter_id,strophe_id,line_id`). Do **not** fetch from wikisource (SDD §2). Group lines by `strophe_id` for the 4-line stanza context.
- **Hunspell files are vendored** — `src/data/ext/ka_GE.aff` (2,363 rules) + `ka_GE.dic` (143K entries). No Spellcheck submodule dependency needed for inputs.
- **Affix expander is stdlib, copy-reusable** — `apps/Anbani.Spellcheck/src/affix.py`: `parse_aff(path)`, `parse_dic(path, aff)`, `expand_entry(aff, stem, flags) -> (valid, forbidden)`. The **stem IS the lemma**, so form→lemma is a trivial re-key. `build_corpus.py` flattens lemma identity away — do **not** reuse it; write a thin new driver over `affix.py` (also copy `normalize.py`'s `fold()`/`is_word()` if mirroring normalization).
- **Bonus clean lemma list** — `apps/Anbani.Spellcheck/raw/dlab_georgian_lemmas.csv` (~39K clean lemmas); use as a lemma allowlist to complement the `.dic` stems (SDD §3 "intersection strategy").
- **Eval seeds exist** — `src/data/ext/questions-words-ka.txt` (analogy 4-tuples), `src/data/rustaveli/test/{related-terms,analogies}.json`. Fold into `eval/probes.json`; hand-verify (never copy watermarked wordnet scores — landmine #16).
- **Corpus ZIM confirmed on node** — `~/.local/share/kiwix-desktop/wikipedia_ka_all_nopic_2026-04.zim` (861MB); GPU currently ~9.7GB/24GB used, ~14.5GB free. `wiktionary_ka` + `wikiquote_ka` also present if ever wanted (not v1).
- **`.gitignore` blocks `*.json`/`*.txt`/`*.tsv`/`*.npy`** via block-all-then-allowlist. Committing `dist/` needs an explicit allow-rule **and** `git add -f` (homoglyph does both) — landmine: silent empty dist commits otherwise.
- **Legacy `src/`** (notebooks: w2v/fasttext/BERT, `src/utils/*.py`) stays as reference; new pipeline is top-level `src/*.py`. Keep `src/data/` (assets). No collision.

## Repo layout to create (mirrors `apps/homoglyph/`)

```
apps/word-embeddings/
  Dockerfile.generate          # CUDA 12.x + torch + transformers + bitsandbytes + libzim + sentencepiece
  Dockerfile.build             # slim python: numpy, umap-learn, scipy, pyarrow (no torch)
  Makefile                     # corpus / vocab / embed / reduce / dist / eval targets, env-var determinism
  requirements-generate.txt    # heavy ML deps
  requirements.txt             # stdlib-lean build/test deps
  src/
    extract_corpus.py          # 1a ZIM -> work/<ds>/sentences.jsonl
    build_vocab.py             # 1b vocab + form->lemma map + occurrence index
    embed_contextual.py        # 1c model forward passes -> work/<ds>/vectors.f32.npy
    reduce_layout.py           # 2a PCA/MRL -> UMAP-2D -> alignment
    build_dist.py              # 2b quantize, neighbors, pack dist/
    evaluate.py                # 3 probe eval + schema/vocab_hash/displacement asserts
    affix_lemmatize.py         # new driver over copied affix.py (form->lemma)
    vendor/affix.py            # copied verbatim from Anbani.Spellcheck (stdlib-only)
  eval/probes.json
  configs/{kawiki-eg,vef-eg,kawiki-g4e4b,kawiki-g4-12b}.yaml
  work/                        # gitignored; stage-1 artifacts live on node
  dist/                        # committed output (add !dist/ allow-rule + git add -f)
  docs/{PRD.md,SDD.md,dist-schema.md}
  .github/workflows/{generate.yml,build.yml,tests.yml}
  tests/
```

CI mirrors homoglyph exactly: `generate.yml` = dispatch-only heavy stage (commits `work`-derived inputs + `dist/`); `build.yml` = on-push stdlib rebuild of `dist/`; `tests.yml` = unittest matrix + a node reader test against committed dist. Bot identity `anbani-bot / bot@users.noreply.github.com`, commits tagged `[skip ci]`, gated on `git diff --cached --quiet`, `permissions: contents: write`. Determinism env: `PYTHONHASHSEED=0`, `PYTHONUTF8=1`, `LC_ALL=C`.

---

## Execution — first pass (cheap-first)

Stage 1 runs **only in Docker on the GPU node** (`ssh george@100.88.125.127`, host `exce`) — never install on host. Mac is edit-only; commit → push → node pulls/builds in Docker. `dist/` build (stage 2) + eval (stage 3) are CPU and run in CI.

### Step 0 — Scaffold (Mac, edit-only)
Create the layout above: Dockerfiles, `Makefile`, requirements, empty `src/*.py` stubs, `configs/*.yaml`, `.github/workflows/*`, `docs/dist-schema.md`, `.gitignore` `!dist/` rule. Copy `Anbani.Spellcheck/src/affix.py` (+`normalize.py`) into `src/vendor/`. Commit + push.

### Step 1 — Corpus extraction (node, CPU — can run while GPU busy)
`extract_corpus.py`: `libzim` iterate `wikipedia_ka_all_nopic_2026-04.zim`, article namespace only, skip redirects → selectolax strip tables/refs/sup/nav/math → paragraph text. **Georgian sentence splitting** (landmine #10): no capitalization cue; split on `[.!?…]`+ws with abbreviation blocklist (`წ. ე.წ. ა.შ. სთ. გვ. მაგ. ე.ი.` + initials); keep 4–60 words; drop <60% chars in **U+10D0–U+10FF** (full block, incl. archaic). Global sentence dedup (hash set). → `work/kawiki/sentences.jsonl`.
For `vef`: parse `vef.csv`, group by `strophe_id`, each 4-line stanza = one context "sentence"; **preserve ჱ ჲ ჳ ჴ ჵ ჶ** — same full-block regex. → `work/vef/sentences.jsonl`.

### Step 2 — Vocab + form→lemma + occurrence index (node/Mac, CPU) — **the QA gate**
`build_vocab.py` + `affix_lemmatize.py`:
- Lemma pool = `.dic` stems ∪ `dlab_georgian_lemmas.csv` ∪ wordnet headwords, ranked by corpus frequency.
- `expand_entry` over each lemma → forms; invert to `form→lemma`; **drop ambiguous forms** (>1 lemma) — landmine #11.
- Lemma freq = Σ surviving-form counts. Occurrence index with Georgian word boundaries `(?<![ა-ჿ])form(?![ა-ჿ])`. Drop lemmas with **<8 occurrences** before the **30,000** cut (≤65,535 for u16 — landmine #17).
- Hard blocklist (checked-in text file): `კატეგორია თარგი ფაილი ვიკიპედია …`, Latin/digit/punct tokens, single letters.
- **Freeze index order**; emit `vocab_hash` (sha256 of label list) — must match across future kawiki datasets (landmine #5).
- `vef` vocab: raw forms from the poem, freq≥3, no lemmatization (archaic morphology defeats modern affix rules), own index space.
- **⚠ Manual QA of vocab top-500 by eye** — this is exactly where the current page failed; do not skip. Extend blocklist during QA.

### Step 3 — Contextual embedding, EmbeddingGemma (node, Docker GPU — fast)
`embed_contextual.py`, `configs/kawiki-eg.yaml` + `vef-eg.yaml`. Per lemma: sample **N=64** occurrence sentences **uniformly at random, seeded** (not first-N — landmine #6), **≤2 per article**. Truncate to ≤96 tokens around target.
- Load EmbeddingGemma-300m bf16. **Prompt template byte-exact** per model card (`title: none | text: {sentence}`) — record verbatim in `meta.json.prompt` (browser P3 must reproduce it — landmine #2). Compute target offsets **after** template prepend.
- **Token-span mean-pool** the target word's subword tokens from the **final-layer** hidden states (offset mapping) — **never the pooled sentence vector** (landmine #1, the #1 silent-failure).
- Average the ≤64 vectors in **fp32** (landmine #7), L2-normalize → `work/<ds>/vectors.f32.npy` (30000×768 / vef×768). Checkpoint by lemma-id ranges.

### Step 4 — Reduce, layout, dist (CI/CPU via `build.yml`)
`reduce_layout.py` + `build_dist.py`:
- Neighbors: exact cosine top-15 in **full 768d** before any reduction.
- 128d int8: **MRL truncation (first 128 + renorm) — valid for EmbeddingGemma** (landmine #3; PCA path reserved for Gemma-4). Per-row symmetric int8: `scale=max(|v|)/127`.
- UMAP per dataset `metric=cosine, n_neighbors=30, min_dist=0.08, n_components=2`, seeded. `kawiki-eg` layout = **reference** (alignment infra built now; the other kawiki models align to it later via `init=reference` + Procrustes — landmine #4). `vef-eg` independent. Standardize to [-1,1]².
- Pack `dist/<ds>/`: `meta.json`, `labels.tsv`, `points.f32`, `neighbors.bin` (15×(u16+u8)), `vectors.i8`, `vscales.f32`; `dist/index.json`. Little-endian, typed-array-readable, **no parquet/DuckDB** (landmine #13). Document in `docs/dist-schema.md` (`"v": 2`).

### Step 5 — Eval + tests (CI/CPU via `tests.yml`)
`evaluate.py` + `eval/probes.json` (~60 probes seeded from SDD §9 list + existing analogy/related-terms sets, hand-verified). Asserts: probe hit-rate ≥70% for `kawiki-eg`, forbidden-hits = 0, schema shapes vs meta, `vocab_hash`, no blocklisted labels, alignment displacement bound (<0.35, exercised once ≥2 kawiki datasets exist). Node reader test parses committed `dist/`. **Eval passing blocks the dist commit.**

### Step 6 — Commit dist, halt
CI commits `dist/kawiki-eg/` + `dist/vef-eg/` (`git add -f`, `[skip ci]`). Report results (vocab QA sample, probe hit-rate vs 32.85% baseline, dist sizes vs 30MB budget). **Stop for review before Gemma-4.**

---

## Subsequent passes (planned, not first session)

- **Pass 2 — Gemma-4 (GPU, long, access ready):** layer ablation {0.5, 0.65, 0.8} on 2K-word subsample per model → lock layer (never last layer — landmine #8); 12B loads **NF4/int8** (won't fit bf16 on 3090 — check free VRAM first, landmine #9); hidden states → fp32. **PCA→128** for the int8 path (not MRL — landmine #3), `d_model` read at runtime. UMAP `init=<reference coords>` + orthogonal Procrustes snap onto `kawiki-eg`; assert displacement <0.35. Adds `kawiki-g4e4b`, `kawiki-g4-12b` to dist (same frozen vocab/`vocab_hash`).
- **Pass 3 — Web P2 (`OdyssevsCom/Anbani.Web.Main`):** `npm i embedding-atlas`; `EmbeddingView` wrapped with `next/dynamic {ssr:false, loading}` (pattern: `pages/[locale]/redactor.js`). New `components/miniapps/Embeddings.js` modeled on `Homoglyph.js` (hero + `SectionHeader`/`Card` explainer, `text-major`/`text-minor`/`brand`/`secondary dark:` tokens, `useEffect` client data-load with placeholders). New `embeddings` i18n namespace: `public/locales/{ka,en}/embeddings.json` + `makeStaticProps(['common','embeddings'])`. Vendor `dist/`→`public/embeddings/` with `*.meta.json` sha256 sidecar; `lib/embeddings.js` memoized `ensureEmbeddings()` (JSON via `homoglyph.js` pattern; typed-array vectors via `spellcheck.js` `arrayBuffer()` pattern; `?v=meta.sha256` cache-bust). **Do a morph-perf spike early** (landmine #14): measure `EmbeddingView` re-ingest FPS on 30K rAF lerp; fallback = crossfade / custom canvas during transition. Playground cosine/analogy over `vectors.i8` in a Web Worker. **Delete `public/emb/` (~100MB) in the same commit that ships the replacement** (landmine #18). Verify via `gh run watch` + beta.anbani.ge — no local site run.
- **Pass 4 — Live query P3 (opt-in):** transformers.js ONNX EmbeddingGemma, WebGPU-gated, template byte-identical to `meta.json.prompt`, MRL→128→int8 dot vs shipped vectors, **kNN-interpolated pin** (no JS UMAP transform — landmine #12). Separately shippable; only on `kawiki-eg`/`vef-eg`.

## Critical files

- Create: `apps/word-embeddings/{Dockerfile.generate,Dockerfile.build,Makefile,requirements*.txt}`, `src/{extract_corpus,build_vocab,affix_lemmatize,embed_contextual,reduce_layout,build_dist,evaluate}.py`, `src/vendor/affix.py`, `configs/*.yaml`, `eval/probes.json`, `.github/workflows/{generate,build,tests}.yml`, `docs/dist-schema.md`, `tests/*`.
- Reuse in place: `src/data/ext/ka_GE.{aff,dic}`, `src/data/rustaveli/raw/vef.csv`, `src/data/ext/questions-words-ka.txt`, `src/data/rustaveli/test/*.json`.
- Copy from: `apps/Anbani.Spellcheck/src/{affix,normalize}.py`, `apps/Anbani.Spellcheck/raw/dlab_georgian_lemmas.csv`; pattern from `apps/homoglyph/.github/workflows/*` + `Makefile`.
- Later (web): `apps/Anbani.Web.Main/components/miniapps/Embeddings.js`, `pages/[locale]/embeddings.js`, `lib/embeddings.js`, `public/locales/{ka,en}/embeddings.json`; delete `public/emb/`.

## Verification

- **Vocab QA (Step 2):** eyeball top-500 labels — zero wiki-markup/Latin/single-letter tokens. Gate before any GPU time.
- **Pipeline proof (Steps 3–5):** `evaluate.py` green — `kawiki-eg` probe hit-rate ≥70% (vs legacy 32.85%), forbidden-hits 0, schema/`vocab_hash`/no-blocklist asserts pass. Spot-check top-5 neighbors for `დედა`, `წითელი`, `მეფე`, `ზღვა` — synonyms/co-hyponyms dominate.
- **Reproducibility:** `meta.json` carries model revision, prompt template, layer, seeds, config hash, `generated_at`; regen is deterministic.
- **Budget:** `dist/` for the two datasets ≪ 30MB total.
- **CI:** `gh run watch` — `build.yml` regenerates dist, `tests.yml` matrix + node reader green. Stage-1 artifacts stay out of git (landmine #19); only `dist/` committed. No local execution (org rule; Docker-on-node exception covers the pipeline only).
