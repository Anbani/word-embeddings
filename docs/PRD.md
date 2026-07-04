# PRD — Georgian LLM Word Embeddings v2 (`anbani.ge/embeddings`)

Status: approved for implementation. Companion doc: [SDD.md](SDD.md) (read both fully before writing code — the SDD carries the load-bearing technical decisions; deviations require explicit sign-off).

## 1. Problem

Current `/embeddings` page iframes a stock TensorBoard Embedding Projector (`public/emb/projector.html`, ~1.9MB HTML blob) with two 2019-era word2vec datasets:

- `kawiki_min` — 10,096 words × 200d, trained on an old kawiki dump. Visibly polluted: top tokens are wiki-markup (კატეგორია, თარგი…), inflected forms scattered as separate points.
- `vef_95_w2v` — ვეფხისტყაოსანი, 14,964 × 300d.

Page ships ~100MB (duplicate `.tsv` + `.bytes` tensors), doesn't match site design/dark-mode/i18n, w2v linkage quality is a decade behind.

## 2. Goal

Replace with a modern, site-native embedding explorer powered by 2026 LLM embeddings that actually understand Georgian, on fresh corpora, with features no off-the-shelf projector has.

## 3. Users

- Georgian-speaking public / students: "show me what a language model thinks ქართული looks like."
- NLP-curious devs: compare embedding models on Georgian concretely.
- Anbani ecosystem: another flagship static showcase alongside /wordnet, /homoglyph, /emoji.

## 4. Datasets shipped (v1)

Same kawiki vocab (≈30K lemmas, identical index order) through three models + one poem corpus:

| id | corpus | model | notes |
|---|---|---|---|
| `kawiki-eg` | kawiki 2026-04 (ZIM) | EmbeddingGemma-300m | flagship / default view; also the browser live-query model |
| `kawiki-g4e4b` | kawiki 2026-04 | Gemma 4 E4B, contextual hidden states | "smarter model" comparison |
| `kawiki-g4-12b` | kawiki 2026-04 | Gemma 4 12B (quantized), contextual hidden states | smartest map |
| `vef-eg` | ვეფხისტყაოსანი full text | EmbeddingGemma-300m | archaic Georgian; own vocab; not aligned to kawiki |

All vectors are **contextual averages**: per word, N sampled corpus sentences containing the word (or its inflected forms — see SDD §4), embed, pool target-word tokens, average. No isolated-word embedding anywhere in v1.

Corpus source: Kiwix ZIMs already on the GPU node (`george@exce:~/.local/share/kiwix-desktop/wikipedia_ka_all_nopic_2026-04.zim`, 861MB; `wiktionary_ka_all_nopic_2026-04.zim` available for glosses). Do NOT download Wikimedia dumps.

Legacy w2v datasets: dropped. Old `public/emb/` deleted entirely (−100MB).

## 5. Page UX (`/embeddings`, `[locale]` routed, static export)

Viewer: Apple **Embedding Atlas** `EmbeddingView` React component (npm `embedding-atlas`), WebGPU with WebGL fallback, styled to anbani.ge (Tailwind theme, dark mode, ka/en i18n via next-i18next). Full-viewport map like current page, waves divider retained.

Features, priority order:

1. **2D map** — precomputed UMAP coords, density rendering, ~30K labeled points, hover label, click select. 60fps pan/zoom on mid-range phone.
2. **Model switcher + animated morph** — segmented control (EmbeddingGemma / Gemma 4 E4B / Gemma 4 12B). Layouts are Procrustes/init-aligned (SDD §6); switching animates each word gliding to its new position (~800ms ease). This is the signature moment of the page — "watch the vocabulary rearrange as the model gets smarter." Corpus switcher separately for ვეფხისტყაოსანი (no morph to/from it).
3. **Word panel** — on select/search: word, frequency, top-15 nearest neighbors (exact cosine in full-dim space, precomputed) with similarity bars; clicking a neighbor navigates. Neighbors shown for the *active model*, so users can flip models and watch the neighbor list improve.
4. **Search** — instant prefix/fuzzy search over vocab labels (client-side, Georgian-aware; no transliteration needed, keyboard page exists for that).
5. **Playground** — cosine similarity between any two vocab words + analogy arithmetic (a − b + c → nearest), computed in-browser over shipped int8 vectors (SDD §7).
6. **Live query pin (opt-in)** — button ("სცადე ნებისმიერი სიტყვა") downloads quantized EmbeddingGemma via transformers.js (WebGPU only, one-time ~200–300MB, Cache API), embeds arbitrary user text, pins it into the map via kNN interpolation (SDD §8). Only on `kawiki-eg` / `vef-eg` datasets; hidden when WebGPU absent. Everything else must work without it.
7. **Explainer section** — below the fold, homoglyph-page style: what embeddings are, how these were made (contextual averaging diagram), model comparison notes, dataset stats. ka + en.

## 6. Non-goals (v1)

- No 3D view; TB projector is fully retired.
- No server/API — pure static, consistent with org policy (api.anbani.ge is down).
- No in-browser t-SNE/UMAP recomputation.
- No embeddings for phrases/sentences beyond the live-query pin.
- No additional corpora (news, wikiquote) — schema must allow adding them later without breaking.

## 7. Performance / size budget

- Initial page JS+data for default dataset: ≤ 6MB transfer (coords+labels+neighbors+meta; int8 vectors may lazy-load on first playground/panel use).
- Each additional dataset lazy-loads on switch, ≤ 6MB.
- Total `public/embeddings/` ≤ 30MB (vs 100MB today).
- Time-to-interactive map on desktop broadband < 3s; mobile mid-range < 6s.
- transformers.js model download never automatic.

## 8. Success criteria

- Eval harness (SDD §9) shows new kawiki maps beat legacy w2v on the probe set (target ≥ 70% probe hit-rate for `kawiki-eg`; legacy baseline was "32.85% on selection").
- Neighbor spot-checks: synonyms/co-hyponyms dominate top-5 for common nouns; no wiki-markup tokens anywhere in vocab.
- Lighthouse perf ≥ 85 on /embeddings (excluding opt-in model download).
- Morph animation smooth (no full-reshuffle chaos — alignment working).

## 9. Phasing

- **P1**: pipeline (all 4 datasets) + dist schema + eval harness. Exit: dist committed, eval passing.
- **P2**: web page with features 1–5 + 7; delete `public/emb/`. Exit: deployed to beta.anbani.ge.
- **P3**: live query pin (feature 6). Separately shippable; do not block P2 on it.

## 10. Ownership / repos

- Pipeline: this repo (`Anbani/word-embeddings`, default branch **master** — org gotcha, others use main). 3-stage layout mirroring `Anbani/homoglyph`: stage 1 generate (GPU Docker on `exce` node), stage 2 build dist (CPU, stdlib-lean), stage 3 tests. CI builds stages 2–3 and commits `dist/`.
- Web: `OdyssevsCom/Anbani.Web.Main` — dist vendored into `public/embeddings/`, new `components/miniapps/Embeddings.js`, existing `[locale]/embeddings.js` route kept (URL/SEO preserved).
- Deploy: existing Actions → wrangler → CF Pages (beta), manual promote to prod. No local execution of the web app; verify via `gh run watch` + beta URL.
