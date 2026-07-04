#!/usr/bin/env python3
"""Stage 1c — contextual word vectors (GPU).

Per lemma: sample up to N=64 occurrence sentences (uniform over the reservoir,
per-article cap), embed each, mean-pool ONLY the target word's subword tokens
from the chosen hidden layer, average in fp32, L2-normalize.

Landmines honored here:
  * #1  token-span mean-pool, NEVER the pooled sentence vector.
  * #2  EmbeddingGemma prompt template applied verbatim; offsets computed AFTER
        the template is prepended. The exact template is echoed to meta later.
  * #6  occurrence sampling is uniform (over the reservoir) with a per-article
        cap — not first-N.
  * #7  bf16 hidden states are cast to fp32 before accumulation.
  * #8  decoder models use hidden layer round(layer_frac * n_layers), never last.
  * #9  12B loads 4-bit NF4 (config quantization: nf4).

Output: work/<ds>/vectors.f32.npy  (count × d_model, L2-normalized, fp32)
Checkpointed: re-running resumes from work/<ds>/vectors.progress.
"""
import argparse
import json
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib  # noqa: E402


# --------------------------------------------------------------------------- data

def load_sentences(corpus):
    """id -> (text, article)."""
    d = {}
    for row in lib.read_jsonl(lib.path("work", corpus, "sentences.jsonl")):
        d[row["id"]] = (row["text"], row.get("article", ""))
    return d


def load_vocab(corpus):
    rows = list(lib.read_jsonl(lib.vocab_file(corpus)))   # collapsed if present
    rows.sort(key=lambda r: r["i"])          # enforce index order
    return rows


def sample_occurrences(occ, sentences, N, per_article_cap, rng):
    """Uniform sample up to N occurrences from the reservoir, <=cap per article."""
    order = occ[:]
    rng.shuffle(order)
    picked = []
    per_article = {}
    for sid, form in order:
        art = sentences.get(sid, ("", ""))[1]
        if per_article.get(art, 0) >= per_article_cap:
            continue
        picked.append((sid, form))
        per_article[art] = per_article.get(art, 0) + 1
        if len(picked) >= N:
            break
    return picked


# --------------------------------------------------------------------------- model

def load_model(model_cfg):
    import torch
    from transformers import AutoModel, AutoTokenizer

    mid = model_cfg["id"]
    revision = model_cfg.get("revision", "main")
    dtype = getattr(torch, model_cfg.get("dtype", "bfloat16"))
    tok = AutoTokenizer.from_pretrained(mid, revision=revision)

    kwargs = {"revision": revision, "torch_dtype": dtype, "output_hidden_states": True}
    if model_cfg.get("quantization") == "nf4":
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True)
        kwargs["device_map"] = "auto"
        model = AutoModel.from_pretrained(mid, **kwargs)
    else:
        model = AutoModel.from_pretrained(mid, **kwargs)
        model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    d_model = model.config.hidden_size
    n_layers = getattr(model.config, "num_hidden_layers", None)
    return tok, model, d_model, n_layers


def pick_layer_index(embed_cfg, n_layers):
    """Return None for 'last' (use last_hidden_state), else a hidden_states index."""
    if embed_cfg.get("layer") == "last":
        return None
    frac = embed_cfg.get("layer_frac", 0.65)
    # hidden_states has n_layers+1 entries (embeddings at 0). Clamp to [1, n_layers].
    idx = int(round(frac * n_layers))
    return max(1, min(n_layers, idx))


# --------------------------------------------------------------------------- pooling

def build_input(template, sentence, form):
    """Return (input_text, char_start, char_end) of the target form in input_text,
    with a word-window guard so the tokenized length stays small."""
    import re
    m = lib.word_boundary(form).search(sentence)
    if m is None:
        # fold mismatch (rare) — fall back to plain find
        idx = sentence.find(form)
        if idx < 0:
            return None
        s, e = idx, idx + len(form)
    else:
        s, e = m.start(), m.end()
    # window guard: keep ~40 words on each side of the form (sentences are already
    # <=60 words, so this rarely fires; the token cap in the tokenizer is the hard
    # limit). Crop on whitespace to preserve offsets simply.
    left = sentence.rfind(" ", 0, max(0, s - 240))
    right = sentence.find(" ", e + 240)
    lo = 0 if left < 0 else left + 1
    hi = len(sentence) if right < 0 else right
    cropped = sentence[lo:hi]
    s -= lo
    e -= lo
    prefix = template.split("{text}")[0]
    text_in = template.replace("{text}", cropped)
    off = len(prefix)
    return text_in, off + s, off + e


def embed_lemma(tok, model, layer_idx, template, occ_samples, sentences, window_tokens, batch_size, d_model):
    import torch
    inputs_meta = []
    for sid, form in occ_samples:
        sent = sentences.get(sid, ("", ""))[0]
        built = build_input(template, sent, form)
        if built is not None:
            inputs_meta.append(built)
    if not inputs_meta:
        return None

    acc = np.zeros(d_model, dtype=np.float64)
    n_used = 0
    device = next(model.parameters()).device
    for i in range(0, len(inputs_meta), batch_size):
        chunk = inputs_meta[i:i + batch_size]
        texts = [c[0] for c in chunk]
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
                  max_length=window_tokens, return_offsets_mapping=True)
        offsets = enc.pop("offset_mapping")
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc)
        hs = out.last_hidden_state if layer_idx is None else out.hidden_states[layer_idx]
        hs = hs.float()  # cast bf16 -> fp32 BEFORE accumulation (landmine #7)
        for r, (_, cs, ce) in enumerate(chunk):
            om = offsets[r].tolist()
            tok_idx = [j for j, (a, b) in enumerate(om)
                       if not (a == 0 and b == 0) and a < ce and b > cs]
            if not tok_idx:
                continue
            vec = hs[r, tok_idx, :].mean(dim=0).cpu().numpy().astype(np.float64)
            acc += vec
            n_used += 1
    if n_used == 0:
        return None
    v = (acc / n_used).astype(np.float32)
    nrm = np.linalg.norm(v)
    if nrm > 0:
        v = v / nrm
    return v


# --------------------------------------------------------------------------- checkpoint

def load_checkpoint(ds, count, d_model):
    vpath = lib.path("work", ds, "vectors.f32.npy")
    ppath = lib.path("work", ds, "vectors.progress")
    if os.path.exists(vpath) and os.path.exists(ppath):
        arr = np.load(vpath)
        done = int(open(ppath).read().strip())
        if arr.shape == (count, d_model):
            lib.log(f"resuming from checkpoint: {done}/{count} done")
            return arr, done
    return np.zeros((count, d_model), dtype=np.float32), 0


def save_checkpoint(ds, arr, done):
    os.makedirs(lib.path("work", ds), exist_ok=True)
    np.save(lib.path("work", ds, "vectors.f32.npy"), arr)
    with open(lib.path("work", ds, "vectors.progress"), "w") as f:
        f.write(str(done))


# --------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description="Contextual word vectors (GPU)")
    ap.add_argument("--config", required=True, help="dataset id, e.g. kawiki-eg")
    ap.add_argument("--limit", type=int, default=None, help="process only first K lemmas (smoke/ablation)")
    args = ap.parse_args()

    cfg = lib.load_config(args.config)
    corpus = cfg["corpus"]
    embed_cfg = cfg["embed"]
    N = embed_cfg.get("N", 64)
    cap = embed_cfg.get("per_article_cap", 2)
    window_tokens = embed_cfg.get("window_tokens", 96)
    batch_size = embed_cfg.get("batch_size", 32)
    template = embed_cfg.get("prompt_template", "{text}")
    base_seed = embed_cfg.get("seed", 20260704)
    ckpt_every = embed_cfg.get("checkpoint_every", 2000)

    lib.log(f"[{cfg['id']}] loading corpus '{corpus}' + model '{cfg['model']['id']}'")
    sentences = load_sentences(corpus)
    vocab = load_vocab(corpus)
    count = len(vocab)
    if args.limit:
        count = min(count, args.limit)

    tok, model, d_model, n_layers = load_model(cfg["model"])
    layer_idx = pick_layer_index(embed_cfg, n_layers)
    lib.log(f"d_model={d_model} n_layers={n_layers} layer_idx={layer_idx} count={count}")

    vectors, done = load_checkpoint(cfg["id"], count, d_model)

    for i in range(done, count):
        row = vocab[i]
        rng = random.Random(base_seed ^ (i * 2654435761 & 0xFFFFFFFF))
        samples = sample_occurrences(row["occ"], sentences, N, cap, rng)
        v = embed_lemma(tok, model, layer_idx, template, samples, sentences,
                        window_tokens, batch_size, d_model)
        if v is not None:
            vectors[i] = v
        if (i + 1) % ckpt_every == 0:
            save_checkpoint(cfg["id"], vectors, i + 1)
            lib.log(f"  [{cfg['id']}] {i + 1}/{count} embedded")

    save_checkpoint(cfg["id"], vectors, count)
    # drop the progress marker on clean completion
    pp = lib.path("work", cfg["id"], "vectors.progress")
    if os.path.exists(pp):
        os.remove(pp)
    lib.log(f"[{cfg['id']}] done -> {lib.path('work', cfg['id'], 'vectors.f32.npy')} ({count}×{d_model})")


if __name__ == "__main__":
    main()
