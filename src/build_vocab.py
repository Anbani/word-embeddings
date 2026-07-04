#!/usr/bin/env python3
"""Stage 1b — vocabulary + form->lemma map + occurrence index.

kawiki: lemma pool = Hunspell .dic stems (affix-expanded to their surface forms)
        ∪ the dlab clean-lemma list. Corpus tokens are mapped form->lemma; the
        lemma's frequency is the sum of its forms' counts. Ambiguous forms
        (mapping to >1 lemma) are dropped. Cut at 30K by frequency (min_occ=8).
vef:    raw surface forms (no lemmatization — archaic morphology defeats the
        modern affix rules). freq>=3.

The vocab INDEX ORDER IS FROZEN here and reused verbatim by every dataset on
this corpus; `vocab_hash` (sha256 of the label list) is asserted equal across
kawiki datasets downstream (landmines #5, #11, #17).

Output: work/<corpus>/vocab.jsonl   {i, word, freq, occ: [[sid, form], ...]}
        work/<corpus>/vocab_meta.json
"""
import argparse
import csv
import json
import os
import random
import re
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib  # noqa: E402
from vendor import affix as affixmod  # noqa: E402
from vendor import normalize as norm  # noqa: E402

OCC_CAP = 400          # reservoir size per lemma (unbiased sample of occurrences)
SEED = 20260704        # reservoir RNG seed (determinism)
TOKEN_RE = re.compile(rf"[{lib.GE}]+")


def _read_wordfile(p):
    out = set()
    if p and os.path.exists(lib.path(p)):
        for line in open(lib.path(p), encoding="utf-8"):
            line = line.strip()
            if line and not line.startswith("#"):
                out.add(norm.fold(line))
    return out


def load_blocklist(cfg_vocab):
    """Wiki-markup blocklist ∪ Georgian stopwords (both filtered from the vocab)."""
    return _read_wordfile(cfg_vocab.get("blocklist")) | _read_wordfile(cfg_vocab.get("stopwords"))


# --------------------------------------------------------------------------- form->lemma

def build_form2lemma(cfg_vocab):
    """Expand every .dic entry to surface forms, invert to form->lemma; union the
    dlab lemma list as identity mappings. Ambiguous forms are dropped."""
    aff_path = lib.path(cfg_vocab["aff"])
    aff = affixmod.parse_aff(aff_path)

    form2lemma = {}
    ambiguous = set()

    def add(form, lemma):
        form = norm.fold(form)
        if not norm.is_word(form) or len(form) < 2:
            return
        prev = form2lemma.get(form)
        if prev is None:
            form2lemma[form] = lemma
        elif prev != lemma:
            ambiguous.add(form)

    n_dic = 0
    for src in cfg_vocab.get("lemma_sources", []):
        p = lib.path(src)
        if not os.path.exists(p):
            lib.log(f"  lemma source missing, skipped: {src}")
            continue
        if src.endswith(".dic"):
            for stem, flags in affixmod.parse_dic(p, aff):
                lemma = norm.fold(stem)
                if not norm.is_word(lemma) or len(lemma) < 2:
                    continue
                valid, _forbidden = affixmod.expand_entry(aff, stem, flags)
                add(lemma, lemma)                 # the stem maps to itself
                for f in valid:
                    add(f, lemma)
                n_dic += 1
                if n_dic % 20000 == 0:
                    lib.log(f"  expanded {n_dic} dic entries; {len(form2lemma)} forms so far")
        else:  # CSV: one lemma per line (col 0), possibly a 'word' header
            with open(p, encoding="utf-8") as f:
                reader = csv.reader(f)
                for row in reader:
                    if not row:
                        continue
                    w = norm.fold(row[0].strip())
                    if w == "word" or not norm.is_word(w) or len(w) < 2:
                        continue
                    add(w, w)

    for f in ambiguous:
        form2lemma.pop(f, None)
    lib.log(f"form->lemma: {len(form2lemma)} forms, {len(ambiguous)} ambiguous dropped, "
            f"{n_dic} dic entries")
    return form2lemma


# --------------------------------------------------------------------------- counting

def count_occurrences(sentences_path, resolve, rng):
    """Stream sentences; map each Georgian token via `resolve(form)->lemma|None`.
    Returns (freq: dict, occ: dict lemma-> reservoir list of [sid, form])."""
    freq = {}
    occ = {}
    n_sent = 0
    for row in lib.read_jsonl(sentences_path):
        sid = row["id"]
        text = row["text"]
        n_sent += 1
        for raw in TOKEN_RE.findall(text):
            form = norm.fold(raw)
            lemma = resolve(form)
            if lemma is None:
                continue
            c = freq.get(lemma, 0) + 1
            freq[lemma] = c
            res = occ.get(lemma)
            if res is None:
                occ[lemma] = [[sid, form]]
            elif len(res) < OCC_CAP:
                res.append([sid, form])
            else:                                  # reservoir algorithm R
                j = rng.randint(0, c - 1)
                if j < OCC_CAP:
                    res[j] = [sid, form]
        if n_sent % 200000 == 0:
            lib.log(f"  scanned {n_sent} sentences; {len(freq)} lemmas seen")
    lib.log(f"counted over {n_sent} sentences; {len(freq)} distinct lemmas/forms")
    return freq, occ


# --------------------------------------------------------------------------- select + write

def select_and_write(cfg, freq, occ):
    v = cfg["vocab"]
    blocklist = load_blocklist(v)
    min_occ = v.get("min_occ", 8)
    min_freq = v.get("min_freq", min_occ)
    target = v.get("target_size", 30000)

    thr = max(min_occ, min_freq)
    # Each candidate is relabeled with its DOMINANT SURFACE FORM (most common form
    # among sampled occurrences) so the display label is a real word, not a bare
    # Hunspell stem (იყ→იყო, რომლი→რომელი). The stem is kept as `lemma`.
    cands = []
    for stem, f in freq.items():
        if f < thr or stem in blocklist or not norm.is_word(stem) or len(stem) < 2:
            continue
        o = occ.get(stem, [])
        dom = Counter(form for _, form in o).most_common(1)
        label = dom[0][0] if dom else stem
        if label in blocklist or not norm.is_word(label) or len(label) < 2:
            continue
        cands.append((label, stem, f, o))
    # freeze order: frequency desc, stem tiebreak (stable regardless of relabeling)
    cands.sort(key=lambda t: (-t[2], t[1]))
    if len(cands) > target:
        cands = cands[:target]
    if len(cands) > 65535:
        lib.log(f"ERROR: vocab {len(cands)} exceeds u16 ceiling 65535")
        sys.exit(3)

    labels = [label for label, _, _, _ in cands]
    vocab_hash = lib.sha256_text("\n".join(labels))

    out_dir = lib.path("work", cfg["name"])
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "vocab.jsonl")
    with open(out_path, "w", encoding="utf-8") as fout:
        for i, (label, stem, f, o) in enumerate(cands):
            fout.write(json.dumps(
                {"i": i, "word": label, "lemma": stem, "freq": f, "occ": o},
                ensure_ascii=False) + "\n")

    meta = {
        "corpus": cfg["name"],
        "count": len(labels),
        "vocab_hash": vocab_hash,
        "min_occ": min_occ,
        "target_size": target,
        "occ_cap": OCC_CAP,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(os.path.join(out_dir, "vocab_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    lib.log(f"vocab: {len(labels)} entries -> {out_path}")
    lib.log(f"vocab_hash: {vocab_hash}")
    lib.log("top 40 by frequency (QA — eyeball for wiki-markup / junk):")
    for label, stem, f, _ in cands[:40]:
        tag = "" if label == stem else f"  (stem: {stem})"
        lib.log(f"  {f:>8}  {label}{tag}")


# --------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description="Build vocab + form->lemma + occurrence index")
    ap.add_argument("--corpus", required=True, help="kawiki | vef")
    args = ap.parse_args()

    cfg = lib.load_config(lib.path("configs", "corpus", args.corpus + ".yaml"))
    sentences_path = lib.path("work", cfg["name"], "sentences.jsonl")
    if not os.path.exists(sentences_path):
        lib.log(f"ERROR: run `make corpus CORPUS={args.corpus}` first ({sentences_path} missing)")
        sys.exit(2)

    rng = random.Random(SEED)

    if cfg["vocab"].get("aff"):        # kawiki: lemmatized
        form2lemma = build_form2lemma(cfg["vocab"])
        resolve = form2lemma.get
    else:                              # vef: raw forms (identity)
        resolve = lambda form: form    # noqa: E731

    freq, occ = count_occurrences(sentences_path, resolve, rng)
    select_and_write(cfg, freq, occ)


if __name__ == "__main__":
    main()
