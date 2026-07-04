#!/usr/bin/env python3
"""Stage 1a — corpus extraction.

kawiki:  read the Georgian Wikipedia ZIM (read-only mount), strip HTML to
         paragraph text, split into Georgian sentences, dedup globally.
vef:     read the in-repo ვეფხისტყაოსანი CSV, group verse lines into stanzas.

Output: work/<corpus>/sentences.jsonl  ->  {"id": int, "article": str, "text": str}

Landmines honored here:
  * #10 Georgian has no capitalization — sentence splitting is punctuation +
        abbreviation-blocklist only; the Georgian-char class is the FULL block
        U+10D0..U+10FF including the archaic letters ვეფხისტყაოსანი needs.
"""
import argparse
import csv
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib  # noqa: E402

PROT = "\x00"  # placeholder for a protected (non-splitting) '.'


def split_sentences(text, abbrevs):
    """Split Georgian paragraph text into sentences.

    No case cue exists, so we split on [.!?…]+ followed by whitespace, after
    neutralizing the periods inside known abbreviations and single-letter
    initials so they don't trigger a false boundary.
    """
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    # Protect multi-char abbreviations (longest first, so "ე.წ." beats "ე.").
    for ab in sorted(abbrevs, key=len, reverse=True):
        if ab in text:
            text = text.replace(ab, ab.replace(".", PROT))
    # Protect single Georgian-letter initials: "ი." (not flanked by other letters).
    text = re.sub(rf"(?<![{lib.GE}])([{lib.GE}])\.", rf"\1{PROT}", text)
    # Split after sentence punctuation followed by whitespace.
    parts = re.split(r"(?<=[.!?…])\s+", text)
    return [p.replace(PROT, ".").strip() for p in parts if p.strip()]


def keep_sentence(text, cfg_extract):
    """Length + Georgian-density gate."""
    nwords = len(text.split())
    if nwords < cfg_extract.get("min_words", 4):
        return False
    mx = cfg_extract.get("max_words")
    if mx is not None and nwords > mx:
        return False
    if lib.georgian_ratio(text) < cfg_extract.get("min_georgian_ratio", 0.60):
        return False
    return True


# --------------------------------------------------------------------------- kawiki

def _html_to_paragraphs(html):
    from selectolax.parser import HTMLParser
    tree = HTMLParser(html)
    if tree.body is None:
        return []
    junk = ("table, sup, style, script, math, figure, figcaption, "
            ".reference, .references, .reflist, .infobox, .navbox, .navbox-inner, "
            ".mw-editsection, .thumb, .gallery, .noprint, .metadata, .hatnote, "
            "ol.references, .mw-empty-elt")
    for node in tree.css(junk):
        node.decompose()
    out = []
    for p in tree.css("p"):
        t = p.text(separator=" ", strip=True)
        if t:
            out.append(t)
    return out


def extract_kawiki(cfg, limit=None):
    from libzim.reader import Archive

    zim_dir = os.environ.get("WE_ZIM_DIR", os.path.expanduser("~/.local/share/kiwix-desktop"))
    zim_path = os.path.join(zim_dir, cfg["zim"])
    if not os.path.exists(zim_path):
        lib.log(f"ERROR: ZIM not found: {zim_path} (set WE_ZIM_DIR)")
        sys.exit(2)
    lib.log(f"opening {zim_path}")
    archive = Archive(zim_path)

    abbrevs = set(cfg["extract"].get("abbrev_blocklist", []))
    cfg_extract = cfg["extract"]

    total_entries = archive.all_entry_count
    lib.log(f"{total_entries} entries; scanning articles…")

    seen = set()              # global dedup by sentence hash
    out_path = lib.path("work", cfg["name"], "sentences.jsonl")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    sid = 0
    n_articles = 0
    with open(out_path, "w", encoding="utf-8") as fout:
        for i in range(total_entries):
            try:
                entry = archive._get_entry_by_id(i)
                if entry.is_redirect:
                    continue
                item = entry.get_item()
                mimetype = item.mimetype or ""
                if "html" not in mimetype:
                    continue
                html = bytes(item.content).decode("utf-8", "ignore")
                title = entry.title or item.title or ""
            except Exception as e:  # noqa: BLE001 — one bad entry must not abort the pass
                continue

            paras = _html_to_paragraphs(html)
            if not paras:
                continue
            n_articles += 1
            for para in paras:
                for sent in split_sentences(para, abbrevs):
                    if not keep_sentence(sent, cfg_extract):
                        continue
                    h = hash(sent)
                    if h in seen:
                        continue
                    seen.add(h)
                    fout.write(lib_json({"id": sid, "article": title, "text": sent}))
                    sid += 1
            if limit and n_articles >= limit:
                lib.log(f"--limit {limit} reached")
                break
            if n_articles % 5000 == 0 and n_articles:
                lib.log(f"  {n_articles} articles, {sid} sentences…")

    lib.log(f"done: {n_articles} articles -> {sid} unique sentences -> {out_path}")


# --------------------------------------------------------------------------- vef

def extract_vef(cfg, limit=None):
    src = lib.path(cfg["source"])
    if not os.path.exists(src):
        lib.log(f"ERROR: vef source not found: {src}")
        sys.exit(2)
    group_key = cfg["extract"].get("group_by", "strophe_id")
    min_ratio = cfg["extract"].get("min_georgian_ratio", 0.60)

    out_path = lib.path("work", cfg["name"], "sentences.jsonl")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    rows = list(csv.DictReader(open(src, encoding="utf-8")))
    # Group consecutive lines by (chapter_id, group_key) so a stanza = one context.
    groups = []
    cur_key = None
    cur_lines = []
    cur_chapter = ""
    for r in rows:
        key = (r.get("chapter_id", ""), r.get(group_key, ""))
        if key != cur_key:
            if cur_lines:
                groups.append((cur_chapter, " ".join(cur_lines)))
            cur_key = key
            cur_lines = []
            cur_chapter = r.get("chapter", "")
        line = (r.get("line") or "").strip().strip('"')
        if line:
            cur_lines.append(line)
    if cur_lines:
        groups.append((cur_chapter, " ".join(cur_lines)))

    sid = 0
    with open(out_path, "w", encoding="utf-8") as fout:
        for chapter, text in groups:
            text = re.sub(r"\s+", " ", text).strip()
            if not text or lib.georgian_ratio(text) < min_ratio:
                continue
            fout.write(lib_json({"id": sid, "article": chapter, "text": text}))
            sid += 1
            if limit and sid >= limit:
                break
    lib.log(f"done: {len(groups)} stanzas -> {sid} contexts -> {out_path}")


# --------------------------------------------------------------------------- util

def lib_json(obj):
    import json
    return json.dumps(obj, ensure_ascii=False) + "\n"


def main():
    ap = argparse.ArgumentParser(description="Extract a corpus to sentences.jsonl")
    ap.add_argument("--corpus", required=True, help="kawiki | vef")
    ap.add_argument("--limit", type=int, default=None, help="cap articles/contexts (smoke test)")
    args = ap.parse_args()

    cfg = lib.load_config(lib.path("configs", "corpus", args.corpus + ".yaml"))
    if args.corpus == "vef":
        extract_vef(cfg, args.limit)
    else:
        extract_kawiki(cfg, args.limit)


if __name__ == "__main__":
    main()
