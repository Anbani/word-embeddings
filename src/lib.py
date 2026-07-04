"""Shared helpers for the word-embeddings pipeline: config loading, Georgian
text constants, deterministic RNG, and jsonl I/O.

Kept dependency-light on purpose: only `yaml` is imported (present in both the
generate and build images). Everything else is stdlib so the CPU build stage
stays slim.
"""
import hashlib
import json
import os
import re
import sys

# NOTE: `yaml` is imported lazily inside load_config so this module stays
# stdlib-only — the CI unit-test matrix imports it without pip-installing deps.

# --------------------------------------------------------------------------- paths

# repo root = parent of src/. Works whether invoked as `python src/foo.py` from
# the repo root or with an absolute path.
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SRC_DIR)


def path(*parts):
    """Absolute path under the repo root."""
    return os.path.join(ROOT, *parts)


# --------------------------------------------------------------------------- Georgian

# The FULL Georgian block U+10D0..U+10FF (ჿ). This deliberately includes the
# archaic letters ჱ ჲ ჳ ჴ ჵ ჶ (U+10F1..U+10F6) that ვეფხისტყაოსანი needs —
# never narrow this to the modern-33 subset (SDD landmine #10).
GE = "ა-ჿ"
GE_RE = re.compile(f"[{GE}]")

# Word-boundary match: `form` not flanked by other Georgian letters. Callers
# build the full pattern per form via `word_boundary(form)`.
_BOUNDARY_L = f"(?<![{GE}])"
_BOUNDARY_R = f"(?![{GE}])"


def word_boundary(form):
    """Compiled regex matching `form` at Georgian word boundaries."""
    return re.compile(_BOUNDARY_L + re.escape(form) + _BOUNDARY_R)


def georgian_ratio(text):
    """Fraction of characters in `text` that are Georgian letters."""
    if not text:
        return 0.0
    ge = sum(1 for c in text if "ა" <= c <= "ჿ")
    return ge / len(text)


# --------------------------------------------------------------------------- config

def load_config(name_or_path):
    """Load a dataset YAML config. Accepts a bare id ('kawiki-eg') or a path."""
    import yaml
    p = name_or_path
    if not os.path.exists(p):
        p = path("configs", name_or_path + ".yaml")
    with open(p, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_config_path"] = os.path.abspath(p)
    return cfg


def config_hash(cfg):
    """Stable sha256 of a config's semantic content (ignores injected `_` keys)."""
    clean = {k: v for k, v in cfg.items() if not k.startswith("_")}
    blob = json.dumps(clean, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- jsonl

def read_jsonl(p):
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(p, rows):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    n = 0
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False))
            f.write("\n")
            n += 1
    return n


# --------------------------------------------------------------------------- misc

def log(*args):
    print(*args, file=sys.stderr, flush=True)


def work_dir(cfg):
    """Per-dataset scratch dir under work/ (gitignored, lives on the GPU node)."""
    d = path("work", cfg["id"])
    os.makedirs(d, exist_ok=True)
    return d
