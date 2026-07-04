# Vendored verbatim from Anbani.Spellcheck/src/normalize.py — do not edit here.
# Re-copy if the upstream changes. Stdlib-only (unicodedata).
"""Shared text normalization — used identically at build time and query time.

A word that goes into the DAWG and a word that is later looked up MUST pass
through the same function, or membership silently breaks. Keep this tiny and
mirror it exactly in `src/spellcheck.js`.

Rules:
  * NFC-normalize (canonical composition).
  * Fold Mtavruli (U+1C90..U+1CBF, Georgian "caps") onto Mkhedruli, so an
    all-caps query validates against a lower-case lexicon.
  * A token is a *word* iff every character is a Georgian letter (Mkhedruli,
    Asomtavruli, or Nuskhuri). Anything with a space/digit/Latin/punctuation is
    not a single dictionary entry and is dropped from the corpus.
"""
import unicodedata

# Mtavruli U+1C90..U+1CBF maps 1:1 onto Mkhedruli U+10D0..U+10FF.
_MTAVRULI_LO, _MTAVRULI_HI = 0x1C90, 0x1CBF
_MTAVRULI_SHIFT = 0x1C90 - 0x10D0


def fold(s):
    """NFC + Mtavruli->Mkhedruli. The single source of truth for a word's form."""
    s = unicodedata.normalize('NFC', s.strip())
    if any(_MTAVRULI_LO <= ord(c) <= _MTAVRULI_HI for c in s):
        s = ''.join(
            chr(ord(c) - _MTAVRULI_SHIFT) if _MTAVRULI_LO <= ord(c) <= _MTAVRULI_HI else c
            for c in s
        )
    return s


def is_georgian_letter(ch):
    o = ord(ch)
    return (0x10D0 <= o <= 0x10FF      # Mkhedruli (modern)
            or 0x10A0 <= o <= 0x10C5   # Asomtavruli (archaic caps)
            or 0x10D0 <= o <= 0x10FA
            or 0x2D00 <= o <= 0x2D25   # Nuskhuri
            or 0x1C90 <= o <= 0x1CBF)  # Mtavruli (folded away by fold(), kept for safety)


def is_word(s):
    """True iff non-empty and entirely Georgian letters (after fold)."""
    return bool(s) and all(is_georgian_letter(c) for c in s)
