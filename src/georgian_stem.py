"""Lightweight Georgian inflectional stemmer.

Georgian is agglutinative and suffixing: nouns/adjectives take case + number +
postposition endings that share the root (დედა, დედამ, დედას, დედები, დედებში…).
This strips those inflectional endings to a common root so morphological variants
collapse to one vocabulary point, and so probe evaluation can match on roots
rather than surface forms.

It is deliberately shallow — it does NOT handle vowel syncope (წელი/წლის keep
different roots) or derivational morphology (that's intentional: დედა vs დედალ
"hen" must stay distinct). stdlib-only so build_vocab, collapse_vocab and
evaluate can all share it.
"""

# Ordered longest-first so multi-char endings win before their single-char tails.
SUFFIXES = [
    "ებისთვის", "ებისგან", "ებამდე", "ისთვის", "ისადმი", "ებთან", "ებში", "ებზე",
    "ებად", "ისგან", "ამდე", "თვის", "ადმი", "ებს", "ები", "ისა", "ის", "ში",
    "ზე", "დან", "თან", "კენ", "გან", "ვით", "ებ", "ად", "მა", "სა", "ს", "ო",
    "მ", "თ", "ა", "ე", "ი",
]
MIN_STEM = 3


def stem(word):
    """Strip inflectional suffixes to a fixpoint, never below MIN_STEM chars."""
    changed = True
    while changed and len(word) > MIN_STEM:
        changed = False
        for s in SUFFIXES:
            if len(word) - len(s) >= MIN_STEM and word.endswith(s):
                word = word[:-len(s)]
                changed = True
                break
    return word
