# Vendored verbatim from Anbani.Spellcheck/src/affix.py — do not edit here.
# Re-copy if the upstream expander changes. Stdlib-only (argparse, sys).
"""Hunspell .aff/.dic affix expander — turns stems + affix rules into surface forms.

This is a focused, deterministic, stdlib-only reimplementation of the subset of
Hunspell that the Georgian dictionary (`ka_GE.{aff,dic}`) actually uses. It is
*not* a general Hunspell engine; features the file never references (COMPOUND*,
AF flag aliasing, COMPLEXPREFIXES, ICONV/OCONV, FULLSTRIP) are intentionally
absent. What it does implement, faithfully:

  * `FLAG long`            — every flag is exactly two characters.
  * `PFX` / `SFX` classes  — strip + add + condition, with cross-product (Y/N).
  * continuation flags     — an affix's added text may carry `/FLAGS`, enabling
                             *further* affixes; crucially these may cross types
                             (a SFX can enable a PFX — this is how Georgian wires
                             preverb+ending circumfixes).
  * twofold suffixing      — up to two suffixes (COMPLEXPREFIXES off => 1 prefix).
  * `NEEDAFFIX` (&v)       — the bare stem is not itself a word; only affixed forms.
  * `CIRCUMFIX` (&x)       — an affix marked &x is valid only paired with another
                             &x affix of the opposite side (preverb <-> ending).
  * `FORBIDDENWORD` (&!)   — forms from such entries are collected separately so
                             the build can subtract them from the union.
  * `NOSUGGEST` (&n)       — irrelevant to membership; the form stays valid.

Determinism: rules are kept in file order; expansion of an entry depends only on
the entry and the parsed .aff. No hashing, no randomness, no clock.
"""
import argparse
import sys

# Hunspell COMPLEXPREFIXES is off here -> at most one prefix and two suffixes.
MAX_PREFIX = 1
MAX_SUFFIX = 2


class Rule:
    """One PFX/SFX line: strip `strip`, append `add`, gated by `cond`."""
    __slots__ = ('strip', 'add', 'add_flags', 'cond', 'cross')

    def __init__(self, strip, add, add_flags, cond, cross):
        self.strip = strip            # chars removed from stem (SFX: end, PFX: start)
        self.add = add                # chars added ('' for the "0" affix)
        self.add_flags = add_flags    # frozenset: continuation flags carried by `add`
        self.cond = cond              # compiled condition atoms (tuple)
        self.cross = cross            # bool: may combine with an affix of the other side


class Aff:
    """Parsed .aff: affix classes plus the special flag names this file declares."""
    __slots__ = ('flag_mode', 'pfx', 'sfx', 'needaffix', 'circumfix',
                 'forbiddenword', 'nosuggest')

    def __init__(self):
        self.flag_mode = 'long'       # only 'long' is exercised; 'single'/'num' supported below
        self.pfx = {}                 # flag -> [Rule, ...]
        self.sfx = {}
        self.needaffix = None
        self.circumfix = None
        self.forbiddenword = None
        self.nosuggest = None


# ----------------------------------------------------------------------------- flags

def parse_flags(s, mode):
    """Decode a flag string per the .aff FLAG mode. Returns a frozenset of flags."""
    if not s:
        return frozenset()
    if mode == 'long':                # two ASCII chars per flag
        return frozenset(s[i:i + 2] for i in range(0, len(s) - 1, 2)) \
            if len(s) % 2 == 0 else frozenset(s[i:i + 2] for i in range(0, len(s), 2))
    if mode == 'num':                 # comma-separated decimal flags
        return frozenset(p for p in s.split(',') if p)
    return frozenset(s)               # 'single': one char per flag


# ----------------------------------------------------------------------------- condition

def compile_condition(s):
    """Compile a Hunspell affix condition into a tuple of single-char matchers.

    Supported atoms: '.' (any), '[abc]' (class), '[^abc]' (negated class), and a
    literal character. '0' means "no condition" (always matches). The rare group
    syntax `(..)` is not used in ka_GE conditions; if seen, atoms fall back to
    matching literally, which can only *over*-generate (safe for a spellchecker).
    """
    if not s or s == '0':
        return ()
    atoms = []
    i, n = 0, len(s)
    while i < n:
        c = s[i]
        if c == '.':
            atoms.append(('.',))
            i += 1
        elif c == '[':
            j = s.find(']', i + 1)
            if j == -1:                       # malformed; treat '[' literally
                atoms.append(('c', c))
                i += 1
                continue
            body = s[i + 1:j]
            neg = body.startswith('^')
            if neg:
                body = body[1:]
            atoms.append(('s', frozenset(body), neg))
            i = j + 1
        else:
            atoms.append(('c', c))
            i += 1
    return tuple(atoms)


def _atom_match(atom, ch):
    t = atom[0]
    if t == '.':
        return True
    if t == 'c':
        return ch == atom[1]
    inset = ch in atom[1]              # ('s', set, neg)
    return (not inset) if atom[2] else inset


def _match_end(word, atoms):
    k = len(atoms)
    if k == 0:
        return True
    if k > len(word):
        return False
    base = len(word) - k
    return all(_atom_match(atoms[i], word[base + i]) for i in range(k))


def _match_start(word, atoms):
    k = len(atoms)
    if k == 0:
        return True
    if k > len(word):
        return False
    return all(_atom_match(atoms[i], word[i]) for i in range(k))


# ----------------------------------------------------------------------------- apply

def apply_sfx(word, rule):
    """Apply a suffix rule to `word`; return the new word or None if inapplicable."""
    if rule.strip and not word.endswith(rule.strip):
        return None
    if not _match_end(word, rule.cond):
        return None
    keep = len(word) - len(rule.strip)
    if keep <= 0:                     # FULLSTRIP off: never strip the whole stem
        return None
    return word[:keep] + rule.add


def apply_pfx(word, rule):
    """Apply a prefix rule to `word`; return the new word or None if inapplicable."""
    if rule.strip and not word.startswith(rule.strip):
        return None
    if not _match_start(word, rule.cond):
        return None
    if len(rule.strip) >= len(word):  # FULLSTRIP off
        return None
    return rule.add + word[len(rule.strip):]


# ----------------------------------------------------------------------------- parse .aff

def parse_aff(path):
    aff = Aff()
    cross_of = {}                     # flag -> bool, from class headers (PFX/SFX ... Y/N count)
    with open(path, encoding='utf-8') as f:
        for raw in f:
            line = raw.rstrip('\n')
            stripped = line.strip()
            if not stripped or stripped.startswith('#'):
                continue
            parts = stripped.split()
            key = parts[0]
            if key == 'FLAG' and len(parts) >= 2:
                aff.flag_mode = parts[1].lower()       # 'long' / 'num' / 'utf-8'/'single'
                if aff.flag_mode in ('utf-8', 'utf8'):
                    aff.flag_mode = 'single'
            elif key == 'NEEDAFFIX' or key == 'PSEUDOROOT':
                aff.needaffix = parts[1]
            elif key == 'CIRCUMFIX':
                aff.circumfix = parts[1]
            elif key == 'FORBIDDENWORD':
                aff.forbiddenword = parts[1]
            elif key == 'NOSUGGEST':
                aff.nosuggest = parts[1]
            elif key in ('PFX', 'SFX'):
                flag = parts[1]
                # Header line: `PFX flag Y|N count`
                if len(parts) >= 4 and parts[2] in ('Y', 'N') and parts[3].isdigit():
                    cross_of[flag] = (parts[2] == 'Y')
                    (aff.pfx if key == 'PFX' else aff.sfx).setdefault(flag, [])
                    continue
                # Rule line: `PFX flag strip add[/flags] cond [morph...]`
                strip = '' if parts[2] == '0' else parts[2]
                add_field = parts[3]
                if '/' in add_field:
                    add_part, flagstr = add_field.split('/', 1)
                    add = '' if add_part == '0' else add_part
                    add_flags = parse_flags(flagstr, aff.flag_mode)
                else:
                    add = '' if add_field == '0' else add_field
                    add_flags = frozenset()
                cond = compile_condition(parts[4]) if len(parts) >= 5 else ()
                rule = Rule(strip, add, add_flags, cond, cross_of.get(flag, True))
                (aff.pfx if key == 'PFX' else aff.sfx).setdefault(flag, []).append(rule)
            # everything else (SET, TRY, REP, MAP, KEY, ...) is suggestion-only -> ignore
    return aff


# ----------------------------------------------------------------------------- parse .dic

def parse_dic(path, aff):
    """Yield (stem, flags:frozenset) for each dictionary entry."""
    with open(path, encoding='utf-8') as f:
        seen_count = False
        for raw in f:
            line = raw.rstrip('\n')
            if not line.strip() or line.lstrip().startswith('#'):
                continue
            if not seen_count and line.strip().isdigit():
                seen_count = True               # leading entry-count line
                continue
            seen_count = True
            token = line.split(None, 1)[0]      # drop trailing morphology fields
            if '/' in token:
                word, flagstr = token.split('/', 1)
                flags = parse_flags(flagstr, aff.flag_mode)
            else:
                word, flags = token, frozenset()
            if word:
                yield word, flags


# ----------------------------------------------------------------------------- expand

def expand_entry(aff, stem, flags):
    """Generate all surface forms of one dictionary entry.

    Returns (valid, forbidden): two sets of strings. `forbidden` is non-empty
    only for entries flagged FORBIDDENWORD; the build subtracts it from the union.

    Model (COMPLEXPREFIXES off): suffixes are inner, the single prefix is outer.
    A suffix may inject continuation flags that enable the outer prefix — that is
    exactly how a Georgian circumfix (preverb + verb ending) is expressed.
    """
    sink = set()
    forbidden_sink = set()
    is_forbidden = aff.forbiddenword is not None and aff.forbiddenword in flags
    target = forbidden_sink if is_forbidden else sink
    cx = aff.circumfix

    def emit(word, cx_pfx, cx_sfx):
        # Circumfix must be balanced: both halves present, or neither.
        if cx_pfx != cx_sfx:
            return
        if word:
            target.add(word)

    # Build the suffixed variants (0, 1, or 2 suffixes) reachable from the stem.
    # Each variant: (word, n_sfx, cx_sfx, cont_flags, cross_all)
    variants = [(stem, 0, False, flags, True)]

    def suffixate(word, avail, n_sfx, cx_sfx, cross_all):
        if n_sfx >= MAX_SUFFIX:
            return
        for fl in avail:
            for rule in aff.sfx.get(fl, ()):           # only real SFX classes
                nw = apply_sfx(word, rule)
                if nw is None:
                    continue
                ncx = cx_sfx or (cx is not None and cx in rule.add_flags)
                ncross = cross_all and rule.cross
                variants.append((nw, n_sfx + 1, ncx, rule.add_flags, ncross))
                suffixate(nw, rule.add_flags, n_sfx + 1, ncx, ncross)

    suffixate(stem, flags, 0, False, True)

    needaffix = aff.needaffix is not None and aff.needaffix in flags
    for word, n_sfx, cx_sfx, cont, cross_all in variants:
        # The form without any prefix (bare stem skipped if it NEEDAFFIX).
        if not (n_sfx == 0 and needaffix):
            emit(word, False, cx_sfx)
        # One prefix, drawn from the stem's own flags and from continuation flags
        # the suffix injected (the preverb hook for circumfixes).
        for fl in (flags | cont):
            for rule in aff.pfx.get(fl, ()):           # only real PFX classes
                if n_sfx > 0 and not (rule.cross and cross_all):
                    continue                           # cross-product needs both Y
                pw = apply_pfx(word, rule)
                if pw is None:
                    continue
                cx_pfx = (cx is not None and cx in rule.add_flags)
                emit(pw, cx_pfx, cx_sfx)

    return sink, forbidden_sink


# ----------------------------------------------------------------------------- CLI

def main():
    ap = argparse.ArgumentParser(description='Expand a Hunspell entry (debug).')
    ap.add_argument('--aff', required=True)
    ap.add_argument('stem')
    ap.add_argument('flags', nargs='?', default='', help='raw flag string, e.g. "PI&v"')
    args = ap.parse_args()
    aff = parse_aff(args.aff)
    valid, forbidden = expand_entry(aff, args.stem, parse_flags(args.flags, aff.flag_mode))
    for w in sorted(valid):
        print(w)
    for w in sorted(forbidden):
        print('FORBIDDEN', w, file=sys.stderr)


if __name__ == '__main__':
    main()
