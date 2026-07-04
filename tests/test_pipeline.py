"""Stdlib-only unit tests for the pure-python pipeline logic. These run on the
CI python matrix without pip-installing anything (no numpy/yaml/torch)."""
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.path.dirname(HERE), "src")
sys.path.insert(0, SRC)

import lib  # noqa: E402
from extract_corpus import split_sentences  # noqa: E402
from vendor import affix, normalize  # noqa: E402


class SentenceSplit(unittest.TestCase):
    def test_basic_split(self):
        s = split_sentences("ერთი წინადადება. მეორე წინადადება.", set())
        self.assertEqual(len(s), 2)

    def test_abbreviation_not_split(self):
        # ე.წ. must not create boundaries; only the final period splits.
        s = split_sentences("მან თქვა ე.წ. სიმართლე და წავიდა.", {"ე.წ."})
        self.assertEqual(len(s), 1)

    def test_single_initial_not_split(self):
        s = split_sentences("ავტორია ი. ჭავჭავაძე დიდი მწერალი.", set())
        self.assertEqual(len(s), 1)

    def test_archaic_letters_preserved(self):
        text = "ჱ ჲ ჳ ჴ ჵ ჶ არს ძველი ანბანი დიდი."
        s = split_sentences(text, set())
        self.assertEqual(len(s), 1)
        for ch in "ჱჲჳჴჵჶ":
            self.assertIn(ch, s[0])


class GeorgianHelpers(unittest.TestCase):
    def test_georgian_ratio(self):
        self.assertAlmostEqual(lib.georgian_ratio("აბგ123"), 0.5)
        self.assertEqual(lib.georgian_ratio(""), 0.0)

    def test_word_boundary(self):
        self.assertIsNotNone(lib.word_boundary("მზე").search("მზე ანათებს"))
        self.assertIsNone(lib.word_boundary("მზე").search("ამზერა"))

    def test_full_block_is_georgian(self):
        # archaic ჶ (U+10F6) counts as Georgian
        self.assertEqual(lib.georgian_ratio("ჶ"), 1.0)


class Normalize(unittest.TestCase):
    def test_mtavruli_fold(self):
        # Mtavruli Ⴀ (U+10A0? no) — use U+1C90 range: fold to Mkhedruli.
        mt = "".join(chr(0x1C90 + i) for i in range(3))   # first 3 Mtavruli
        mk = "".join(chr(0x10D0 + i) for i in range(3))
        self.assertEqual(normalize.fold(mt), mk)

    def test_is_word(self):
        self.assertTrue(normalize.is_word("სიტყვა"))
        self.assertFalse(normalize.is_word("word"))
        self.assertFalse(normalize.is_word("სიტყვა1"))


class Affix(unittest.TestCase):
    def test_parse_flags_long(self):
        fl = affix.parse_flags("ABCD", "long")
        self.assertEqual(fl, frozenset({"AB", "CD"}))

    def test_apply_sfx(self):
        rule = affix.Rule(strip="", add="ს", add_flags=frozenset(), cond=(), cross=True)
        self.assertEqual(affix.apply_sfx("კატა", rule), "კატას")

    def test_expand_smoke_on_vendored_aff(self):
        aff_path = lib.path("src", "data", "ext", "ka_GE.aff")
        if not os.path.exists(aff_path):
            self.skipTest("vendored ka_GE.aff missing")
        aff = affix.parse_aff(aff_path)
        valid, forbidden = affix.expand_entry(aff, "სახლი", frozenset())
        self.assertIn("სახლი", valid)          # bare stem present (no NEEDAFFIX)


class ConfigHash(unittest.TestCase):
    def test_sha256_text_stable(self):
        self.assertEqual(lib.sha256_text("აბგ"), lib.sha256_text("აბგ"))
        self.assertNotEqual(lib.sha256_text("ა"), lib.sha256_text("ბ"))


if __name__ == "__main__":
    unittest.main()
