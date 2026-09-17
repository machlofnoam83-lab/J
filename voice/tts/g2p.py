"""Hebrew Grapheme-to-Phoneme — written from zero.

Converts Hebrew (and embedded English/numbers) text into a phoneme sequence our
synthesiser can voice. This is the piece most "Hebrew TTS" projects skip and
then sound wrong; we do it properly:

  * final forms folded (ם -> m)
  * niqqud honoured when present (kamatz/patach -> a, segol -> e, ...)
  * geresh digraphs: ג' -> J, ג׳ -> J, ז' -> Z, צ' -> CH, ד' -> D, ת' -> T
  * gershayim (") inside a word = acronym, read letter by letter
  * silent he at word end, silent alef, shva na/nach heuristics
  * schwa handling, dagesh doubling (optional), beged-kefet softening
  * numbers spelled out in Hebrew (feminine/masculine aware enough for speech)
  * Latin words transliterated to the same phoneme inventory

Phoneme inventory (SAMPA-ish, ours):
  consonants: p b t d k g f v s z S(x) h m n l r y w H(X) A(ʔ) J(dZ) CH(tS) T(ts)
  vowels:     a e i o u  (+ long markers via prosody, not extra symbols)
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# ------------------------------------------------------------- inventory ----
PHONEMES: Dict[str, str] = {
    "p": "פ", "b": "ב", "t": "ת", "d": "ד", "k": "כ", "g": "ג", "f": "פֿ", "v": "בֿ/ו",
    "s": "ס/שׂ", "z": "ז", "S": "ש", "h": "ה", "m": "מ", "n": "נ", "l": "ל",
    "r": "ר", "y": "י", "w": "ו", "x": "ח/כ", "A": "א", "J": "ג'", "CH": "צ'",
    "T": "צ", "Q": "ק",
    "a": "ַ/ָ", "e": "ֶ/ֵ", "i": "ִ/י", "o": "וֹ", "u": "וּ",
    "_": "הפסקה",
}

VOWELS = ("a", "e", "i", "o", "u")
CONSONANTS = ("p", "b", "t", "d", "k", "g", "f", "v", "s", "z", "S", "h", "m",
              "n", "l", "r", "y", "w", "x", "A", "J", "CH", "T", "Q")

# niqqud -> vowel
NIQQUD_MAP = {
    "\u05b7": "a",   # patach
    "\u05b8": "a",   # kamatz
    "\u05b6": "e",   # segol
    "\u05b5": "e",   # tsere
    "\u05b4": "i",   # hirik
    "\u05b9": "o",   # holam
    "\u05bb": "u",   # kubuts
    "\u05b0": "",    # shva (resolved later: na vs nach)
    "\u05bc": "",    # dagesh (handled as doubling flag)
    "\u05bf": "v",   # rafe
    "\u05c1": "",    # shin dot -> S
    "\u05c2": "",    # sin dot  -> s
}

FINAL_TO_REGULAR = {"ך": "כ", "ם": "מ", "ן": "נ", "ץ": "צ", "ף": "פ"}

# base letter -> default phoneme (no niqqud)
BASE = {
    "א": "A", "ב": "v", "ג": "g", "ד": "d", "ה": "h", "ו": "w", "ז": "z",
    "ח": "x", "ט": "t", "י": "y", "כ": "x", "ל": "l", "מ": "m", "נ": "n",
    "ס": "s", "ע": "A", "פ": "f", "צ": "T", "ק": "Q", "ר": "r", "ש": "S",
    "ת": "t",
}
# beged-kefet with dagesh -> hard stop
HARD = {"ב": "b", "ג": "g", "ד": "d", "כ": "k", "פ": "p", "ת": "t"}
SOFT = {"ב": "v", "ג": "g", "ד": "d", "כ": "x", "פ": "f", "ת": "t"}

GERESH_MAP = {"ג": "J", "ז": "Z", "צ": "CH", "ד": "d", "ת": "t", "צ׳": "CH", "ח": "x", "ט": "t"}

LATIN_MAP = {
    "a": "a", "b": "b", "c": "k", "d": "d", "e": "e", "f": "f", "g": "g",
    "h": "h", "i": "i", "j": "J", "k": "k", "l": "l", "m": "m", "n": "n",
    "o": "o", "p": "p", "q": "Q", "r": "r", "s": "s", "t": "t", "u": "u",
    "v": "v", "w": "w", "x": "ks", "y": "i", "z": "z",
}

ONES = ["", "אחת", "שתיים", "שלוש", "ארבע", "חמש", "שש", "שבע", "שמונה", "תשע"]
ONES_M = ["", "אחד", "שניים", "שלושה", "ארבעה", "חמישה", "שישה", "שבעה", "שמונה", "תשעה"]
TEENS = ["עשר", "אחת עשרה", "שתים עשרה", "שלוש עשרה", "ארבע עשרה", "חמש עשרה",
         "שש עשרה", "שבע עשרה", "שמונה עשרה", "תשע עשרה"]
TENS = ["", "", "עשרים", "שלושים", "ארבעים", "חמישים", "שישים", "שבעים", "שמונים", "תשעים"]
HUNDREDS = ["", "מאה", "מאתיים", "שלוש מאות", "ארבע מאות", "חמש מאות",
            "שש מאות", "שבע מאות", "שמונה מאות", "תשע מאות"]


@dataclass
class Phoneme:
    sym: str                 # our phoneme symbol
    text: str = ""           # source graphemes (for debugging / alignment)
    stressed: bool = False
    word_start: bool = False
    word_end: bool = False
    is_vowel: bool = False
    pause: float = 0.0       # seconds of silence to insert BEFORE this phoneme


def number_to_hebrew(n: int, feminine: bool = True) -> str:
    """Spell an integer in Hebrew words (good enough for natural speech)."""
    n = int(n)
    if n == 0:
        return "אפס"
    if n < 0:
        return "מינוס " + number_to_hebrew(-n, feminine)
    if n < 1_000_000_000:
        parts: List[str] = []
        millions = n // 1_000_000
        if millions:
            parts.append((number_to_hebrew(millions, False) if millions > 2 else
                          ("מיליון" if millions == 1 else "שני מיליון")) +
                         (" מיליון" if millions > 2 else ""))
            n %= 1_000_000
        thousands = n // 1000
        if thousands:
            if thousands == 1:
                parts.append("אלף")
            elif thousands == 2:
                parts.append("אלפיים")
            else:
                parts.append(number_to_hebrew(thousands, True) + " אלפים" if thousands < 11
                             else number_to_hebrew(thousands, False) + " אלף")
            n %= 1000
        hundreds = n // 100
        if hundreds:
            parts.append(HUNDREDS[hundreds] if hundreds <= 9 else number_to_hebrew(hundreds, True))
            n %= 100
        if n >= 10:
            if n < 20:
                parts.append(TEENS[n - 10])
                n = 0
            else:
                parts.append(TENS[n // 10])
                n %= 10
        if n:
            parts.append((ONES if feminine else ONES_M)[n])
        return " ".join(p for p in parts if p)
    return str(n)


def split_words(text: str) -> List[str]:
    return re.findall(r"[\w'\u05f3\u05f4\u0591-\u05bc\u05d0-\u05ea]+|[^\s\w]", text, re.UNICODE)


class HebrewG2P:
    """Grapheme -> phoneme converter."""

    def __init__(self, spell_acronyms: bool = True, keep_pauses: bool = True) -> None:
        self.spell_acronyms = spell_acronyms
        self.keep_pauses = keep_pauses

    # ---------------------------------------------------------------- main --
    # NOTE: \b does NOT fire between a Hebrew letter and a digit (both are \w),
    # so we assert digit boundaries explicitly.
    CLOCK_RE = re.compile(r"(?<![\d:])(\d{1,2}):(\d{2})(?![\d:])")

    def convert(self, text: str) -> List[Phoneme]:
        out: List[Phoneme] = []
        text = text.replace("\u200f", "").replace("\u200e", "")
        def _clock(m: "re.Match[str]") -> str:
            phrase = _clock_phrase(m.group(1), m.group(2))
            head = text[: m.start()].rstrip()
            if re.search(r"(השעה|בשעה|השעה היא|בשעה)$", head[-14:]):
                return phrase.replace("השעה ", "", 1)
            return phrase
        text = self.CLOCK_RE.sub(_clock, text)
        for chunk in re.split(r"(\s+)", text):
            if not chunk:
                continue
            if chunk.isspace():
                continue
            for token in re.findall(r"[^\s]+", chunk):
                out.extend(self._token(token))
        return _post_process(out)

    # ------------------------------------------------------------- tokens --
    def _token(self, tok: str) -> List[Phoneme]:
        # punctuation -> pause markers
        if tok in (",", ";", ":", "־"):
            return [Phoneme("_", tok, pause=0.14)]
        if tok in (".", "!", "?", "…"):
            return [Phoneme("_", tok, pause=0.30)]
        if tok in ("(", "[", "{", '"', "'"):
            return []
        if tok in (")", "]", "}", "—", "-"):
            return [Phoneme("_", tok, pause=0.10)]

        # pure numbers -> Hebrew words
        if re.fullmatch(r"-?\d+", tok):
            words = number_to_hebrew(int(tok))
            return self._hebrew_words(words)
        if re.fullmatch(r"-?\d+\.\d+", tok):
            whole, frac = tok.split(".")
            ph = self._hebrew_words(number_to_hebrew(int(whole)))
            ph.append(Phoneme("_", ".", pause=0.10))
            ph += self._hebrew_words("נקודה")
            for ch in frac:
                ph += self._hebrew_words(number_to_hebrew(int(ch)))
            return ph
        if tok.endswith("%"):
            return self._hebrew_words(number_to_hebrew(int(float(tok[:-1]))) + " אחוז")

        # latin / mixed
        if re.fullmatch(r"[A-Za-z][A-Za-z0-9_' .-]*", tok):
            return self._latin(tok)

        # A trailing/leading punctuation mark must not be glued to the word,
        # otherwise the lexicon lookup misses ("בבוקר." != "בבוקר").
        core = tok.strip(".,!?;:\"'()[]{}—–-־")
        if core and core != tok:
            head = tok[: len(tok) - len(tok.lstrip(".,!?;:\"'()[]{}—–-־"))]
            tail = tok[len(core) + len(head):]
            out = self._punct(head)
            out += self._hebrew_words(core) if re.search(r"[\u05d0-\u05ea]", core) else self._latin(core)
            out += self._punct(tail)
            return out

        return self._hebrew_words(tok)

    def _punct(self, marks: str) -> List["Phoneme"]:
        out: List["Phoneme"] = []
        for m in marks:
            if m in (".", "!", "?", "…"):
                out.append(Phoneme("_", m, pause=0.30))
            elif m in (",", ";", ":", "־"):
                out.append(Phoneme("_", m, pause=0.14))
            elif m in ("—", "–", "-"):
                out.append(Phoneme("_", m, pause=0.10))
        return out

    # ------------------------------------------------------------ hebrew ----
    def _hebrew_words(self, text: str) -> List[Phoneme]:
        out: List[Phoneme] = []
        for word in re.findall(r"[^\s]+", text):
            out.extend(self._hebrew_word(word))
            out.append(Phoneme("_", " ", pause=0.05))
        return out

    def _hebrew_word(self, word: str) -> List[Phoneme]:
        # acronym with gershayim: read letters
        if "\u05f4" in word and self.spell_acronyms:
            letters = [c for c in word if c in BASE]
            seq: List[Phoneme] = []
            for i, ch in enumerate(letters):
                seq.extend(self._letter_name(ch))
                if i < len(letters) - 1:
                    seq.append(Phoneme("_", "-", pause=0.09))
            return seq

        w = "".join(FINAL_TO_REGULAR.get(c, c) for c in word)
        if w in LEXICON:                       # a fully vocalised known word wins
            phs = [Phoneme(x, w, is_vowel=x in VOWELS) for x in LEXICON[w]]
            if phs:
                phs[0].word_start = True
                phs[-1].word_end = True
            self._mark_stress(phs)
            return phs
        pre, w = self._split_prefixes(w)

        i = 0
        phs: List[Phoneme] = []
        n = len(w)
        while i < n:
            ch = w[i]
            geresh = w[i + 1] in ("\u05f3", "'") if i + 1 < n else False
            dagesh = w[i + 1] == "\u05bc" if i + 1 < n else False
            niqqud = ""
            j = i + 1
            while j < n and w[j] in NIQQUD_MAP and w[j] != "\u05bc":
                niqqud += w[j]
                j += 1
            if dagesh:
                j = max(j, i + 2)

            if geresh and ch in GERESH_MAP:
                phs.append(Phoneme(GERESH_MAP[ch], ch + "\u05f3"))
                i = j + 1 if geresh else j
                if i <= j - 1:
                    i = j
                continue

            if ch in BASE:
                sym = self._consonant(ch, dagesh=dagesh, word_final=(j >= n), niqqud=niqqud)
                phs.append(Phoneme(sym, ch))
                # vowels from niqqud
                for mark in niqqud:
                    v = NIQQUD_MAP.get(mark)
                    if v:
                        phs.append(Phoneme(v, mark, is_vowel=True))
                # shva: voice it when word-initial or after a long vowel (shva na)
                if "\u05b0" in niqqud and self._shva_is_na(phs):
                    phs.append(Phoneme("e", "\u05b0", is_vowel=True))
                i = j
                continue

            if ch.isdigit():
                phs.extend(self._hebrew_words(number_to_hebrew(int(ch))))
                i += 1
                continue
            if ch in LATIN_MAP:
                for s in LATIN_MAP[ch.lower()]:
                    phs.append(Phoneme(s, ch, is_vowel=s in VOWELS))
                i += 1
                continue
            i += 1

        if not phs:
            return []
        phs[0].word_start = True
        phs[-1].word_end = True
        self._mark_stress(phs)
        return pre + self._insert_default_vowels(phs, w)

    # Single-letter prefixes that are their own syllable in Hebrew speech.
    PREFIXES: Tuple[Tuple[str, str, str], ...] = (
        ("ו", "v", "a"),    # and
        ("ב", "b", "e"),    # in
        ("כ", "k", "e"),    # as
        ("ל", "l", "e"),    # to
        ("מ", "m", "i"),    # from
        ("ש", "S", "e"),    # that
        ("ה", "h", "a"),    # the
    )

    def _split_prefixes(self, word: str) -> Tuple[List[Phoneme], str]:
        """Peel a single-letter prefix off a word (only when it is unambiguous)."""
        out: List[Phoneme] = []
        if len(word) < 3:
            return out, word
        for letter, cons, vowel in self.PREFIXES:
            if word.startswith(letter) and word[1] in BASE and len(word) >= 4:
                # do not strip when the rest would start with the same letter
                # (e.g. "ללמד") or when the word is a known lexicon entry
                if word[1] == letter:
                    continue
                if word in LEXICON:
                    break
                out = [Phoneme(cons, letter), Phoneme(vowel, letter, is_vowel=True)]
                return out, word[1:]
        return out, word

    def _consonant(self, ch: str, dagesh: bool, word_final: bool, niqqud: str) -> str:
        if dagesh and ch in HARD:
            return HARD[ch]
        if ch == "ש":
            return "s" if "\u05c2" in niqqud else "S"
        if ch == "ו":
            return "w"
        if ch == "ה":
            return "" if word_final and not niqqud else "h"   # silent final he
        if ch == "א":
            return "A"
        if ch == "ע":
            return "A"
        if ch == "ח":
            return "x"      # always a fricative, word-final included
        if ch == "כ":
            return "x" if not dagesh else "k"
        if ch == "ט":
            return "t"
        if ch == "ק":
            return "Q"
        return BASE.get(ch, "")

    @staticmethod
    def _shva_is_na(phs: List[Phoneme]) -> bool:
        if len(phs) <= 1:
            return True
        prev = phs[-2] if len(phs) >= 2 else None
        if prev is None:
            return True
        return bool(prev.is_vowel and prev.sym in ("a", "e", "i", "o", "u"))

    def _insert_default_vowels(self, phs: List[Phoneme], word: str) -> List[Phoneme]:
        """Unvocalised Hebrew needs implied vowels. We use a light heuristic:
        keep consonant clusters that are legal in Hebrew, and add a short 'a'/'e'
        where a cluster of 3+ consonants would otherwise be unpronounceable.
        Words listed in the lexicon get their exact vocalisation."""
        lex = LEXICON.get(word)
        if lex:
            return [Phoneme(s, word, is_vowel=s in VOWELS) for s in lex]
        out: List[Phoneme] = []
        run = 0
        for p in phs:
            if p.sym == "_":
                run = 0
                out.append(p)
                continue
            if p.is_vowel:
                run = 0
                out.append(p)
                continue
            run += 1
            out.append(p)
            if run >= 3 and p.sym not in ("l", "r", "m", "n", "y", "w"):
                out.append(Phoneme("a", "", is_vowel=True))
                run = 1
        return out

    @staticmethod
    def _mark_stress(phs: List[Phoneme]) -> None:
        """Hebrew stress is usually ultimate (last syllable), often penultimate."""
        vowels = [p for p in phs if p.is_vowel]
        if not vowels:
            return
        idx = len(vowels) - 1
        if len(vowels) >= 2 and vowels[-1].sym == "a":
            # many segolate nouns stress the penult
            idx = len(vowels) - 2
        vowels[idx].stressed = True

    def _letter_name(self, ch: str) -> List[Phoneme]:
        for s in LETTER_NAMES.get(ch, ""):
            if s in PHONEMES:
                yield_ph = Phoneme(s, ch, is_vowel=s in VOWELS)
                return [yield_ph]
        seq = LEXICON.get(LETTER_SPELL.get(ch, ""), "")
        return [Phoneme(s, ch, is_vowel=s in VOWELS) for s in (seq or [])]

    # ------------------------------------------------------------- latin ----
    def _latin(self, word: str) -> List[Phoneme]:
        out: List[Phoneme] = []
        low = word.lower()
        for ch in low:
            if ch in LATIN_MAP:
                for s in LATIN_MAP[ch]:
                    out.append(Phoneme(s, ch, is_vowel=s in VOWELS))
            elif ch == " ":
                out.append(Phoneme("_", " ", pause=0.05))
        if out:
            out[0].word_start = True
            out[-1].word_end = True
        return out


LETTER_SPELL = {
    "א": "alef", "ב": "bet", "ג": "gimel", "ד": "dalet", "ה": "he", "ו": "vav",
    "ז": "zayin", "ח": "het", "ט": "tet", "י": "yod", "כ": "kaf", "ל": "lamed",
    "מ": "mem", "נ": "nun", "ס": "samekh", "ע": "ayin", "פ": "pe", "צ": "tsadi",
    "ק": "qof", "ר": "resh", "ש": "shin", "ת": "tav",
}
LETTER_NAMES = {
    "א": "A", "ב": "b", "ג": "g", "ד": "d", "ה": "h", "ו": "v", "ז": "z",
    "ח": "x", "ט": "t", "י": "y", "כ": "k", "ל": "l", "מ": "m", "נ": "n",
    "ס": "s", "ע": "A", "פ": "p", "צ": "T", "ק": "Q", "ר": "r", "ש": "S", "ת": "t",
}

# Exact vocalisations for the words JARVIS says most. Extend freely — this is
# our lexicon, not a downloaded one.
LEXICON: Dict[str, List[str]] = {
    "אלף": ["A", "l", "a", "f"], "בית": ["b", "a", "y", "i", "t"],
    "גימל": ["g", "i", "m", "e", "l"], "דלת": ["d", "a", "l", "e", "t"],
    "הא": ["h", "e", "A"], "וו": ["v", "a", "w"], "זין": ["z", "a", "y", "i", "n"],
    "חית": ["x", "i", "t"], "טית": ["t", "i", "t"], "יוד": ["y", "u", "d"],
    "כף": ["k", "a", "f"], "למד": ["l", "a", "m", "e", "d"], "מם": ["m", "e", "m"],
    "נון": ["n", "u", "n"], "סמך": ["s", "a", "m", "e", "x"], "עין": ["A", "y", "i", "n"],
    "פא": ["p", "e", "A"], "צדיק": ["T", "a", "d", "i", "Q"], "קוף": ["Q", "u", "f"],
    "ריש": ["r", "e", "S"], "שין": ["S", "i", "n"], "תו": ["t", "a", "w"],
    "שלום": ["S", "a", "l", "u", "m"], "אדוני": ["A", "d", "u", "n", "i"],
    "אדונים": ["A", "d", "u", "n", "i", "m"], "מערכת": ["m", "a", "A", "r", "e", "k", "e", "t"],
    "מערכות": ["m", "a", "A", "r", "a", "k", "u", "t"], "מחשב": ["m", "a", "x", "S", "e", "v"],
    "קוד": ["Q", "u", "d"], "זיכרון": ["z", "i", "Q", "a", "r", "u", "n"],
    "מוח": ["m", "u", "A", "x"], "קול": ["Q", "u", "l"], "דיבור": ["d", "i", "b", "u", "r"],
    "שעה": ["S", "a", "A", "a"], "יום": ["y", "u", "m"], "היום": ["h", "a", "y", "u", "m"],
    "תודה": ["t", "u", "d", "a"], "בבקשה": ["b", "e", "v", "a", "Q", "S", "a"],
    "כן": ["k", "e", "n"], "לא": ["l", "u", "A"], "אני": ["A", "n", "i"],
    "אתה": ["A", "t", "a"], "זה": ["z", "e", "h"], "מה": ["m", "a"],
    "איך": ["e", "y", "x"], "למה": ["l", "a", "m", "a"], "עכשיו": ["A", "k", "S", "a", "w"],
    "מיד": ["m", "i", "y", "a", "d"], "פותח": ["p", "u", "t", "e", "A", "x"],
    "סוגר": ["s", "u", "g", "e", "r"], "בודק": ["b", "u", "d", "e", "Q"],
    "מריץ": ["m", "a", "r", "i", "T"], "כותב": ["Q", "u", "t", "e", "v"],
    "קורא": ["Q", "u", "r", "e"], "מחפש": ["m", "x", "a", "p", "e", "S"],
    "שומר": ["S", "u", "m", "e", "r"], "מוחק": ["m", "u", "x", "e", "Q"],
    "מחשב": ["m", "e", "x", "a", "S", "e", "v"], "חושב": ["x", "u", "S", "e", "v"],
    "מתכנן": ["m", "i", "t", "Q", "a", "n", "e", "n"], "מבצע": ["m", "e", "v", "a", "T", "e", "A"],
    "מאמת": ["m", "e", "A", "a", "m", "e", "t"], "מאזין": ["m", "a", "A", "z", "i", "n"],
    "מדבר": ["m", "e", "d", "a", "b", "e", "r"], "עונה": ["u", "n", "e"],
    "לומד": ["l", "u", "m", "e", "d"], "הכול": ["h", "a", "k", "u", "l"],
    "פעילות": ["p", "e", "A", "i", "l", "u", "t"], "פעיל": ["p", "a", "A", "i", "l"],
    "תקין": ["t", "a", "Q", "i", "n"], "תקינות": ["t", "a", "Q", "i", "n", "u", "t"],
    "בדיקה": ["b", "e", "d", "i", "Q", "a"], "בדיקות": ["b", "e", "d", "i", "Q", "u", "t"],
    "תוצאה": ["t", "u", "T", "a", "a"], "שגיאה": ["S", "g", "i", "a", "a"],
    "תיקון": ["t", "i", "Q", "u", "n"], "הרשאה": ["h", "a", "r", "S", "a", "a"],
    "אישור": ["i", "S", "u", "r"], "ממתין": ["m", "i", "m", "t", "i", "n"],
    "הוראות": ["h", "u", "r", "a", "u", "t"], "משימה": ["m", "a", "S", "i", "m", "a"],
    "סוכן": ["s", "u", "g", "e", "n"], "מתכנת": ["m", "i", "t", "Q", "a", "n", "e", "t"],
    "פייתון": ["p", "i", "t", "u", "n"], "קובץ": ["Q", "u", "v", "a", "T"],
    "קבצים": ["Q", "e", "v", "a", "T", "i", "m"], "תיקייה": ["t", "i", "Q", "i", "y", "a"],
    "תהליך": ["t", "a", "h", "a", "l", "i", "x"], "תהליכים": ["t", "a", "h", "a", "l", "i", "x", "i", "m"],
    "מסך": ["m", "a", "s", "a", "x"], "חלון": ["x", "a", "l", "u", "n"],
    "רשת": ["r", "e", "S", "e", "t"], "שרת": ["S", "a", "r", "e", "t"],
    "דיסק": ["d", "i", "s", "Q"], "סוללה": ["s", "u", "l", "a", "l", "a"],
    "כל": ["k", "u", "l"], "שלי": ["S", "e", "l", "i"], "שלך": ["S", "e", "l", "x", "a"],
    "אנחנו": ["A", "n", "a", "x", "n", "u"], "הם": ["h", "e", "m"], "היא": ["h", "i"],
    "הוא": ["h", "u"], "את": ["e", "t"], "עם": ["i", "m"], "על": ["a", "l"],
    "או": ["u"], "אם": ["i", "m"], "אז": ["a", "z"], "אבל": ["A", "v", "a", "l"],
    "כי": ["Q", "i"], "גם": ["g", "a", "m"], "רק": ["r", "a", "Q"], "עוד": ["u", "d"],
    "כבר": ["k", "e", "v", "a", "r"], "כאן": ["Q", "a", "n"], "שם": ["S", "a", "m"],
    "מאוד": ["m", "e", "u", "d"], "קצת": ["Q", "a", "T", "a", "t"], "הרבה": ["h", "a", "r", "b", "e"],
    "יש": ["y", "e", "S"], "אין": ["e", "n"], "צריך": ["T", "a", "r", "i", "x"],
    "יכול": ["y", "a", "x", "u", "l"], "יכולה": ["y", "x", "u", "l", "a"],
    "רוצה": ["r", "u", "T", "e"], "יודע": ["y", "u", "d", "e", "A"],
    "חושב": ["x", "u", "S", "e", "v"], "מבין": ["m", "e", "v", "i", "n"],
    "בדקתי": ["b", "a", "d", "a", "Q", "t", "i"], "מצאתי": ["m", "a", "T", "a", "t", "i"],
    "פתחתי": ["p", "a", "t", "a", "x", "t", "i"], "סגרתי": ["s", "a", "g", "a", "r", "t", "i"],
    "כתבתי": ["Q", "a", "t", "a", "v", "t", "i"], "קראתי": ["Q", "a", "r", "a", "t", "i"],
    "שמרתי": ["S", "a", "m", "a", "r", "t", "i"], "הפעלתי": ["h", "i", "f", "A", "l", "a", "t", "i"],
    "ביצעתי": ["b", "i", "T", "a", "A", "t", "i"], "אימות": ["i", "m", "u", "t"],
    "תכנון": ["t", "i", "Q", "n", "u", "n"], "ניתוח": ["n", "i", "t", "u", "A", "x"],
    "מעבד": ["m", "e", "A", "b", "a", "d"], "זיכרון": ["z", "i", "Q", "a", "r", "u", "n"],
    "קבצי": ["Q", "u", "v", "T", "e"], "נתיב": ["n", "a", "t", "i", "v"],
    "אבטחה": ["a", "v", "t", "a", "x", "a"], "הרשאות": ["h", "a", "r", "S", "a", "u", "t"],
    "יומן": ["y", "u", "m", "a", "n"], "ביקורת": ["b", "i", "Q", "u", "r", "e", "t"],
    "מחשבון": ["m", "a", "x", "S", "e", "v", "u", "n"],
    "דפדפן": ["d", "a", "f", "d", "e", "p", "a", "n"],
    "פנוי": ["p", "a", "n", "u", "y"], "פעילים": ["p", "e", "A", "i", "l", "i", "m"],
    "בבוקר": ["b", "a", "b", "u", "Q", "e", "r"], "בערב": ["b", "a", "A", "e", "r", "e", "v"],
    "בלילה": ["b", "a", "l", "a", "y", "l", "a"], "בצהריים": ["b", "a", "T", "a", "h", "u", "r", "a", "y", "i", "m"],
    "אחוז": ["A", "x", "u", "z"], "אחוזים": ["A", "x", "u", "z", "i", "m"],
    "דקות": ["d", "a", "Q", "u", "t"], "דקה": ["d", "a", "Q", "a"],
    "בדיוק": ["b", "e", "d", "i", "y", "u", "Q"], "ו": ["v", "a"],
    "אפס": ["e", "f", "e", "s"], "אחד": ["e", "x", "a", "d"], "אחת": ["A", "x", "a", "t"],
    "שניים": ["S", "n", "a", "y", "i", "m"], "שתיים": ["S", "t", "a", "y", "i", "m"],
    "עשר": ["e", "s", "e", "r"], "מאה": ["m", "e", "A", "a"], "אלף": ["e", "l", "e", "f"],
}


def _clock_phrase(hh: str, mm: str) -> str:
    h, m = int(hh), int(mm)
    hour_word = number_to_hebrew(h, feminine=True)
    if m == 0:
        return f"השעה {hour_word} בדיוק"
    if m == 30:
        return f"השעה {hour_word} ושלושים דקות"
    return f"השעה {hour_word} ו{number_to_hebrew(m, feminine=True)} דקות"


def _drop_vowel_letters(phs: List[Phoneme]) -> List[Phoneme]:
    """In unvocalised Hebrew, yod after i/e and vav after u/o are vowel letters,
    not consonants. Dropping them is what makes the synthesis sound native."""
    out: List[Phoneme] = []
    for i, p in enumerate(phs):
        prev = out[-1] if out else None
        nxt = phs[i + 1] if i + 1 < len(phs) else None
        if prev is not None and prev.is_vowel and p.sym in ("y", "w"):
            pairs = (("y", "i"), ("y", "e"), ("w", "u"), ("w", "o"))
            if (p.sym, prev.sym) in pairs and (nxt is None or nxt.sym == "_" or not nxt.is_vowel):
                continue
        out.append(p)
    return out


def _post_process(phs: List[Phoneme]) -> List[Phoneme]:
    phs = _drop_vowel_letters(phs)
    out: List[Phoneme] = []
    for p in phs:
        if not p.sym:
            continue
        # collapse duplicate pauses
        if p.sym == "_" and out and out[-1].sym == "_":
            out[-1].pause = max(out[-1].pause, p.pause)
            continue
        out.append(p)
    while out and out[-1].sym == "_":
        out.pop()
    while out and out[0].sym == "_":
        out.pop(0)
    return out


G2P = HebrewG2P()


def convert(text: str) -> List[Phoneme]:
    return G2P.convert(text)


def to_symbols(text: str) -> str:
    return " ".join(p.sym for p in convert(text))


if __name__ == "__main__":
    samples = [
        "שלום אדוני, כל המערכות פעילות.",
        "השעה היא 10:30 בבוקר.",
        "פתח את המחשבון ומחק את הקובץ tmp.txt",
        "JARVIS is online.",
        "יש לי 15 אחוז זיכרון פנוי, ו־3 תהליכים פעילים.",
        "צה\"ל", "ג'רוויס", "צ'ק",
        "כתבתי פונקציה בפייתון והיא עברה בדיקה.",
    ]
    for s in samples:
        print(f"\n{s}\n   -> {to_symbols(s)}")
