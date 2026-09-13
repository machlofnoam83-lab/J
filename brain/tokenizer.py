"""JARVIS byte-level BPE tokenizer — implemented from scratch.

No ``sentencepiece``, no ``tiktoken``, no downloads. This is our own merge
algorithm, our own vocabulary, our own Hebrew normalizer.

Design notes
------------
* **Byte-level base** (like GPT-2) so *any* UTF-8 text — Hebrew, English,
  emoji, source code — is representable without ``<unk>``.
* **Hebrew-aware pre-normalizer**: final forms folded, niqqud handled as a
  separate channel, geresh/gershayim/maqaf stripped, whitespace canonicalised.
* **Pre-tokenizer regex** keeps Hebrew letters together, splits Latin words,
  numbers, code punctuation and indentation-significant newlines.
* Vocabulary is stored as a single JSON file: ``models/tokenizer.json``.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

# ---------------------------------------------------------------- specials --
PAD, EOS, USER, ASSISTANT, TOOL, TOOL_RESULT, THINK, PLAN, SEP, BOT = (
    "<|pad|>", "<|eos|>", "<|user|>", "<|assistant|>", "<|tool|>",
    "<|tool_result|>", "<|think|>", "<|plan|>", "<|sep|>", "<|bot|>",
)
SPECIAL_TOKENS: Tuple[str, ...] = (
    PAD, EOS, USER, ASSISTANT, TOOL, TOOL_RESULT, THINK, PLAN, SEP, BOT,
)
PAD_ID, EOS_ID = 0, 1

# Byte <-> unicode-symbol bijection (GPT-2 style) so every byte is printable.
def _byte_encoder() -> Dict[int, str]:
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("\xa1"), ord("\xac") + 1))
        + list(range(ord("\xae"), ord("\xff") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


BYTE_ENC = _byte_encoder()
BYTE_DEC = {v: k for k, v in BYTE_ENC.items()}

# Hebrew final-form folding (ם -> מ etc.)
FINAL_FORMS = {"ך": "כ", "ם": "מ", "ן": "נ", "ץ": "צ", "ף": "פ"}

# Niqqud marks are stripped by default but can be preserved via keep_niqqud.
NIQQUD = re.compile(r"[\u0591-\u05bd\u05bf-\u05c5\u05c7\u05c4\u05b0-\u05bc]")
HEBREW_GERESH = re.compile(r"[\u05f3\u05f4\u0591-\u05af]|[׳״]")
_SPECIAL_RE = re.compile("(" + "|".join(re.escape(t) for t in SPECIAL_TOKENS) + ")")

# Pre-tokenizer: Hebrew runs | Latin words | numbers | newlines+indent | other
_PAT = re.compile(
    r"""[\u05d0-\u05ea]+            # hebrew letters run
      | [A-Za-z_][A-Za-z0-9_]*     # identifier-ish latin
      | \d+(?:\.\d+)?              # numbers
      | \n[ \t]*                   # newline + indentation (matters for code)
      | [^\s\w]+                   # punctuation clusters
      | \s                         # single whitespace
    """,
    re.VERBOSE | re.UNICODE,
)


def normalize(text: str, keep_niqqud: bool = False, fold_finals: bool = True) -> str:
    """Canonicalise Hebrew/English text before tokenization."""
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\t", "    ")
    if fold_finals:
        text = "".join(FINAL_FORMS.get(ch, ch) for ch in text)
    if not keep_niqqud:
        text = NIQQUD.sub("", text)
    text = HEBREW_GERESH.sub("", text)
    text = re.sub(r"[ ]{2,}", " ", text)
    return text.strip() if text.strip() else text


class Tokenizer:
    """Byte-level BPE with Hebrew normalisation."""

    def __init__(
        self,
        vocab: Dict[str, int],
        merges: List[str],
        keep_niqqud: bool = False,
    ) -> None:
        self.vocab = vocab
        self.inverse = {i: t for t, i in vocab.items()}
        self.merges = list(merges)                       # JSON-serialisable
        self._rank = {tuple(m.split(" ")): i for i, m in enumerate(self.merges)}
        self.keep_niqqud = keep_niqqud
        self._special_re = re.compile(
            "(" + "|".join(re.escape(s) for s in SPECIAL_TOKENS) + ")"
        )
        self._cache: Dict[str, List[int]] = {}

    # ------------------------------------------------------------ factories --
    @classmethod
    def train(
        cls,
        texts: Iterable[str],
        vocab_size: int = 16384,
        keep_niqqud: bool = False,
        min_frequency: int = 2,
        progress: bool = True,
    ) -> "Tokenizer":
        """Learn merges from a corpus of raw strings.

        Uses an incremental pair-heap: after each merge only the affected words
        are re-scanned, so training is ~O(total symbols * merges_touched)
        instead of re-counting every pair in the corpus each round.
        """
        counts: Counter = Counter()
        for raw in texts:
            # Special tokens are atomic — they must never be split into byte
            # pieces and fed to the merge learner, or they swamp the counts.
            norm = normalize(_SPECIAL_RE.sub(" ", raw), keep_niqqud=keep_niqqud)
            for piece in _PAT.findall(norm):
                word = "".join(BYTE_ENC[b] for b in piece.encode("utf-8"))
                counts[word] += 1

        vocab: Dict[str, int] = {t: i for i, t in enumerate(SPECIAL_TOKENS)}
        for b in range(256):
            vocab[BYTE_ENC[b]] = len(vocab)

        words: Dict[Tuple[str, ...], int] = {
            tuple(w): c for w, c in counts.items() if c >= min_frequency
        }
        if not words:
            return cls(vocab, [], keep_niqqud=keep_niqqud)

        pairs: Counter = Counter()
        for word, freq in words.items():
            for p in set(zip(word, word[1:])):
                pairs[p] += freq

        merges: List[str] = []
        target_merges = max(0, vocab_size - len(vocab))

        # symbol -> set(words containing it): keeps each merge step near O(1)
        sym_words: Dict[str, set] = defaultdict(set)
        for w in words:
            for s in w:
                sym_words[s].add(w)

        while len(merges) < target_merges and pairs:
            best, best_freq = max(pairs.items(), key=lambda kv: (kv[1], kv[0]))
            if best_freq < min_frequency:
                break
            merged = best[0] + best[1]
            if merged in vocab:
                # stale/duplicate pair: drop it and keep searching
                pairs.pop(best, None)
                continue
            merges.append(f"{best[0]} {best[1]}")
            vocab[merged] = len(vocab)
            if len(vocab) >= vocab_size:
                break

            affected = sym_words.get(best[0], set()) & sym_words.get(best[1], set())
            affected = {w for w in affected if w in words}
            for w in affected:
                f = words[w]
                for p in set(zip(w, w[1:])):
                    pairs[p] -= f
                    if pairs[p] <= 0:
                        pairs.pop(p, None)
                nw = _merge_word(w, best)
                del words[w]
                for s in w:
                    sym_words[s].discard(w)
                words[nw] = words.get(nw, 0) + f
                for s in nw:
                    sym_words[s].add(nw)
                for p in set(zip(nw, nw[1:])):
                    pairs[p] += f
            # NOTE: never unconditionally drop sym_words[best[0]] / [best[1]].
            # Those symbols still occur inside `merged` and inside every word
            # that was not rewritten. Deleting the keys starves all future pairs
            # containing them and freezes vocabulary growth at ~650 merges.
            for key in (best[0], best[1]):
                if not sym_words.get(key):
                    sym_words.pop(key, None)

            if progress and len(merges) % 1000 == 0:
                print(f"  [bpe] merges={len(merges)}/{target_merges} vocab={len(vocab)} "
                      f"words={len(words)}")

        return cls(vocab, merges, keep_niqqud=keep_niqqud)

    # ---------------------------------------------------------------- codec --
    def _bpe(self, token: str) -> List[str]:
        if token in self._cache:
            return self._cache[token]
        word = tuple(token)
        if len(word) == 1:
            self._cache[token] = list(word)
            return list(word)
        while True:
            candidates = [
                (self._rank[p], i) for i, p in enumerate(zip(word, word[1:])) if p in self._rank
            ]
            if not candidates:
                break
            rank, idx = min(candidates)
            new_word: List[str] = []
            i = 0
            while i < len(word):
                if i == idx:
                    new_word.append(word[i] + word[i + 1])
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = tuple(new_word)
            if len(word) == 1:
                break
        out = list(word)
        self._cache[token] = out
        return out

    def encode(self, text: str, add_specials: bool = False) -> List[int]:
        if add_specials:
            text = f"{BOT} {text} {EOS}"
        ids: List[int] = []
        for part in self._special_re.split(text):
            if not part:
                continue
            if part in self.vocab and part.startswith("<|"):
                ids.append(self.vocab[part])
                continue
            clean = _SPECIAL_RE.sub(" ", part)   # never feed atomic specials to BPE
            for piece in _PAT.findall(normalize(clean, keep_niqqud=self.keep_niqqud)):
                sym = "".join(BYTE_ENC[b] for b in piece.encode("utf-8"))
                for unit in self._bpe(sym):
                    if unit in self.vocab:
                        ids.append(self.vocab[unit])
                    else:
                        # graceful byte-level fallback: never silently drop content
                        for ch in unit:
                            ids.append(self.vocab[BYTE_ENC[BYTE_DEC[ch]]])
        return ids

    def decode(self, ids: Sequence[int], skip_specials: bool = True) -> str:
        chars: List[str] = []
        for i in ids:
            tok = self.inverse.get(int(i))
            if tok is None:
                continue
            if tok.startswith("<|"):
                if not skip_specials:
                    chars.append(tok)
                continue
            chars.append(tok)
        joined = "".join(chars)
        # The byte-encoder maps some byte values to chr(256+) (GPT-2 trick),
        # so latin-1 is NOT the right inverse table — BYTE_DEC is.
        try:
            raw = bytes(BYTE_DEC[c] for c in joined if c in BYTE_DEC)
        except Exception:
            return joined
        return raw.decode("utf-8", errors="replace")

    # ------------------------------------------------------------- helpers --
    def encode_chat(self, user: str, history: Sequence[Tuple[str, str]] = ()) -> List[int]:
        """Render a conversation into ids using our special tokens."""
        parts: List[str] = []
        for u, a in history[-4:]:
            parts.append(f"{USER} {u} {EOS} {ASSISTANT} {a} {EOS}")
        parts.append(f"{USER} {user} {EOS} {ASSISTANT}")
        return self.encode(" ".join(parts))

    def __len__(self) -> int:
        return len(self.vocab)

    # ---------------------------------------------------------------- I/O ----
    def save(self, path: Path | str) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps(
                {
                    "version": 1,
                    "kind": "jarvis-byte-bpe",
                    "keep_niqqud": self.keep_niqqud,
                    "special_tokens": list(SPECIAL_TOKENS),
                    "vocab": self.vocab,
                    "merges": self.merges,
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return p

    @classmethod
    def load(cls, path: Path | str) -> "Tokenizer":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(data["vocab"], data["merges"], keep_niqqud=data.get("keep_niqqud", False))


def _merge_word(word: Tuple[str, ...], pair: Tuple[str, str]) -> Tuple[str, ...]:
    a, b = pair
    out: List[str] = []
    i = 0
    while i < len(word):
        if i < len(word) - 1 and word[i] == a and word[i + 1] == b:
            out.append(a + b)
            i += 2
        else:
            out.append(word[i])
            i += 1
    return tuple(out)


# ---------------------------------------------------------------- utility ----
def build_tiny_default(vocab_size: int = 1024) -> Tokenizer:
    """A minimal working tokenizer so the system never hard-fails before training."""
    return Tokenizer.train(
        [
            "שלום עולם, JARVIS online. all systems operational.",
            "אני כאן כדי לעזור לך לתכנת, לחשב, ולשלוט במחשב.",
            "def hello(): return 'world'",
        ]
        * 40,
        vocab_size=vocab_size,
        progress=False,
    )


def stats(tok: Tokenizer, samples: Sequence[str]) -> Dict[str, float]:
    """Compression diagnostics used by tools/selftest.py."""
    total_chars = sum(len(s) for s in samples)
    total_toks = sum(len(tok.encode(s)) for s in samples)
    return {
        "vocab_size": len(tok),
        "chars": total_chars,
        "tokens": total_toks,
        "chars_per_token": round(total_chars / max(total_toks, 1), 3),
    }
