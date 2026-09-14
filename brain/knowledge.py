"""JARVIS knowledge store + from-scratch semantic retrieval.

No external embedding model is available (and none is wanted): we build our own
**hashed character-n-gram TF-IDF vectoriser**. It is fast, deterministic, needs
no training, and works surprisingly well for Hebrew retrieval because Hebrew
morphology is carried by letter n-grams.

Pipeline:
    knowledge_base.yaml  ->  entries  ->  vectors (numpy)  ->  cosine top-k
The reasoning loop uses this to ground every factual answer, which is what keeps
a small neural model honest.
"""

from __future__ import annotations

import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from brain.tokenizer import normalize  # noqa: E402


@dataclass
class Fact:
    id: str
    topic: str
    he: str
    en: str = ""
    tags: Sequence[str] = ()
    qa: Sequence[Tuple[str, str]] = ()
    extra: Dict[str, Any] = field(default_factory=dict)

    def text(self) -> str:
        return " ".join(filter(None, [self.he, self.en, self.topic, " ".join(self.tags)]))


# ------------------------------------------------------- topical relevance --
# Character-n-gram cosine is easily fooled by a shared *request frame*:
# "ספר לי על חורים שחורים" and "ספר לי בדיחה" both start with "tell me", and
# that prefix alone can push an unrelated joke above the grounding threshold.
# A grounded answer is a promise, so we require the two sides to share at least
# one word that actually carries topic — after the polite frame is stripped.

_REQUEST_FRAMES = (
    "אני רוצה לדעת", "אני רוצה לשמוע", "האם אתה יכול", "אתה יכול", "אפשר לקבל",
    "ספר לי", "ספרי לי", "תגיד לי", "תגידי לי", "אמור לי", "תן לי", "תני לי",
    "הסבר לי", "תסביר לי", "אשמח לדעת", "בבקשה", "נא",
    "tell me", "can you", "could you", "i want to know", "i would like to know",
    "please", "explain",
)

# Raw list. Do NOT compare tokens against this directly: `content_tokens`
# normalises and stems before testing membership, and normalisation rewrites the
# Hebrew final forms (ך ם ן ץ ף -> כ מ נ צ פ). Fourteen entries here contain such
# a letter, so `איך` normalises to `איכ` and never matched — "how" survived as a
# topical word. That one defect let "איך מתקנים ברז דולף" match "איך מתקנים באג?"
# at 0.736 and answer a plumbing question with debugging advice. `_STOPWORDS`
# below is this list pushed through the same pipeline the tokens go through.
_STOPWORDS_RAW = frozenset("""
    מה מי איך מתי למה כמה על של את זה זו הזו הזאת אני אתה את הם הן יש אין האם או גם כי אם
    לי לך אותי אצלי שם כאן כל מאוד קצת יותר הכי בין עם בלי לפני אחרי תודה בבקשה שלום
    הוא היא היה הייתה יהיה יכול יכולה צריך רוצה יודע
    היום מחר אתמול עכשיו תמיד פעם שוב כבר משהו כלום אף אחד
    today tomorrow yesterday now always again already something nothing
    the a an is are was were be been do does did you i we they he she it me my your our
    to of in on at for with about into from that this these those and or but not no
    can could should would will may might tell know get
""".split())

_HE_PREFIXES = ("ובה", "וה", "וב", "כש", "מה", "שה", "ה", "ו", "ב", "כ", "ל", "מ", "ש")

_PLACEHOLDER = re.compile(r"\{[a-z_]+\}")


_HE_SUFFIXES = ("ימ", "ות", "ה")     # ם is normalised to מ, so ים appears as ימ


def _stem(tok: str) -> str:
    """Very light Hebrew/English stemming — enough to see that חור and החורים agree."""
    if re.search(r"[\u0590-\u05FF]", tok):
        for pre in _HE_PREFIXES:
            if tok.startswith(pre) and len(tok) - len(pre) >= 3:
                tok = tok[len(pre):]
                break
        for suf in _HE_SUFFIXES:
            if tok.endswith(suf) and len(tok) - len(suf) >= 3:
                return tok[:-len(suf)]
        return tok
    if tok.endswith("s") and len(tok) > 4:
        return tok[:-1]
    return tok


def _build_stopwords(raw: Iterable[str]) -> frozenset:
    """Run the raw list through the same normalise+stem pipeline tokens use.

    Comparing a normalised token against a raw stopword silently fails for every
    entry containing a Hebrew final-form letter, because `normalize` rewrites
    ך ם ן ץ ף to כ מ נ צ פ. Each raw entry is therefore stored in all three forms
    it can legitimately appear as — raw, normalised, and stemmed — so membership
    testing agrees with what `content_tokens` actually produces.
    """
    out = set()
    for w in raw:
        w = w.strip()
        if not w:
            continue
        out.add(w)
        n = normalize(w).lower()
        out.add(n)
        out.add(_stem(n))
    return frozenset(out)


_STOPWORDS = _build_stopwords(_STOPWORDS_RAW)


def content_tokens(text: str) -> set:
    """Words that carry topic: request frames and function words removed."""
    t = f" {normalize(str(text)).lower()} "
    for frame in _REQUEST_FRAMES:
        t = t.replace(f" {frame} ", " ")
    toks = re.findall(r"[A-Za-z\u0590-\u05FF][A-Za-z\u0590-\u05FF'\-]{1,}", t)
    return {_stem(w) for w in toks
            if w not in _STOPWORDS and _stem(w) not in _STOPWORDS and len(_stem(w)) >= 2}


def shares_topic(a: str, b: str) -> bool:
    """True when two strings have at least one substantive word in common.

    KNOWN LIMIT, measured rather than assumed. This gate is necessary but not
    sufficient: it stops unrelated questions from matching on a polite shared
    prefix, but it cannot stop two questions that share a *generic* word while
    differing in subject. Measured on held-out probes:

        "איך מתקנים ברז דולף"  -> "איך מתקנים באג?"        score 0.736
        "מה השורשים של המשפחה" -> "מה השורש הריבועי של 144?" score 0.254

    Neither a score threshold nor an overlap-fraction threshold separates these
    from genuine paraphrases. The 0.736 leak scores *higher* than the best true
    paraphrase (0.612), and the overlap-fraction distributions fully overlap
    (true min 0.00, false max 0.50). Character n-grams plus token overlap simply
    do not encode that "ברז" and "באג" are different subjects while "מתקנים" is
    the same verb — that needs semantic vectors or part-of-speech-aware matching.

    Fixing the stopword normalisation below (see `_build_stopwords`) cut false
    answers on uncovered topics from 3/6 to 2/9. Going further by tuning on that
    22-sample probe set would be overfitting to the probe, so the limit is
    recorded here instead of hidden behind a threshold that looks principled.
    """
    ta, tb = content_tokens(a), content_tokens(b)
    if not ta or not tb:
        return False
    if ta & tb:
        return True
    # allow a stem/inflection match (חור ~ חורים, token ~ tokens). Hebrew roots are
    # short, so a 3-letter stem is enough there; Latin needs 4 to stay meaningful.
    for x in ta:
        for y in tb:
            short, long = (x, y) if len(x) <= len(y) else (y, x)
            floor = 3 if re.search(r"[\u0590-\u05FF]", short) else 4
            if len(short) >= floor and long.startswith(short):
                return True
    return False


# ------------------------------------------------------------- vectoriser ---
class HashedNgramVectorizer:
    """Character n-gram hashing vectoriser with sublinear TF and IDF.

    Deterministic (no random projections), dependency-free beyond numpy, and
    language-agnostic — which matters because we cannot download anything.
    """

    def __init__(self, dim: int = 4096, ngram: Tuple[int, int] = (2, 4)) -> None:
        self.dim = dim
        self.ngram = ngram
        self.idf: Optional[np.ndarray] = None
        self._word_idf: Dict[str, float] = {}

    # ---------------------------------------------------------------- feats --
    @staticmethod
    def _grams(text: str, lo: int, hi: int) -> Iterable[str]:
        padded = f" {text} "
        for n in range(lo, hi + 1):
            for i in range(len(padded) - n + 1):
                yield padded[i: i + n]

    def _hash(self, gram: str) -> int:
        # FNV-1a: stable across processes and machines (Python's hash() is not)
        h = 0xCBF29CE484222325
        for ch in gram.encode("utf-8"):
            h ^= ch
            h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
        return h % self.dim

    def transform_one(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        text = normalize(text).lower()
        counts: Dict[int, int] = {}
        for gram in self._grams(text, *self.ngram):
            idx = self._hash(gram)
            counts[idx] = counts.get(idx, 0) + 1
        for idx, c in counts.items():
            vec[idx] = 1.0 + math.log(c)
        for w in re.findall(r"[\w]+", text):
            idf = self._word_idf.get(w)
            if idf:
                vec[self._hash(w) % self.dim] += idf
        if self.idf is not None:
            vec *= self.idf
        norm = float(np.linalg.norm(vec))
        return vec / norm if norm else vec

    def transform(self, texts: Sequence[str]) -> np.ndarray:
        if self.idf is None:
            self.fit(texts)
        return np.vstack([self.transform_one(t) for t in texts]) if texts else np.zeros((0, self.dim), np.float32)

    def fit(self, texts: Sequence[str]) -> "HashedNgramVectorizer":
        df = np.zeros(self.dim, dtype=np.float32)
        word_df: Dict[str, int] = {}
        for t in texts:
            seen = set()
            n = normalize(t).lower()
            for gram in self._grams(n, *self.ngram):
                seen.add(self._hash(gram))
            for idx in seen:
                df[idx] += 1
            for w in set(re.findall(r"[\w]+", n)):
                word_df[w] = word_df.get(w, 0) + 1
        N = max(1, len(texts))
        self.idf = np.log((1 + N) / (1 + df)) + 1.0
        self._word_idf = {w: math.log((1 + N) / (1 + c)) + 1.0 for w, c in word_df.items()}
        return self


# ------------------------------------------------------------------- store ---
class KnowledgeStore:
    """Loads knowledge_base.yaml and answers grounded retrieval queries."""

    def __init__(self, path: Path | str = ROOT / "brain/knowledge_base.yaml", dim: int = 4096) -> None:
        self.path = Path(path)
        self.facts: List[Fact] = []
        self.vectorizer = HashedNgramVectorizer(dim=dim)
        self._matrix: Optional[np.ndarray] = None
        self._qa_index: List[Tuple[str, str, int]] = []   # (question, answer, fact idx)
        self.load()

    def load(self) -> int:
        if not self.path.exists():
            return 0
        raw = yaml.safe_load(self.path.read_text(encoding="utf-8")) or {}
        self.facts = []
        for e in raw.get("entries", []):
            self.facts.append(Fact(
                id=str(e.get("id", f"fact{len(self.facts)}")),
                topic=str(e.get("topic", "")),
                he=str(e.get("he", "")),
                en=str(e.get("en", "")),
                tags=tuple(e.get("tags", []) or []),
                qa=tuple((q, a) for q, a in (e.get("qa", []) or [])),
                extra={k: v for k, v in e.items()
                       if k not in ("id", "topic", "he", "en", "tags", "qa")},
            ))
        docs = [f.text() for f in self.facts] + [q for q, _a, _i in self._qa_pairs()]
        self.vectorizer.fit(docs)
        self._matrix = self.vectorizer.transform([f.text() for f in self.facts])
        self._qa_index = [(q, a, i) for i, f in enumerate(self.facts) for q, a in f.qa]
        return len(self.facts)

    def _qa_pairs(self) -> List[Tuple[str, str, int]]:
        out = []
        for i, f in enumerate(self.facts):
            for q, a in f.qa:
                out.append((q, a, i))
        return out

    # -------------------------------------------------------------- queries --
    def search(self, query: str, k: int = 3, min_score: float = 0.08,
               require_overlap: bool = False) -> List[Dict[str, Any]]:
        if self._matrix is None or not self.facts:
            return []
        qv = self.vectorizer.transform_one(query)
        sims = self._matrix @ qv
        order = np.argsort(-sims)[: k * 3]
        hits: List[Dict[str, Any]] = []
        for idx in order:
            score = float(sims[idx])
            if score < min_score:
                continue
            f = self.facts[int(idx)]
            if require_overlap and not shares_topic(query, f.text()):
                continue
            hits.append({"id": f.id, "topic": f.topic, "score": round(score, 4),
                         "text_he": f.he, "text_en": f.en, "tags": list(f.tags)})
            if len(hits) >= k:
                break
        return hits

    def search_qa(self, query: str, k: int = 3, min_score: float = 0.25,
                  require_overlap: bool = True, skip_templates: bool = True) -> List[Dict[str, Any]]:
        """Look for a near-duplicate question we already know the answer to.

        A high character-n-gram score is *not* sufficient: the candidate must also
        share a substantive word with the query, otherwise a polite shared prefix
        ("ספר לי…") can sell an unrelated canned answer as grounded truth.
        """
        if not self._qa_index:
            return []
        qv = self.vectorizer.transform_one(query)
        scored = []
        for q, a, fi in self._qa_index:
            if skip_templates and _PLACEHOLDER.search(a):
                # some KB answers are templates for the skill layer ("השעה היא
                # {time}"); publishing one raw would show the user literal braces.
                # Those questions are answered deterministically by their intent.
                continue
            s = float(qv @ self.vectorizer.transform_one(q))
            if s >= min_score and (not require_overlap or shares_topic(query, q)):
                scored.append((s, q, a, self.facts[fi].id, self.facts[fi].topic))
        scored.sort(reverse=True)
        return [{"score": round(s, 4), "question": q, "answer": a, "fact_id": fid, "topic": tp}
                for s, q, a, fid, tp in scored[:k]]

    def by_id(self, fact_id: str) -> Optional[Fact]:
        for f in self.facts:
            if f.id == fact_id:
                return f
        return None

    def topics(self) -> List[str]:
        return sorted({f.topic for f in self.facts if f.topic})

    def stats(self) -> Dict[str, Any]:
        return {"facts": len(self.facts), "qa_pairs": len(self._qa_index),
                "topics": len(self.topics()), "dim": self.vectorizer.dim}


if __name__ == "__main__":
    ks = KnowledgeStore()
    print("knowledge store:", ks.stats())
    for q in ["מה השורש הריבועי", "איך אתה מדבר בלי API", "מי בנה אותך",
              "הרשאות ואבטחה", "מה זה רקורסיה"]:
        hits = ks.search(q, k=2)
        qa = ks.search_qa(q, k=1)
        print(f"\nQ: {q}")
        for h in hits:
            print(f"   [{h['score']:.3f}] {h['id']} / {h['topic']}: {h['text_he'][:70]}...")
        for h in qa:
            print(f"   QA [{h['score']:.3f}] {h['question']} -> {h['answer'][:60]}")
