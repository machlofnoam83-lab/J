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
    def search(self, query: str, k: int = 3, min_score: float = 0.08) -> List[Dict[str, Any]]:
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
            hits.append({"id": f.id, "topic": f.topic, "score": round(score, 4),
                         "text_he": f.he, "text_en": f.en, "tags": list(f.tags)})
            if len(hits) >= k:
                break
        return hits

    def search_qa(self, query: str, k: int = 3, min_score: float = 0.25) -> List[Dict[str, Any]]:
        """Look for a near-duplicate question we already know the answer to."""
        if not self._qa_index:
            return []
        qv = self.vectorizer.transform_one(query)
        scored = []
        for q, a, fi in self._qa_index:
            s = float(qv @ self.vectorizer.transform_one(q))
            if s >= min_score:
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
