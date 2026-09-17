"""Hybrid retrieval: BM25 + character n-grams + phrase evidence + MMR diversity.

Why three signals instead of one
--------------------------------
Each of these fails in a way the others cover, and the failures are the ones that
actually matter in Hebrew:

* **BM25** is exact and explainable, but Hebrew is heavily inflected, so
  "ההרשאות" and "הרשאה" share a stem only sometimes. Our stemmer is deliberately
  light, which means lexical matching alone misses real matches.
* **Character n-grams** catch morphology and typos ("הרשאות" ↔ "הרשאה" share
  most 2–4 grams) but are noise-happy: two unrelated sentences about "המערכת"
  score well because the word is long and common.
* **Phrase evidence** — the query appearing as a contiguous span in the chunk —
  is the strongest single signal we have and is nearly impossible to fake, so it
  gets a large additive bonus rather than being folded into a weighted average
  where it could be averaged away.

The final ranking then applies **MMR**, because the top-k by relevance alone is
usually five overlapping chunks of the same file, and an answer assembled from
those repeats itself while missing the second source that would have made it
useful.

Every weight below is a plain number in :data:`WEIGHTS`, not a magic constant
buried in an expression, so it can be tuned (and tested) without reading the
scoring code.
"""

from __future__ import annotations

import json
import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from brain.knowledge import HashedNgramVectorizer  # noqa: E402
from brain.rag.index import RagIndex, term_counts, tokenize  # noqa: E402
from brain.tokenizer import normalize  # noqa: E402

# Weights sum to 1.00 so a score is directly readable as "how good is this hit"
# on a 0–1 scale. An earlier version used unnormalised additive weights and
# divided BM25 by the best candidate's score — which forced the top hit to 1.0
# on that component *no matter how irrelevant it was*, and made every query look
# equally confident. Saturating transforms fixed it; see ``_bm25_component``.
WEIGHTS = {
    "bm25": 0.34,          # lexical, stem-matched
    "ngram": 0.14,         # morphological / typo tolerant
    "phrase": 0.22,        # contiguous query span found in the chunk
    "coverage": 0.20,      # fraction of query content terms present
    "path": 0.06,          # query term appears in file name / title
    "heading": 0.04,       # query term appears in the enclosing heading
}
BM25_SAT = 8.0             # BM25 score at which the lexical component hits 0.5
NGRAM_FLOOR = 0.10         # below this, char-ngram similarity is noise
REL_GATE = 0.45            # drop hits scoring under this fraction of the best
# Asking a Hebrew prose question should not be answered by a code fragment that
# happens to share stems with it. Measured, not guessed: on this repository
# "מה זה רקורסיה" ranked a list of arithmetic test cases first without it.
PROSE_QUERY_CODE_PENALTY = 0.55
MMR_LAMBDA = 0.72          # 1.0 = pure relevance, 0.0 = pure diversity
NGRAM_FIT_SAMPLE = 20_000  # cap on chunks used to fit IDF (cost is linear)


@dataclass
class Hit:
    """One retrieved chunk with the breakdown that produced its score."""

    chunk_id: int
    doc_id: int
    path: str
    title: str
    text: str
    start_line: int
    end_line: int
    heading: str
    kind: str
    score: float
    parts: Dict[str, float] = field(default_factory=dict)
    matched_terms: Tuple[str, ...] = ()

    @property
    def file(self) -> str:
        return Path(self.path).name

    @property
    def span(self) -> str:
        if self.end_line == self.start_line:
            return f"שורה {self.start_line}"
        return f"שורות {self.start_line}–{self.end_line}"

    @property
    def citation(self) -> str:
        name = Path(self.path).name
        return f"{name} · {self.span}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id, "doc_id": self.doc_id, "path": self.path,
            "file": Path(self.path).name, "title": self.title, "text": self.text,
            "start_line": self.start_line, "end_line": self.end_line,
            "heading": self.heading, "kind": self.kind,
            "score": round(self.score, 4),
            "parts": {k: round(v, 4) for k, v in self.parts.items()},
            "matched_terms": list(self.matched_terms),
            "citation": self.citation, "span": self.span,
        }


class Retriever:
    """Ranks indexed chunks against a query. Stateless except for the vectoriser."""

    def __init__(self, index: RagIndex, *, dim: int = 4096) -> None:
        self.index = index
        self.dim = dim
        self.vec = HashedNgramVectorizer(dim=dim)
        self._vec_ready = False
        self._vocab: Optional[List[str]] = None
        self._vocab_grams: Optional[Dict[str, List[str]]] = None
        self._vocab_rev: Optional[List[Tuple[str, str]]] = None
        self.last_expansions: Dict[str, List[str]] = {}

    # ------------------------------------------------------------ vectoriser --
    def fit_vectorizer(self, *, force: bool = False,
                       sample: int = NGRAM_FIT_SAMPLE) -> Dict[str, Any]:
        """Fit (or restore) the n-gram IDF weights.

        The IDF arrays are cached in the index's ``meta`` table: refitting on
        every start would re-hash every gram in the corpus for no benefit, since
        the weights only change when the corpus changes.
        """
        if not force:
            cached = self._load_cached_vectorizer()
            if cached:
                return {"source": "cache", "chunks": cached}
        texts: List[str] = []
        for _cid, text in self.index.all_chunk_texts():
            texts.append(text)
            if len(texts) >= sample:
                break
        if not texts:
            self._vec_ready = True
            return {"source": "empty", "chunks": 0}
        self.vec.fit(texts)
        self._vec_ready = True
        self._store_cached_vectorizer(len(texts))
        return {"source": "fit", "chunks": len(texts)}

    def _store_cached_vectorizer(self, n: int) -> None:
        try:
            payload = {
                "n": n,
                "idf": [round(float(x), 6) for x in self.vec.idf] if self.vec.idf is not None else [],
                "word_idf": {k: round(float(v), 6) for k, v in self.vec._word_idf.items()},
            }
            self.index._set_meta("ngram_idf", json.dumps(payload, ensure_ascii=False))
            self.index._conn.commit()
        except (TypeError, ValueError, OSError):
            pass  # a cache we cannot write is not an error, just a slower start

    def _load_cached_vectorizer(self) -> int:
        raw = self.index.get_meta("ngram_idf")
        if not raw:
            return 0
        try:
            payload = json.loads(raw)
            idf = payload.get("idf") or []
            if len(idf) != self.dim:
                return 0
            self.vec.idf = np.asarray(idf, dtype=np.float32)
            self.vec._word_idf = {k: float(v) for k, v in (payload.get("word_idf") or {}).items()}
            self._vec_ready = True
            return int(payload.get("n") or 0)
        except (ValueError, TypeError, json.JSONDecodeError):
            return 0

    def invalidate_vectorizer(self) -> None:
        """Call after the corpus changed so the next search refits IDF."""
        self.index._set_meta("ngram_idf", "")
        self.index._conn.commit()
        self._vec_ready = False
        self._vocab = None
        self._vocab_grams = None
        self._vocab_rev = None

    # ------------------------------------------------- morphological expand --
    #: An expanded term counts for this much of a literal one.
    EXPANSION_WEIGHT = 0.55
    #: Minimum character-trigram Jaccard to consider two terms related.
    EXPAND_JACCARD = 0.45
    #: Maximum expansions admitted per query term (kept small on purpose: a wide
    #: expansion turns a precise question into a topic search).
    EXPAND_MAX_PER_TERM = 4

    def _load_vocab(self) -> None:
        if self._vocab is not None:
            return
        self._vocab = self.index.vocabulary()
        grams: Dict[str, List[str]] = {}
        for t in self._vocab:
            for g in _trigrams(t):
                grams.setdefault(g, []).append(t)
        self._vocab_grams = grams
        # Reversed vocabulary, so "ends with" becomes a prefix bisect. Hebrew
        # prefixes attach to the front of a word, which is exactly the side our
        # stemmer strips — so the residue of a query word is usually a *suffix*
        # of the indexed word ("ריצ" ⊂ "הריץ"). Without this rule the expansion
        # found nothing for the most common Hebrew inflection there is.
        self._vocab_rev = sorted((t[::-1], t) for t in self._vocab)

    def expand_terms(self, terms: Sequence[str]) -> Tuple[Dict[str, float], Dict[str, List[str]]]:
        """Map query terms onto the indexed vocabulary.

        Hebrew is heavily inflected and our stemmer is deliberately light, so
        "מריצים" (they run) and "להריץ" (to run) do not reduce to the same token.
        Rather than ship a Hebrew morphological analyser we cannot download, we
        ask the *index itself* which of its own terms look like the query term —
        by prefix and by character-trigram overlap. It costs one vocabulary scan
        (cached) and it is auditable: the expansions are reported alongside the
        answer, so a wrong guess is visible rather than silent.
        """
        self._load_vocab()
        weights: Dict[str, float] = {t: 1.0 for t in terms}
        trace: Dict[str, List[str]] = {}
        if not self._vocab:
            return weights, trace
        for term in terms:
            if term in weights and term in set(self._vocab):
                trace.setdefault(term, [])
                continue
            cand: set = set()
            # 1. prefix — "הרשא" ⊂ "הרשאות".
            lo, hi = _bisect_prefix(self._vocab, term)
            cand.update(self._vocab[lo:hi])
            # 2. suffix — "ריצ" ⊂ "הריץ". This is the rule that matters most for
            #    Hebrew, because the stemmer strips prefixes and so hands us the
            #    tail of the word.
            rterm = term[::-1]
            rev = self._vocab_rev or []
            keys = [k for k, _t in rev]
            rlo, rhi = _bisect_prefix(keys, rterm)
            for _k, t in rev[rlo:rhi]:
                if len(t) - len(term) <= 3:
                    cand.add(t)
            # 3. character-trigram overlap — catches everything in between.
            for g in _trigrams(term):
                for t in (self._vocab_grams or {}).get(g, ()):
                    if t != term:
                        cand.add(t)
            tg = _trigram_set(term)
            scored: List[Tuple[float, str]] = []
            for c in cand:
                if c == term or abs(len(c) - len(term)) > max(3, len(term)):
                    continue
                j = _jaccard(tg, _trigram_set(c))
                if j >= self.EXPAND_JACCARD:
                    scored.append((j, c))
            scored.sort(reverse=True)
            chosen = [c for _j, c in scored[:self.EXPAND_MAX_PER_TERM]]
            if chosen:
                trace[term] = chosen
                for c in chosen:
                    # Never let an expansion outweigh the user's own word.
                    weights[c] = max(weights.get(c, 0.0), self.EXPANSION_WEIGHT)
        return weights, trace

    # ---------------------------------------------------------------- scoring --
    def _ensure_vectorizer(self) -> None:
        if not self._vec_ready:
            self.fit_vectorizer()

    def search(self, query: str, *, k: int = 8, candidates: int = 120,
               path_filter: str = "", kinds: Sequence[str] = (),
               diverse: bool = True) -> List[Hit]:
        """Return the top-k chunks for ``query``, best first.

        ``candidates`` bounds how many BM25 hits are re-scored with the
        (comparatively expensive) n-gram vectoriser — the classic
        retrieve-then-rerank shape.
        """
        query = (query or "").strip()
        if not query:
            return []
        q_terms = tokenize(query)
        q_unique = sorted(set(q_terms))
        if not q_unique:
            return []

        # Morphological expansion widens *recall* only. Coverage and the phrase
        # bonus below stay bound to the user's literal words, so a match found
        # purely through a guessed expansion can never present itself as strong
        # evidence — it shows up as a lower score, and the expansion is reported.
        weights, expansions = self.expand_terms(q_unique)
        self.last_expansions = expansions
        lookup = sorted(weights)
        postings = self.index.postings_for(lookup)
        if not postings:
            return []
        df = self.index.doc_freq(lookup)
        bm25 = self.index.bm25_scores(lookup, postings, df, term_weights=weights)
        if not bm25:
            return []

        ranked = sorted(bm25.items(), key=lambda kv: kv[1], reverse=True)[:candidates]
        rows = self.index.fetch_chunks([cid for cid, _ in ranked])
        if not rows:
            return []

        self._ensure_vectorizer()
        qv = self.vec.transform_one(query)
        nq = normalize(query).lower()
        phrase_probes = _phrase_probes(query)
        q_path_terms = {t for t in q_unique if len(t) >= 3}
        prose_query = _is_prose_query(query)

        hits: List[Hit] = []
        for cid, b in ranked:
            row = rows.get(cid)
            if row is None:
                continue
            if path_filter and path_filter.lower() not in row["path"].lower():
                continue
            if kinds and row["kind"] not in kinds and row["doc_kind"] not in kinds:
                continue
            text = row["text"]
            ntext = normalize(text).lower()

            nv = float(qv @ self.vec.transform_one(text)) if self._vec_ready else 0.0
            nv = max(0.0, min(1.0, nv))
            nv = max(0.0, (nv - NGRAM_FLOOR) / (1.0 - NGRAM_FLOOR))

            phrase = 0.0
            n_qwords = max(1, len(nq.split()))
            for probe in phrase_probes:
                if phrase_in(ntext, probe):
                    # Scale by how much of the question the probe covers. A flat
                    # 1.0 here is what let a two-word prefix ("איכ מתקנימ") score
                    # as strongly as a verbatim match of the whole question, and
                    # that is how "איך מתקנים אופניים" found a docstring example
                    # about fixing a leaking tap and called it an answer.
                    phrase = max(phrase, len(probe.split()) / n_qwords)
                    continue
                # tolerate a single intervening word ("הקובץ הגדול" vs "הקובץ")
                if len(probe) > 12 and " " in probe:
                    head, _, tail = probe.partition(" ")
                    if phrase_in(ntext, head) and phrase_in(ntext, tail):
                        phrase = max(phrase, 0.5 * len(probe.split()) / n_qwords)

            present = [t for t in q_unique if t in (postings.get(cid) or {})]
            # ^1.4 rather than linear: matching 1 of 4 query terms is much less
            # than a quarter of an answer, and linear coverage rewarded it as if
            # it were.
            coverage = (len(present) / len(q_unique)) ** 1.4

            path_blob = f"{row['path']} {row.get('title') or ''}".lower()
            path_hit = 1.0 if any(t in path_blob for t in q_path_terms) else 0.0
            head_blob = normalize(row.get("heading") or "").lower()
            head_hit = 1.0 if any(t in head_blob for t in q_path_terms) else 0.0

            parts = {
                "bm25": WEIGHTS["bm25"] * _bm25_component(b),
                "ngram": WEIGHTS["ngram"] * nv,
                "phrase": WEIGHTS["phrase"] * phrase,
                "coverage": WEIGHTS["coverage"] * coverage,
                "path": WEIGHTS["path"] * path_hit,
                "heading": WEIGHTS["heading"] * head_hit,
            }
            score = sum(parts.values())
            kind = row.get("kind") or "prose"
            if prose_query and kind in ("code", "data"):
                score *= PROSE_QUERY_CODE_PENALTY
                parts["code_penalty"] = score - sum(parts.values())
            if score <= 0:
                continue
            hits.append(Hit(
                chunk_id=cid, doc_id=int(row["doc_id"]), path=row["path"],
                title=row.get("title") or "", text=text,
                start_line=int(row["start_line"]), end_line=int(row["end_line"]),
                heading=row.get("heading") or "", kind=kind,
                score=score, parts=parts, matched_terms=tuple(present),
            ))

        if not hits:
            return []
        hits.sort(key=lambda h: h.score, reverse=True)
        # A hit far below the best one is not a second opinion, it is noise that
        # the answer composer would happily quote as a source.
        gate = REL_GATE * hits[0].score
        hits = [h for h in hits if h.score >= gate] or hits[:1]
        if diverse and len(hits) > 1:
            hits = _mmr(hits, k, self.vec, qv)
        return hits[:k]


def _trigrams(term: str) -> List[str]:
    """Character trigrams of a padded term, used for morphological similarity."""
    padded = f" {term} "
    if len(padded) < 3:
        return [padded]
    return [padded[i:i + 3] for i in range(len(padded) - 2)]


def _trigram_set(term: str) -> frozenset:
    return frozenset(_trigrams(term))


def _jaccard(a: frozenset, b: frozenset) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return inter / (len(a) + len(b) - inter)


def _bisect_prefix(sorted_terms: Sequence[str], prefix: str) -> Tuple[int, int]:
    """Slice bounds of ``sorted_terms`` whose entries start with ``prefix``."""
    import bisect
    lo = bisect.bisect_left(sorted_terms, prefix)
    hi = bisect.bisect_right(sorted_terms, prefix + "\uffff")
    return lo, hi


def _bm25_component(b: float) -> float:
    """Saturating map from a raw BM25 score to [0,1).

    ``b / (b + SAT)`` is 0.5 at ``SAT`` and asymptotes to 1 — so a weak match
    stays visibly weak instead of being rescaled to "best of a bad lot".
    """
    if b <= 0:
        return 0.0
    return b / (b + BM25_SAT)


def _is_prose_query(query: str) -> bool:
    """True when the question is natural language rather than a code identifier."""
    q = query.strip()
    if not q:
        return False
    if re.search(r"[(){}\[\];=<>]|\.py|\.js|\bdef\b|\bclass\b|->|::", q):
        return False
    heb = len(re.findall(r"[\u0590-\u05FF]", q))
    return heb >= 3


def _phrase_probes(query: str, max_probes: int = 6) -> List[str]:
    """Contiguous spans of the query worth looking for verbatim in a chunk.

    Longest first: finding the whole question in a file is far stronger evidence
    than finding one of its words.
    """
    words = normalize(query).lower().split()
    words = [w for w in words if w]
    if not words:
        return []
    probes: List[str] = []
    for n in range(min(len(words), 6), 1, -1):
        for i in range(0, len(words) - n + 1):
            span = " ".join(words[i:i + n])
            if len(span) >= 6:
                probes.append(span)
        if len(probes) >= max_probes:
            break
    if not probes and words:
        probes = [w for w in words if len(w) >= 6][:max_probes]
    return probes[:max_probes]


def phrase_in(haystack: str, probe: str) -> bool:
    """True when ``probe`` occurs in ``haystack`` as a whole word or phrase.

    A raw substring test matched ``permission`` inside the identifier
    ``syncPermissions`` and counted that as phrase evidence for a question about
    the permission firewall — the strongest signal in the scorer, awarded for a
    variable name. Latin probes therefore need boundaries. Hebrew probes must
    *not* get them: Hebrew attaches prepositions and articles directly to the
    word ("בחומת" contains "חומת" legitimately).
    """
    if not probe:
        return False
    if probe.isascii():
        return re.search(rf"(?<![a-z0-9_]){re.escape(probe)}(?![a-z0-9_])", haystack) is not None
    return probe in haystack


def _mmr(hits: Sequence[Hit], k: int, vec: HashedNgramVectorizer,
         qv: np.ndarray) -> List[Hit]:
    """Maximal Marginal Relevance: relevance tempered by redundancy.

    Without it, asking about a topic discussed in one long file returns five
    consecutive chunks of that file — high relevance, near-zero information.
    """
    if not hits:
        return []
    vecs: Dict[int, np.ndarray] = {}
    for h in hits:
        vecs[h.chunk_id] = vec.transform_one(h.text)
    selected: List[Hit] = []
    pool = list(hits)
    top = max((h.score for h in pool), default=1.0) or 1.0
    while pool and len(selected) < k:
        best, best_val = None, -1e9
        for h in pool:
            red = 0.0
            for s in selected:
                red = max(red, float(vecs[h.chunk_id] @ vecs[s.chunk_id]))
            val = MMR_LAMBDA * (h.score / top) - (1 - MMR_LAMBDA) * red
            if val > best_val:
                best, best_val = h, val
        if best is None:
            break
        selected.append(best)
        pool.remove(best)
    return selected


if __name__ == "__main__":  # pragma: no cover - manual probe
    idx = RagIndex(Path("data/rag_index.sqlite3"))
    r = Retriever(idx)
    r.fit_vectorizer()
    for q in ["חומת הרשאות", "איך בנוי מנוע הדיבור", "permission firewall"]:
        print(f"\nQ: {q}")
        for h in r.search(q, k=3):
            print(f"  [{h.score:.3f}] {h.citation}  {h.text[:70]!r}")
