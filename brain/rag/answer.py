"""Grounded answer composition — extractive by construction, citeable by law.

The single design decision that matters
---------------------------------------
JARVIS has a 6.2M-parameter model that is good at classification and *not* at
free-form Hebrew generation. If we let it write the answer, the citations would
be decoration: the model could say something no source says, and the user would
have no way to tell.

So this composer **cannot** paraphrase. Every sentence in the body of an answer
is copied verbatim out of a retrieved chunk, and the answer carries the file and
line range each sentence came from. :func:`verify_answer` re-checks that
property from the citations alone — it is not a comment, it is the invariant the
test-suite enforces, because "grounded" is exactly the kind of claim that rots
the moment someone edits the composer.

When nothing clears the evidence threshold we say so plainly and name the
nearest files, rather than emitting a confident sentence about nothing.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from brain.rag.chunk import split_sentences  # noqa: E402
from brain.rag.index import tokenize  # noqa: E402
from brain.rag.retrieve import Hit, _phrase_probes, phrase_in  # noqa: E402
from brain.tokenizer import normalize  # noqa: E402

# A hit must be at least this relevant before we are willing to quote it.
MIN_HIT_SCORE = 0.22
# A sentence must clear this to be quoted at all.
MIN_SENTENCE_SCORE = 0.10
# Below this the evidence is noise and we say "I don't know" instead of quoting
# something that merely shares words with the question.
NONE_CONFIDENCE = 0.25
GROUNDED_CONFIDENCE = 0.45
# How much of the answer body we are willing to speak aloud.
MAX_BODY_SENTENCES = 4
MAX_BODY_CHARS = 900

_NUM_RE = re.compile(r"\d+(?:[.,]\d+)?")


@dataclass
class Citation:
    """Where a quoted sentence came from. This is the whole point of the layer."""

    path: str
    file: str
    start_line: int
    end_line: int
    heading: str
    quote: str
    score: float
    chunk_id: int = 0
    matched_terms: Tuple[str, ...] = ()

    @property
    def span_he(self) -> str:
        if self.end_line == self.start_line:
            return f"שורה {self.end_line}"
        return f"שורות {self.start_line}–{self.end_line}"

    @property
    def label(self) -> str:
        return f"{self.file} · {self.span_he}"

    def to_dict(self) -> Dict[str, Any]:
        return {"path": self.path, "file": self.file, "start_line": self.start_line,
                "end_line": self.end_line, "heading": self.heading, "quote": self.quote,
                "score": round(self.score, 4), "chunk_id": self.chunk_id,
                "span_he": self.span_he, "label": self.label,
                "matched_terms": list(self.matched_terms)}


@dataclass
class GroundedAnswer:
    """The result of a RAG query: text, speech, evidence, and honesty flags."""

    query: str
    text: str
    speak: str
    grounded: bool
    answer_type: str                 # grounded | weak | none
    confidence: float
    citations: List[Citation] = field(default_factory=list)
    hits: List[Hit] = field(default_factory=list)
    reason: str = ""
    stats: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query, "text": self.text, "speak": self.speak,
            "grounded": self.grounded, "answer_type": self.answer_type,
            "confidence": round(self.confidence, 4), "reason": self.reason,
            "citations": [c.to_dict() for c in self.citations],
            "sources": [{"path": h.path, "file": h.file if hasattr(h, "file")
                         else Path(h.path).name, "start_line": h.start_line,
                         "end_line": h.end_line, "score": round(h.score, 4),
                         "heading": h.heading,
                         "parts": {k: round(v, 4) for k, v in h.parts.items()}}
                        for h in self.hits],
            "stats": self.stats,
        }


# ------------------------------------------------------------------ scoring --
def _sentence_score(sentence: str, q_terms: Sequence[str], probes: Sequence[str],
                    q_numbers: Sequence[str], position: float, rank: float,
                    heading: str, q_words: int = 1) -> float:
    """How well one candidate sentence answers the question.

    Deliberately dominated by *evidence* (term coverage, verbatim phrase,
    matching numbers) rather than fluency: we are choosing what to quote, not
    writing prose.
    """
    s_terms = set(tokenize(sentence))
    if not s_terms:
        return 0.0
    qs = set(q_terms)
    coverage = len(qs & s_terms) / len(qs) if qs else 0.0

    evidence = 1.0 * coverage
    ns = normalize(sentence).lower()
    for probe in probes:
        if phrase_in(ns, probe):
            # Same scaling as the retriever: a probe covering two of the
            # question's three words is two-thirds of the evidence, not all of
            # it. Flat 1.0 let a shared two-word prefix quote an unrelated
            # docstring example as if it were the answer.
            evidence += 0.75 * (len(probe.split()) / max(1, q_words))
            break

    if q_numbers:
        s_nums = set(_NUM_RE.findall(sentence))
        if any(n in s_nums for n in q_numbers):
            evidence += 0.60

    head_terms = set(tokenize(heading)) if heading else set()
    if head_terms and qs & head_terms:
        evidence += 0.20

    # Position and source rank *modulate* evidence — they are not evidence. An
    # earlier version added them unconditionally, which handed 0.40 to a
    # sentence that shared no word at all with the question, and that free
    # credit is exactly what made "מה זה רקורסיה" quote an unrelated test table.
    if evidence <= 0:
        return 0.0
    score = evidence * (0.75 + 0.15 * (1.0 - position) + 0.10 * (1.0 - rank))

    n = len(sentence.strip())
    if n < 24:
        score *= 0.55          # a stub is rarely an answer
    elif n > 600:
        score *= 0.85          # a wall of text is not speakable
    # Saturate into [0,1). Without this the raw sum reached ~2.9 and every
    # threshold downstream was meaningless, because "high score" just meant
    # "long sentence with common words in it".
    return score / (score + 1.0)


def _dedupe_key(sentence: str) -> str:
    """Collapse the overlap region: adjacent chunks repeat boundary sentences."""
    n = re.sub(r"\s+", " ", normalize(sentence).lower()).strip(" .!?…׃")
    return n[:160]


CODE_WINDOW_LINES = 5      # how many consecutive code lines to quote at once


def _code_windows(hit: Hit) -> List[str]:
    """Quote units for code/data chunks.

    Running a sentence splitter over source code produces fragments that stop in
    the middle of an expression — technically verbatim, useless as evidence. A
    contiguous window of whole lines is the smallest honest unit of code.
    """
    lines = [l for l in hit.text.split("\n")]
    if not lines:
        return []
    step = max(1, CODE_WINDOW_LINES // 2)
    out: List[str] = []
    for i in range(0, len(lines), step):
        window = lines[i:i + CODE_WINDOW_LINES]
        text = "\n".join(window).strip("\n")
        if text.strip():
            out.append(text)
        if i + CODE_WINDOW_LINES >= len(lines):
            break
    return out


# ---------------------------------------------------------------- composing --
def compose_answer(query: str, hits: Sequence[Hit], *,
                   k_sentences: int = MAX_BODY_SENTENCES,
                   max_chars: int = MAX_BODY_CHARS,
                   min_hit_score: float = MIN_HIT_SCORE,
                   min_sentence_score: float = MIN_SENTENCE_SCORE) -> GroundedAnswer:
    """Build a cited, extractive answer from retrieved chunks.

    Returns a :class:`GroundedAnswer` with ``answer_type`` in
    ``{"grounded", "weak", "none"}`` — the caller (and the HUD) can refuse to
    present a ``weak`` answer as fact.
    """
    query = (query or "").strip()
    q_terms = tokenize(query)
    q_unique = set(q_terms)
    probes = _phrase_probes(query)
    q_numbers = _NUM_RE.findall(query)
    q_words = max(1, len(normalize(query).lower().split()))

    # NOTE on a gate that was tried and removed. A hard "chunk must contain
    # ≥60% of the query's content terms" rule looks obviously right and measured
    # badly: on this repository the queries the corpus *can* answer legitimately
    # sit at 0.33–0.50 coverage ("איך מוסיפים סקיל חדש" → 0.33, because the docs
    # say "skill" in English while the question says "סקיל"), while an off-topic
    # question reached 1.00. The gate destroyed true positives and caught
    # nothing. Coverage is kept as a *scoring* signal, where it belongs, and the
    # honest-refusal decision is made on confidence instead.
    usable = [h for h in hits if h.score >= min_hit_score]

    # -------------------------------------------------- collect candidates --
    candidates: List[Tuple[float, int, int, str, Hit]] = []
    seen: set = set()
    for rank, hit in enumerate(usable):
        if hit.kind in ("code", "data"):
            units = _code_windows(hit)
        else:
            units = split_sentences(hit.text)
        if not units:
            continue
        for pos, sent in enumerate(units):
            key = _dedupe_key(sent)
            if not key or key in seen:
                continue
            sc = _sentence_score(sent, q_terms, probes, q_numbers,
                                 pos / max(1, len(units) - 1),
                                 rank / max(1, len(usable) - 1),
                                 hit.heading, q_words)
            if sc < min_sentence_score:
                continue
            seen.add(key)
            candidates.append((sc, rank, pos, sent, hit))

    if not candidates:
        return _no_answer(query, hits, "no sentence in the index clears the evidence bar")

    candidates.sort(key=lambda c: (c[0], -c[1]), reverse=True)
    # At most one quote per chunk: two sentences from the same span are one
    # source, and listing them twice made a single file look like corroboration.
    picked: List[Tuple[float, int, int, str, Hit]] = []
    used_chunks: set = set()
    for cand in candidates:
        if cand[4].chunk_id in used_chunks:
            continue
        used_chunks.add(cand[4].chunk_id)
        picked.append(cand)
        if len(picked) >= k_sentences * 2:
            break
    chosen = picked[:k_sentences]
    # Re-order for reading: by source rank, then position inside that source.
    chosen.sort(key=lambda c: (c[1], c[2]))

    citations: List[Citation] = []
    body: List[str] = []
    total = 0
    for sc, rank, _pos, sent, hit in chosen:
        if total + len(sent) > max_chars and body:
            continue
        body.append(sent)
        total += len(sent) + 2
        citations.append(Citation(
            path=hit.path, file=Path(hit.path).name, start_line=hit.start_line,
            end_line=hit.end_line, heading=hit.heading, quote=sent, score=sc,
            chunk_id=hit.chunk_id, matched_terms=hit.matched_terms))

    top = usable[0]
    best_sentence = max(c[0] for c in chosen)
    coverage = (len(set(top.matched_terms)) / len(set(q_terms))) if q_terms else 0.0
    phrase = 1.0 if top.parts.get("phrase", 0) > 0 else 0.0
    # Each term is already bounded in [0,1], so this is a real probability-like
    # number rather than a rescaled "best of whatever came back".
    confidence = round(min(0.99,
                           0.40 * min(1.0, top.score / 0.55)
                           + 0.35 * best_sentence
                           + 0.15 * coverage
                           + 0.10 * phrase), 4)

    if confidence < NONE_CONFIDENCE:
        return _no_answer(query, hits,
                          f"best evidence scored {confidence:.2f}, below {NONE_CONFIDENCE}")

    files = sorted({c.file for c in citations})
    n_files = len(files)
    where = files[0] if n_files == 1 else f"{n_files} קבצים"

    body_he = "\n".join(f"[{i}] {s}" for i, s in enumerate(body, start=1))
    src_he = "\n".join(f"    · {c.label}" for c in citations)
    text = (f"מצאתי את זה ב{where}.\n\n{body_he}\n\nמקורות:\n{src_he}")

    speak = _speakable(citations, where, code=citations[0].quote.count("\n") >= 2)
    answer_type = "grounded" if confidence >= GROUNDED_CONFIDENCE else "weak"

    return GroundedAnswer(
        query=query, text=text, speak=speak, grounded=True, answer_type=answer_type,
        confidence=confidence, citations=citations, hits=list(usable[:8]),
        reason=f"{len(citations)} quoted sentence(s) from {n_files} file(s)",
        stats={"candidates": len(candidates), "usable_hits": len(usable),
               "top_hit_score": round(top.score, 4),
               "best_sentence_score": round(best_sentence, 4),
               "term_coverage": round(coverage, 4),
               "phrase": bool(top.parts.get("phrase", 0) > 0),
               "quoted_chars": total},
    )


def _speakable(citations: Sequence[Citation], where: str,
               code: bool = False) -> str:
    """A short spoken version — the HUD reads this, not the whole body.

    Source code is never read aloud: a TTS engine pronouncing ``self._lock``
    character by character is not an answer, it is noise. For a code source we
    name the file and the line range and stop.
    """
    if not citations:
        return "לא מצאתי את זה בקבצים שלך."
    first = citations[0]
    if code:
        return f"מצאתי את זה בקובץ {first.file}, {first.span_he}. הקטע מוצג על המסך."
    quote = first.quote.strip()
    if len(quote) > 240:
        quote = quote[:237].rsplit(" ", 1)[0] + "…"
    return f"מצאתי את זה בקובץ {first.file}, {first.span_he}. {quote}"


def _no_answer(query: str, hits: Sequence[Hit], reason: str) -> GroundedAnswer:
    """The honest failure path. Names the nearest files instead of guessing."""
    if hits:
        near = ", ".join(sorted({Path(h.path).name for h in hits[:3]}))
        text = (f"לא מצאתי תשובה לשאלה הזו בקבצים המאונדקסים. "
                f"הקרובים ביותר שהגעתי אליהם: {near} — אבל אף משפט בהם לא עונה על השאלה, "
                f"ואני לא ממציא תשובה.")
        speak = "לא מצאתי תשובה בקבצים שלך. אני לא ממציא."
    else:
        text = ("האינדקס לא החזיר אף קטע רלוונטי. יכול להיות שהקבצים לא מאונדקסים עדיין — "
                "אפשר להריץ אינדוקס מחדש ולנסות שוב.")
        speak = "אין לי עדיין קבצים מאונדקסים בנושא הזה."
    return GroundedAnswer(
        query=query, text=text, speak=speak, grounded=False, answer_type="none",
        confidence=0.0, citations=[], hits=list(hits[:5]), reason=reason,
        stats={"usable_hits": len(hits)},
    )


# --------------------------------------------------------------- invariant ---
def verify_answer(answer: GroundedAnswer, sources: Optional[Dict[str, str]] = None) -> Tuple[bool, List[str]]:
    """Re-check the grounding invariant from the citations alone.

    Every quoted sentence must appear verbatim in the file it is attributed to
    (after whitespace normalisation). ``sources`` maps path -> file text; when
    omitted the check runs against the retrieved chunk text, which is the weaker
    but always-available form. The test-suite passes real file contents so the
    claim being checked is the one the user sees.
    """
    problems: List[str] = []
    if answer.grounded and not answer.citations:
        problems.append("grounded answer with no citations")
    if not answer.grounded and answer.citations:
        problems.append("ungrounded answer carrying citations")
    for i, c in enumerate(answer.citations):
        if not c.quote.strip():
            problems.append(f"citation {i}: empty quote")
        if c.start_line < 1 or c.end_line < c.start_line:
            problems.append(f"citation {i}: impossible span {c.start_line}-{c.end_line}")
        if sources is not None:
            text = sources.get(c.path)
            if text is None:
                problems.append(f"citation {i}: source file {c.path!r} not provided")
                continue
            # Compare the RAW strings, collapsing whitespace only. Normalising
            # both sides here — which is what the first version did — made the
            # check blind to exactly the damage it exists to catch: a quote
            # mangled into "קבצימ" still matched a file containing "קבצים"
            # because both sides folded final forms before comparing.
            a = re.sub(r"\s+", " ", c.quote).strip()
            b = re.sub(r"\s+", " ", text.replace("\r\n", "\n").replace("\r", "\n"))
            if a and a not in b:
                problems.append(f"citation {i}: quote not found verbatim in {c.path}")
    return (not problems), problems
