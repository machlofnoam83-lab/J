"""Hebrew-aware chunking with exact, citation-grade line spans.

A RAG system that cannot say *where* an answer came from is a rumour engine, so
every chunk carries the 1-based inclusive line range it was cut from, and those
ranges are computed from the real line array rather than estimated from
character offsets (an estimate drifts the first time a file contains a multi-byte
Hebrew character and a tab in the same paragraph).

Two strategies, because prose and code have different seams:

* **prose** — units are paragraphs (runs of consecutive non-blank lines). A
  paragraph is a semantic unit in Hebrew writing exactly as in English, and
  cutting mid-paragraph is what produces answers that stop before the "but".
* **code / data** — units are single lines. Splitting Python on ``.`` would
  shatter every float literal and every sentence-shaped comment.

Both strategies share the same assembler, including a genuine *overlap*: a new
chunk re-includes trailing units from its predecessor, so a fact straddling a
boundary is retrievable from either side.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

# Sentence enders. ``׃`` (sof pasuq) is included for religious/liturgical text.
_ENDERS = ".!?…׃"
# A heading in markdown, or a top-level definition in code — used for context.
_MD_HEADING = re.compile(r"^\s{0,3}(#{1,6})\s+(.*)$")
_CODE_HEADING = re.compile(
    r"^(?:def|class|function|async\s+def|const|let|var|struct|enum|interface|"
    r"public|private|protected|namespace|module|package|@app\.route|@REGISTRY\.register)\b"
)
# "e.g." / "i.e." / "דוג'." — a period that does not end a sentence.
_ABBREV = re.compile(r"(?:\b(?:[a-zA-Z]\.){1,4})$")


@dataclass
class Chunk:
    """One retrievable unit, with the exact place it came from."""

    ordinal: int
    start_line: int                 # 1-based, inclusive
    end_line: int                   # 1-based, inclusive
    text: str
    heading: str = ""
    kind: str = "prose"
    path: str = ""
    score_boost: float = 0.0
    terms: Tuple[str, ...] = field(default_factory=tuple)

    @property
    def n_chars(self) -> int:
        return len(self.text)

    @property
    def n_lines(self) -> int:
        return self.end_line - self.start_line + 1

    @property
    def span(self) -> str:
        if self.n_lines == 1:
            return f"שורה {self.start_line}"
        return f"שורות {self.start_line}–{self.end_line}"

    def to_dict(self) -> dict:
        return {"ordinal": self.ordinal, "start_line": self.start_line,
                "end_line": self.end_line, "lines": self.n_lines, "chars": self.n_chars,
                "heading": self.heading, "kind": self.kind, "path": self.path,
                "text": self.text}


# ------------------------------------------------------------------- units ---
@dataclass
class _Unit:
    start: int                      # 1-based first line
    end: int                        # 1-based last line
    text: str
    heading: str = ""


def _prose_units(lines: Sequence[str]) -> List[_Unit]:
    """Paragraphs: maximal runs of consecutive non-blank lines."""
    units: List[_Unit] = []
    start: Optional[int] = None
    buf: List[str] = []
    heading = ""
    for i, line in enumerate(lines, start=1):
        m = _MD_HEADING.match(line)
        if m:
            heading = m.group(2).strip()
        stripped = line.strip()
        if not stripped:
            if start is not None:
                units.append(_Unit(start, i - 1, "\n".join(buf), heading))
                start, buf = None, []
            continue
        if start is None:
            start = i
            buf = [line.rstrip()]
        else:
            buf.append(line.rstrip())
    if start is not None:
        units.append(_Unit(start, len(lines), "\n".join(buf), heading))
    return units


def _line_units(lines: Sequence[str]) -> List[_Unit]:
    """Code/data: every line is its own unit, so spans stay line-exact."""
    units: List[_Unit] = []
    heading = ""
    for i, line in enumerate(lines, start=1):
        if _CODE_HEADING.match(line) or _MD_HEADING.match(line):
            heading = line.strip()[:120]
        if not line.strip():
            continue
        units.append(_Unit(i, i, line.rstrip(), heading))
    return units


def _hard_split(unit: _Unit, max_chars: int) -> List[_Unit]:
    """Split an oversized unit on sentence boundaries, keeping line math exact.

    Prose paragraphs can legitimately exceed ``max_chars`` (a pasted log line, a
    single-line JSON document). We split on sentence enders first and only fall
    back to a raw character window when no ender exists — and in both cases the
    line range of each fragment is derived by counting the newlines it consumed.
    """
    out: List[_Unit] = []
    text = unit.text
    pieces: List[str] = []
    cur = ""
    for ch in text:
        cur += ch
        if ch in _ENDERS:
            pieces.append(cur)
            cur = ""
    if cur.strip():
        pieces.append(cur)
    if not pieces:
        pieces = [text]

    # Merge tiny fragments, then split any fragment that is still too big.
    merged: List[str] = []
    for p in pieces:
        if merged and len(merged[-1]) + len(p) <= max_chars:
            merged[-1] += p
        else:
            merged.append(p)
    final: List[str] = []
    for p in merged:
        while len(p) > max_chars:
            final.append(p[:max_chars])
            p = p[max_chars:]
        if p.strip():
            final.append(p)

    line_cursor = unit.start
    for frag in final:
        consumed = frag.count("\n")
        out.append(_Unit(line_cursor, line_cursor + consumed, frag.rstrip(), unit.heading))
        line_cursor += consumed
    # The last fragment must end exactly where the unit ended.
    if out:
        out[-1].end = unit.end
    return out


# --------------------------------------------------------------- sentences ---
def split_sentences(text: str) -> List[str]:
    """Sentence splitter that survives Hebrew punctuation and abbreviations."""
    text = text.strip()
    if not text:
        return []
    out: List[str] = []
    buf = ""
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        buf += ch
        if ch in _ENDERS:
            nxt = text[i + 1] if i + 1 < n else " "
            prev = buf[:-1].rstrip()
            # a decimal ("3.14") or a version ("1.2.3") is not a sentence end
            decimal = ch == "." and i + 1 < n and text[i + 1].isdigit() \
                and i > 0 and text[i - 1].isdigit()
            # The abbreviation test must see the period itself: "e.g" is not an
            # abbreviation pattern, "e.g." is.
            if not decimal and (nxt in (" ", "\n", "\t") or i + 1 >= n) \
                    and not _ABBREV.search(buf) and prev:
                out.append(buf.strip())
                buf = ""
        i += 1
    if buf.strip():
        out.append(buf.strip())
    return [s for s in out if s]


# ---------------------------------------------------------------- assembler --
def chunk_text(text: str, *, kind: str = "prose", target: int = 900,
               max_chars: int = 1600, overlap: int = 160,
               max_chunks: int = 500, path: str = "") -> List[Chunk]:
    """Cut ``text`` into retrieval units.

    ``target``    soft chunk size we aim for
    ``max_chars`` hard ceiling; a unit bigger than this is split further
    ``overlap``   how many trailing characters of the previous chunk to repeat
    """
    if not text or not text.strip():
        return []
    lines = text.split("\n")
    units = _prose_units(lines) if kind == "prose" else _line_units(lines)
    if not units:
        units = [_Unit(1, len(lines), text.strip())]

    # Oversized units are exploded before assembly so the assembler only ever
    # sees pieces it can place.
    prepared: List[_Unit] = []
    for u in units:
        if len(u.text) > max_chars:
            prepared.extend(_hard_split(u, max_chars))
        elif u.text.strip():
            prepared.append(u)

    sep = "\n\n" if kind == "prose" else "\n"
    chunks: List[Chunk] = []
    current: List[_Unit] = []
    current_len = 0
    heading = ""
    fresh = False                      # has anything *new* been added since flush?

    def tail_units() -> List[_Unit]:
        """The overlap carried into the next chunk, capped at ``overlap`` chars.

        Carrying whole units is wrong for prose: one paragraph is 100–500 chars,
        so a requested overlap of 160 silently became "repeat the entire previous
        paragraph". When the last unit is bigger than the budget we take a
        character-level suffix instead, snapped to a word boundary, and derive
        its start line from the newlines it contains — the span stays exact.
        """
        if overlap <= 0 or not current:
            return []
        acc: List[_Unit] = []
        total = 0
        for u in reversed(current):
            if total + len(u.text) > overlap and acc:
                break
            acc.insert(0, u)
            total += len(u.text) + len(sep)
            if total >= overlap:
                break
        if len(acc) == 1 and len(acc[0].text) > max(overlap, 1):
            u = acc[0]
            frag = u.text[-overlap:]
            cut = frag.find(" ")
            if 0 <= cut < len(frag) - 5:
                frag = frag[cut + 1:]
            start_line = u.end - frag.count("\n")
            acc = [_Unit(start_line, u.end, frag.strip(), u.heading)]
        return [a for a in acc if a.text.strip()]

    def flush() -> None:
        nonlocal current, current_len, heading, fresh
        if not current:
            return
        if not fresh:
            # Everything left in ``current`` is carried-over overlap. Emitting it
            # would append a chunk whose text is entirely a repeat of the tail of
            # the previous one — duplicate postings, duplicate retrieval hits and
            # a citation pointing at a span the reader already saw.
            current, current_len, fresh = [], 0, False
            return
        body = sep.join(u.text for u in current).strip()
        if body:
            chunks.append(Chunk(
                ordinal=len(chunks), start_line=current[0].start, end_line=current[-1].end,
                text=body, heading=heading or current[-1].heading, kind=kind, path=path,
            ))
        tail = tail_units()
        current = tail
        current_len = sum(len(u.text) + len(sep) for u in tail)
        heading = tail[-1].heading if tail else ""
        fresh = False

    for u in prepared:
        add = len(u.text) + (len(sep) if current else 0)
        if current and current_len + add > max_chars:
            flush()
            add = len(u.text) + (len(sep) if current else 0)
        current.append(u)
        current_len += add
        fresh = True
        if u.heading:
            # Set *after* the possible flush above, so this unit's heading lands
            # on the chunk that actually contains it.
            heading = u.heading
        if current_len >= target:
            flush()
    flush()

    # Degenerate guard: a file with no usable unit still gets one chunk so it is
    # at least findable by path/title, rather than silently vanishing.
    if not chunks and text.strip():
        chunks.append(Chunk(ordinal=0, start_line=1, end_line=max(1, len(lines)),
                            text=text.strip()[:max_chars], heading="", kind=kind, path=path))
    return chunks[:max_chunks]


if __name__ == "__main__":  # pragma: no cover - manual probe
    import sys
    from pathlib import Path

    sample = Path(sys.argv[1]).read_text(encoding="utf-8", errors="replace") if len(sys.argv) > 1 else (
        "# כותרת ראשית\n\n"
        "זהו פסקה ראשונה עם כמה משפטים. היא מספיק ארוכה כדי לבדוק את החלוקה. "
        "משפט שלישי כאן.\n\n"
        "פסקה שנייה. עוד קצת טקסט בעברית כדי לראות שהחפיפה עובדת.\n"
    )
    for c in chunk_text(sample, kind="prose", target=80, overlap=20):
        print(f"[{c.ordinal}] {c.span} ({c.n_chars} chars) heading={c.heading!r}")
        print("   " + c.text.replace("\n", " ⏎ ")[:110])
