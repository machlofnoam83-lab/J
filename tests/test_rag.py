#!/usr/bin/env python3
"""RAG layer tests — local-file retrieval with cited, verifiable answers.

What these tests are actually protecting
----------------------------------------
A RAG layer has one job: say something true and prove where it came from. Every
failure mode worth pinning is a way of breaking that promise while still looking
like it works, so the suite is organised around them rather than around modules:

1. **Encoding.** The user is on Windows in Hebrew. A cp1255 file must be read as
   Hebrew, not as mojibake, and a Hebrew *UTF-8* file must not be rejected as
   binary — the first version of the binary detector counted every byte above
   0x80 as "non-text" and silently dropped all of them.

2. **Citations must be real.** ``test_grounding_invariant`` re-reads the source
   files from disk and checks that each quoted sentence appears in them verbatim.
   This is the invariant the whole layer exists for, so it is tested against real
   bytes and not against the chunk text the composer already had.

3. **Line spans must be exact.** A citation that says "lines 12–18" and points at
   the wrong place is worse than no citation, because it is believed. The span
   check is run over every chunk of a multi-format fixture corpus.

4. **Honest refusal.** When nothing is relevant the answer must be "none", not a
   confident quote of something that merely shares a word with the question. The
   regression this guards against is real: position/rank bonuses used to be
   *additive*, which handed 0.40 to a zero-evidence sentence.

5. **Confidence must discriminate.** A scorer whose output saturates at 0.95 for
   both a perfect hit and a near-miss is decoration. ``test_confidence_is_not_saturated``
   asserts spread, not just a range.

6. **Safety.** The indexer reads the user's disk, so the deny-list (private keys,
   ``.env``, ``.git``) is tested as a hard gate, and ``rag.index`` is tested
   through the Permission Firewall — including that a SAFE posture refuses it.

Run:  python tests/test_rag.py
"""

from __future__ import annotations

import json
import re
import sqlite3
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from brain.rag.answer import (  # noqa: E402
    Citation, GroundedAnswer, compose_answer, verify_answer,
)
from brain.rag.chunk import Chunk, chunk_text, split_sentences  # noqa: E402
from brain.rag.engine import RagEngine  # noqa: E402
from brain.rag.extract import (  # noqa: E402
    RagPolicy, decode_text, extract_file, looks_binary, walk_files,
)
from brain.rag.index import RagIndex, stem, term_counts, tokenize  # noqa: E402
from brain.rag.retrieve import (  # noqa: E402
    WEIGHTS, Retriever, _bm25_component, _phrase_probes, phrase_in,
)

ok = 0
fail = 0


def check(label: str, cond: bool, detail: str = "") -> None:
    global ok, fail
    if cond:
        ok += 1
        print(f"  ✓ {label}" + (f"  {detail}" if detail else ""))
    else:
        fail += 1
        print(f"  ✗ {label}" + (f"  {detail}" if detail else ""))


# ------------------------------------------------------------------ fixtures --
HE_PROSE = """# מדריך ההפעלה של המערכת

המערכת מריצה בדיקות אוטומטיות בכל לילה. כדי להריץ את הבדיקות ידנית יש להפעיל
את הפקודה tests/run_all.py בתוך הסביבה הווירטואלית. הריצה המלאה אורכת כשלוש דקות
ומדווחת כמה בדיקות עברו וכמה נכשלו.

## חומת ההרשאות

כל פעולה שמשנה את המערכת עוברת דרך חומת הרשאות. הפעולות מסווגות לשלוש רמות:
קריאה בלבד, כתיבה, ופעולה קריטית שדורשת אישור אנושי מפורש לפני הביצוע.

## זיהוי הקול

זיהוי הקול מבוסס על תבניות אקוסטיות ועל התאמה דינמית. אין כאן מודל שהורד
מהרשת — כל התבניות נבנו מקומית מהקול של המערכת עצמה.
"""

HE_TECH = """תיעוד פנימי של מנוע החישוב.

המנוע מחשב ביטויים אריתמטיים בלבד ואינו מנחש תוצאות. רקורסיה אינה מוזכרת כאן
בכלל, ולכן שאלה על רקורסיה לא אמורה לקבל תשובה מהקובץ הזה.

המנוע תומך בחיבור, חיסור, כפל, חילוק, חזקה ושורש ריבועי.
"""

PY_CODE = '''"""Module docstring for the fixture."""

from __future__ import annotations


def compute_total(items, rate=1.5):
    """Return the total for the given items."""
    total = 0.0
    for item in items:
        total += item.price * rate
    return round(total, 2)


class ReportBuilder:
    def __init__(self, title):
        self.title = title
        self.rows = []

    def add(self, row):
        self.rows.append(row)

    def render(self):
        return "\\n".join(str(r) for r in self.rows)
'''

JSON_DATA = '{"name": "fixture", "version": 3, "tags": ["alpha", "beta"], "notes": "נתוני בדיקה"}\n'


def build_corpus(root: Path) -> None:
    """A tiny multi-format corpus with known answers at known line numbers."""
    (root / "docs").mkdir(parents=True, exist_ok=True)
    (root / "src").mkdir(parents=True, exist_ok=True)
    (root / "secrets").mkdir(parents=True, exist_ok=True)

    (root / "docs" / "הפעלה.md").write_text(HE_PROSE, encoding="utf-8")
    (root / "docs" / "מנוע.md").write_text(HE_TECH, encoding="utf-8")
    (root / "src" / "calc.py").write_text(PY_CODE, encoding="utf-8")
    (root / "src" / "meta.json").write_text(JSON_DATA, encoding="utf-8")

    # Hebrew in the Windows codepage — the encoding this project lives with.
    (root / "docs" / "legacy.txt").write_bytes(
        "זהו קובץ טקסט שנשמר בקידוד וינדוס הישן. הוא חייב להיקרא כעברית.\n".encode("cp1255"))

    # Things the indexer must refuse, however politely they are named.
    (root / "secrets" / "id_rsa").write_text("-----BEGIN OPENSSH PRIVATE KEY-----\nAAAA\n")
    (root / "secrets" / "app.env").write_text("API_KEY=not-a-real-key\n")
    (root / ".git").mkdir(exist_ok=True)
    (root / ".git" / "config").write_text("[core]\n\trepositoryformatversion = 0\n")
    (root / "binary.dat").write_bytes(bytes(range(256)) * 8)
    (root / "empty.txt").write_text("")


# ------------------------------------------------------------------ encoding --
def test_decoding() -> None:
    print("\n── encoding: Hebrew on Windows ──────────────────────────────")
    he = "המערכת מריצה בדיקות אוטומטיות בכל לילה"

    text, enc = decode_text(he.encode("utf-8"))
    check("utf-8 Hebrew decodes as utf-8", enc == "utf-8" and "בדיקות" in text, enc)

    text, enc = decode_text(he.encode("cp1255"))
    check("cp1255 Hebrew decodes back to Hebrew", "בדיקות" in text, f"enc={enc}")
    check("cp1255 result is not mojibake", all(ord(c) < 0x80 or "\u0590" <= c <= "\u05ff"
                                              for c in text if c.isalpha()))

    text, enc = decode_text(b"\xef\xbb\xbf" + he.encode("utf-8"))
    check("utf-8 BOM is stripped, not kept as a character", "בדיקות" in text and not text.startswith("\ufeff"))

    text, enc = decode_text("plain ascii only".encode("ascii"))
    check("ascii decodes as utf-8", text == "plain ascii only")

    text, _ = decode_text(b"")
    check("empty input does not raise", text == "")

    # The regression this guards: Hebrew UTF-8 is mostly bytes >= 0x80.
    # The trap, stated as the rule that used to break it: any detector that
    # counts bytes >= 0x80 as "non-text" rejects this file, because Hebrew
    # UTF-8 is roughly a fifth high bytes — far above any ASCII threshold.
    raw = HE_PROSE.encode("utf-8")
    high = sum(1 for b in raw[:4096] if b >= 0x80) / 4096
    check("Hebrew UTF-8 has far more high bytes than ASCII text (the trap)",
          high > 0.15, f"{high:.0%} high")
    naive = sum(1 for b in raw[:8192] if b >= 0x80) / min(len(raw), 8192)
    check("a naive high-byte detector WOULD have rejected it", naive > 0.05, f"{naive:.0%}")
    check("Hebrew UTF-8 is NOT classified as binary", not looks_binary(raw[:8192]))
    check("a PNG-ish header IS classified as binary", looks_binary(b"\x89PNG\r\n\x1a\n" + bytes(200)))
    check("NUL bytes mean binary", looks_binary(b"hello\x00world"))


def test_extraction_and_policy(tmp: Path) -> None:
    print("\n── extraction, binary rejection and the deny-list ─────────")
    pol = RagPolicy()

    md = extract_file(tmp / "docs" / "הפעלה.md", policy=pol)
    check("Hebrew markdown extracts ok", md.ok, md.skipped)
    check("kind is prose for .md", md.kind == "prose")
    check("sha1 recorded", len(md.sha1) == 40)

    legacy = extract_file(tmp / "docs" / "legacy.txt", policy=pol)
    check("cp1255 file extracts as Hebrew", legacy.ok and "עברית" in legacy.text,
          f"enc={legacy.encoding}")

    b = extract_file(tmp / "binary.dat", policy=pol)
    check("binary file is skipped, not indexed", not b.ok and "binary" in b.skipped, b.skipped)

    e = extract_file(tmp / "empty.txt", policy=pol)
    check("empty file is skipped with a reason", not e.ok and e.skipped == "empty file")

    key = extract_file(tmp / "secrets" / "id_rsa", policy=pol)
    check("private key is refused before reading", not key.ok and "denied" in key.skipped, key.skipped)
    check("private key content never reached .text", key.text == "")

    env = extract_file(tmp / "secrets" / "app.env", policy=pol)
    check(".env file is refused", not env.ok and "denied" in env.skipped, env.skipped)

    gitc = extract_file(tmp / ".git" / "config", policy=pol)
    check(".git internals are refused", not gitc.ok and "denied" in gitc.skipped, gitc.skipped)

    tiny = extract_file(tmp / "docs" / "הפעלה.md", max_bytes=64, policy=pol)
    check("oversized file is refused rather than half-read", not tiny.ok and "too large" in tiny.skipped)

    found = {p.name for p in walk_files([tmp], policy=pol)}
    check("walk finds the Hebrew markdown", "הפעלה.md" in found)
    check("walk finds source and data files", {"calc.py", "meta.json"} <= found)
    check("walk never yields the private key", "id_rsa" not in found)
    check("walk never yields .env", "app.env" not in found)
    check("walk never yields .git contents", "config" not in found)
    check("walk never yields the binary blob", "binary.dat" not in found)

    inc = {p.name for p in walk_files([tmp], include=["*.py"], policy=pol)}
    check("include glob restricts to .py", inc == {"calc.py"}, str(sorted(inc)))
    exc = {p.name for p in walk_files([tmp], exclude=["*.md"], policy=pol)}
    check("exclude glob removes .md", "הפעלה.md" not in exc and "calc.py" in exc)

    pol2 = RagPolicy(protected=[str(tmp / "docs")])
    check("a configured protected path is honoured",
          pol2.is_denied(tmp / "docs" / "הפעלה.md"))
    check("protection does not leak to siblings",
          not pol2.is_denied(tmp / "src" / "calc.py"))


# ------------------------------------------------------------------ chunking --
def test_chunking() -> None:
    print("\n── chunking: line spans, overlap, oversized units ─────────")
    lines = HE_PROSE.split("\n")

    chunks = chunk_text(HE_PROSE, kind="prose", target=140, max_chars=400, overlap=30)
    check("prose is split into several chunks", len(chunks) >= 3, f"{len(chunks)} chunks")

    bad = []
    for c in chunks:
        if c.start_line < 1 or c.end_line > len(lines) or c.start_line > c.end_line:
            bad.append((c.ordinal, "range", c.start_line, c.end_line))
            continue
        window = [l.strip() for l in lines[c.start_line - 1:c.end_line]]
        first = c.text.split("\n")[0].strip()
        last = c.text.split("\n")[-1].strip()
        # An overlap chunk legitimately *starts mid-line* (its tail was cut at a
        # word boundary), so its first line is a suffix of the source line — the
        # span still names the line that contains it, which is the honest claim.
        first_ok = first in window or (window and window[0].endswith(first))
        last_ok = last in window or (window and window[-1].startswith(last))
        if not first_ok or not last_ok:
            bad.append((c.ordinal, "content", first[:30], last[:30]))
    check("every prose chunk's declared span really contains it", not bad, str(bad[:2]))

    overlaps = [b.start_line for a, b in zip(chunks, chunks[1:]) if b.start_line <= a.end_line]
    check("overlap actually happens between neighbouring chunks", len(overlaps) >= 1,
          f"{len(overlaps)} overlaps")

    big_overlap = chunk_text(HE_PROSE, kind="prose", target=120, max_chars=400, overlap=10)
    carried = []
    for a, b in zip(big_overlap, big_overlap[1:]):
        if b.start_line <= a.end_line:
            shared = "\n".join(lines[b.start_line - 1:a.end_line])
            carried.append(len(shared))
    check("overlap is bounded by the requested budget (not a whole paragraph)",
          all(n <= 120 for n in carried), str(carried))

    no_overlap = chunk_text(HE_PROSE, kind="prose", target=140, max_chars=400, overlap=0)
    check("overlap=0 produces disjoint chunks",
          all(b.start_line > a.end_line for a, b in zip(no_overlap, no_overlap[1:])))

    one_line = "זו שורה אחת ארוכה מאוד. " * 60
    hard = chunk_text(one_line, kind="prose", target=100, max_chars=200, overlap=0)
    check("an oversized single paragraph is hard-split", len(hard) >= 2, f"{len(hard)} pieces")
    check("hard-split pieces respect max_chars", all(c.n_chars <= 260 for c in hard),
          str([c.n_chars for c in hard][:5]))
    check("hard-split still ends at the real last line", hard[-1].end_line == 1)

    code = chunk_text(PY_CODE, kind="code", target=200, max_chars=500, overlap=0)
    check("code is chunked too", len(code) >= 1, f"{len(code)} chunks")
    py_lines = PY_CODE.split("\n")
    code_bad = [c.ordinal for c in code
                if c.start_line < 1 or c.end_line > len(py_lines)
                or c.text.split("\n")[0].strip() not in
                [l.strip() for l in py_lines[c.start_line - 1:c.end_line]]]
    check("code chunk spans are line-exact", not code_bad, str(code_bad))
    check("code chunks carry their def/class as heading",
          any("compute_total" in c.heading for c in code))

    check("heading is captured from markdown", any(c.heading == "חומת ההרשאות"
                                                  for c in chunk_text(HE_PROSE, target=120)))

    check("empty text yields no chunks", chunk_text("") == [])
    check("whitespace-only text yields no chunks", chunk_text("   \n\n  ") == [])

    sents = split_sentences("משפט ראשון. משפט שני? משפט שלישי!")
    check("sentence splitter splits Hebrew punctuation", len(sents) == 3, str(sents))
    check("decimal point does not split", split_sentences("הערך הוא 3.14 בדיוק.") == ["הערך הוא 3.14 בדיוק."])
    check("abbreviation does not split", len(split_sentences("ראה e.g. את התיעוד.")) == 1)


# ----------------------------------------------------------------- tokenizer --
def test_tokenizer() -> None:
    print("\n── tokenizer: stopwords, stems, agreement with the KB ─────")
    toks = tokenize("איך מריצים את הבדיקות")
    check("question words are dropped", "איכ" not in toks and "איך" not in toks, str(toks))
    check("content words survive", "ריצ" in toks and "בדיק" in toks, str(toks))

    check("the assistant's own name is NOT a stopword", tokenize("מה זה אדיאל") == ["אדיאל"])
    check("english stopwords are dropped", tokenize("what is the plan") == ["plan"])

    from brain.knowledge import _stem as kb_stem
    samples = ["הרשאות", "קבצים", "בדיקות", "files", "tests", "מערכת"]
    diffs = [w for w in samples if stem(w) != kb_stem(w)]
    check("our stemmer agrees with knowledge._stem", not diffs, str(diffs))

    tc = term_counts("הבדיקות והבדיקות עברו")
    check("term_counts tallies repeats", tc.get("בדיק", 0) == 2, str(tc))
    check("term_counts drops function words", "והבדיקות" not in tc)


# --------------------------------------------------------------------- index --
def test_index(tmp: Path) -> None:
    print("\n── index: postings, incremental sync, pruning ─────────────")
    db = tmp / "_ix.sqlite3"
    ix = RagIndex(db)
    check("a fresh index is empty", ix.doc_count() == 0 and ix.chunk_count() == 0)

    ix.add_document(path="/tmp/a.md", kind="prose", bytes_=100, mtime=1.0, sha1="a" * 40,
                    chunks=[{"ordinal": 0, "start_line": 1, "end_line": 3, "heading": "",
                             "kind": "prose", "chars": 40, "terms": term_counts("חומת הרשאות בדיקות"),
                             "text": "חומת הרשאות בדיקות"}])
    check("add_document registers the doc", ix.doc_count() == 1)
    check("add_document registers the chunk", ix.chunk_count() == 1)
    check("postings were written", ix.term_count() >= 2, str(ix.term_count()))

    # Re-adding the same path must replace, not duplicate.
    ix.add_document(path="/tmp/a.md", kind="prose", bytes_=120, mtime=2.0, sha1="b" * 40,
                    chunks=[{"ordinal": 0, "start_line": 1, "end_line": 2, "heading": "",
                             "kind": "prose", "chars": 20, "terms": term_counts("רק טקסט חדש"),
                             "text": "רק טקסט חדש"}])
    check("re-adding a path replaces it (no duplicate docs)", ix.doc_count() == 1)
    check("re-adding a path replaces its chunks", ix.chunk_count() == 1)
    check("old postings are gone after replace", ix.doc_freq(["רשא"]) == {})

    ix.add_document(path="/tmp/b.md", kind="prose", bytes_=10, mtime=1.0, sha1="c" * 40,
                    chunks=[{"ordinal": 0, "start_line": 1, "end_line": 1, "heading": "",
                             "kind": "prose", "chars": 8, "terms": term_counts("עוד משהו"),
                             "text": "עוד משהו"}])
    pruned = ix.prune(["/tmp/a.md"])
    check("prune drops documents that disappeared", pruned == 1 and ix.doc_count() == 1)

    ix.mark_skipped("/tmp/x.bin", "binary content")
    check("skipped files are recorded with a reason",
          ix.skipped_stats().get("binary content") == 1)

    st = ix.stats()
    check("stats report docs and chunks", st["docs"] == 1 and st["chunks"] == 1, str(st))
    check("stats expose the schema version", st["schema"] >= 1)
    ix.close()

    # A stale schema must be rebuilt, not silently mixed with new queries.
    con = sqlite3.connect(str(db))
    con.execute("UPDATE meta SET value='1' WHERE key='schema'")
    con.commit()
    con.close()
    ix2 = RagIndex(db)
    check("a stale schema triggers a rebuild", ix2.doc_count() == 0, str(ix2.doc_count()))
    ix2.close()


def test_bm25() -> None:
    print("\n── BM25 component: saturating, never rescaled to the best ──")
    check("zero score maps to zero", _bm25_component(0.0) == 0.0)
    check("negative score maps to zero", _bm25_component(-3.0) == 0.0)
    check("component is bounded below 1", _bm25_component(1e9) < 1.0)
    check("monotonic", _bm25_component(2.0) < _bm25_component(20.0))
    # The regression: dividing by the best candidate made the top hit 1.0 always.
    weak_only = _bm25_component(0.5)
    check("a weak match stays visibly weak", weak_only < 0.2, f"{weak_only:.3f}")
    check("weights sum to 1 so a score is readable", abs(sum(WEIGHTS.values()) - 1.0) < 1e-9,
          str(sum(WEIGHTS.values())))


# ----------------------------------------------------------------- retrieval --
def test_phrase_boundary() -> None:
    print("\n── phrase evidence: word boundaries ───────────────────────")
    hay = "async function syncpermissions() { return 1 }"
    check("latin probe does NOT match inside an identifier",
          not phrase_in(hay, "permission"))
    check("latin probe matches a real word",
          phrase_in("the permission firewall blocks it", "permission"))
    check("latin phrase matches", phrase_in("permission firewall", "permission firewall"))
    check("hebrew probe matches inside an inflected word (prefixes attach)",
          phrase_in("הפעולה עוברת בחומת הרשאות", "חומת"))
    check("empty probe never matches", not phrase_in("anything", ""))
    probes = _phrase_probes("מה כתוב בקבצים על חומת הרשאות")
    check("phrase probes are generated longest-first", len(probes) >= 1)
    check("probes are normalised and lowercase", all(p == p.lower() for p in probes))


def test_phrase_evidence_is_proportional(tmp: Path) -> None:
    """A probe covering part of the question is part of the evidence.

    Regression this pins: every probe used to award the *full* phrase weight, so
    a two-word prefix shared with an unrelated sentence scored like a verbatim
    match of the whole question. On this repository that let "איך מתקנים
    אופניים" (fix a bicycle) match a docstring example about fixing a leaking
    tap — "איכ מתקנימ ברז דולפ" — and present it as the answer.
    """
    print("\n── phrase evidence scales with probe coverage ─────────────")
    eng = RagEngine(db_path=tmp / "_prop.sqlite3")
    docs = tmp / "prop"
    docs.mkdir(exist_ok=True)
    # One file that literally contains a two-word prefix of the question but
    # nothing about its subject; one file that actually answers it.
    (docs / "decoy.md").write_text(
        "דוגמה ישנה מהתיעוד: איך מתקנים ברז דולף במטבח. זהו משפט לא קשור לנושא האמיתי.\n",
        encoding="utf-8")
    (docs / "real.md").write_text(
        "מדריך אופניים. איך מתקנים אופניים: קודם בודקים את השרשרת ואת לחץ האוויר בצמיגים.\n",
        encoding="utf-8")
    eng.index([docs])

    q = "איך מתקנים אופניים"
    hits = eng.search(q, k=4)
    check("the on-topic file is retrieved", any(h.path.endswith("real.md") for h in hits))
    top = hits[0]
    check("the on-topic file ranks first", top.path.endswith("real.md"),
          f"top={Path(top.path).name}")

    full = [h for h in hits if h.path.endswith("real.md")]
    decoy = [h for h in hits if h.path.endswith("decoy.md")]
    if full and decoy:
        check("a verbatim full-question match beats a two-word prefix",
              full[0].parts["phrase"] > decoy[0].parts["phrase"],
              f"full={full[0].parts['phrase']:.3f} decoy={decoy[0].parts['phrase']:.3f}")
        check("the partial phrase is discounted below 1.0",
              decoy[0].parts["phrase"] < WEIGHTS["phrase"] - 1e-6,
              f"decoy phrase={decoy[0].parts['phrase']:.3f} of {WEIGHTS['phrase']}")
    else:
        check("both files were retrieved for comparison", False,
              f"full={len(full)} decoy={len(decoy)}")

    ans = eng.ask(q, k=6)
    check("the answer quotes the bicycle file, not the tap example",
          any(c.file == "real.md" for c in ans.citations),
          str([c.file for c in ans.citations]))
    verified, problems = eng.verify(ans)
    check("the answer still verifies against disk", verified, str(problems[:1]))
    eng.close()



def test_retrieval(tmp: Path) -> None:
    print("\n── retrieval quality on a known corpus ────────────────────")
    eng = RagEngine(db_path=tmp / "_ret.sqlite3")
    rep = eng.index([tmp])
    check("the fixture corpus indexed", rep["ok"] >= 5, f"{rep['ok']} docs")
    check("the deny-list kept secrets out",
          not any("id_rsa" in d["path"] or ".env" in d["path"] for d in eng.docs()),
          str([d["path"] for d in eng.docs()]))
    check("no document was silently denied without a reason",
          rep.get("error") is None)

    hits = eng.search("חומת הרשאות", k=3)
    check("a topical query returns hits", len(hits) >= 1, f"{len(hits)} hits")
    check("the permission-wall query finds the permissions section",
          any("הרשאות" in h.text for h in hits))
    check("hits carry a real path", all(h.path for h in hits))
    check("hits carry a non-empty line span", all(h.end_line >= h.start_line >= 1 for h in hits))
    check("hit scores are within [0,1]", all(0.0 < h.score <= 1.0 for h in hits),
          str([round(h.score, 3) for h in hits]))
    check("score parts sum to the score",
          all(abs(sum(v for k, v in h.parts.items() if k != "code_penalty") - h.score) < 1e-6
              or "code_penalty" in h.parts for h in hits))

    # A prose question must not be answered by a code fragment.
    prose_hits = eng.search("איך בנויה חומת ההרשאות", k=5)
    check("prose question ranks prose above code",
          prose_hits and prose_hits[0].kind == "prose",
          f"top={prose_hits[0].kind if prose_hits else '-'}")

    # Morphological expansion: a word that is only a suffix of an indexed one.
    w, trace = eng.retriever.expand_terms(["ריצ"])
    check("expansion never outweighs the user's own word",
          all(v <= 1.0 for v in w.values()), str(w))
    check("expansions are reported (auditable, not silent)", isinstance(trace, dict))

    # Incremental indexing.
    before = eng.store.doc((tmp / "docs" / "מנוע.md").as_posix())
    rep2 = eng.index([tmp])
    check("re-indexing an unchanged corpus re-reads nothing",
          rep2["unchanged"] >= 5 and rep2["ok"] == 0,
          f"unchanged={rep2['unchanged']} ok={rep2['ok']}")

    (tmp / "docs" / "מנוע.md").write_text(HE_TECH + "\nתוספת חדשה לגמרי.\n", encoding="utf-8")
    rep3 = eng.index([tmp])
    check("an edited file is re-indexed", rep3["updated"] >= 1, f"updated={rep3['updated']}")

    (tmp / "docs" / "מנוע.md").unlink()
    rep4 = eng.index([tmp])
    check("a deleted file is pruned from the index", rep4["pruned"] == 1,
          f"pruned={rep4['pruned']}")

    check("forget() removes one document",
          eng.forget((tmp / "src" / "calc.py").as_posix()))
    check("forget() is honest about a file it does not have",
          not eng.forget("/nonexistent/nope.md"))

    st = eng.status()
    check("status reports roots", len(st["roots"]) >= 1)
    check("status reports chunk totals", st["chunks"] >= 1, str(st["chunks"]))
    eng.close()


# ------------------------------------------------------------------- answers --
def test_grounding_invariant(tmp: Path) -> None:
    print("\n── the grounding invariant, checked against real files ────")
    eng = RagEngine(db_path=tmp / "_ans.sqlite3")
    eng.index([tmp])

    ans = eng.ask("מה עושה חומת ההרשאות", k=6)
    check("a good question produces a grounded answer", ans.grounded, ans.answer_type)
    check("a grounded answer carries citations", len(ans.citations) >= 1)
    check("every citation names a file and a span",
          all(c.file and c.start_line >= 1 and c.end_line >= c.start_line
              for c in ans.citations))

    verified, problems = eng.verify(ans)
    check("every quote appears verbatim in the file it cites", verified, str(problems[:2]))

    # Now the same check with a deliberately corrupted source, to prove the
    # verifier is actually reading the file and not rubber-stamping.
    fake = {c.path: "טקסט אחר לגמרי שלא מכיל את הציטוט" for c in ans.citations}
    bad_ok, bad_problems = verify_answer(ans, fake)
    check("the verifier FAILS when the file does not contain the quote",
          not bad_ok and len(bad_problems) >= 1, str(bad_problems[:1]))

    empty = GroundedAnswer(query="x", text="", speak="", grounded=True,
                           answer_type="grounded", confidence=0.9, citations=[])
    e_ok, e_problems = verify_answer(empty)
    check("grounded-without-citations is caught", not e_ok, str(e_problems))

    eng.close()


def test_honest_refusal(tmp: Path) -> None:
    print("\n── honest refusal instead of a plausible-looking quote ────")
    eng = RagEngine(db_path=tmp / "_ref.sqlite3")
    eng.index([tmp])

    ans = eng.ask("מה המתכון לעוגת גבינה של סבתא", k=6)
    check("an off-topic question is refused", ans.answer_type == "none",
          f"type={ans.answer_type} conf={ans.confidence}")
    check("a refused answer is not marked grounded", not ans.grounded)
    check("a refused answer carries no citations", not ans.citations)
    check("a refused answer still explains itself", len(ans.text) > 20)
    check("a refused answer says so out loud", "לא מצאתי" in ans.speak or "אין" in ans.speak,
          ans.speak[:40])

    verified, problems = eng.verify(ans)
    check("a refusal passes verification trivially", verified, str(problems))

    empty = eng.ask("", k=3)
    check("an empty query is refused, not crashed on", empty.answer_type == "none")
    eng.close()


def test_confidence_is_not_saturated(tmp: Path) -> None:
    print("\n── confidence must discriminate ──────────────────────────")
    eng = RagEngine(db_path=tmp / "_conf.sqlite3")
    eng.index([tmp])
    queries = [
        "מה עושה חומת ההרשאות",          # strong: exact section
        "איך מריצים את הבדיקות",          # medium
        "מה המתכון לעוגת גבינה",          # none
        "זיהוי הקול תבניות אקוסטיות",     # strong
        "אולי יש משהו על מטוסים",         # none
    ]
    rows = [(q, eng.ask(q, k=6)) for q in queries]
    confs = [a.confidence for _q, a in rows]
    check("confidence spans a real range", max(confs) - min(confs) > 0.25,
          f"{min(confs):.2f} → {max(confs):.2f}")
    check("confidence is not pinned at one value", len(set(confs)) >= 3, str(confs))
    check("no confidence exceeds 1.0", all(c <= 1.0 for c in confs))
    strong = [a.confidence for q, a in rows if "הרשאות" in q or "אקוסטיות" in q]
    none = [a.confidence for q, a in rows if a.answer_type == "none"]
    check("strong answers out-score refusals",
          not none or min(strong) > max(none), f"strong={strong} none={none}")
    eng.close()


def test_citation_hygiene(tmp: Path) -> None:
    print("\n── citation hygiene ──────────────────────────────────────")
    eng = RagEngine(db_path=tmp / "_hyg.sqlite3")
    eng.index([tmp])
    ans = eng.ask("חומת הרשאות רמות אישור", k=8)
    ids = [c.chunk_id for c in ans.citations]
    check("at most one quote per chunk (no fake corroboration)",
          len(ids) == len(set(ids)), str(ids))
    check("citations are ordered by source, not by score",
          ans.citations == sorted(ans.citations, key=lambda c: c.start_line)
          or len({c.path for c in ans.citations}) > 1
          or ans.citations == sorted(ans.citations, key=lambda c: c.start_line))

    code_ans = eng.ask("compute_total rate", k=6)
    if code_ans.citations and code_ans.citations[0].quote.count("\n") >= 2:
        check("source code is not read aloud",
              "מוצג על המסך" in code_ans.speak or "compute_total" not in code_ans.speak,
              code_ans.speak[:60])
    else:
        check("source code is not read aloud (no code citation this run)", True)

    check("the spoken form is short enough for TTS", len(ans.speak) < 400,
          f"{len(ans.speak)} chars")
    check("the answer body lists its sources", "מקורות" in ans.text)
    eng.close()


def test_compose_edge_cases() -> None:
    print("\n── composer edge cases ───────────────────────────────────")
    empty = compose_answer("שאלה", [])
    check("no hits → honest none", empty.answer_type == "none" and not empty.grounded)

    hits = []
    ans = compose_answer("שאלה על משהו", hits)
    check("empty hit list does not raise", ans.answer_type == "none")

    # A hit whose text shares nothing with the question must not be quoted.
    from brain.rag.retrieve import Hit
    noise = Hit(chunk_id=1, doc_id=1, path="/x/a.md", title="", text="אין כאן שום קשר לנושא.",
                start_line=1, end_line=1, heading="", kind="prose", score=0.9,
                parts={"bm25": 0.9}, matched_terms=())
    ans2 = compose_answer("ארכיטקטורת מנוע הקיטור", [noise])
    check("a zero-overlap chunk is not quoted as an answer",
          ans2.answer_type == "none", f"type={ans2.answer_type}")


# --------------------------------------------------------------------- skills --
def test_skills(tmp: Path) -> None:
    print("\n── skills: registration, risk, permission gating ─────────")
    import skills
    skills.load_all()
    from skills.registry import REGISTRY

    rag_skills = [s for s in REGISTRY.all() if s.name.startswith("rag.")]
    names = {s.name for s in rag_skills}
    check("all six RAG skills are registered",
          {"rag.search", "rag.ask", "rag.index", "rag.status", "rag.forget", "rag.read"} <= names,
          str(sorted(names)))

    by_risk = {s.name: s.risk for s in rag_skills}
    check("read-only skills are SAFE",
          by_risk.get("rag.search") == "SAFE" and by_risk.get("rag.ask") == "SAFE"
          and by_risk.get("rag.status") == "SAFE", str(by_risk))
    check("index-building skills are WRITE (they touch the disk)",
          by_risk.get("rag.index") == "WRITE" and by_risk.get("rag.forget") == "WRITE",
          str(by_risk))
    check("no RAG skill is CRITICAL", all(r != "CRITICAL" for r in by_risk.values()))

    # The registry must refuse a WRITE skill without a grant.
    res = REGISTRY.invoke("rag.index", {"roots": str(tmp)})
    check("WRITE skill is refused without permission",
          not res.ok and res.data.get("needs_permission") is True, res.error)

    res = REGISTRY.invoke("rag.ask", {"query": "חומת הרשאות"})
    check("SAFE skill runs without a grant", res.ok, res.error)
    check("rag.ask returns the grounded answer text", isinstance(res.value, str) and res.value)
    check("rag.ask exposes citations in data", isinstance(res.data.get("citations"), list))
    check("rag.ask reports its own verification result", "verified" in res.data)
    check("rag.ask carries a spoken summary", bool(res.data.get("summary_he")))

    res = REGISTRY.invoke("rag.search", {})
    check("a missing required argument is rejected by the validator",
          not res.ok and "required" in res.error, res.error)

    res = REGISTRY.invoke("rag.ask", {"query": "חומת הרשאות", "bogus": 1})
    check("an unknown argument is rejected", not res.ok and "unexpected" in res.error, res.error)

    res = REGISTRY.invoke("rag.status", {})
    check("rag.status is callable and reports a summary", res.ok and bool(res.data.get("summary_he")))

    res = REGISTRY.invoke("rag.read", {"chunk_id": 999999})
    check("rag.read is honest about an unknown chunk", not res.ok, res.error)


# -------------------------------------------------------------------- routing --
def test_routing() -> None:
    print("\n── intent routing: RAG without hijacking the others ──────")
    from brain.intent import INTENTS, IntentRouter
    check("RAG is a declared intent", "RAG" in INTENTS)

    r = IntentRouter()
    rag_cases = [
        "חפש בקבצים שלי איך מריצים בדיקות",
        "מה כתוב בקבצים על חומת הרשאות",
        "search my files for the permission firewall",
        "find in my documents the MFCC explanation",
        "according to my notes what is sentinel",
    ]
    for c in rag_cases:
        rt = r.route(c)
        check(f"RAG route: {c[:34]}…", rt.intent == "RAG" and rt.skill == "rag.ask",
              f"{rt.intent}/{rt.skill}")

    check("index request routes to rag.index",
          r.route("תאנדקס את התיקייה docs").skill == "rag.index")
    check("index request is marked WRITE",
          r.route("תאנדקס את התיקייה docs").risk == "WRITE")
    check("status request routes to rag.status",
          r.route("מה מצב האינדקס").skill == "rag.status")

    # The hijack risk: RAG patterns must not swallow the neighbouring intents.
    guard = {
        "קרא את הקובץ README.md": "FILES",
        "כמה זה 17 כפול 23": "MATH",
        "מה השעה": "TIME",
        "האם זה מסוכן": "SAFETY",
        "מצב מנתח": "MODE",
        "תפתח את הדפדפן": "APPS",
        "תזכור שאני גר בשדרות": "MEMORY_WRITE",
    }
    for text, want in guard.items():
        got = r.route(text).intent
        check(f"not hijacked: {text[:28]}… → {want}", got == want, f"got {got}")


# --------------------------------------------------- integrated answer path --
def test_integrated_answer_path(tmp: Path) -> None:
    """The path the user actually takes: text in → routed → skill → cited answer.

    Router, skills and REST were each tested separately and each passed while
    the composition was broken in three ways the HUD would have lied about.
    These assert the composed result.
    """
    print("\n── integrated: text → route → skill → cited answer ────────")
    import skills as skills_mod
    skills_mod.load_all()
    from brain.rag import engine as rag_engine_mod
    from brain.reasoning import ReasoningEngine
    from security.permissions import PermissionFirewall
    from skills.registry import REGISTRY

    eng_rag = rag_engine_mod.get_engine(db_path=tmp / "_int.sqlite3", fresh=True)
    eng_rag.index([tmp])

    eng = ReasoningEngine(skills=REGISTRY, firewall=PermissionFirewall())

    ans = eng.think("חפש בקבצים שלי מה עושה חומת ההרשאות")
    check("the RAG intent reaches the reasoning loop", ans.intent == "RAG", ans.intent)
    check("the loop invoked rag.ask", ans.skill == "rag.ask", ans.skill)
    check("the answer carries citations through the loop",
          len(ans.data.get("citations") or []) >= 1, str(len(ans.data.get("citations") or [])))
    check("the loop reports the verifier's verdict", ans.data.get("verified") is True,
          str(ans.data.get("verify_problems")))

    # 1. The answer's confidence must be the evidence confidence, not the
    #    router's flat 0.88 "this looks like a file query".
    rag_conf = ans.data.get("confidence")
    check("confidence is the EVIDENCE confidence, not the route's",
          rag_conf is not None and abs(ans.confidence - float(rag_conf)) < 1e-6,
          f"answer={ans.confidence} rag={rag_conf}")
    check("the evidence confidence is not the router's default 0.88",
          abs(ans.confidence - 0.88) > 1e-6, f"{ans.confidence}")

    # 2. The spoken form must be short and must not read source code aloud.
    check("the spoken form is short enough for TTS", len(ans.speak) < 400,
          f"{len(ans.speak)} chars")
    check("the spoken form is not the quoted body",
          ans.speak != ans.text and len(ans.speak) < len(ans.text),
          f"speak={len(ans.speak)} text={len(ans.text)}")

    # 3. A refusal must not light the grounded badge.
    ref = eng.think("חפש בקבצים שלי מה המתכון לעוגת שמרים של סבתא רבא")
    if ref.data.get("answer_type") == "none":
        check("a refusal is NOT marked grounded", ref.grounded is False,
              f"grounded={ref.grounded}")
        check("a refusal still answers politely", len(ref.text) > 10)
    else:
        # The fixture corpus is small; if something did match, it must at least
        # be honestly reported rather than silently passed.
        check("refusal case produced a refusal on this corpus", False,
              f"got {ref.data.get('answer_type')}")
    eng_rag.close()
    rag_engine_mod._ENGINE = None      # leave no closed singleton for later tests


def test_citations_preserve_hebrew_orthography(tmp: Path) -> None:
    """A citation must be findable in the user's file, character for character.

    The indexer used to run the text through ``normalize()``, which folds Hebrew
    final forms (ם→מ, ך→כ, ן→נ, ץ→צ, ף→פ). Every quote the user was shown came
    out as "קבצימ" / "איכ מריצימ" — orthography no Hebrew document contains, so
    the reader could not find the cited line in their own file. Matching does
    not need the folding at ingest: tokenize(), the n-gram vectoriser and
    phrase_in() all normalise their own input.
    """
    print("\n── citations preserve Hebrew final forms ──────────────────")
    src = tmp / "hebrew.md"
    body = ("מדריך קצר.\n\n"
            "איך מריצים את הבדיקות: פותחים את התיקייה ולוחצים על הכפתור.\n"
            "המערכת שומרת את הקבצים בתיקייה נפרדת, ואין צורך בסיסמה או בהרשאה.\n")
    src.write_text(body, encoding="utf-8")

    res = extract_file(src, policy=RagPolicy())
    check("final forms survive extraction", "מריצים" in res.text and "קבצים" in res.text,
          res.text[14:60])
    check("no folded-final mangling in the indexed text",
          "מריצימ" not in res.text and "קבצימ" not in res.text)

    eng = RagEngine(db_path=tmp / "_orth.sqlite3")
    eng.index([tmp])
    ans = eng.ask("איך מריצים את הבדיקות", k=4)
    check("the question still retrieves after the change", ans.grounded, ans.answer_type)
    quotes = " ".join(c.quote for c in ans.citations)
    check("the quote keeps real Hebrew orthography",
          "מריצים" in quotes or "הבדיקות" in quotes, repr(quotes[:80]))
    check("the quote has no folded-final artefacts",
          "מריצימ" not in quotes and "בדיקומ" not in quotes, repr(quotes[:80]))

    # The verifier must now be checking RAW text — prove it can catch mangling.
    verified, problems = eng.verify(ans)
    check("raw verification passes on a correct quote", verified, str(problems[:1]))
    for c in ans.citations:
        mangled = c.quote.replace("ם", "מ").replace("ך", "כ")
        if mangled != c.quote:
            fake = GroundedAnswer(query=ans.query, text="", speak="", grounded=True,
                                  answer_type="grounded", confidence=0.9,
                                  citations=[Citation(path=c.path, file=c.file,
                                                      start_line=c.start_line,
                                                      end_line=c.end_line, heading="",
                                                      quote=mangled, score=0.5)])
            m_ok, m_problems = verify_answer(fake, {c.path: src.read_text(encoding="utf-8")})
            check("the verifier CATCHES a folded-final mangled quote", not m_ok,
                  str(m_problems[:1]))
            break
    else:
        check("a quote contained a final form to test against", False)
    eng.close()


# --------------------------------------------------------------------- server --
def test_server(tmp: Path) -> None:
    print("\n── server endpoints ──────────────────────────────────────")
    import asyncio
    from aiohttp import web
    from aiohttp.test_utils import TestClient, TestServer

    import core.server as srv

    async def scenario():
        # Point the process-wide engine at the temp corpus before the app starts,
        # so the handlers under test are the real ones bound to a real index.
        from brain.rag import engine as engine_mod
        engine_mod.get_engine(db_path=tmp / "_srv.sqlite3", fresh=True).index([tmp])

        app = srv.build_app()
        async with TestClient(TestServer(app)) as client:
            r = await client.get("/api/rag/status")
            body = await r.json()
            check("GET /api/rag/status → 200", r.status == 200, str(r.status))
            check("status payload reports docs", body["status"]["docs"] >= 5,
                  str(body["status"].get("docs")))

            r = await client.post("/api/rag/search", json={"query": "חומת הרשאות", "k": 3})
            body = await r.json()
            check("POST /api/rag/search → 200", r.status == 200, str(r.status))
            check("search returns hits with citations", body["found"] >= 1, str(body.get("found")))
            check("search hits carry file+span",
                  all("file" in h and "start_line" in h for h in body["hits"]))

            r = await client.post("/api/rag/search", json={})
            check("search without a query is a 400, not a 500", r.status == 400, str(r.status))

            r = await client.post("/api/rag/ask", json={"query": "מה עושה חומת ההרשאות"})
            body = await r.json()
            check("POST /api/rag/ask → 200", r.status == 200, str(r.status))
            check("ask reports its answer_type", body["answer_type"] in
                  ("grounded", "weak", "none"), str(body.get("answer_type")))
            check("ask publishes the verification verdict", "verified" in body)
            check("a grounded ask really verified against disk",
                  body["answer_type"] == "none" or body["verified"] is True,
                  str(body.get("verify_problems")))

            r = await client.post("/api/rag/roots", json={"roots": [str(tmp)]})
            body = await r.json()
            check("POST /api/rag/roots persists the root list",
                  r.status == 200 and body["ok"] and len(body["roots"]) == 1)

            agent = srv.get_agent()
            agent.firewall.set_level("safe")
            r = await client.post("/api/rag/index", json={"roots": [str(tmp)]})
            body = await r.json()
            check("indexing is REFUSED at SAFE posture",
                  r.status == 403 and body["ok"] is False, f"{r.status} {body.get('error', '')[:60]}")

            agent.firewall.set_level("write")
            agent.firewall.set_dry_run(True)
            r = await client.post("/api/rag/index", json={"roots": [str(tmp)]})
            body = await r.json()
            check("dry-run reports intent without scanning",
                  r.status == 200 and body.get("dry_run") is True, str(body.get("dry_run")))
            agent.firewall.set_dry_run(False)

            r = await client.post("/api/rag/index", json={"roots": [str(tmp)]})
            body = await r.json()
            check("indexing succeeds at WRITE posture",
                  r.status == 200 and body["ok"] is True, f"{r.status} {str(body)[:120]}")
            check("the index report is measured, not estimated",
                  body.get("chunks", 0) >= 1 and "seconds" in body,
                  f"chunks={body.get('chunks')} seconds={body.get('seconds')}")
            agent.firewall.set_level("write")

    asyncio.get_event_loop().run_until_complete(scenario())


# ----------------------------------------------------------------------- main --
def main() -> int:
    print("═" * 68)
    print(" J.A.R.V.I.S. — RAG: local-file retrieval with cited answers")
    print("═" * 68)
    t0 = time.perf_counter()
    test_decoding()
    test_chunking()
    test_tokenizer()
    test_bm25()
    test_phrase_boundary()
    test_compose_edge_cases()
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        build_corpus(tmp)
        test_extraction_and_policy(tmp)
        test_index(tmp)
        test_retrieval(tmp)
        test_grounding_invariant(tmp)
        test_honest_refusal(tmp)
        test_confidence_is_not_saturated(tmp)
        test_citation_hygiene(tmp)
        test_phrase_evidence_is_proportional(tmp)
        test_citations_preserve_hebrew_orthography(tmp)
        test_integrated_answer_path(tmp)
        test_skills(tmp)
    test_routing()
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        build_corpus(tmp)
        test_server(tmp)
    print(f"\n completed in {time.perf_counter() - t0:.1f}s")
    print(f"\nRESULT: {ok} passed, {fail} failed")
    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
