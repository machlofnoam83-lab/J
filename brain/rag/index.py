"""The RAG index: SQLite, inverted postings, incremental sync.

Why a hand-rolled inverted index instead of "just scan every chunk"
-------------------------------------------------------------------
Scanning is honest and dependency-free, and it is also the reason most hobby RAG
implementations fall over the first time the user points them at a real folder:
cost is O(chunks × query_terms) on *every* keystroke. An inverted index makes
candidate generation O(matching postings), which is what lets the HUD answer
while you are still typing.

SQLite is the right store here because it is in the standard library, it is one
file, it survives a crash mid-write (WAL + transactions), and the user can open
it with any SQLite browser to audit exactly what JARVIS indexed — which matters
when the indexer has read their documents.

Nothing is stored that we cannot rebuild: ``tools/build_rag_index.py`` recreates
the whole index from the files on disk. That is why it lives in ``data/`` and is
git-ignored.
"""

from __future__ import annotations

import json
import math
import re
import sqlite3
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from brain.tokenizer import normalize  # noqa: E402

SCHEMA_VERSION = 3

# BM25's standard constants. Not tuned on a held-out set — they are the values
# the original paper found robust across collections, and we say so.
BM25_K1 = 1.5
BM25_B = 0.75

_TOKEN_RE = re.compile(r"[A-Za-z\u0590-\u05FF][A-Za-z\u0590-\u05FF'\-]{1,}")
_HE_PREFIXES = ("ובה", "וה", "וב", "כש", "מה", "שה", "ה", "ו", "ב", "כ", "ל", "מ", "ש")
_HE_SUFFIXES = ("ימ", "ות", "ה")

# Function words carry no retrieval signal but dominate Hebrew text; without
# removing them the top posting list for almost any query is the same list.
_STOPWORDS_RAW = """
אני אתה את היא הוא אנחנו אתם הן הם זה זו אלה אלו ששל של שלי שלך שלו שלה שלנו
על עם בלי בין אם או אבל כי אז גם רק עוד כבר מאוד יותר הכי איך מה מי מתי איפה לא
כן אין יש להיות היה הייתה היו להיות אני את לי לו לה לנו לכם להם כל כמה כזה ככה
היום מחר אתמול עכשיו תודה בבקשה שלום אדוני
the a an and or of to in for on with is are was were be been this that it as at by
from you your i we they he she his her its not no yes please thanks
what which who whom whose when where why how whose
"""
# NOTE: the assistant's own names (אדיאל / גרוויס / jarvis) are deliberately NOT
# stopwords. They were at first, which meant "מה זה אדיאל" tokenised to nothing
# and returned zero candidates — the one question the persona should answer best.


def stem(tok: str) -> str:
    """Very light Hebrew/English stemming (mirrors ``brain/knowledge._stem``).

    Duplicated rather than imported because that helper is private to the
    knowledge module; the two must agree, and ``tests/test_rag.py`` asserts they
    do, so the duplication cannot silently drift.
    """
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


def _build_stopwords(raw: str) -> frozenset:
    """Store every stopword in the forms it can actually appear as.

    ``tokenize`` compares *normalised* tokens against this set, and
    ``normalize`` rewrites final forms (ך→כ, ם→מ, …). A raw entry like "איך"
    therefore never matched the token "איכ", so the question word survived into
    every query as if it carried topic. ``brain/knowledge.py`` documents this
    exact trap; this is the same fix.
    """
    out: set = set()
    for w in raw.split():
        w = w.strip()
        if not w:
            continue
        out.add(w)
        n = normalize(w).lower()
        out.add(n)
        out.add(stem(n))
    return frozenset(x for x in out if x)


_STOPWORDS = _build_stopwords(_STOPWORDS_RAW)

def tokenize(text: str, *, drop_stopwords: bool = True) -> List[str]:
    """Normalised, stemmed content terms — the unit of the inverted index."""
    t = f" {normalize(str(text)).lower()} "
    out: List[str] = []
    for w in _TOKEN_RE.findall(t):
        if drop_stopwords and (w in _STOPWORDS or stem(w) in _STOPWORDS):
            continue
        s = stem(w)
        if len(s) >= 2:
            out.append(s)
    return out


def term_counts(text: str) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for t in tokenize(text):
        counts[t] = counts.get(t, 0) + 1
    return counts


SCHEMA = """
CREATE TABLE IF NOT EXISTS docs (
    doc_id     INTEGER PRIMARY KEY,
    path       TEXT UNIQUE NOT NULL,
    root       TEXT DEFAULT '',
    kind       TEXT DEFAULT 'prose',
    bytes      INTEGER DEFAULT 0,
    mtime      REAL DEFAULT 0,
    sha1       TEXT DEFAULT '',
    chars      INTEGER DEFAULT 0,
    lines      INTEGER DEFAULT 0,
    chunks     INTEGER DEFAULT 0,
    encoding   TEXT DEFAULT '',
    title      TEXT DEFAULT '',
    indexed_at REAL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id   INTEGER PRIMARY KEY,
    doc_id     INTEGER NOT NULL,
    ordinal    INTEGER DEFAULT 0,
    start_line INTEGER DEFAULT 1,
    end_line   INTEGER DEFAULT 1,
    heading    TEXT DEFAULT '',
    kind       TEXT DEFAULT 'prose',
    chars      INTEGER DEFAULT 0,
    nterms     INTEGER DEFAULT 0,
    text       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
CREATE TABLE IF NOT EXISTS postings (
    term     TEXT NOT NULL,
    chunk_id INTEGER NOT NULL,
    tf       INTEGER NOT NULL,
    PRIMARY KEY (term, chunk_id)
);
CREATE TABLE IF NOT EXISTS skipped (
    path   TEXT PRIMARY KEY,
    reason TEXT DEFAULT '',
    at     REAL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);
"""


class RagIndex:
    """A single SQLite file holding every indexed document and its postings."""

    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.executescript(SCHEMA)
        self._conn.commit()
        self._ensure_version()

    # ------------------------------------------------------------- lifecycle --
    def _ensure_version(self) -> None:
        row = self._conn.execute("SELECT value FROM meta WHERE key='schema'").fetchone()
        have = int(row["value"]) if row else 0
        if have != SCHEMA_VERSION:
            # A stale schema is worse than no schema: silently mixing v2 rows
            # with v3 queries produces results that look fine and are not.
            for t in ("postings", "chunks", "docs", "skipped"):
                self._conn.execute(f"DROP TABLE IF EXISTS {t}")
            self._conn.executescript(SCHEMA)
            self._set_meta("schema", str(SCHEMA_VERSION))
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.commit()
                self._conn.close()
            except sqlite3.Error:
                pass

    def _set_meta(self, key: str, value: str) -> None:
        self._conn.execute(
            "INSERT INTO meta(key,value) VALUES(?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, value))

    def get_meta(self, key: str, default: str = "") -> str:
        row = self._conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return row["value"] if row else default

    # ---------------------------------------------------------------- writes --
    def add_document(self, *, path: str, root: str = "", kind: str = "prose",
                     bytes_: int = 0, mtime: float = 0.0, sha1: str = "",
                     encoding: str = "", title: str = "",
                     chunks: Sequence[Dict[str, Any]]) -> int:
        """Insert or replace one document and all of its chunks + postings."""
        with self._lock:
            self.remove_document(path, commit=False)
            cur = self._conn.execute(
                "INSERT INTO docs(path,root,kind,bytes,mtime,sha1,chars,lines,chunks,"
                "encoding,title,indexed_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                (path, root, kind, bytes_, mtime, sha1,
                 sum(c["chars"] for c in chunks),
                 max((c["end_line"] for c in chunks), default=0),
                 len(chunks), encoding, title, time.time()))
            doc_id = int(cur.lastrowid)
            for c in chunks:
                cc = self._conn.execute(
                    "INSERT INTO chunks(doc_id,ordinal,start_line,end_line,heading,kind,"
                    "chars,nterms,text) VALUES(?,?,?,?,?,?,?,?,?)",
                    (doc_id, c.get("ordinal", 0), c["start_line"], c["end_line"],
                     c.get("heading", ""), c.get("kind", kind), c["chars"],
                     len(c.get("terms", ())), c["text"]))
                cid = int(cc.lastrowid)
                tf = c.get("terms") or {}
                if isinstance(tf, dict):
                    self._conn.executemany(
                        "INSERT OR REPLACE INTO postings(term,chunk_id,tf) VALUES(?,?,?)",
                        [(t, cid, n) for t, n in tf.items()])
            self._conn.commit()
            return doc_id

    def mark_skipped(self, path: str, reason: str) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO skipped(path,reason,at) VALUES(?,?,?) "
                "ON CONFLICT(path) DO UPDATE SET reason=excluded.reason, at=excluded.at",
                (path, reason, time.time()))
            self._conn.commit()

    def remove_document(self, path: str, commit: bool = True) -> bool:
        with self._lock:
            row = self._conn.execute("SELECT doc_id FROM docs WHERE path=?", (path,)).fetchone()
            if not row:
                return False
            doc_id = int(row["doc_id"])
            self._conn.execute(
                "DELETE FROM postings WHERE chunk_id IN (SELECT chunk_id FROM chunks WHERE doc_id=?)",
                (doc_id,))
            self._conn.execute("DELETE FROM chunks WHERE doc_id=?", (doc_id,))
            self._conn.execute("DELETE FROM docs WHERE doc_id=?", (doc_id,))
            self._conn.execute("DELETE FROM skipped WHERE path=?", (path,))
            if commit:
                self._conn.commit()
            return True

    def prune(self, keep: Iterable[str]) -> int:
        """Drop documents whose file is no longer in the scanned set."""
        keep = set(keep)
        with self._lock:
            rows = self._conn.execute("SELECT path FROM docs").fetchall()
            gone = [r["path"] for r in rows if r["path"] not in keep]
            for p in gone:
                self.remove_document(p, commit=False)
            self._conn.commit()
            return len(gone)

    # ----------------------------------------------------------------- reads --
    def doc(self, path: str) -> Optional[Dict[str, Any]]:
        row = self._conn.execute("SELECT * FROM docs WHERE path=?", (path,)).fetchone()
        return dict(row) if row else None

    def docs(self, limit: int = 0) -> List[Dict[str, Any]]:
        q = "SELECT * FROM docs ORDER BY path"
        if limit:
            q += f" LIMIT {int(limit)}"
        return [dict(r) for r in self._conn.execute(q).fetchall()]

    def chunk_count(self) -> int:
        return int(self._conn.execute("SELECT COUNT(*) AS n FROM chunks").fetchone()["n"])

    def doc_count(self) -> int:
        return int(self._conn.execute("SELECT COUNT(*) AS n FROM docs").fetchone()["n"])

    def term_count(self) -> int:
        return int(self._conn.execute(
            "SELECT COUNT(DISTINCT term) AS n FROM postings").fetchone()["n"])

    def avg_doc_len(self) -> float:
        row = self._conn.execute(
            "SELECT AVG(nterms) AS a FROM chunks WHERE nterms>0").fetchone()
        return float(row["a"] or 0.0)

    def doc_freq(self, terms: Sequence[str]) -> Dict[str, int]:
        if not terms:
            return {}
        qmarks = ",".join("?" * len(terms))
        rows = self._conn.execute(
            f"SELECT term, COUNT(*) AS df FROM postings WHERE term IN ({qmarks}) GROUP BY term",
            tuple(terms)).fetchall()
        return {r["term"]: int(r["df"]) for r in rows}

    def postings_for(self, terms: Sequence[str]) -> Dict[int, Dict[str, int]]:
        """chunk_id -> {term: tf} for every chunk containing any query term."""
        if not terms:
            return {}
        qmarks = ",".join("?" * len(terms))
        rows = self._conn.execute(
            f"SELECT chunk_id, term, tf FROM postings WHERE term IN ({qmarks})",
            tuple(terms)).fetchall()
        out: Dict[int, Dict[str, int]] = {}
        for r in rows:
            out.setdefault(int(r["chunk_id"]), {})[r["term"]] = int(r["tf"])
        return out

    def fetch_chunks(self, chunk_ids: Sequence[int]) -> Dict[int, Dict[str, Any]]:
        if not chunk_ids:
            return {}
        qmarks = ",".join("?" * len(chunk_ids))
        rows = self._conn.execute(
            f"""SELECT c.chunk_id, c.doc_id, c.ordinal, c.start_line, c.end_line,
                       c.heading, c.kind, c.chars, c.nterms, c.text,
                       d.path AS path, d.title AS title, d.root AS root, d.kind AS doc_kind
                FROM chunks c JOIN docs d ON d.doc_id=c.doc_id
                WHERE c.chunk_id IN ({qmarks})""", tuple(chunk_ids)).fetchall()
        return {int(r["chunk_id"]): dict(r) for r in rows}

    def all_chunk_texts(self, batch: int = 500) -> Iterable[Tuple[int, str]]:
        """Stream every chunk's text — used to fit the n-gram vectoriser."""
        last = 0
        while True:
            rows = self._conn.execute(
                "SELECT chunk_id, text FROM chunks WHERE chunk_id>? ORDER BY chunk_id LIMIT ?",
                (last, batch)).fetchall()
            if not rows:
                return
            for r in rows:
                last = int(r["chunk_id"])
                yield last, r["text"]

    def skipped_stats(self) -> Dict[str, int]:
        rows = self._conn.execute(
            "SELECT reason, COUNT(*) AS n FROM skipped GROUP BY reason ORDER BY n DESC").fetchall()
        return {r["reason"]: int(r["n"]) for r in rows}

    def stats(self) -> Dict[str, Any]:
        row = self._conn.execute(
            """SELECT COUNT(*) AS docs,
                      COALESCE(SUM(chunks),0) AS chunks,
                      COALESCE(SUM(bytes),0) AS bytes,
                      COALESCE(SUM(chars),0) AS chars
               FROM docs""").fetchone()
        return {
            "db_path": str(self.db_path),
            "docs": int(row["docs"]),
            "chunks": int(row["chunks"]),
            "indexed_bytes": int(row["bytes"]),
            "indexed_chars": int(row["chars"]),
            "distinct_terms": self.term_count(),
            "avg_chunk_terms": round(self.avg_doc_len(), 2),
            "skipped": int(self._conn.execute(
                "SELECT COUNT(*) AS n FROM skipped").fetchone()["n"]),
            "schema": SCHEMA_VERSION,
        }

    def bm25_scores(self, terms: Sequence[str], postings: Dict[int, Dict[str, int]],
                    doc_freq: Dict[str, int],
                    term_weights: Optional[Dict[str, float]] = None) -> Dict[int, float]:
        """Okapi BM25 over the candidate set. Pure SQLite+math, no numpy.

        ``term_weights`` lets morphological expansions contribute at a discount
        (an expanded term is a guess, not the user's word) without changing the
        formula.
        """
        if not postings:
            return {}
        n_docs = max(1, self.chunk_count())
        avgdl = self.avg_doc_len() or 1.0
        scores: Dict[int, float] = {}
        for cid, tfs in postings.items():
            dl = sum(tfs.values()) or 1
            s = 0.0
            for term, tf in tfs.items():
                df = doc_freq.get(term, 0)
                if df <= 0:
                    continue
                idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
                contrib = idf * (tf * (BM25_K1 + 1)) / (
                    tf + BM25_K1 * (1 - BM25_B + BM25_B * dl / avgdl))
                if term_weights is not None:
                    contrib *= term_weights.get(term, 1.0)
                s += contrib
            if s > 0:
                scores[cid] = s
        return scores

    def vocabulary(self) -> List[str]:
        """Every distinct indexed term — the basis for morphological expansion."""
        return [r["term"] for r in self._conn.execute(
            "SELECT DISTINCT term FROM postings ORDER BY term").fetchall()]

    def vacuum(self) -> None:
        with self._lock:
            self._conn.execute("VACUUM")
            self._conn.commit()

    def to_json(self) -> str:
        return json.dumps(self.stats(), ensure_ascii=False, indent=2)
