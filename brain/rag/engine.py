"""RagEngine — the one object the skills, the router and the server talk to.

Lifecycle
    ``index()``   walk the configured roots, extract, chunk, upsert, prune,
                  then refit the n-gram IDF weights so scoring reflects the new
                  corpus. Incremental: a file whose size+mtime+sha1 are
                  unchanged is not re-read.
    ``search()``  retrieval only — for the HUD's "show me where" panel.
    ``ask()``     retrieval + extractive composition — a cited answer.

Two safety properties live here rather than in the callers, because a caller can
forget them:

* every path is passed through :class:`RagPolicy` before a byte is read, and
* indexing is *never* automatic. ``auto_index_on_boot`` defaults to False and
  ``roots`` defaults to empty, so a fresh install indexes nothing until the user
  names a folder.
"""

from __future__ import annotations

import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from brain.rag.answer import GroundedAnswer, compose_answer, verify_answer  # noqa: E402
from brain.rag.chunk import chunk_text  # noqa: E402
from brain.rag.extract import (  # noqa: E402
    ExtractResult, RagPolicy, extract_file, human_bytes, walk_files,
)
from brain.rag.index import RagIndex, term_counts  # noqa: E402
from brain.rag.retrieve import Hit, Retriever  # noqa: E402
from core.bus import BUS  # noqa: E402
from core.config import CONFIG  # noqa: E402


#: The command frame the router matches on, which is *not* part of the question.
#: Left in the query it becomes three more content terms ("חפש", "קבצים", "שלי")
#: competing with the two that actually matter, and BM25 dilution plus term
#: coverage turn a good hit into a refusal. Measured: "מה עושה חומת ההרשאות"
#: answered at 0.65 confidence, while "חפש בקבצים שלי מה עושה חומת ההרשאות" —
#: the phrasing the router recognises — answered nothing at all.
_QUERY_FRAMES = re.compile(
    r"(חפש(י)?( לי)? ב(תוך )?(ה)?(קבצים|מסמכים|תיקייה|הערות)( שלי| שלך)?"
    r"|מצא(י)?( לי)? ב(תוך )?(ה)?(קבצים|מסמכים|תיקייה)( שלי| שלך)?"
    r"|מה כתוב ב(תוך )?(ה)?(קבצים|מסמכים)"
    r"|(לפי|מתוך|על פי) (ה)?(קבצים|מסמכים|הערות)( שלי| שלך)?"
    r"|(ה)?(קבצים|מסמכים) (שלי|שלך) (אומרים|מראים)"
    r"|search (my|the|in my|in the) (files|documents|docs|notes)"
    r"|grep my (files|documents|notes)"
    r"|find (it |this )?in my (files|documents|notes)"
    r"|according to my (files|documents|notes)"
    r"|what (do|does) my (files|documents|notes) say)", re.I)


def strip_query_frame(query: str) -> str:
    """Remove the "search my files" wrapper, keeping the actual question.

    Falls back to the original text when stripping would leave nothing, so a
    bare "מה כתוב בקבצים" still searches on its own words instead of becoming
    an empty query.
    """
    q = (query or "").strip()
    if not q:
        return ""
    stripped = re.sub(r"\s{2,}", " ", _QUERY_FRAMES.sub(" ", q)).strip(" ,.;:!?·—-")
    return stripped if len(stripped) >= 2 else q


class RagEngine:
    """Facade over index + retriever + composer."""

    def __init__(self, *, db_path: Optional[Path | str] = None, config=None) -> None:
        self.config = config or CONFIG.rag
        self.db_path = Path(db_path or self.config.db_path)
        self.store = RagIndex(self.db_path)
        self.retriever = Retriever(self.store, dim=int(self.config.ngram_dim))
        self.policy = RagPolicy(protected=tuple(CONFIG.security.protected_paths))
        self.last_index: Dict[str, Any] = {}

    # ------------------------------------------------------------------ state --
    @property
    def roots(self) -> List[Path]:
        return [Path(r).expanduser() for r in (self.config.roots or ())]

    def set_roots(self, roots: Sequence[Path | str]) -> List[str]:
        """Persist a new root list. Returns the normalised absolute paths."""
        clean: List[str] = []
        for r in roots:
            p = Path(str(r).strip().strip('"').strip("'")).expanduser()
            if not str(p):
                continue
            try:
                p = p.resolve()
            except OSError:
                pass
            if p not in [Path(c) for c in clean]:
                clean.append(str(p))
        self.config.roots = tuple(clean)
        self.store._set_meta("roots", "\n".join(clean))
        self.store._conn.commit()
        return clean

    def _effective_roots(self, roots: Optional[Sequence[Path | str]] = None) -> List[Path]:
        if roots:
            return [Path(r).expanduser() for r in roots]
        stored = [r for r in (self.store.get_meta("roots") or "").split("\n") if r.strip()]
        if stored:
            return [Path(r) for r in stored]
        return self.roots

    @property
    def closed(self) -> bool:
        return bool(getattr(self.store, "closed", False))

    def close(self) -> None:
        self.store.close()

    # ----------------------------------------------------------------- index ---
    def index(self, roots: Optional[Sequence[Path | str]] = None, *,
              force: bool = False, progress=None) -> Dict[str, Any]:
        """Scan, extract, chunk and upsert. Returns a full, honest report."""
        t0 = time.perf_counter()
        use_roots = self._effective_roots(roots)
        if roots:
            self.set_roots([str(r) for r in use_roots])
        report: Dict[str, Any] = {
            "roots": [str(r) for r in use_roots], "ok": 0, "unchanged": 0,
            "updated": 0, "added": 0, "skipped": 0, "pruned": 0, "denied": 0,
            "chunks": 0, "chars": 0, "bytes": 0, "errors": [], "skip_reasons": {},
            "seconds": 0.0,
        }
        if not use_roots:
            report["error"] = "no roots configured — pass roots or set rag.roots"
            return report

        BUS.emit("rag.index.start", {"roots": report["roots"]}, source="rag")
        seen: List[str] = []
        for n, path in enumerate(walk_files(
                use_roots, include=tuple(self.config.include),
                exclude=tuple(self.config.exclude), policy=self.policy,
                max_bytes=int(self.config.max_file_bytes),
                max_files=int(self.config.max_files),
                allow_self_feedback=bool(self.config.allow_self_feedback))):
            posix = path.as_posix()
            seen.append(posix)
            existing = self.store.doc(posix)
            try:
                st = path.stat()
            except OSError:
                continue
            if existing and not force \
                    and abs(float(existing["mtime"]) - st.st_mtime) < 1e-6 \
                    and int(existing["bytes"]) == st.st_size:
                report["unchanged"] += 1
                report["chunks"] += int(existing["chunks"])
                continue

            res = extract_file(path, max_bytes=int(self.config.max_file_bytes),
                               policy=self.policy)
            if not res.ok:
                reason = res.skipped or "unknown"
                if reason.startswith("denied"):
                    report["denied"] += 1
                report["skipped"] += 1
                report["skip_reasons"][reason] = report["skip_reasons"].get(reason, 0) + 1
                self.store.mark_skipped(posix, reason)
                continue

            chunks = self._chunk(res)
            if not chunks:
                report["skipped"] += 1
                report["skip_reasons"]["no chunks"] = report["skip_reasons"].get("no chunks", 0) + 1
                self.store.mark_skipped(posix, "no chunks")
                continue

            was_present = existing is not None
            self.store.add_document(
                path=posix, root=str(use_roots[0]), kind=res.kind, bytes_=res.bytes,
                mtime=res.mtime, sha1=res.sha1, encoding=res.encoding,
                title=_title(path, chunks),
                chunks=[{"ordinal": c.ordinal, "start_line": c.start_line,
                         "end_line": c.end_line, "heading": c.heading,
                         "kind": c.kind, "chars": c.n_chars,
                         "terms": term_counts(c.text), "text": c.text} for c in chunks])
            report["ok"] += 1
            report["added" if not was_present else "updated"] += 1
            report["chunks"] += len(chunks)
            report["chars"] += len(res.text)
            report["bytes"] += res.bytes
            if progress and report["ok"] % 25 == 0:
                progress(report["ok"], n + 1, posix)

        report["pruned"] = self.store.prune(seen)
        # Corpus changed (or first build) → IDF weights must be refit, otherwise
        # n-gram scores keep reflecting the old collection.
        if report["ok"] or force or not self.retriever._vec_ready:
            fit = self.retriever.fit_vectorizer(force=True)
            report["vectorizer"] = fit
        report["seconds"] = round(time.perf_counter() - t0, 3)
        report["totals"] = self.store.stats()
        report["totals"]["indexed_size"] = human_bytes(report["bytes"])
        report["summary_he"] = (
            f"אינדקסתי {report['ok']} קבצים ({report['chunks']} קטעים, "
            f"{human_bytes(report['bytes'])}) ב־{report['seconds']} שניות."
            + (f" דילגתי על {report['skipped']}." if report["skipped"] else "")
            + (f" הסרתי {report['pruned']} קבצים שנעלמו." if report["pruned"] else ""))
        self.last_index = report
        BUS.emit("rag.index.done", {k: report[k] for k in
                                    ("ok", "added", "updated", "unchanged", "skipped",
                                     "pruned", "chunks", "seconds")}, source="rag")
        return report

    def _chunk(self, res: ExtractResult) -> List[Any]:
        return chunk_text(
            res.text, kind=res.kind, path=res.path.as_posix(),
            target=int(self.config.chunk_target), max_chars=int(self.config.chunk_max),
            overlap=int(self.config.chunk_overlap))

    # ---------------------------------------------------------------- search --
    def search(self, query: str, *, k: Optional[int] = None,
               path_filter: str = "", kinds: Sequence[str] = ()) -> List[Hit]:
        return self.retriever.search(
            strip_query_frame(query), k=int(k or self.config.top_k),
            candidates=int(self.config.rerank_candidates),
            path_filter=path_filter, kinds=kinds)

    def ask(self, query: str, *, k: Optional[int] = None,
            path_filter: str = "", kinds: Sequence[str] = ()) -> GroundedAnswer:
        t0 = time.perf_counter()
        hits = self.search(query, k=k, path_filter=path_filter, kinds=kinds)
        ans = compose_answer(query, hits, min_hit_score=float(self.config.min_hit_score))
        ans.stats["ms"] = round((time.perf_counter() - t0) * 1000, 1)
        BUS.emit("rag.ask", {"query": query, "type": ans.answer_type,
                             "grounded": ans.grounded, "confidence": ans.confidence,
                             "citations": len(ans.citations)}, source="rag")
        return ans

    def verify(self, answer: GroundedAnswer) -> Tuple[bool, List[str]]:
        """Check the grounding invariant against the real files on disk."""
        sources: Dict[str, str] = {}
        for c in answer.citations:
            if c.path in sources:
                continue
            try:
                sources[c.path] = Path(c.path).read_text(encoding="utf-8", errors="replace")
            except OSError:
                sources[c.path] = ""
        return verify_answer(answer, sources)

    # ----------------------------------------------------------------- misc ---
    def forget(self, path: str) -> bool:
        p = Path(path).expanduser()
        return self.store.remove_document(p.as_posix()) or self.store.remove_document(str(p))

    def status(self) -> Dict[str, Any]:
        st = self.store.stats()
        st.update({
            "enabled": bool(self.config.enabled),
            "roots": [str(r) for r in self._effective_roots()],
            "top_k": int(self.config.top_k),
            "chunk_target": int(self.config.chunk_target),
            "last_index": {k: self.last_index.get(k) for k in
                           ("ok", "added", "updated", "unchanged", "skipped",
                            "pruned", "chunks", "seconds")} if self.last_index else {},
            "skip_reasons": self.store.skipped_stats(),
            "vectorizer_ready": bool(self.retriever._vec_ready),
        })
        return st

    def stats(self) -> Dict[str, Any]:
        return self.store.stats()

    def docs(self, limit: int = 200) -> List[Dict[str, Any]]:
        return self.store.docs(limit)


def _title(path: Path, chunks: Sequence[Any]) -> str:
    """First markdown heading, else the file name — used as a retrieval signal."""
    for c in chunks:
        h = (getattr(c, "heading", "") or "").strip().lstrip("#").strip()
        if h:
            return h[:120]
    return path.stem.replace("_", " ").replace("-", " ")[:120]


# ------------------------------------------------------------------ singleton --
_ENGINE: Optional[RagEngine] = None


def get_engine(*, db_path: Optional[Path | str] = None, fresh: bool = False) -> RagEngine:
    """Process-wide engine. ``fresh=True`` is for tests that need a clean index."""
    global _ENGINE
    stale = _ENGINE is not None and _ENGINE.closed
    if _ENGINE is None or fresh or stale \
            or (db_path and str(Path(db_path)) != str(_ENGINE.db_path)):
        if _ENGINE is not None and not stale:
            try:
                _ENGINE.close()
            except Exception:
                pass
        # A closed engine used to be handed straight back, because the only test
        # was ``is None``. Anything sharing the process-wide handle then failed
        # with "Cannot operate on a closed database" long after whoever closed
        # it had moved on.
        _ENGINE = RagEngine(db_path=db_path)
    return _ENGINE


class _LazyRag:
    """Import-time-free handle so ``import brain.rag`` never opens a database."""

    def __getattr__(self, item: str) -> Any:
        return getattr(get_engine(), item)


RAG = _LazyRag()
