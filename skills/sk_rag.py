"""RAG skills — index, search and answer over the user's own local files.

Risk levels follow the same logic as the filesystem skills:

* ``rag.search`` / ``rag.ask`` / ``rag.status`` are **SAFE** — they read an index
  JARVIS already built, and never touch the disk.
* ``rag.index`` and ``rag.forget`` are **WRITE** — they open and read real files
  and mutate the index. They go through the Permission Firewall, and every path
  they touch has already passed :class:`RagPolicy`, so a private key can be
  requested by name and still be refused.

Note what is deliberately *absent*: there is no skill that lets the model choose
which files to read at answer time. Retrieval only ever returns chunks that were
indexed during an explicit, user-initiated ``rag.index``. That keeps the blast
radius of a bad routing decision at "quotes something already on the index",
never "reads something new off the disk mid-conversation".
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from brain.rag.engine import get_engine  # noqa: E402
from skills.registry import REGISTRY, SkillResult  # noqa: E402


def _split_roots(raw: Any) -> List[str]:
    """Accept a list, a comma/newline separated string, or nothing."""
    if raw is None:
        return []
    if isinstance(raw, (list, tuple)):
        return [str(x).strip() for x in raw if str(x).strip()]
    return [p.strip().strip('"').strip("'") for p in str(raw).replace("\n", ",").split(",")
            if p.strip()]


@REGISTRY.register(
    "rag.search", risk="SAFE", agent="argus",
    description_he="מחפש בקבצים המקומיים המאונדקסים ומחזיר קטעים עם מקור מדויק",
    description_en="Search the indexed local files and return chunks with exact sources",
    required=("query",),
    triggers_he=("חפש בקבצים", "מצא בקבצים שלי", "חפש במסמכים", "יש בקבצים"),
    triggers_en=("search my files", "grep my documents", "find in my files"),
)
def rag_search(query: str = "", k: int = 8, path_filter: str = "") -> SkillResult:
    eng = get_engine()
    hits = eng.search(query, k=k, path_filter=path_filter)
    if not hits:
        return SkillResult(ok=True, value="לא מצאתי קטעים רלוונטיים בקבצים המאונדקסים.",
                           data={"query": query, "found": 0, "hits": [],
                                 "summary_he": "לא מצאתי קטעים רלוונטיים בקבצים המאונדקסים.",
                                 "status": eng.status()})
    listing = "\n".join(
        f"[{i}] {h.file} · {h.span} (ציון {h.score:.2f}) — {h.text[:120].replace(chr(10), ' ')}"
        for i, h in enumerate(hits, start=1))
    return SkillResult(ok=True, value=f"מצאתי {len(hits)} קטעים:\n{listing}", data={
        "query": query, "found": len(hits), "hits": [h.to_dict() for h in hits],
        "expansions": eng.retriever.last_expansions,
        "summary_he": (f"מצאתי {len(hits)} קטעים. החזק ביותר: "
                       f"{hits[0].file} {hits[0].span}.")})


@REGISTRY.register(
    "rag.ask", risk="SAFE", agent="argus",
    description_he="עונה על שאלה מתוך הקבצים המקומיים, עם ציטוט מקור ושורות",
    description_en="Answer a question from the local files with cited sources",
    required=("query",),
    triggers_he=("מה כתוב בקבצים", "תמצא לי תשובה בקבצים", "לפי הקבצים שלי"),
    triggers_en=("according to my files", "answer from my documents"),
)
def rag_ask(query: str = "", k: int = 8, path_filter: str = "") -> SkillResult:
    eng = get_engine()
    ans = eng.ask(query, k=k, path_filter=path_filter)
    ok, problems = eng.verify(ans)
    return SkillResult(
        ok=True, value=ans.text,
        data={
            "query": query, "answer_type": ans.answer_type, "grounded": ans.grounded,
            "confidence": ans.confidence, "speak": ans.speak,
            "citations": [c.to_dict() for c in ans.citations],
            "sources": ans.to_dict()["sources"],
            "stats": ans.stats,
            "expansions": eng.retriever.last_expansions,
            # The invariant is checked against the real files, not asserted.
            "verified": ok, "verify_problems": problems,
            "summary_he": ans.speak,
        })


@REGISTRY.register(
    "rag.index", risk="WRITE", agent="hermes",
    description_he="סורק תיקייה מקומית ובונה ממנה אינדקס חיפוש (קריאה בלבד לקבצים)",
    description_en="Scan a local folder and build the search index (files are only read)",
    triggers_he=("תאנדקס את הקבצים", "בנה אינדקס", "סרוק את התיקייה"),
    triggers_en=("index my files", "rebuild the index"),
)
def rag_index(roots: Any = "", force: bool = False) -> SkillResult:
    eng = get_engine()
    root_list = _split_roots(roots)
    if not root_list and not eng.roots and not eng._effective_roots():
        return SkillResult(ok=False, error="לא הוגדרה תיקייה לאינדוקס — יש להעביר roots")
    missing = [r for r in root_list if not Path(r).expanduser().exists()]
    if missing:
        return SkillResult(ok=False, error=f"נתיבים לא קיימים: {', '.join(missing)}")
    report = eng.index(root_list or None, force=bool(force))
    if report.get("error"):
        return SkillResult(ok=False, error=report["error"], data={"report": report})
    return SkillResult(ok=True, value=report.get("summary_he") or "האינדוקס עודכן.", data={
        "report": {k: report[k] for k in ("ok", "added", "updated", "unchanged",
                                          "skipped", "pruned", "denied", "chunks",
                                          "seconds") if k in report},
        "totals": report.get("totals", {}),
        "skip_reasons": report.get("skip_reasons", {}),
        "summary_he": (f"אינדקסתי {report.get('ok', 0)} קבצים "
                       f"({report.get('chunks', 0)} קטעים) ב־{report.get('seconds', 0)} שניות."),
    })


@REGISTRY.register(
    "rag.status", risk="SAFE", agent="argus",
    description_he="מציג את מצב אינדקס הקבצים: כמה קבצים, כמה קטעים, אילו תיקיות",
    description_en="Report the state of the local-file index",
    triggers_he=("מה מצב האינדקס", "כמה קבצים אינדקסת"),
    triggers_en=("index status", "how many files indexed"),
)
def rag_status() -> SkillResult:
    eng = get_engine()
    st = eng.status()
    docs = st.get("docs", 0)
    if not docs:
        summary = "עדיין לא אינדקסתי אף קובץ — צריך להגדיר תיקייה ולהריץ אינדוקס."
    else:
        summary = (f"האינדקס כולל {docs} קבצים ו־{st.get('chunks', 0)} קטעים "
                   f"מתוך {len(st.get('roots', []))} תיקיות.")
    return SkillResult(ok=True, value=summary, data={"status": st, "summary_he": summary})


@REGISTRY.register(
    "rag.forget", risk="WRITE", agent="hermes",
    description_he="מוחק קובץ אחד מהאינדקס (הקובץ עצמו לא נמגע)",
    description_en="Remove one file from the index (the file itself is untouched)",
    required=("path",),
    triggers_he=("תשכח את הקובץ מהאינדקס", "הסר מהאינדקס"),
    triggers_en=("forget this file", "remove from index"),
)
def rag_forget(path: str = "") -> SkillResult:
    eng = get_engine()
    removed = eng.forget(path)
    return SkillResult(ok=removed,
                       value=f"הסרתי את {Path(path).name} מהאינדקס." if removed
                       else f"{Path(path).name} לא נמצא באינדקס.",
                       error="" if removed else "not in index",
                       data={"path": str(path), "removed": removed})


@REGISTRY.register(
    "rag.read", risk="SAFE", agent="argus",
    description_he="מחזיר קטע מאונדקס לפי מזהה, כולל שורות — לציטוט מדויק",
    description_en="Return one indexed chunk by id, with its line span",
    required=("chunk_id",),
    triggers_he=("הראה את הקטע", "פתח את המקור"),
    triggers_en=("show the chunk", "open the source"),
)
def rag_read(chunk_id: int = 0) -> SkillResult:
    eng = get_engine()
    rows = eng.store.fetch_chunks([int(chunk_id)])
    row = rows.get(int(chunk_id))
    if not row:
        return SkillResult(ok=False, error=f"chunk {chunk_id} is not in the index")
    return SkillResult(ok=True, value=row["text"], data={
        "path": row["path"], "start_line": row["start_line"], "end_line": row["end_line"],
        "heading": row["heading"], "kind": row["kind"],
        "summary_he": f"{Path(row['path']).name} שורות {row['start_line']}–{row['end_line']}.",
    })
