"""The researcher, and the posture switches — all offline, all local.

`research.query` is the answer to "a researcher that finds me everything I need".
There is no web to search on an offline machine, so "everything" means every
source this build actually owns: the knowledge base, the memory palace, the
conversation transcript, the file tree, and the skill catalogue. Each hit comes
back tagged with the source it came from and the line or record it matched, so
an answer can be checked instead of trusted.

Search is deliberately simple — token overlap and substring, scored — because a
fancier ranker with no training data would be theatre. What matters here is
coverage and provenance, not a cosine score nobody can audit.

`mode.set` / `mode.get` / `mode.list` expose the twelve postures in brain/modes.py
to voice, chat and the HUD through the same registry everything else uses.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List

from skills.registry import REGISTRY, SkillResult

ROOT = Path(__file__).resolve().parents[1]

# Directories the file sweep must never enter: they are build noise, not content.
_SKIP_DIRS = {".git", ".venv", "node_modules", "__pycache__", "models", ".arena",
              ".cache", "build", "dist", "trash"}
_TEXT_EXT = {".py", ".md", ".txt", ".yaml", ".yml", ".json", ".html", ".css", ".js", ".bat"}
_MAX_FILES = 400
_MAX_FILE_BYTES = 200_000

_STOP = {"של", "על", "את", "אתה", "מה", "מהו", "מהי", "האם", "יש", "אין", "לי",
         "בבקשה", "תמצא", "חפש", "חקור", "בשבילי", "כל", "הכל", "אודות",
         "לא", "כן", "גם", "רק", "הוא", "היא", "זה", "זאת", "אותו", "אותה",
         "שלי", "שלך", "שלנו", "היה", "היתה", "יהיה", "יכול", "צריך", "אפשר",
         "מאוד", "קצת", "כמה", "איפה", "מתי", "למה", "כדי", "עם", "בלי",
         "בין", "מן", "אל", "עד", "אחרי", "לפני", "קיים", "קיימת", "עוד",
         "the", "a", "an", "about", "for", "me", "find", "search", "research",
         "what", "is", "of", "and", "to", "in", "on", "please", "tell",
         "there", "here", "any", "all", "everything", "know", "have", "you"}

# Substring matching is right for Hebrew, where prefixes glue onto words
# ("זיכרון" inside "הזיכרון"), but it makes short common tokens match almost any
# sentence. Two guards keep a nonsense topic from "finding" ten hits: a token only
# counts if it is at least three characters and not a function word, and a
# haystack only scores if either two distinct tokens hit or one long distinctive
# one does. Measured, without these "קסנומורף_לא_קיים" returned hits on the word
# "לא" alone — a confident non-answer, the exact failure this project audits for.
# Tokens shorter than four characters are never evidence: in Hebrew that is the
# length where function words and glued prefixes live, and substring-matching
# them lights up almost every sentence. Four characters and out of the stoplist
# is where a token starts to mean something, so a single such match is enough to
# keep a hit — "חומת" appearing in a firewall entry is real evidence, while the
# two-token-or-six-character rule an earlier cut used threw it away and returned
# zero results for a perfectly good question.
_MIN_TOK = 4


def _tokens(text: str) -> List[str]:
    words = re.findall(r"[A-Za-zא-ת]{%d,}" % _MIN_TOK, (text or "").lower())
    kept = [w for w in words if w not in _STOP]
    return kept or words


def _matched(query_tokens: List[str], hay: str) -> List[str]:
    if not query_tokens or not hay:
        return []
    low = hay.lower()
    return [t for t in dict.fromkeys(query_tokens) if t in low]


def _score(query_tokens: List[str], hay: str) -> float:
    hit = _matched(query_tokens, hay)
    if not hit:
        return 0.0
    # longer matched tokens are stronger evidence; a short haystack concentrates
    # the evidence more than a long one
    s = float(sum(len(t) for t in hit))
    return s * (1.5 if len(hay) < 240 else 1.0)


@REGISTRY.register(
    "research.query", risk="SAFE", agent="argus",
    description_he="חוקר נושא בכל המקורות המקומיים ומחזיר תמצית עם מקורות",
    triggers_he=("תחקור לי", "חפש לי כל", "מה יש לך על", "research"),
    required=("topic",),
)
def research_query(topic: str = "", limit: int = 4) -> SkillResult:
    """Search every local source for a topic and return a sourced digest."""
    q = (topic or "").strip()
    if not q:
        return SkillResult(ok=False, error="לא ניתן לחקור בלי נושא — מה לחפש, אדוני?")
    toks = _tokens(q)
    if not toks:
        return SkillResult(ok=False, error=f"הנושא {q!r} לא הכיל מילים לחיפוש.")
    k = max(1, min(int(limit or 4), 8))
    found: Dict[str, List[Dict[str, Any]]] = {}
    t0 = time.perf_counter()

    # ── knowledge base ────────────────────────────────────────────────────
    try:
        from brain.knowledge import KnowledgeStore
        kb = KnowledgeStore()
        # The vectoriser will return *something* for any input, including noise at
        # ~0.08 cosine for a string that means nothing. A hit is therefore kept
        # only on strong cosine or on real token overlap — otherwise "find me
        # everything about xenomorph" comes back with three confident paragraphs
        # about CPU caches, which is the non-answer this project exists to avoid.
        for hit in kb.search(q, k=k, min_score=0.05):
            body = f"{hit.get('text_he','')} {hit.get('text_en','')} {hit.get('topic','')}"
            if float(hit.get("score", 0.0)) < 0.25 and _score(toks, body) <= 0:
                continue
            found.setdefault("knowledge", []).append({
                "score": round(float(hit.get("score", 0.0)), 3),
                "text": str(hit.get("text_he") or hit.get("text_en") or "")[:220],
                "topic": str(hit.get("topic", ""))[:60],
                "id": hit.get("id") or ""})
        for qa in kb.search_qa(q, k=k, min_score=0.35):
            found.setdefault("knowledge_qa", []).append({
                "score": round(float(qa.get("score", 0.0)), 3),
                "question": str(qa.get("question", ""))[:120],
                "answer": str(qa.get("answer", ""))[:220]})
    except Exception as exc:
        found.setdefault("knowledge", []).append({"score": 0.0, "text": f"שגיאה: {exc}"})

    # ── memory palace ─────────────────────────────────────────────────────
    try:
        from brain.memory import MemoryPalace
        from core.config import CONFIG
        mp = MemoryPalace(CONFIG.memory.db_path)
        for hit in mp.recall(q, k=k, min_score=0.03):
            # recall() at a low floor returns its top-k for *any* input, the same
            # vector-noise trap the knowledge gate closes: a random nonce was
            # "remembered" twice. Token overlap is the evidence test here too.
            body = str(getattr(hit, "content", hit))
            if _score(toks, body) <= 0:
                continue
            found.setdefault("memory", []).append({
                "score": round(float(getattr(hit, "score", 0.0)), 3),
                "text": body[:220]})
        for f in mp.all_facts():
            s = _score(toks, f"{f.get('key','')} {f.get('value','')}")
            if s > 0:
                found.setdefault("memory_facts", []).append({
                    "score": round(s, 2), "key": str(f.get("key", ""))[:80],
                    "value": str(f.get("value", ""))[:160]})
    except Exception as exc:
        found.setdefault("memory", []).append({"score": 0.0, "text": f"שגיאה: {exc}"})

    # ── conversation transcript ────────────────────────────────────────────
    try:
        tpath = Path(getattr(__import__("core.config", fromlist=["CONFIG"]).CONFIG.memory,
                             "transcript_path", str(ROOT / "data" / "transcript.jsonl")))
        if tpath.exists():
            rows = []
            for line in tpath.read_text(encoding="utf-8", errors="replace").splitlines():
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                hay = f"{rec.get('user','')} {rec.get('assistant','')}"
                s = _score(toks, hay)
                if s > 0:
                    rows.append((s, rec))
            rows.sort(key=lambda r: -r[0])
            for s, rec in rows[:k]:
                found.setdefault("transcript", []).append({
                    "score": round(s, 2),
                    "user": str(rec.get("user", ""))[:110],
                    "assistant": str(rec.get("assistant", ""))[:160]})
    except Exception as exc:
        found.setdefault("transcript", []).append({"score": 0.0, "text": f"שגיאה: {exc}"})

    # ── file tree ──────────────────────────────────────────────────────────
    try:
        hits: List[Dict[str, Any]] = []
        scanned = 0
        for p in ROOT.rglob("*"):
            if scanned >= _MAX_FILES:
                break
            if not p.is_file() or p.suffix.lower() not in _TEXT_EXT:
                continue
            if any(part in _SKIP_DIRS for part in p.parts):
                continue
            scanned += 1
            rel = str(p.relative_to(ROOT))
            s = _score(toks, rel) * 2.0
            best_line = ""
            try:
                if p.stat().st_size <= _MAX_FILE_BYTES:
                    for i, ln in enumerate(p.read_text(encoding="utf-8",
                                                      errors="replace").splitlines()):
                        ls = _score(toks, ln)
                        if ls > s:
                            s, best_line = ls, f"{i+1}: {ln.strip()[:140]}"
            except OSError:
                continue
            if s > 0:
                hits.append({"score": round(s, 2), "path": rel, "line": best_line})
        hits.sort(key=lambda h: -h["score"])
        if hits:
            found["files"] = hits[:k]
    except Exception as exc:
        found.setdefault("files", []).append({"score": 0.0, "text": f"שגיאה: {exc}"})

    # ── skill catalogue ────────────────────────────────────────────────────
    try:
        import skills as skills_pkg
        skills_pkg.load_all()
        rows = []
        for s in REGISTRY.schema():
            sc = _score(toks, f"{s.get('name','')} {s.get('description','')}")
            if sc > 0:
                rows.append({"score": round(sc, 2), "skill": s.get("name", ""),
                             "risk": s.get("risk", ""),
                             "desc": str(s.get("description", ""))[:140]})
        rows.sort(key=lambda r: -r["score"])
        if rows:
            found["skills"] = rows[:k]
    except Exception:
        pass

    total = sum(len(v) for v in found.values())
    ms = round((time.perf_counter() - t0) * 1000, 1)
    srcs = ", ".join(found.keys()) if found else "אף מקור"
    if not total:
        return SkillResult(
            ok=True,
            value=(f"חיפשתי את {q!r} בכל המקורות המקומיים — בסיס ידע, זיכרון, תמליל, "
                   f"עץ הקבצים וקטלוג הכלים — ולא מצאתי דבר. זה אומר שאין לי על זה "
                   f"ידע מקומי, לא שהנושא לא קיים."),
            data={"topic": q, "sources_searched": 5, "hits": 0, "ms": ms,
                  "groups": {}, "note": "offline: no web sources are consulted"})

    summary = (f"מצאתי {total} ממצאים על {q!r} ב־{len(found)} מקורות ({srcs}), "
               f"בתוך {ms}ms. הבולטים:")
    for group, rows in list(found.items())[:3]:
        top = rows[0]
        snippet = (top.get("text") or top.get("answer") or top.get("value")
                   or top.get("line") or top.get("desc") or top.get("path")
                   or top.get("assistant") or "")
        summary += f"\n· {group}: {str(snippet)[:150]}"

    return SkillResult(ok=True, value=summary, data={
        "topic": q, "sources_searched": 5, "hits": total, "ms": ms,
        "groups": found, "note": "offline: no web sources are consulted"})


@REGISTRY.register(
    "memory.consolidate", risk="SAFE", agent="mnemosyne",
    description_he="מרכז זיכרון: מעלה נושאים שחוזרים על עצמם לעובדות קבועות ומנקה דעיכה",
    triggers_he=("רכז זיכרון", "גיבוש זיכרון", "consolidate"),
)
def memory_consolidate(min_support: int = 3, prune: bool = False) -> SkillResult:
    """Promote recurring conversational themes into durable semantic facts."""
    from brain.memory import MemoryPalace
    from core.config import CONFIG
    mp = MemoryPalace(CONFIG.memory.db_path,
                      decay_half_life_days=CONFIG.memory.decay_half_life_days)
    rep = mp.consolidate(min_support=int(min_support or 3), prune=bool(prune))
    n_promoted = len(rep["promoted"])
    n_refreshed = len(rep["refreshed"])
    if not n_promoted and not n_refreshed:
        return SkillResult(ok=True,
                           value=(f"סרקתי {rep['episodes_scanned']} שיחות ולא מצאתי נושא "
                                  f"שחוזר לפחות {rep['min_support']} פעמים — אין מה לרכז. "
                                  f"הזיכרון האפיזודי נשאר כשהוא."),
                           data=rep)
    names = ", ".join(k.split(".", 1)[1] for k in rep["promoted"][:6])
    return SkillResult(ok=True,
                       value=(f"ריכזתי: {n_promoted} נושאים חדשים הפכו לעובדות "
                              f"({names}) ו־{n_refreshed} קיימים רועננו, מתוך "
                              f"{rep['episodes_scanned']} שיחות. ניקוי דעיכה: "
                              f"{rep['pruned']['deleted'] if not rep['pruned']['dry_run'] else 0} נמחקו."),
                       data=rep)


# ------------------------------------------------------------------- modes --
@REGISTRY.register(
    "mode.set", risk="SAFE", agent="argus",
    description_he="מחליף את מצב הפעולה של JARVIS (חוקר, מתכנת, מזכיר…)",
    triggers_he=("מצב חוקר", "מצב מתכנת", "עבור למצב", "switch mode"),
    required=("mode",),
)
def mode_set(mode: str = "") -> SkillResult:
    from brain.modes import STATE
    # Resolve a spoken phrase to a mode id first, then actually switch. An
    # earlier cut wrote `by_trigger(mode) or set(mode)`, which resolved the
    # phrase and then short-circuited — reporting the new posture while the
    # state never moved. The reply was confident and the switch was imaginary.
    trig = STATE.by_trigger(mode)
    m = STATE.set(trig.id if trig is not None else mode)
    if m is None:
        known = ", ".join(f"{x.id} ({x.he})" for x in __import__(
            "brain.modes", fromlist=["MODES"]).MODES.values())
        return SkillResult(ok=False, error=f"אין מצב בשם {mode!r}. המצבים: {known}")
    return SkillResult(ok=True,
                       value=f"עברתי למצב {m.he} ({m.en}). {m.persona_he} {m.desc_he}",
                       data={"mode": m.id, "he": m.he, "en": m.en, "desc": m.desc_he})


@REGISTRY.register(
    "mode.get", risk="SAFE", agent="argus",
    description_he="מדווח באיזה מצב JARVIS נמצא עכשיו",
    triggers_he=("באיזה מצב אתה", "מה המצב שלך"),
)
def mode_get() -> SkillResult:
    from brain.modes import STATE
    st = STATE.stats()
    m = STATE.get()
    return SkillResult(ok=True,
                       value=(f"אני במצב {m.he} ({m.en}) כבר {st['held_seconds']:.0f} שניות, "
                              f"אדוני. {m.desc_he}"),
                       data=st)


@REGISTRY.register(
    "mode.list", risk="SAFE", agent="argus",
    description_he="מציג את כל מצבי הפעולה האפשריים",
    triggers_he=("אילו מצבים יש", "רשימת מצבים"),
)
def mode_list() -> SkillResult:
    from brain.modes import STATE
    rows = STATE.schema()
    lines = "\n".join(f"· {r['he']} ({r['id']}) — {r['desc']}"
                      + ("  ← פעיל" if r["active"] else "") for r in rows)
    return SkillResult(ok=True, value=f"{len(rows)} מצבי פעולה:\n{lines}",
                       data={"count": len(rows), "modes": rows})
