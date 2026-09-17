"""JARVIS Memory Palace — persistent, local, four kinds of memory.

  * episodic    — what happened, in which conversation, when
  * semantic    — durable facts about the world and about the user
  * procedural  — skills that were learned and can be replayed
  * vector      — semantic recall over everything above

Every record carries ``confidence`` and timestamps. Recall applies a
**forgetting curve** (exponential decay with a configurable half-life) so old,
unused, low-confidence memories fade exactly the way human memory does — and so
the store never grows into noise.

Storage is plain SQLite: portable, single-file, no server, no cloud.
"""

from __future__ import annotations

import json
import math
import re
import sqlite3
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from brain.knowledge import HashedNgramVectorizer, _stem, content_tokens  # noqa: E402
from brain.tokenizer import normalize  # noqa: E402
from core.bus import BUS, T  # noqa: E402

# Words that appear in nearly every exchange and would therefore "consolidate"
# out of pure address rather than topic: the assistant's form of address, its own
# name, pleasantry. Promoting "אדוני" to a fact about the user's world would be
# the consolidation equivalent of a transcript counting "the" as a theme.
_CONSOL_STOP = {"אדוני", "jarvis", "jrovis", "גרוויס", "שלום", "בבקשה", "תודה",
                "אני", "אתה", "את", "של", "על", "מה", "יש", "היה", "זה", "כן",
                "לא", "גם", "רק", "עכשיו", "המערכת", "מערכת", "המשתמש",
                # conversational verbs and possessives recur because people talk,
                # not because a topic exists; measured, they were the top
                # "themes" of the first cut
                "דיברנו", "דיבר", "דיברת", "אמר", "אמרתי", "אמרנו", "אמרת",
                "סיפר", "סיפרתי", "שאל", "שאלתי", "ביקש", "ביקשתי", "עשה",
                "עשיתי", "רוצה", "יכול", "צריך", "שלי", "שלך", "שלנו", "איך",
                "היה", "היתה", "יהיה", "נמצא", "קרה", "קורה",
                # normalized spellings: the tokenizer maps final ך/ם/ן to כ/מ/נ,
                # so a stop entry written in print form would never match
                "איכ", "כיכ", "שליכ"}

# Glued prefixes that are safe to peel: the definite article, "and", "in", "to",
# "like", "that". Deliberately NOT "מ", because mem opens as many roots as it
# closes as a preposition, and peeling it turns "מוזיקה" into noise.
_CONSOL_PREFIXES = ("ו", "ה", "ב", "ל", "כ", "ש")


def _consolidation_tokens(text: str) -> Dict[str, str]:
    """normalized word -> itself, for theme mining.

    Grouping peels only unambiguous glued prefixes so "המוזיקה" and "מוזיקה"
    count as one theme, and drops anything shorter than four characters or on
    the stop list. The first cut grouped by crude stem and promoted fragments
    like "איכ" and "דיברנו" to facts; peeling less and stopping more is what
    keeps a promoted fact reading like a topic.
    """
    t = f" {normalize(str(text)).lower()} "
    out: Dict[str, str] = {}
    for w in re.findall(r"[A-Za-z\u0590-\u05FF][A-Za-z\u0590-\u05FF'\-]{2,}", t):
        if w in _CONSOL_STOP:
            continue
        key = w
        for _ in range(2):
            if len(key) >= 4 and key[0] in _CONSOL_PREFIXES:
                key = key[1:]
            else:
                break
        if len(key) >= 4 and key not in _CONSOL_STOP:
            out.setdefault(key, w)
    return out


def strip_recall_frame(text: str) -> str:
    """Drop interrogative and recall frame words, keeping survivors verbatim.

    `recall` used to vectorise the raw query, and the vectoriser is character
    n-gram based, so the frame carried as much weight as the content. Every
    "מה אמרתי על X" therefore looked like every other one: measured across 15
    probes, genuine content matches scored 0.46-0.71 while frame-only matches
    against an unrelated memory clustered at 0.05-0.28, and one query —
    "סיפרתי לך על רקס" — matched a memory about a *project* at 0.227. The frame
    words, not the topic, were doing the matching.

    Only frame words are removed and the rest is left exactly as written, because
    stored memories were vectorised raw; stemming the query would break character
    n-gram agreement with them. `content_tokens` decides what is content, so this
    reuses the KB's stopword list and normalisation rather than growing a second
    one that can drift out of sync — the same divergence that caused 14 stopwords
    to silently stop matching.

    Falls back to the original text when nothing survives, so a query made
    entirely of frame words still produces a vector instead of an empty one.
    """
    raw = (text or "").strip()
    if not raw:
        return ""
    content = content_tokens(raw)
    if not content:
        return raw
    kept = []
    for w in raw.split():
        n = normalize(w).lower()
        if n in content or _stem(n) in content:
            kept.append(w)
    return " ".join(kept) or raw

SCHEMA = """
CREATE TABLE IF NOT EXISTS episodes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    kind TEXT NOT NULL,
    role TEXT,
    content TEXT NOT NULL,
    meta TEXT,
    importance REAL DEFAULT 0.5,
    accesses INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS facts (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    ts REAL NOT NULL,
    updated REAL NOT NULL,
    confidence REAL DEFAULT 0.8,
    source TEXT DEFAULT 'user',
    accesses INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS skills (
    name TEXT PRIMARY KEY,
    pattern TEXT NOT NULL,
    solution TEXT NOT NULL,
    successes INTEGER DEFAULT 0,
    failures INTEGER DEFAULT 0,
    ts REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_ep_kind ON episodes(kind);
CREATE INDEX IF NOT EXISTS idx_ep_ts ON episodes(ts);
"""


@dataclass
class RecallHit:
    text: str
    score: float
    kind: str
    row_id: int
    meta: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {"text": self.text, "score": round(self.score, 4), "kind": self.kind,
                "id": self.row_id, "meta": self.meta}


class MemoryPalace:
    def __init__(self, db_path: Path | str, decay_half_life_days: float = 21.0,
                 dim: int = 4096) -> None:
        self.path = Path(db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.half_life = max(0.1, decay_half_life_days) * 86400.0
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.executescript(SCHEMA)
            self._conn.commit()
        self.vectorizer = HashedNgramVectorizer(dim=dim)
        self._cache_dirty = True
        self._cache_texts: List[str] = []
        self._cache_meta: List[Dict[str, Any]] = []
        self._cache_vecs: Optional[np.ndarray] = None

    # ------------------------------------------------------------- episodic --
    def remember_episode(self, content: str, kind: str = "dialogue", role: str = "",
                         importance: float = 0.5, meta: Optional[Dict[str, Any]] = None) -> int:
        now = time.time()
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO episodes(ts, kind, role, content, meta, importance) VALUES (?,?,?,?,?,?)",
                (now, kind, role, content, json.dumps(meta or {}, ensure_ascii=False), float(importance)),
            )
            self._conn.commit()
            self._cache_dirty = True
        BUS.emit(T.MEMORY_WRITE, {"kind": kind, "id": cur.lastrowid, "chars": len(content)}, source="memory")
        return int(cur.lastrowid or 0)

    def remember_exchange(self, user: str, assistant: str, meta: Optional[Dict[str, Any]] = None) -> None:
        self.remember_episode(user, kind="user", role="user", importance=0.6, meta=meta)
        self.remember_episode(assistant, kind="assistant", role="jarvis", importance=0.6, meta=meta)

    def recent_episodes(self, n: int = 12, kinds: Sequence[str] = ()) -> List[Dict[str, Any]]:
        q = "SELECT * FROM episodes"
        args: List[Any] = []
        if kinds:
            q += f" WHERE kind IN ({','.join('?' * len(kinds))})"
            args.extend(kinds)
        q += " ORDER BY id DESC LIMIT ?"
        args.append(int(n))
        with self._lock:
            rows = self._conn.execute(q, args).fetchall()
        return [self._row_to_ep(r) for r in reversed(rows)]

    @staticmethod
    def _row_to_ep(r: sqlite3.Row) -> Dict[str, Any]:
        try:
            meta = json.loads(r["meta"] or "{}")
        except Exception:
            meta = {}
        return {"id": r["id"], "ts": r["ts"], "kind": r["kind"], "role": r["role"],
                "content": r["content"], "importance": r["importance"], "meta": meta}

    # ------------------------------------------------------------- semantic --
    def remember_fact(self, key: str, value: str, confidence: float = 0.9,
                      source: str = "user") -> None:
        now = time.time()
        with self._lock:
            self._conn.execute(
                """INSERT INTO facts(key, value, ts, updated, confidence, source)
                   VALUES (?,?,?,?,?,?)
                   ON CONFLICT(key) DO UPDATE SET value=excluded.value,
                     updated=excluded.updated, confidence=excluded.confidence,
                     source=excluded.source""",
                (key, value, now, now, float(confidence), source),
            )
            self._conn.commit()
            self._cache_dirty = True
        BUS.emit(T.MEMORY_WRITE, {"kind": "fact", "key": key}, source="memory")

    def recall_fact(self, key: str) -> Optional[str]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM facts WHERE key=?", (key,)
            ).fetchone()
            if row is None:
                return None
            self._conn.execute("UPDATE facts SET accesses=accesses+1 WHERE key=?", (key,))
            self._conn.commit()
        decay = self._decay(time.time() - row["updated"])
        if decay * row["confidence"] < 0.05:
            return None
        return str(row["value"])

    def all_facts(self) -> List[Dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute("SELECT * FROM facts ORDER BY updated DESC").fetchall()
        out = []
        now = time.time()
        for r in rows:
            strength = self._decay(now - r["updated"]) * r["confidence"]
            out.append({"key": r["key"], "value": r["value"], "confidence": r["confidence"],
                        "source": r["source"], "updated": r["updated"],
                        "strength": round(strength, 4)})
        return out

    # ----------------------------------------------------------- procedural --
    def learn_skill(self, name: str, pattern: str, solution: str) -> None:
        with self._lock:
            self._conn.execute(
                """INSERT INTO skills(name, pattern, solution, ts) VALUES (?,?,?,?)
                   ON CONFLICT(name) DO UPDATE SET pattern=excluded.pattern,
                     solution=excluded.solution, ts=excluded.ts""",
                (name, pattern, solution, time.time()),
            )
            self._conn.commit()
            self._cache_dirty = True

    def skill_outcome(self, name: str, success: bool) -> None:
        col = "successes" if success else "failures"
        with self._lock:
            self._conn.execute(f"UPDATE skills SET {col}={col}+1 WHERE name=?", (name,))
            self._conn.commit()

    def find_skill(self, pattern: str) -> Optional[Dict[str, Any]]:
        vec = self.vectorizer
        with self._lock:
            rows = self._conn.execute("SELECT * FROM skills").fetchall()
        if not rows:
            return None
        vec.fit([r["pattern"] for r in rows])
        qv = vec.transform_one(pattern)
        best, best_score = None, -1.0
        for r in rows:
            s = float(qv @ vec.transform_one(r["pattern"]))
            total = r["successes"] + r["failures"]
            prior = (r["successes"] + 1) / (total + 2)
            score = s * (0.5 + 0.5 * prior)
            if score > best_score:
                best, best_score = r, score
        if best is None or best_score < 0.3:
            return None
        return {"name": best["name"], "pattern": best["pattern"], "solution": best["solution"],
                "score": round(best_score, 4), "successes": best["successes"],
                "failures": best["failures"]}

    # --------------------------------------------------------------- recall --
    def _decay(self, age_seconds: float) -> float:
        return math.pow(0.5, max(0.0, age_seconds) / self.half_life)

    def _rebuild_cache(self) -> None:
        with self._lock:
            eps = self._conn.execute(
                "SELECT id, ts, kind, content, importance FROM episodes ORDER BY id DESC LIMIT 4000"
            ).fetchall()
            facts = self._conn.execute("SELECT key, value, updated, confidence FROM facts").fetchall()
        texts: List[str] = []
        metas: List[Dict[str, Any]] = []
        now = time.time()
        for r in eps:
            texts.append(r["content"])
            metas.append({"kind": "episode:" + r["kind"], "id": r["id"],
                          "weight": self._decay(now - r["ts"]) * (0.5 + r["importance"])})
        for r in facts:
            texts.append(f"{r['key']}: {r['value']}")
            metas.append({"kind": "fact", "id": r["key"],
                          "weight": self._decay(now - r["updated"]) * r["confidence"]})
        if texts:
            self.vectorizer.fit(texts)
            self._cache_vecs = self.vectorizer.transform(texts)
        else:
            self._cache_vecs = np.zeros((0, self.vectorizer.dim), dtype=np.float32)
        self._cache_texts, self._cache_meta = texts, metas
        self._cache_dirty = False

    def recall(self, query: str, k: int = 5, min_score: float = 0.05) -> List[RecallHit]:
        if self._cache_dirty or self._cache_vecs is None:
            self._rebuild_cache()
        if self._cache_vecs is None or not len(self._cache_texts):
            return []
        qv = self.vectorizer.transform_one(strip_recall_frame(query))
        sims = self._cache_vecs @ qv
        weights = np.array([m["weight"] for m in self._cache_meta], dtype=np.float32)
        scored = sims * (0.35 + 0.65 * weights)
        order = np.argsort(-scored)[: k * 4]
        hits: List[RecallHit] = []
        for i in order:
            s = float(scored[i])
            if s < min_score:
                continue
            m = self._cache_meta[int(i)]
            hits.append(RecallHit(text=self._cache_texts[int(i)], score=s,
                                  kind=m["kind"], row_id=m["id"], meta={"weight": round(float(weights[int(i)]), 3)}))
            if len(hits) >= k:
                break
        BUS.emit(T.MEMORY_RECALL, {"query": query[:60], "hits": len(hits)}, source="memory")
        return hits

    # ------------------------------------------------------------ maintenance --
    # --------------------------------------------------------- consolidation --
    def consolidate(self, min_support: int = 3, scan: int = 200,
                    prune: bool = False) -> Dict[str, Any]:
        """Mine recurring themes out of recent episodes into semantic facts.

        Episodes are episodic memory: what happened, in order, decaying. A theme
        that keeps coming back across separate conversations is no longer an
        event — it is something true about the user's world, and keeping it only
        as N fading episodes means the N+1'th mention starts from zero. So themes
        appearing in at least `min_support` recent episodes are promoted (or
        refreshed, if already promoted) into a semantic fact whose confidence
        grows with the support, sourced as "consolidation" so its origin stays
        visible.

        This is the offline analogue of sleep consolidation: nothing new enters
        the system, recurring structure is merely moved to where retrieval is
        cheap. Nothing is deleted here unless `prune` is set, and even then only
        through forget()'s decay threshold.
        """
        eps = self.recent_episodes(n=int(scan))
        counts: Dict[str, int] = {}
        word: Dict[str, str] = {}
        snippet: Dict[str, str] = {}
        for ep in eps:
            body = str(ep.get("content", ""))
            for stem, raw in _consolidation_tokens(body).items():
                counts[stem] = counts.get(stem, 0) + 1
                word.setdefault(stem, raw)
                snippet.setdefault(stem, body[:140])

        existing = {f.get("key") for f in self.all_facts()}
        promoted: List[str] = []
        refreshed: List[str] = []
        for stem, c in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
            if c < int(min_support):
                continue
            key = f"consolidated.{stem}"
            value = (f"הנושא '{word.get(stem, stem)}' עלה ב־{c} מתוך {len(eps)} "
                     f"השיחות שנבדקו לאחרונה. דוגמה: {snippet.get(stem, '')}")
            confidence = min(0.95, 0.45 + 0.12 * c)
            self.remember_fact(key, value, confidence=confidence, source="consolidation")
            (refreshed if key in existing else promoted).append(key)

        pruned = self.forget(dry_run=not prune)
        return {"episodes_scanned": len(eps), "themes_seen": len(counts),
                "min_support": int(min_support),
                "promoted": promoted, "refreshed": refreshed,
                "pruned": pruned}

    def forget(self, dry_run: bool = True, threshold: float = 0.02) -> Dict[str, Any]:
        """Prune memories whose decayed strength fell below the threshold."""
        now = time.time()
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, ts, importance, accesses FROM episodes"
            ).fetchall()
        doomed = []
        for r in rows:
            strength = self._decay(now - r["ts"]) * (0.5 + r["importance"]) * (1 + 0.1 * r["accesses"])
            if strength < threshold:
                doomed.append(int(r["id"]))
        if not dry_run and doomed:
            with self._lock:
                self._conn.executemany("DELETE FROM episodes WHERE id=?", [(i,) for i in doomed])
                self._conn.commit()
                self._cache_dirty = True
        return {"candidates": len(doomed), "deleted": 0 if dry_run else len(doomed),
                "dry_run": dry_run}

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            eps = self._conn.execute("SELECT COUNT(*) c FROM episodes").fetchone()["c"]
            facts = self._conn.execute("SELECT COUNT(*) c FROM facts").fetchone()["c"]
            skills = self._conn.execute("SELECT COUNT(*) c FROM skills").fetchone()["c"]
        return {"episodes": eps, "facts": facts, "skills": skills,
                "db_bytes": self.path.stat().st_size if self.path.exists() else 0,
                "half_life_days": round(self.half_life / 86400, 1)}

    def close(self) -> None:
        with self._lock:
            self._conn.close()
