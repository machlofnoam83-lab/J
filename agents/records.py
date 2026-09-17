"""ediyel records — the people dossier.

The face gallery answers "is this the same face I have seen before?". That is a
vector comparison and nothing more: it knows that face #7 was enrolled as Dana,
and it knows nothing else about her. This agent holds the rest — the story, the
age, the birthday, the notes you actually care about when someone walks in.

Two stores, deliberately separate:

  * ``vision/faces.py`` owns the **vector** (4096 floats, privacy-sensitive,
    never leaves the machine, never shown in the HUD)
  * this module owns the **dossier** (a name, a story, a photo path, free text)

They are joined by ``person_id`` when a face is enrolled, and by name when it is
not. A dossier without a face is perfectly legal — you can keep a record of your
mother without ever pointing a camera at her.

Photos are *referenced*, never copied. The path is stored and the bytes are not
duplicated into JSON, because a dossier file that grows a megabyte per person
stops being something you can read or back up.

Everything here is local. Nothing is uploaded, nothing is fetched, and the file
is plain JSON you can open in a text editor and delete.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

# Sits beside the face gallery so one folder holds "what JARVIS knows about
# people" and one deletion removes all of it.
def default_file() -> Path:
    """Where the dossier lives.

    Read at call time, not at import time. Binding this to a module constant
    looked tidier and was wrong: anything that set ``JARVIS_RECORDS_DIR`` after
    the first import — which is every test, and any tool that configures the
    environment before building a store — silently wrote to the production file
    instead. Resolving it here makes the override actually override.
    """
    return Path(os.environ.get("JARVIS_RECORDS_DIR",
                               str(Path("data") / "records"))) / "people.json"


# Kept as a name other modules can import, but it is a snapshot — prefer
# default_file(). Nothing in this module reads it.
DEFAULT_DIR = Path(os.environ.get(
    "JARVIS_RECORDS_DIR", Path("data") / "records"))
DEFAULT_FILE = DEFAULT_DIR / "people.json"

# A dossier is a person, not a paragraph. This keeps the store honest when
# someone dictates a whole conversation into the story field.
MAX_STORY = 4000
MAX_NAME = 80

# Fields a user may set. Anything else is rejected rather than silently stored,
# because a dossier you cannot predict the shape of is one you cannot search.
TEXT_FIELDS = ("name", "story", "note", "role", "relationship",
               "phone", "email", "birthday", "photo", "tags")


def _now() -> float:
    return time.time()


def _clean(value: Any, limit: int) -> str:
    s = str(value or "").strip()
    return s[:limit]


@dataclass
class Person:
    """One dossier. ``id`` is stable; everything else is editable."""

    id: str
    name: str
    story: str = ""
    age: Optional[int] = None
    birthday: str = ""
    note: str = ""
    role: str = ""                    # what they are to you, not a permission
    relationship: str = ""
    phone: str = ""
    email: str = ""
    photo: str = ""                   # a path, not bytes
    tags: List[str] = field(default_factory=list)
    face_id: str = ""                 # join key into vision/faces.py
    created: float = field(default_factory=_now)
    updated: float = field(default_factory=_now)

    # ------------------------------------------------------------- derived --

    @property
    def has_photo(self) -> bool:
        if not self.photo:
            return False
        try:
            return Path(self.photo).is_file()
        except Exception:                                  # pragma: no cover
            return False

    @property
    def age_now(self) -> Optional[int]:
        """Stored age, corrected by the birthday when one is known.

        A stored age goes stale the day after you write it. If a birthday is
        present it wins, because that one never needs updating.
        """
        if self.birthday:
            years = _years_since(self.birthday)
            if years is not None:
                return years
        return self.age

    @property
    def summary_he(self) -> str:
        """One Hebrew sentence — what JARVIS says when this person walks in."""
        bits = [self.name]
        if self.relationship:
            bits.append(f"({self.relationship})")
        age = self.age_now
        if age is not None:
            bits.append(f"בן/בת {age}")
        head = " ".join(bits)
        if self.story:
            first = re.split(r"[.\n!?]", self.story.strip())[0].strip()
            if first:
                head += f" — {first[:140]}"
        return head

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["age_now"] = self.age_now
        d["has_photo"] = self.has_photo
        d["summary_he"] = self.summary_he
        return d

    def brief(self) -> Dict[str, Any]:
        """What a list view needs — never the whole story."""
        return {"id": self.id, "name": self.name, "age": self.age_now,
                "relationship": self.relationship, "role": self.role,
                "has_photo": self.has_photo, "face_id": self.face_id,
                "summary_he": self.summary_he[:120]}


def _parse_birthday(birthday: str) -> Optional[tuple]:
    """(year, month, day) from the handful of shapes people actually type.

    Returns None rather than guessing: a reminder fired on the wrong day is
    worse than no reminder, because it teaches the user to ignore the others.
    """
    s = str(birthday or "").strip()
    if not s:
        return None
    m = re.search(r"(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})", s)
    if m:
        year, month, day = int(m.group(1)), int(m.group(2)), int(m.group(3))
    else:
        m = re.search(r"(\d{1,2})[-/.](\d{1,2})[-/.](\d{4})", s)
        if not m:
            return None
        day, month, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
    if not (1 <= month <= 12 and 1 <= day <= 31 and 1900 <= year <= 2200):
        return None
    return year, month, day


def days_until_birthday(birthday: str, today: Optional[Any] = None) -> Optional[int]:
    """Days until the next occurrence, 0 meaning today.

    Feb 29 rolls to Feb 28 in a common year rather than being skipped — someone
    born on the 29th still has a birthday, and silently dropping them from the
    list once every four years is the kind of small wrongness nobody notices
    until they do.
    """
    parsed = _parse_birthday(birthday)
    if parsed is None:
        return None
    import datetime as dt
    year, month, day = parsed
    today = today or dt.date.today()
    for y in (today.year, today.year + 1):
        try:
            nxt = dt.date(y, month, day)
        except ValueError:
            nxt = dt.date(y, month, 28)          # Feb 29 in a common year
        if nxt >= today:
            return (nxt - today).days
    return None                                  # pragma: no cover


def _years_since(birthday: str) -> Optional[int]:
    """Parse a handful of honest date shapes. Returns None rather than guessing."""
    s = str(birthday or "").strip()
    if not s:
        return None
    m = re.search(r"(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})", s)
    if not m:
        m = re.search(r"(\d{1,2})[-/.](\d{1,2})[-/.](\d{4})", s)
        if not m:
            return None
        day, month, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
    else:
        year, month, day = int(m.group(1)), int(m.group(2)), int(m.group(3))
    if not (1 <= month <= 12 and 1 <= day <= 31 and 1900 <= year <= 2200):
        return None
    t = time.localtime()
    age = t.tm_year - year - ((t.tm_mon, t.tm_mday) < (month, day))
    return age if 0 <= age <= 130 else None


class RecordsStore:
    """JSON-backed dossier store. No database, no network, no surprises."""

    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = Path(path) if path else default_file()
        self._people: Dict[str, Person] = {}
        self._load()

    # ------------------------------------------------------------------- IO --

    def _load(self) -> None:
        if not self.path.is_file():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            # A corrupt dossier must not take the assistant down, and it must
            # not be silently overwritten either — keep the bad file to inspect.
            try:
                self.path.rename(self.path.with_suffix(".corrupt.json"))
            except Exception:                              # pragma: no cover
                pass
            return
        for item in (raw.get("people") or []):
            try:
                p = Person(**{k: v for k, v in item.items()
                              if k in Person.__dataclass_fields__})
                self._people[p.id] = p
            except Exception:                              # pragma: no cover
                continue

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"version": 1, "updated": _now(),
                   "people": [p.to_dict() for p in self._people.values()]}
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                       encoding="utf-8")
        tmp.replace(self.path)

    # ------------------------------------------------------------------ CRUD --

    def add(self, *, name: str, story: str = "", age: Optional[int] = None,
            birthday: str = "", photo: str = "", face_id: str = "",
            note: str = "", role: str = "", relationship: str = "",
            phone: str = "", email: str = "",
            tags: Optional[List[str]] = None) -> Optional[Person]:
        """Create a dossier. Returns None when there is no name to key on."""
        clean = _clean(name, MAX_NAME)
        if not clean:
            return None
        pid = f"p{int(_now() * 1000) % 10**10:010d}"
        while pid in self._people:                          # pragma: no cover
            pid = f"p{int(_now() * 1000) % 10**10:010d}"
        p = Person(id=pid, name=clean, story=_clean(story, MAX_STORY),
                   age=_as_age(age), birthday=_clean(birthday, 40),
                   photo=_clean(photo, 500), face_id=_clean(face_id, 64),
                   note=_clean(note, MAX_STORY), role=_clean(role, 80),
                   relationship=_clean(relationship, 80),
                   phone=_clean(phone, 40), email=_clean(email, 120),
                   tags=[_clean(t, 40) for t in (tags or []) if _clean(t, 40)])
        self._people[pid] = p
        self._save()
        return p

    def update(self, person_id: str, **changes: Any) -> Optional[Person]:
        p = self._people.get(person_id)
        if p is None:
            return None
        for key, value in changes.items():
            if key not in TEXT_FIELDS and key not in ("age", "face_id", "tags"):
                continue
            if key == "age":
                p.age = _as_age(value)
            elif key == "tags":
                p.tags = [_clean(t, 40) for t in (value or []) if _clean(t, 40)]
            elif key == "story":
                p.story = _clean(value, MAX_STORY)
            elif key == "name":
                cleaned = _clean(value, MAX_NAME)
                if cleaned:
                    p.name = cleaned
            else:
                setattr(p, key, _clean(value, 500))
        p.updated = _now()
        self._save()
        return p

    def remove(self, person_id: str) -> bool:
        if person_id not in self._people:
            return False
        del self._people[person_id]
        self._save()
        return True

    def get(self, person_id: str) -> Optional[Person]:
        return self._people.get(person_id)

    def list(self) -> List[Dict[str, Any]]:
        return [p.brief() for p in sorted(self._people.values(),
                                          key=lambda x: x.name.lower())]

    # ---------------------------------------------------------------- search --

    def find(self, query: str) -> List[Person]:
        """Find a person by name, relationship, tag or a word in their story.

        Ranked, not filtered: an exact name match beats a story mention, because
        "tell me about Dana" means Dana and not whoever once mentioned her.
        """
        q = _clean(query, 200).lower()
        if not q:
            return []
        scored: List[tuple] = []
        for p in self._people.values():
            score = 0
            if p.name.lower() == q:
                score += 100
            elif q in p.name.lower():
                score += 60
            if q in p.relationship.lower():
                score += 30
            if q in p.role.lower():
                score += 20
            if any(q in t.lower() for t in p.tags):
                score += 25
            if q in p.story.lower():
                score += 10
            if q in p.note.lower():
                score += 5
            if score:
                scored.append((score, p.name.lower(), p))
        scored.sort(key=lambda x: (-x[0], x[1]))
        return [p for _, _, p in scored]

    def by_face(self, face_id: str) -> Optional[Person]:
        """The join back from the camera: who is this face?"""
        fid = _clean(face_id, 64)
        if not fid:
            return None
        for p in self._people.values():
            if p.face_id == fid:
                return p
        return None

    def link_face(self, person_id: str, face_id: str) -> Optional[Person]:
        """Attach an enrolled face to an existing dossier."""
        return self.update(person_id, face_id=_clean(face_id, 64))

    def stats(self) -> Dict[str, Any]:
        with_face = sum(1 for p in self._people.values() if p.face_id)
        with_photo = sum(1 for p in self._people.values() if p.has_photo)
        with_story = sum(1 for p in self._people.values() if p.story.strip())
        return {"people": len(self._people), "with_face": with_face,
                "with_photo": with_photo, "with_story": with_story,
                "path": str(self.path)}


def _as_age(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        n = int(str(value).strip())
    except Exception:
        return None
    return n if 0 <= n <= 130 else None


_STORE: Optional[RecordsStore] = None


def get_store(fresh: bool = False) -> RecordsStore:
    """Process-wide dossier store."""
    global _STORE
    if fresh or _STORE is None:
        _STORE = RecordsStore()
    return _STORE


class RecordsAgent:
    """The agent the brain talks to. Thin by design — the store does the work.

    Follows the same shape as the other agents (``name``/``role``/``handle``) so
    the orchestrator does not need a special case for it.
    """

    name = "ediyel_records"
    role = "סוכן הרשומות — תיק אישי לכל אדם"

    def __init__(self, store: Optional[RecordsStore] = None) -> None:
        self.store = store or get_store()
        self.history: List[Dict[str, Any]] = []

    def handle(self, task: str) -> Dict[str, Any]:
        """Free-text entry point, mirroring the other agents."""
        t0 = time.perf_counter()
        hits = self.store.find(task)
        out = {
            "ok": bool(hits),
            "agent": self.name,
            "text_he": (hits[0].summary_he if hits
                        else "אין לי רשומה על האדם הזה."),
            "people": [p.to_dict() for p in hits[:5]],
            "ms": (time.perf_counter() - t0) * 1000,
        }
        self.history.append({"task": task[:200], "ok": out["ok"],
                             "t": time.time()})
        return out

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "role": self.role, **self.store.stats()}


AGENT: Optional[RecordsAgent] = None


def get_agent(fresh: bool = False) -> RecordsAgent:
    global AGENT
    if fresh or AGENT is None:
        AGENT = RecordsAgent()
    return AGENT
