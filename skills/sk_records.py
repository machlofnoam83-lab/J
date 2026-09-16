"""Skills for ediyel records — the people dossier.

Risk levels here follow one rule: reading a dossier is SAFE, changing one is
WRITE. Deleting is WRITE rather than CRITICAL because a dossier is a note you
took, not a permission — losing one is annoying, not dangerous. Enrolment is the
thing that grants power, and that stays CRITICAL in ``sk_vision``.

Every skill degrades to a plain refusal rather than an exception, because the
dossier file can be missing, empty, or corrupt and none of those should stop the
rest of the assistant from working.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from skills.registry import REGISTRY, SkillResult


def _store():
    from agents.records import get_store
    return get_store()


def _as_age(value: Any) -> Optional[int]:
    from agents.records import _as_age as conv
    return conv(value)


def _people_text(people: List[Dict[str, Any]], limit: int = 12) -> str:
    if not people:
        return "אין רשומות עדיין."
    lines = []
    for p in people[:limit]:
        age = p.get("age")
        rel = p.get("relationship") or ""
        bit = f"{p['name']}"
        if rel:
            bit += f" ({rel})"
        if age is not None:
            bit += f", {age}"
        if p.get("has_photo"):
            bit += ", יש תמונה"
        if p.get("face_id"):
            bit += ", מזוהה במצלמה"
        lines.append(f"• {bit}")
    if len(people) > limit:
        lines.append(f"• ועוד {len(people) - limit}")
    return "\n".join(lines)


# --------------------------------------------------------------------- query --

@REGISTRY.register(
    "records.find",
    risk="SAFE",
    description_he="חיפוש אדם ברשומות לפי שם, קרבה, תגית או מילה בסיפור",
    description_en="Find a person in the dossier store",
    required=("query",),
    triggers_he=("מי זה", "תגיד לי על", "מה אתה יודע על", "רשומה של"),
    triggers_en=("who is", "tell me about", "what do you know about"),
)
def records_find(query: str = "", **_: Any) -> SkillResult:
    t0 = time.perf_counter()
    q = str(query or "").strip()
    if not q:
        return SkillResult(ok=False, error="missing required argument 'query'",
                           skill="records.find",
                           ms=(time.perf_counter() - t0) * 1000)
    try:
        hits = _store().find(q)
    except Exception as exc:                               # pragma: no cover
        return SkillResult(ok=False, error=f"records unavailable: {exc}",
                           skill="records.find",
                           ms=(time.perf_counter() - t0) * 1000)
    ms = (time.perf_counter() - t0) * 1000
    if not hits:
        return SkillResult(
            ok=True, skill="records.find", ms=ms,
            value={"query": q, "count": 0},
            data={"text_he": f"אין לי רשומה על «{q}»."})
    top = hits[0]
    extra = "" if len(hits) == 1 else f"\n\nעוד {len(hits) - 1} התאמות."
    return SkillResult(
        ok=True, skill="records.find", ms=ms,
        value={"query": q, "count": len(hits),
               "people": [p.to_dict() for p in hits[:5]]},
        data={"text_he": top.summary_he + extra})


@REGISTRY.register(
    "records.list",
    risk="SAFE",
    description_he="רשימת האנשים ברשומות",
    description_en="List everyone in the dossier store",
    triggers_he=("מי רשום", "רשימת אנשים", "כל הרשומות"),
    triggers_en=("list people", "who is registered"),
)
def records_list(**_: Any) -> SkillResult:
    t0 = time.perf_counter()
    try:
        st = _store()
        people = st.list()
    except Exception as exc:                               # pragma: no cover
        return SkillResult(ok=False, error=f"records unavailable: {exc}",
                           skill="records.list",
                           ms=(time.perf_counter() - t0) * 1000)
    return SkillResult(
        ok=True, skill="records.list", ms=(time.perf_counter() - t0) * 1000,
        value={"count": len(people), "people": people},
        data={"text_he": _people_text(people), "stats": st.stats()})


@REGISTRY.register(
    "records.whoami",
    risk="SAFE",
    description_he="מי אני לפי הפנים — מחבר את המצלמה לרשומה",
    description_en="Who am I, joining the camera to the dossier",
    triggers_he=("מי אני", "אתה יודע מי אני"),
    triggers_en=("who am i",),
)
def records_whoami(**_: Any) -> SkillResult:
    """The join that makes the two stores worth having.

    Always allowed, even with no face: being told "I do not know you yet" is the
    thing that prompts someone to enrol, and gating it would close the only door
    an unscanned person has.
    """
    t0 = time.perf_counter()
    ms = lambda: (time.perf_counter() - t0) * 1000         # noqa: E731
    try:
        from brain.presence import get_presence
        p = get_presence().current()
    except Exception:                                      # pragma: no cover
        p = None

    if p is None or not getattr(p, "known", False):
        return SkillResult(
            ok=True, skill="records.whoami", ms=ms(),
            value={"known": False},
            data={"text_he": "אני עדיין לא יודע מי אתה. תגיד «רשום אותי בשם …» "
                             "ואחבר אותך לרשומה."})

    face_id = str(getattr(p, "identity", "") or "")
    try:
        rec = _store().by_face(face_id)
    except Exception:                                      # pragma: no cover
        rec = None
    if rec is None:
        return SkillResult(
            ok=True, skill="records.whoami", ms=ms(),
            value={"known": True, "name": p.name, "dossier": False},
            data={"text_he": f"זיהיתי אותך כ‑{p.name}, אבל אין עליך עדיין "
                             f"רשומה. אפשר להוסיף אחת."})
    return SkillResult(
        ok=True, skill="records.whoami", ms=ms(),
        value={"known": True, "name": p.name, "dossier": True,
               "person": rec.to_dict()},
        data={"text_he": rec.summary_he})


# ------------------------------------------------------------------- mutate --

@REGISTRY.register(
    "records.add",
    risk="WRITE",
    description_he="הוספת אדם לרשומות — שם, גיל, סיפור, תמונה",
    description_en="Add a person to the dossier store",
    required=("name",),
    triggers_he=("תוסיף לרשומות", "תשמור רשומה", "תתעד את"),
    triggers_en=("add a record", "remember this person"),
)
def records_add(name: str = "", story: str = "", age: Any = None,
                birthday: str = "", photo: str = "", relationship: str = "",
                role: str = "", note: str = "", phone: str = "",
                email: str = "", tags: Any = None,
                face_id: str = "", **_: Any) -> SkillResult:
    t0 = time.perf_counter()
    ms = lambda: (time.perf_counter() - t0) * 1000          # noqa: E731
    clean = str(name or "").strip()
    if not clean:
        return SkillResult(ok=False, error="missing required argument 'name'",
                           skill="records.add", ms=ms(),
                           data={"text_he": "אני צריך שם כדי לפתוח רשומה."})

    # A photo path that does not exist is a mistake, not a preference. Storing
    # it anyway would leave a dossier that claims a picture it cannot show.
    if photo:
        from pathlib import Path
        if not Path(str(photo)).is_file():
            return SkillResult(
                ok=False, error=f"photo not found: {photo}",
                skill="records.add", ms=ms(),
                data={"text_he": f"הקובץ {photo} לא נמצא, אז לא שמרתי אותו "
                                 f"כתמונה. הרשומה עצמה יכולה להישמר בלי תמונה."})

    tag_list = tags if isinstance(tags, (list, tuple)) else (
        [t.strip() for t in str(tags).split(",") if t.strip()] if tags else [])
    try:
        person = _store().add(name=clean, story=story, age=_as_age(age),
                              birthday=birthday, photo=photo,
                              relationship=relationship, role=role, note=note,
                              phone=phone, email=email, tags=tag_list,
                              face_id=face_id)
    except Exception as exc:                               # pragma: no cover
        return SkillResult(ok=False, error=f"could not save: {exc}",
                           skill="records.add", ms=ms())
    if person is None:
        return SkillResult(ok=False, error="a name is required",
                           skill="records.add", ms=ms())
    return SkillResult(
        ok=True, skill="records.add", ms=ms(), value=person.to_dict(),
        data={"text_he": f"שמרתי רשומה על {person.name}."
                         + (f" {person.summary_he}" if person.story else "")})


@REGISTRY.register(
    "records.update",
    risk="WRITE",
    description_he="עדכון רשומה קיימת",
    description_en="Update an existing dossier",
    required=("person_id",),
    triggers_he=("תעדכן את הרשומה", "תוסיף פרט"),
    triggers_en=("update the record",),
)
def records_update(person_id: str = "", **kwargs: Any) -> SkillResult:
    # The parameter must be named exactly ``kwargs``: the registry's validator
    # only lets unknown keys through when it sees that name, so ``**changes``
    # made every real update fail with "unexpected arguments".
    t0 = time.perf_counter()
    ms = lambda: (time.perf_counter() - t0) * 1000          # noqa: E731
    pid = str(person_id or "").strip()
    if not pid:
        return SkillResult(ok=False,
                           error="missing required argument 'person_id'",
                           skill="records.update", ms=ms())
    changes = dict(kwargs)
    changes.pop("person_id", None)
    if "age" in changes:
        changes["age"] = _as_age(changes.get("age"))
    try:
        person = _store().update(pid, **changes)
    except Exception as exc:                               # pragma: no cover
        return SkillResult(ok=False, error=f"could not save: {exc}",
                           skill="records.update", ms=ms())
    if person is None:
        return SkillResult(ok=False, error=f"no such person: {pid}",
                           skill="records.update", ms=ms(),
                           data={"text_he": "לא מצאתי את הרשומה הזאת."})
    return SkillResult(
        ok=True, skill="records.update", ms=ms(), value=person.to_dict(),
        data={"text_he": f"עדכנתי את הרשומה של {person.name}."})


@REGISTRY.register(
    "records.remove",
    risk="WRITE",
    description_he="מחיקת רשומה",
    description_en="Delete a dossier",
    required=("person_id",),
    triggers_he=("תמחק את הרשומה", "תשכח אותו"),
    triggers_en=("delete the record", "forget him"),
)
def records_remove(person_id: str = "", **_: Any) -> SkillResult:
    t0 = time.perf_counter()
    pid = str(person_id or "").strip()
    if not pid:
        return SkillResult(ok=False,
                           error="missing required argument 'person_id'",
                           skill="records.remove",
                           ms=(time.perf_counter() - t0) * 1000)
    try:
        st = _store()
        person = st.get(pid)
        ok = st.remove(pid)
    except Exception as exc:                               # pragma: no cover
        return SkillResult(ok=False, error=f"could not delete: {exc}",
                           skill="records.remove",
                           ms=(time.perf_counter() - t0) * 1000)
    if not ok:
        return SkillResult(ok=False, error=f"no such person: {pid}",
                           skill="records.remove",
                           ms=(time.perf_counter() - t0) * 1000,
                           data={"text_he": "לא מצאתי את הרשומה הזאת."})
    return SkillResult(
        ok=True, skill="records.remove",
        ms=(time.perf_counter() - t0) * 1000, value={"removed": pid},
        data={"text_he": f"מחקתי את הרשומה של {person.name if person else pid}."})


@REGISTRY.register(
    "records.link",
    risk="WRITE",
    description_he="חיבור פנים שנרשמו לרשומה קיימת",
    description_en="Attach an enrolled face to a dossier",
    required=("person_id", "face_id"),
    triggers_he=("תחבר את הפנים", "תשייך לרשומה"),
    triggers_en=("link the face",),
)
def records_link(person_id: str = "", face_id: str = "", **_: Any) -> SkillResult:
    t0 = time.perf_counter()
    ms = lambda: (time.perf_counter() - t0) * 1000          # noqa: E731
    pid, fid = str(person_id or "").strip(), str(face_id or "").strip()
    if not pid or not fid:
        return SkillResult(ok=False,
                           error="both 'person_id' and 'face_id' are required",
                           skill="records.link", ms=ms())
    try:
        person = _store().link_face(pid, fid)
    except Exception as exc:                               # pragma: no cover
        return SkillResult(ok=False, error=f"could not link: {exc}",
                           skill="records.link", ms=ms())
    if person is None:
        return SkillResult(ok=False, error=f"no such person: {pid}",
                           skill="records.link", ms=ms(),
                           data={"text_he": "לא מצאתי את הרשומה הזאת."})
    return SkillResult(
        ok=True, skill="records.link", ms=ms(), value=person.to_dict(),
        data={"text_he": f"חיברתי את הפנים לרשומה של {person.name}. "
                         f"מעכשיו כשהמצלמה תזהה אותו — אדע מי זה."})
