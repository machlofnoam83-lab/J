"""Vision skills — how the brain asks what the camera sees.

Before this module the brain had no way to consult the camera at all. ``vision/``
ran, ``FaceGate`` pushed a level into the firewall, and the reasoning engine had
no idea anybody was in the room: it could not answer "who is in front of you",
could not greet by name, and could not explain a refusal in terms the user could
act on.

These skills read ``brain.presence``, which the server feeds on every camera
frame. They never run detection themselves — a question about the room should
cost microseconds, not another pass over a frame — and they never grant anything.
Reporting who is present and changing what is permitted are different jobs, and
only ``vision.faces.FaceGate`` does the second one.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any, Dict, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from brain.presence import get_presence  # noqa: E402
from skills.registry import REGISTRY, SkillResult  # noqa: E402


def _tracker():
    return get_presence()


def _gallery() -> Optional[Any]:
    """The face store, if one has been opened. Never opens one itself."""
    try:
        from core.server import get_faces
        store, _gate = get_faces()
        return store
    except Exception:
        return None


@REGISTRY.register(
    "vision.who", risk="SAFE", agent="jarvis",
    description_he="אומר מי עומד מול המצלמה עכשיו, ומה רמת ההרשאה שלו",
    description_en="Reports who is in front of the camera right now and their level",
    triggers_he=("מי מולי", "מי זה", "מי אתה רואה", "מי נמצא כאן", "מי מול המצלמה"),
    triggers_en=("who is this", "who is in front of you", "who do you see"),
)
def vision_who() -> SkillResult:
    p = _tracker().current()
    if p.known and p.name and p.name != "—":
        role_he = "הבעלים" if p.is_owner else "אורח"
        text = (f"{p.name} מול המצלמה — {role_he}, הרשאה {p.level}, "
                f"ביטחון {round(p.confidence * 100)}%.")
    elif p.people:
        text = ("יש מישהו מול המצלמה אבל לא זיהיתי אותו. "
                "אפשר לרשום אותו דרך פאנל הראייה.")
    else:
        text = "אף אחד לא מול המצלמה כרגע."
    if p.stale and (p.known or p.people):
        text += f" (התצפית בת {round(p.age)} שניות — ייתכן שכבר לא מעודכנת.)"
    return SkillResult(ok=True, value=text, data={
        "presence": p.to_dict(), "text_he": text})


@REGISTRY.register(
    "vision.scene", risk="SAFE", agent="jarvis",
    description_he="מתאר מה המצלמה רואה: כמה אנשים, תאורה, מרחק, ואיכות הזיהוי",
    description_en="Describes the scene: how many people, lighting, distance, quality",
    triggers_he=("מה אתה רואה", "מה יש מולך", "תאר את החדר", "מה המצלמה רואה"),
    triggers_en=("what do you see", "describe the scene"),
)
def vision_scene() -> SkillResult:
    p = _tracker().current()
    text = p.summary_he or "אין לי פריים מהמצלמה — המצלמה כבויה או שעדיין לא נקלט כלום."
    return SkillResult(ok=True, value=text, data={
        "presence": p.to_dict(), "text_he": text,
        "people": p.people, "quality": round(p.quality, 3),
        "lighting_liveness": p.liveness})


@REGISTRY.register(
    "vision.gallery", risk="SAFE", agent="jarvis",
    description_he="מפרט מי רשום לזיהוי פנים, מי הבעלים, וכמה דגימות לכל אחד",
    description_en="Lists enrolled faces, who the owner is, and sample counts",
    triggers_he=("מי רשום", "מי אתה מכיר", "רשימת הפנים", "מי הבעלים"),
    triggers_en=("who is enrolled", "list faces", "who is the owner"),
)
def vision_gallery() -> SkillResult:
    store = _gallery()
    if store is None:
        return SkillResult(ok=False, error="גלריית הפנים לא זמינה כרגע",
                           data={"text_he": "גלריית הפנים לא זמינה כרגע."})
    people = store.list()
    owner = store.owner()
    if not people:
        text = ("אף אחד לא רשום לזיהוי פנים. ההרשמה הראשונה תהפוך לבעלים "
                "עם הרשאות מלאות.")
        return SkillResult(ok=True, value=text, data={"text_he": text, "people": [],
                                                      "owner": None})
    lines = []
    for per in people:
        mark = "★ " if per.get("role") == "owner" else ""
        lines.append(f"{mark}{per.get('name')} — {per.get('level')}, "
                     f"{per.get('samples')} דגימות")
    text = "רשומים לזיהוי: " + "; ".join(lines) + "."
    if owner is not None:
        text += f" הבעלים הוא {owner.name}."
    return SkillResult(ok=True, value=text, data={
        "text_he": text, "people": people,
        "owner": owner.to_dict() if owner is not None else None,
        "count": len(people)})


@REGISTRY.register(
    "vision.permission", risk="SAFE", agent="jarvis",
    description_he="מסביר איזו הרשאה פעילה עכשיו ולמה — לפי מי שמול המצלמה",
    description_en="Explains which permission level is active right now and why",
    triggers_he=("מה מותר לי", "מה ההרשאה", "למה אתה לא מרשה", "איזו רמה"),
    triggers_en=("what am I allowed", "why not allowed", "what level"),
)
def vision_permission(risk: str = "WRITE") -> SkillResult:
    tr = _tracker()
    p = tr.current()
    risk = str(risk or "WRITE").upper()
    allowed = tr.allows(risk)
    why = tr.explain(risk)
    if allowed:
        text = (f"ההרשאה הפעילה היא {p.level}, ו‑{risk} מותר בה."
                if p.known else
                f"אף אחד לא מזוהה מול המצלמה; {risk} עדיין מותר כי חומת האש "
                f"מוגדרת ל‑{p.level}.")
    else:
        text = why or f"{risk} לא מותר ברמה הנוכחית ({p.level})."
    return SkillResult(ok=True, value=text, data={
        "text_he": text, "allowed": allowed, "requested": risk,
        "active_level": p.level, "reason_he": why,
        "presence": p.to_dict()})


# ──────────────────────────────────────────────────────────── enrolment ──
# This is the one vision skill that changes the world rather than describing it,
# and it is CRITICAL for a concrete reason: the first face into an empty gallery
# becomes the owner with CRITICAL, so "register me" is not a convenience, it is
# the granting of every permission in the system. The firewall therefore asks a
# human in the HUD before it runs, and refuses outright while the firewall sits
# at SAFE — which is exactly the state an unidentified stranger produces.

_NAME_STOP = frozenset({
    "אותי", "אותו", "אותה", "את", "של", "לי", "זה", "הזה", "הזו", "בבקשה",
    "פנים", "פרצוף", "מצלמה", "גלריה", "רשום", "תרשום", "תכיר", "תוסיף",
})


def _clean_name(raw: str) -> str:
    """A name is a word a person would actually be called. Never guess one."""
    for tok in re.split(r"[\s,.;:!?\-]+", str(raw or "").strip()):
        tok = tok.strip("״׳\"'")
        if not tok or tok.lower() in _NAME_STOP:
            continue
        if not re.search(r"[A-Za-z\u0590-\u05FF]", tok):
            continue
        return tok[:24]
    return ""


@REGISTRY.register(
    "vision.enroll", risk="CRITICAL", agent="jarvis",
    description_he="רושם את הפנים שמופיעות עכשיו מול המצלמה לזיהוי — ההרשמה הראשונה הופכת לבעלים",
    description_en="Enrols the face currently in front of the camera; the first enrolment becomes the owner",
    required=("name",),
    triggers_he=("רשום אותי", "תכיר אותי", "תרשום את הפנים", "תוסיף אותי לגלריה"),
    triggers_en=("enroll me", "register my face", "remember my face"),
)
def vision_enroll(name: str = "", note: str = "") -> SkillResult:
    store = _gallery()
    if store is None:
        return SkillResult(ok=False, error="גלריית הפנים לא זמינה",
                           data={"text_he": "גלריית הפנים לא זמינה כרגע."})

    p = _tracker().current()

    # Guard 1 — there has to be a face. The vector comes from the frame the
    # camera actually saw; this skill never runs detection itself.
    if not p.has_face:
        text = ("אין לי פנים לרשום — המצלמה לא קלטה פרצוף בפריים האחרון, "
                "או שהתצפית כבר ישנה מדי.")
        return SkillResult(ok=False, error="no face available to enrol",
                           data={"text_he": text})

    # Guard 2 — never enrol something the liveness check called an artefact.
    # Enrolling a photograph of a person would plant their identity in the
    # gallery permanently, which is far worse than a single failed unlock.
    if p.liveness == "suspect":
        text = ("לא רשמתי. הפריים נראה כמו מסך או תמונה ולא כמו פנים חיות, "
                "ורישום של תצלום היה שותל זהות קבועה בגלריה.")
        return SkillResult(ok=False, error="liveness suspect — refusing to enrol",
                           data={"text_he": text, "withheld": "liveness"})

    # Guard 3 — a name is a fact about the world, not something to invent.
    clean = _clean_name(name)
    if not clean:
        text = ("אני צריך שם כדי לרשום. תגיד למשל «רשום אותי בשם דנה», "
                "ואני לא ממציא שם בעצמי.")
        return SkillResult(ok=False, error="no usable name given",
                           data={"text_he": text})

    empty_before = not store.list()
    try:
        person = store.enroll(clean, p.vector, level="SAFE",
                              note=str(note or "")[:120])
    except Exception as exc:
        return SkillResult(ok=False, error=f"enrolment failed: {exc}",
                           data={"text_he": f"הרישום נכשל: {exc}"})

    # Refresh the firewall and the brain together, so the very next question is
    # answered as the person who was just enrolled rather than as a stranger.
    try:
        from core.server import get_faces
        _store, gate = get_faces()
        gate.observe(None)
        from vision.faces import Match
        _tracker().observe(Match(person.id, person.name, person.level,
                                 p.confidence, 0.0, 1, True, role=person.role,
                                 vector=p.vector))
    except Exception:                                  # pragma: no cover
        pass

    # Open the dossier in the same breath. The gallery now holds a vector and a
    # name; the records store holds the story. Linking them here means the user
    # never has to copy an id by hand, and "who is this" can be answered with
    # more than a label the moment a face is recognised.
    #
    # Best-effort by design: a records store that will not open must not undo an
    # enrolment that already succeeded.
    dossier = None
    try:
        from agents.records import get_store
        rec_store = get_store()
        existing = rec_store.by_face(person.id)
        if existing is None:
            dossier = rec_store.add(name=person.name, face_id=person.id,
                                    relationship="owner" if person.role == "owner" else "")
    except Exception:                                  # pragma: no cover
        dossier = None

    if person.role == "owner":
        text = (f"נרשמת כ‑{person.name} — וזו ההרשמה הראשונה, אז אתה הבעלים "
                f"עם הרשאות מלאות ({person.level}).")
    else:
        text = (f"נרשמת כ‑{person.name}. ההרשאה שלך {person.level} — "
                f"הבעלים הוא {(store.owner() or type('', (), {'name': '—'})).name}.")
    if dossier is not None:
        text += " פתחתי גם רשומה — אפשר להוסיף לה גיל, סיפור ותמונה."
    return SkillResult(ok=True, value=text, data={
        "text_he": text, "person": person.to_dict(),
        "is_owner": person.role == "owner",
        "was_empty_gallery": empty_before,
        "level": person.level, "samples": len(person.vectors),
        "dossier_id": (dossier.id if dossier is not None else None)})
