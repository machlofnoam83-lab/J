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
