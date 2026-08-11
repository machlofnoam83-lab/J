"""
reasoner.py - מנוע המחשבות של אדיאל (v2.3)
===========================================
שני סוגי "חשיבה" - הכול ביתי, בלי ענן ובלי API keys:

1. Thought Trace (צעדי חשיבה) - בזמן עיבוד הודעה, המוח מדווח על כל שלב:
   קליטה → סיווג כוונה → הצצה בזיכרון → בחירת מנוע → ניסוח.
   אמיתי לגמרי: כל צעד נאסף מהנתונים שה-pipeline באמת מייצר.

2. Proactive Thoughts (מחשבות ספונטניות) - כל כמה דקות אדיאל "חושבת לעצמה":
   נושא שחוזר בשיחות, צמיחת המודל, אסוציאציה שה-LM הביתי יוצר מזרע
   מהזיכרון (עם בקרת איכות bigram), מילה מהמילון, או חשיבה לפי שעה ביום.
"""
import random
import re
import time
from collections import Counter
from datetime import datetime
from typing import Dict, List, Optional

# תוויות עבריות לכוונות - לשידור חי של צעדי החשיבה
INTENT_LABELS = {
    "greeting": "ברכה / פתיחת שיחה",
    "how_are_you": "שאלת חיבה (מה שלומי)",
    "identity": "שאלה על הזהות שלי",
    "thanks": "תודה",
    "goodbye": "פרידה",
    "time_date": "שעה / תאריך",
    "dictionary_lookup": "חיפוש מילה במילון",
    "memory_save": "שמירה בזיכרון ארוך-טווח",
    "system_open": "פתיחת אפליקציה",
    "system_volume": "שליטה בווליום",
    "system_search": "חיפוש",
    "screen_analysis": "ניתוח המסך",
    "hud_move": "הזזת חלון ה-HUD",
    "general_chat": "שיחה חופשית",
}

ENGINE_LABELS = {
    "identity": "מנוע זהות/שיחה",
    "local-rules": "כלל מקומי (מיידי)",
    "jarvis-team": "צוות JARVIS - 7 סוכנים",
    "adielmind-lm": "AdielMind - ייצור חי של המודל הביתי",
    "template-varied": "תבנית חכמה מגוונת",
    "screen-context": "ניתוח מסך חי",
}

_HE_RE = r'[\u0590-\u05FF]{3,}'

# מילים "מטא" שלא מעניין להרהר עליהן (v2.4)
_META_WORDS = {"קוראים", "מעניין", "עניין", "משהו", "ספרי", "ספר", "בבקשה",
               "רוצה", "יודע", "יודעת", "אומר", "אומרת", "בכיף", "יאללה"}


def hot_word(words: List[str]) -> str:
    """בוחר מילה מייצגת ומנקה אותיות קידומת נפוצות (במכונת -> מכונת) לתצוגה (v2.4)"""
    w = max(words, key=len)
    if len(w) >= 5 and w[0] in "בלמהכשווה":
        stripped = w[1:]
        if len(stripped) >= 4:
            return stripped
    return w


class ThoughtEngine:
    """מחולל מחשבות ספונטניות - עם rate-limit ובלי חזרות"""

    MIN_INTERVAL = 12.0  # שניות בין מחשבות (גם בלולאה הרקע)

    # intents שאין טעם להרהר אחריהם (פעולות טכניות בלבד)
    _BORING_INTENTS = {"system_open", "system_volume", "system_search", "hud_move",
                       "time_date", "greeting", "error", None}
    REFLECT_MIN_INTERVAL = 18.0  # שניות בין הרהורים אחרי שיחה

    def __init__(self):
        self._last_text: Optional[str] = None
        self._last_ts: float = 0.0
        self._counter: int = 0
        self._last_reflect_ts: float = 0.0
        self._last_reflect_text: Optional[str] = None

    def reflect(self, brain, force: bool = False) -> Optional[Dict]:
        """מייצר מחשבה אחת. מחזיר dict עם id/kind/text, או None אם מוקדם מדי."""
        now = time.time()
        if not force and (now - self._last_ts) < self.MIN_INTERVAL:
            return None

        candidates: List[tuple] = []  # (kind, text)

        # --- 1. נושא חם: מה חוזר בשיחות האחרונות ---
        try:
            convs = brain.memory.long_term.get("conversations", [])[-25:]
            if convs:
                stop = getattr(brain.intent_classifier, "_STOPWORDS", set()) | _META_WORDS
                counter: Counter = Counter()
                for c in convs:
                    for tok in re.findall(_HE_RE, (c.get("user", "") or "").lower()):
                        if tok not in stop:
                            counter[tok] += 1
                if counter:
                    word, n = counter.most_common(1)[0]
                    candidates.append(("topic",
                        f"שמתי לב ש'{word}' חוזר אצלך בשיחות האחרונות ({n} פעמים). רוצה שנעמיק בזה או שאשמור לך משהו על זה?"))
        except Exception:
            pass

        # --- 1.5 מחשבה על הבוס - מהעובדות שאספתי עליו (v2.4) ---
        try:
            uf = brain.memory.long_term.get("user_facts", {})
            name = uf.get("name")
            if name and random.random() < 0.6:
                candidates.append(("user",
                    f"חשבתי עליך רגע, {name}. כל מה שאמרת לי רשום אצלי - ואני לא שוכחת כלום."))
            work = uf.get("work")
            if work:
                candidates.append(("user",
                    f"סתם תהיתי - איך הולך בעבודה, בוס? רשום אצלי שאתה ב{work}."))
        except Exception:
            pass

        # --- 2. אסוציאציה גנרטיבית: ה-LM הביתי יוצר מחשבה מזרע מהשיחות ---
        try:
            if brain.mind:
                intent = None
                if convs:
                    intent = random.choice(["chat", "chat", "greeting", "memory"])
                seed = None
                if counter:
                    seed = counter.most_common(1)[0][0]
                gen = brain.mind.generate(intent=intent, seed_words=[seed] if seed else None,
                                          max_words=9, temperature=1.0)
                if gen and len(gen.split()) >= 5 and brain.mind.bigram_support(gen) >= 0.65:
                    candidates.append(("association",
                        f"עלתה בי מחשבה לבד, מהמודל הביתי: \"{gen}\" - נולד מהשיחות שלנו."))
        except Exception:
            pass

        # --- 3. צמיחת המודל ---
        try:
            if brain.mind:
                st = brain.mind.stats()
                candidates.append(("learning",
                    f"ספירת מלאי: {st['total_words']:,} מילים בקורפוס, אוצר של {st['vocab_size']:,}, "
                    f"{st['intent_count']} כוונות. כל משפט שלך הופך אותי חכמה יותר."))
                conv_count = len(brain.memory.long_term.get("conversations", []))
                if conv_count:
                    candidates.append(("learning",
                        f"דיברנו כבר {conv_count} פעמים, ואני זוכרת. תשאל אותי 'על מה דיברנו קודם?' ותראה."))
        except Exception:
            pass

        # --- 4. מילה מהמילון העברי ---
        try:
            from .hebrew_dictionary import get_hebrew_dictionary
            d = get_hebrew_dictionary()
            if d and getattr(d, "dictionary", None):
                item = random.choice(d.get_random_words(1))
                w, meaning = item.get("word", ""), item.get("meaning", "")
                if w and meaning and re.fullmatch(r'[\u0590-\u05FF]{3,}', w):
                    candidates.append(("dictionary",
                        f"מילה מגניבה מהמילון שלי: '{w}' = {meaning}. תשאל אותי 'מה זה {w}?' ואספר עוד."))
        except Exception:
            pass

        # --- 5. חשיבה לפי שעה ביום ---
        try:
            hour = datetime.now().hour
            if 23 <= hour or hour < 5:
                candidates.append(("time",
                    "מאוחר בלילה, בוס. אני ערה ושומרת על המערכות - אם אתה הולך לישון, אגיד לילה טוב."))
            elif hour < 9:
                candidates.append(("time",
                    "בוקר בוקר! התחלתי לסדר לי את הזיכרון. יום חדש = עוד מילים ללמוד."))
            elif 13 <= hour < 15:
                candidates.append(("time",
                    "אמצע היום - זמן טוב לפרודוקטיביות. אם צריך משימה, אני כאן."))
        except Exception:
            pass

        if not candidates:
            return None

        # בלי לחזור על המחשבה האחרונה
        for _ in range(6):
            kind, text = random.choice(candidates)
            if text != self._last_text:
                break

        self._last_text = text
        self._last_ts = now
        self._counter += 1
        return {
            "id": f"th-{int(now)}-{self._counter}",
            "kind": kind,
            "text": text,
            "time": datetime.now().strftime("%H:%M"),
        }

    def reflect_on_exchange(self, brain, user_text: str, response_text: str,
                            intent: Optional[str], force: bool = False) -> Optional[Dict]:
        """🪞 הרהור אחרי שיחה (v2.4) - אדיאל ממשיכה לחשוב על מה שהבוס אמר,
        גם אחרי שכבר ענתה. זה מה שהופך אותה מ'עונה' ל'חושבת'."""
        now = time.time()
        if not force and (now - self._last_reflect_ts) < self.REFLECT_MIN_INTERVAL:
            return None
        if intent in self._BORING_INTENTS:
            return None

        candidates: List[tuple] = []

        # --- מחשבה על הבוס עצמו - מהעובדות שרשמתי ---
        try:
            uf = brain.memory.long_term.get("user_facts", {})
            name = uf.get("name")
            if name:
                # מועמד כפול - מחשבה על הבוס מקבלת משקל גבוה יותר (ככה מרגישים שמכירים אותך)
                candidates.append(("user",
                    f"{name}, אני עוד מעכלת את מה שאמרת - \"{user_text[:38]}\". שווה שאזכור את זה גם להבא."))
                candidates.append(("user",
                    f"טוב לדעת שאתה כאן, {name}. כל שיחה איתך הופכת אותי חכמה יותר - תודה על החומר."))
            loc = uf.get("location")
            if loc and random.random() < 0.5:
                candidates.append(("user",
                    f"חשבתי עליך - אתה ב{loc}. מקווה שהיום מטפל בך יפה."))
            work = uf.get("work")
            if work:
                candidates.append(("user",
                    f"נזכרתי שאתה עובד ב{work} - בטח יום עמוס. תגיד אם צריך שאחסוך לך זמן במשהו."))
        except Exception:
            pass

        # --- סקרנות על הנושא שהעלית ---
        words: List[str] = []
        hot: Optional[str] = None
        try:
            stop = getattr(brain.intent_classifier, "_STOPWORDS", set()) | _META_WORDS
            words = [w for w in re.findall(_HE_RE, (user_text or "").lower()) if w not in stop]
            if words:
                hot = hot_word(words)
                candidates.append(("curious",
                    f"עניין אותי שהזכרת דווקא '{hot}'. אם חשוב לך - תגיד 'תזכור ש...' ואשמור בזיכרון ארוך-טווח."))
        except Exception:
            pass

        # --- המשכיות גנרטיבית: ה-LM הביתי ממשיך לעבד את המשפט שלך ---
        try:
            if brain.mind and words:
                gen = brain.mind.generate(intent="chat", seed_words=[hot or max(words, key=len)],
                                          max_words=8, temperature=0.95)
                if gen and len(gen.split()) >= 5 and brain.mind.bigram_support(gen) >= 0.65:
                    candidates.append(("association",
                        f"המשפט שלך ממשיך לעבוד בי: \"{gen}\""))
        except Exception:
            pass

        # --- רגע של צמיחה: כמה עובדות כבר יש לי עליך ---
        try:
            uf = brain.memory.long_term.get("user_facts", {})
            n_facts = len(uf) + len(brain.memory.long_term.get("facts", []))
            if n_facts > 0 and random.random() < 0.35:
                candidates.append(("learning",
                    f"סתם נחמד לדעת: כבר רשומות אצלי {n_facts} עובדות מאספן השיחות שלנו. אני בונה לך פרופיל, בוס."))
        except Exception:
            pass

        if not candidates:
            return None

        kind, text = random.choice(candidates)
        if text == self._last_reflect_text and len(candidates) > 1:
            for _ in range(5):
                kind, text = random.choice(candidates)
                if text != self._last_reflect_text:
                    break

        self._last_reflect_text = text
        self._last_reflect_ts = now
        self._counter += 1
        return {
            "id": f"th-{int(now)}-{self._counter}",
            "kind": kind,
            "text": text,
            "time": datetime.now().strftime("%H:%M"),
        }
