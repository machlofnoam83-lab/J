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


class ThoughtEngine:
    """מחולל מחשבות ספונטניות - עם rate-limit ובלי חזרות"""

    MIN_INTERVAL = 12.0  # שניות בין מחשבות (גם בלולאה הרקע)

    def __init__(self):
        self._last_text: Optional[str] = None
        self._last_ts: float = 0.0
        self._counter: int = 0

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
                stop = getattr(brain.intent_classifier, "_STOPWORDS", set())
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
                                          max_words=11, temperature=1.05)
                if gen and len(gen.split()) >= 4 and brain.mind.bigram_support(gen) >= 0.5:
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
                if w and meaning:
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
