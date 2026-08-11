"""
AdielMind - מודל שפה גנרטיבי ביתי, נבנה מאפס (v1.0)
ללא שירותי ענן, ללא API keys, ללא torch - pure Python.

איך זה עובד:
- Markov chain מסדר 1-3 עם interpolation (uni/bi/tri-grams)
- מותנה ב-intent: מודל לכל כוונה (greeting, chat, knowledge...) + מודל גלובלי
- דגימה עם temperature + nucleus (top-p) + ענישת חזרות
- מתאמן מ: תבניות האישיות, המילון העברי, וכל שיחה אמיתית (online!)
- נשמר לדיסק ונטען מחדש - אדיאל ממשיכה להיות חכמה יותר מפעם לפעם
"""
import os
import re
import json
import math
import random
from collections import Counter, defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
LM_FILE = os.path.join(DATA_DIR, "adiel_lm.json")

# משקלי interpolation - כמה כל רמה תורמת לברירת המילה הבאה
W_TRI, W_BI, W_UNI = 6.0, 3.0, 1.0
INTENT_BOOST = 5.0       # כמה מודל ה-intent מוגזם יחסית לגלובלי
GLOBAL_WEIGHT_INTENTED = 0.7  # כשיש intent - הגלובלי מקבל משקל נמוך יותר (פחות רעש מילון)


class WordTokenizer:
    """טוקנייזר מילים עברי/אנגלי - שומר סלנג, מנקה רעש"""
    WORD_RE = re.compile(r'[א-ת]+[׳״]?[א-ת]*|[a-zA-Z]+|\d+')

    @classmethod
    def tokenize(cls, text: str) -> List[str]:
        text = (text or "").strip().lower()
        # סמני פיסוק הופכים לטוקן סוף-משפט כדי שהמודל ידע לעצור
        text = re.sub(r'[.!?…]+', ' . ', text)
        tokens = cls.WORD_RE.findall(text)
        return tokens


class NGramCounts:
    """מונה n-grams למודל אחד (גלובלי או per-intent)"""
    __slots__ = ("uni", "bi", "tri", "total_words", "starters")

    def __init__(self):
        self.uni: Counter = Counter()
        self.bi: Dict[str, Counter] = defaultdict(Counter)
        self.tri: Dict[str, Counter] = defaultdict(Counter)  # מפתח: "w1 w2"
        self.starters: Counter = Counter()  # מילים שפותחות משפטים
        self.total_words = 0

    def add_sentence(self, tokens: List[str]):
        if not tokens:
            return
        self.starters[tokens[0]] += 1
        seq = tokens
        for i, w in enumerate(seq):
            self.uni[w] += 1
            self.total_words += 1
            if i >= 1:
                self.bi[seq[i - 1]][w] += 1
            if i >= 2:
                self.tri[f"{seq[i - 2]} {seq[i - 1]}"][w] += 1

    # --- Persistence ---
    def to_dict(self) -> Dict:
        return {
            "uni": dict(self.uni),
            "bi": {k: dict(v) for k, v in self.bi.items()},
            "tri": {k: dict(v) for k, v in self.tri.items()},
            "starters": dict(self.starters),
            "total_words": self.total_words,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "NGramCounts":
        m = cls()
        m.uni = Counter(d.get("uni", {}))
        m.bi = defaultdict(Counter, {k: Counter(v) for k, v in d.get("bi", {}).items()})
        m.tri = defaultdict(Counter, {k: Counter(v) for k, v in d.get("tri", {}).items()})
        m.starters = Counter(d.get("starters", {}))
        m.total_words = d.get("total_words", 0)
        return m


class AdielLM:
    """המודל הגנרטיבי הביתי של אדיאל"""

    VERSION = 1

    def __init__(self, path: str = LM_FILE):
        self.path = path
        self.global_model = NGramCounts()
        self.intent_models: Dict[str, NGramCounts] = {}
        self.trained_at: Optional[str] = None
        self.total_sentences = 0
        self.vocab_size = 0
        os.makedirs(os.path.dirname(self.path), exist_ok=True)

    # ---------- אימון ----------
    def add_text(self, text: str, intent: str = "chat"):
        """הוספת טקסט לקורפוס - לימוד אונליין מכל שיחה"""
        tokens = WordTokenizer.tokenize(text)
        if len(tokens) < 2:
            return
        self.global_model.add_sentence(tokens)
        self.intent_models.setdefault(intent, NGramCounts()).add_sentence(tokens)
        self.total_sentences += 1
        self.vocab_size = len(self.global_model.uni)

    def add_corpus(self, corpus: List[Tuple[str, str]]):
        """corpus = list of (text, intent)"""
        for text, intent in corpus:
            self.add_text(text, intent)
        self.trained_at = datetime.now().isoformat()

    # ---------- דגימה ----------
    def _candidate_pool(self, w1: Optional[str], w2: Optional[str], intent: Optional[str]) -> Counter:
        """בריכת מועמדים למילה הבאה עם ציונים משוקללים"""
        pool: Counter = Counter()

        def merge(model: NGramCounts, boost: float):
            if w1 and w2:
                tri = model.tri.get(f"{w1} {w2}")
                if tri:
                    for w, c in tri.items():
                        pool[w] += c * W_TRI * boost
            if w2:
                bi = model.bi.get(w2)
                if bi:
                    for w, c in bi.items():
                        pool[w] += c * W_BI * boost
            if not pool:
                for w, c in model.uni.most_common(200):
                    pool[w] += c * W_UNI * 0.01 * boost

        has_intent = bool(intent and intent in self.intent_models)
        merge(self.global_model, GLOBAL_WEIGHT_INTENTED if has_intent else 1.0)
        if has_intent:
            merge(self.intent_models[intent], INTENT_BOOST)
        return pool

    @staticmethod
    def _sample(pool: Counter, temperature: float, used: Counter, top_p: float = 0.95) -> Optional[str]:
        if not pool:
            return None
        items = list(pool.items())
        # temperature + ענישת חזרות
        weights = []
        for w, c in items:
            c = c / max(temperature, 0.05)
            if used[w] >= 2:
                c *= 0.15
            elif used[w] == 1:
                c *= 0.6
            weights.append(c)
        # nucleus: נשמור את המועמדים המובילים שמכסים top_p מהמסה
        total = sum(weights)
        if total <= 0:
            return None
        ranked = sorted(zip(items, weights), key=lambda x: x[1], reverse=True)
        kept, acc = [], 0.0
        for (w, _), wt in ranked:
            kept.append((w, wt))
            acc += wt
            if acc / total >= top_p:
                break
        kept_total = sum(wt for _, wt in kept)
        r = random.random() * kept_total
        acc = 0.0
        for w, wt in kept:
            acc += wt
            if r <= acc:
                return w
        return kept[-1][0]

    def generate(self, intent: Optional[str] = None, seed_words: Optional[List[str]] = None,
                 max_words: int = 22, temperature: float = 0.9) -> str:
        """ייצור משפט/ים - עם נביאה מההקשר של המשתמש אם אפשר"""
        if self.global_model.total_words == 0:
            return ""

        # נקד: מילים אחרונות של המשתמש שקיימות בקורפוס
        w1 = w2 = None
        if seed_words:
            seeds = [w for w in WordTokenizer.tokenize(" ".join(seed_words)) if w in self.global_model.uni]
            if len(seeds) >= 2:
                w1, w2 = seeds[-2], seeds[-1]
            elif seeds:
                w2 = seeds[-1]

        if w2 is None:
            # התחלה רנדומלית מפתיחות משפטים ידועות
            starters = self.global_model.starters
            if intent and intent in self.intent_models and self.intent_models[intent].starters:
                starters = self.intent_models[intent].starters + starters
            if starters:
                w2 = random.choices(list(starters.keys()), weights=list(starters.values()), k=1)[0]

        if w2 is None:
            return ""

        out: List[str] = [] if seed_words and (w1 or w2) else [w2]
        used: Counter = Counter(out)
        for _ in range(max_words):
            pool = self._candidate_pool(w1, w2, intent)
            nxt = self._sample(pool, temperature, used)
            if not nxt:
                break
            if nxt == ".":
                break
            # עצירה על לופ של טרי-גרם
            if len(out) >= 5 and out[-3:] == ([w1, w2, nxt] if w1 else [w2, nxt]):
                break
            out.append(nxt)
            used[nxt] += 1
            w1, w2 = w2, nxt

        text = " ".join(out).strip()
        if text and not text.endswith((".", "!", "?", ":")):
            text += random.choice([".", "!", ""])
        return text

    # ---------- סטטיסטיקה / שמירה ----------
    def bigram_support(self, text: str) -> float:
        """אחוז הביגרמים בטקסט שקיימים בקורפוס (count>=2) - מדד איכות לייצור"""
        tokens = WordTokenizer.tokenize(text)
        if len(tokens) < 2:
            return 0.0
        supported = 0
        for i in range(1, len(tokens)):
            c = self.global_model.bi.get(tokens[i - 1], {}).get(tokens[i], 0)
            if c >= 2:
                supported += 1
        return supported / (len(tokens) - 1)

    def stats(self) -> Dict:
        return {
            "version": self.VERSION,
            "total_words": self.global_model.total_words,
            "vocab_size": len(self.global_model.uni),
            "total_sentences": self.total_sentences,
            "intents": sorted(self.intent_models.keys()),
            "intent_count": len(self.intent_models),
            "trained_at": self.trained_at,
        }

    def save(self):
        try:
            data = {
                "version": self.VERSION,
                "trained_at": self.trained_at or datetime.now().isoformat(),
                "total_sentences": self.total_sentences,
                "global": self.global_model.to_dict(),
                "intents": {k: m.to_dict() for k, m in self.intent_models.items()},
            }
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)
        except Exception as e:
            print(f"[AdielLM] Save failed: {e}")

    def load(self) -> bool:
        if not os.path.exists(self.path):
            return False
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if data.get("version") != self.VERSION:
                return False
            self.global_model = NGramCounts.from_dict(data.get("global", {}))
            self.intent_models = {k: NGramCounts.from_dict(v) for k, v in data.get("intents", {}).items()}
            self.total_sentences = data.get("total_sentences", 0)
            self.trained_at = data.get("trained_at")
            self.vocab_size = len(self.global_model.uni)
            return self.global_model.total_words > 0
        except Exception as e:
            print(f"[AdielLM] Load failed: {e}")
            return False


def build_default_corpus() -> List[Tuple[str, str]]:
    """קורפוס הבסיס - אישיות + מילון + ידע גנרי. הכול ביתי!"""
    corpus: List[Tuple[str, str]] = []

    # 1) תבניות האישיות - לכל אחת ה-intent שלה
    try:
        from .personality import LOCAL_RESPONSES
        intent_map = {
            "greeting": "greeting", "greeting_returning": "greeting",
            "how_are_you": "howareyou", "acknowledge": "ack",
            "hud_dock": "hud", "hud_center": "hud", "hud_hide": "hud",
            "screen_analyzing": "screen", "screen_code": "screen", "screen_error": "screen",
            "memory_save": "memory", "memory_remember": "memory",
            "help": "chat", "thanks": "thanks", "goodbye": "goodbye",
            "fallback_chat": "chat", "follow_up": "chat", "learning_proposal": "learning",
            "error": "error",
        }
        for key, responses in LOCAL_RESPONSES.items():
            intent = intent_map.get(key, "chat")
            for r in responses:
                # ננקה placeholders כדי שלא יווצרו טוקנים שבורים
                clean = re.sub(r'\{[a-z_]+\}', '', r).strip()
                if clean:
                    corpus.append((clean, intent))
    except Exception as e:
        print(f"[AdielLM] Personality corpus failed: {e}")

    # 2) המילון העברי - משפטי ידע אמיתיים
    try:
        from .hebrew_dictionary import get_hebrew_dictionary
        hd = get_hebrew_dictionary()
        words = getattr(hd, "dictionary", None) or getattr(hd, "words", {})
        count = 0
        if isinstance(words, dict):
            for word, entry in words.items():
                if isinstance(entry, dict):
                    meaning = entry.get("meaning", "")
                    example = entry.get("example", "")
                else:
                    meaning, example = str(entry), ""
                if meaning:
                    corpus.append((f"{word} זה {meaning}", "knowledge"))
                    corpus.append((f"המילה {word} פירושה {meaning}", "knowledge"))
                    count += 1
                if example:
                    corpus.append((example, "knowledge"))
                if count >= 400:  # מספיק לבסיס
                    break
    except Exception as e:
        print(f"[AdielLM] Dictionary corpus skipped: {e}")

    # 3) ידע שיחתי גנרי - זרימה טבעית
    generic = [
        ("מה קורה בוס אני כאן אם צריך משהו", "greeting"),
        ("איך אני יכולה לעזור לך היום", "chat"),
        ("ספר לי עוד על זה זה נשמע מעניין", "chat"),
        ("אני זוכרת את כל מה שאתה אומר לי", "memory"),
        ("אתה יכול לבקש ממני לפתוח אפליקציות לחפש בגוגל ולקרוא את המסך", "chat"),
        ("אני לומדת מילים חדשות מכל שיחה שלנו", "learning"),
        ("בוא נעשה את זה ביחד צעד אחר צעד", "chat"),
        ("יש לי רעיון טוב בשבילך בוס", "chat"),
        ("תגיד לי מה אתה צריך ואני אמצא דרך", "chat"),
        ("כל יום איתך אני נהיית חכמה יותר", "learning"),
        ("אני מנתחת את הנתונים וחוזרת אליך עם תשובה", "chat"),
        ("זו שאלה טובה תן לי רגע לחשוב על זה", "chat"),
        ("אפשר גם לכתוב לי במקלדת אם אין מיקרופון", "chat"),
        ("אני עובדת במלוא העוצמה כל המערכות ירוקות", "ack"),
        ("סגור בוס אני מטפלת בזה עכשיו", "ack"),
    ]
    corpus.extend(generic)

    return corpus


# Singleton
_lm_instance: Optional[AdielLM] = None


def get_adiel_lm() -> AdielLM:
    global _lm_instance
    if _lm_instance is None:
        _lm_instance = AdielLM()
        if _lm_instance.load():
            s = _lm_instance.stats()
            print(f"[AdielLM] 🧠 מודל נטען מהדיסק: {s['total_words']:,} מילים, vocab {s['vocab_size']}, {s['intents']} — ממשיכה ללמוד!")
        else:
            corpus = build_default_corpus()
            _lm_instance.add_corpus(corpus)
            _lm_instance.save()
            s = _lm_instance.stats()
            print(f"[AdielLM] 🧠 מודל חדש אומן מאפס: {s['total_words']:,} מילים, vocab {s['vocab_size']}, {s['intent_count']} כוונות")
    return _lm_instance
