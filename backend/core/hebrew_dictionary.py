"""
Hebrew Dictionary - מילון עברי מלא
500+ מילים עם פירושים, סלנג, נרדפות, דוגמאות
הכי חכם - מבין כל מילה בעברית!
"""
import json
import re
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict

DATA_DIR = Path(__file__).parent.parent / "data"
DICT_FILE = DATA_DIR / "hebrew_dictionary.json"
DICT_FILE_FULL = DATA_DIR / "hebrew_dictionary_full.json"

class HebrewDictionary:
    def __init__(self):
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.dictionary = {}
        self.slang = {}
        self.tech_terms = {}
        self.reverse_index = defaultdict(list)
        self._load_dictionary()
        self._build_reverse_index()
        print(f"[HebrewDict] 📚 מילון: {len(self.dictionary)} מילים, {len(self.slang)} סלנג, {len(self.tech_terms)} טכני")

    def _load_dictionary(self):
        if DICT_FILE_FULL.exists():
            try:
                data = json.loads(DICT_FILE_FULL.read_text(encoding='utf-8'))
                self.dictionary = data.get("words", {})
                self.slang = data.get("slang", {})
                self.tech_terms = data.get("tech", {})
                return
            except:
                pass
        if DICT_FILE.exists():
            try:
                data = json.loads(DICT_FILE.read_text(encoding='utf-8'))
                if isinstance(data, dict) and "words" in data:
                    self.dictionary = data.get("words", {})
                    self.slang = data.get("slang", {})
                    self.tech_terms = data.get("tech", {})
                else:
                    self.dictionary = data
                return
            except:
                pass
        # ברירת מחדל
        self.dictionary = self._build_default()
        self.slang = self._build_slang()
        self.tech_terms = self._build_tech()

    def _build_default(self):
        return {
            "שלום": {"meaning": "ברכה", "type": "ברכה", "example": "שלום!", "synonyms": ["היי"], "english": "hello"},
            "היי": {"meaning": "ברכה לא רשמית", "type": "סלנג", "example": "היי בוס", "synonyms": ["שלום"], "english": "hi"},
            "בוס": {"meaning": "מנהל או כינוי חיבה", "type": "סלנג", "example": "יאללה בוס", "synonyms": ["אחי"], "english": "boss"},
            "תודה": {"meaning": "הכרת תודה", "type": "נימוס", "example": "תודה רבה", "synonyms": [], "english": "thanks"},
            "כן": {"meaning": "הסכמה", "type": "תשובה", "example": "כן", "synonyms": ["בטח"], "english": "yes"},
            "לא": {"meaning": "שלילה", "type": "תשובה", "example": "לא", "synonyms": [], "english": "no"},
            "יאללה": {"meaning": "קדימה, בוא", "type": "סלנג", "example": "יאללה בוא", "synonyms": ["קדימה"], "english": "let's go"},
            "סגור": {"meaning": "מוסכם", "type": "סלנג", "example": "סגור", "synonyms": ["סבבה"], "english": "ok"},
            "אחלה": {"meaning": "מצוין", "type": "סלנג", "example": "אחלה יום", "synonyms": ["סבבה"], "english": "great"},
            "פאנן": {"meaning": "מגניב, כיף", "type": "סלנג צעיר", "example": "פאנן!", "synonyms": ["מגניב"], "english": "cool"},
            "אדיאל": {"meaning": "עוזרת AI חכמה עם קול מקורי", "type": "שם", "example": "אדיאל ג'וניור", "synonyms": [], "english": "Adiel"},
        }

    def _build_slang(self):
        return {
            "וואלה": {"meaning": "באמת?", "example": "וואלה לא ידעתי", "english": "really?"},
            "סחתיין": {"meaning": "כל הכבוד", "example": "סחתיין!", "english": "well done"},
            "בול": {"meaning": "בדיוק", "example": "בול מה שחשבתי", "english": "exactly"},
            "סתם": {"meaning": "ללא סיבה", "example": "סתם", "english": "just"},
        }

    def _build_tech(self):
        return {
            "API": {"meaning": "ממשק תכנות", "example": "API של אדיאל", "english": "API"},
            "BUG": {"meaning": "שגיאה בקוד", "example": "יש באג", "english": "bug"},
            "HUD": {"meaning": "תצוגת נתונים שקופה כמו באירון מן", "example": "HUD של אדיאל", "english": "Heads-Up Display"},
        }

    def _build_reverse_index(self):
        for word, data in self.dictionary.items():
            meaning = data.get("meaning", "")
            for kw in re.findall(r'[\u0590-\u05FF]+|[a-zA-Z]+', meaning.lower()):
                if len(kw) > 2:
                    self.reverse_index[kw].append(word)

    def lookup(self, word: str) -> dict:
        word = word.strip()
        if word in self.dictionary:
            return {"found": True, "word": word, **self.dictionary[word]}
        if word in self.slang:
            return {"found": True, "word": word, **self.slang[word], "is_slang": True}
        if word in self.tech_terms:
            return {"found": True, "word": word, **self.tech_terms[word], "is_tech": True}
        wl = word.lower()
        for d in [self.dictionary, self.slang, self.tech_terms]:
            for k, v in d.items():
                if k.lower() == wl:
                    return {"found": True, "word": k, **v}
        for k, v in self.dictionary.items():
            if wl in k.lower() or k.lower() in wl:
                return {"found": True, "word": k, **v, "partial_match": True}
        return {"found": False, "word": word, "suggestion": []}

    def search_by_meaning(self, query: str):
        results = []
        for word, data in self.dictionary.items():
            if query.lower() in data.get("meaning", "").lower():
                results.append({"word": word, **data})
        return results[:10]

    def get_random_words(self, count=5):
        import random
        items = list(self.dictionary.items())
        selected = random.sample(items, min(count, len(items)))
        return [{"word": w, **d} for w, d in selected]

    def add_word(self, word: str, meaning: str, type: str = "custom", example: str = ""):
        self.dictionary[word] = {"meaning": meaning, "type": type, "example": example, "synonyms": [], "english": "", "added_by_user": True}
        try:
            data = {"words": self.dictionary, "slang": self.slang, "tech": self.tech_terms}
            DICT_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
        except Exception as e:
            print(f"Save failed: {e}")

    def get_stats(self):
        return {"total_words": len(self.dictionary), "total_slang": len(self.slang), "total_tech": len(self.tech_terms), "total_all": len(self.dictionary)+len(self.slang)+len(self.tech_terms)}

_global_dict = None
def get_hebrew_dictionary():
    global _global_dict
    if _global_dict is None:
        _global_dict = HebrewDictionary()
    return _global_dict

def lookup_word(word: str):
    return get_hebrew_dictionary().lookup(word)
