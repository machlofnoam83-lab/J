"""
Learning Engine - Adiel Learns & Remembers
מערכת למידה מתמשכת - זיכרון חכם + אוצר מילים מתעדכן + אינטנטים חדשים

- כל שיחה הופכת לחכמה יותר
- לומדת מילים חדשות, סלנג, העדפות
- זוכרת עובדות, פרויקטים, אנשים
- מציעה עדכונים אבל לא מעדכנת בלי אישור
"""
import os
import re
import json
import math
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

DATA_DIR = Path(__file__).parent.parent / "data"
VOCAB_FILE = DATA_DIR / "vocabulary.json"
LEARNED_INTENTS_FILE = DATA_DIR / "learned_intents.json"
USER_PROFILE_FILE = DATA_DIR / "user_profile.json"
CONVERSATION_SUMMARY_FILE = DATA_DIR / "conversation_summaries.json"

class VocabularyManager:
    """מנהל אוצר מילים שגדל עם הזמן"""
    def __init__(self):
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.vocab = self._load_json(VOCAB_FILE, {
            "words": {},  # word -> {meaning, learned_at, usage_count, example}
            "slang": {},  # slang -> {meaning, formal_equivalent}
            "custom_commands": {},  # user custom phrases -> intent
            "total_learned": 0
        })

    def _load_json(self, path: Path, default: Dict):
        if path.exists():
            try:
                return json.loads(path.read_text(encoding='utf-8'))
            except:
                return default
        return default

    def _save(self):
        try:
            VOCAB_FILE.write_text(json.dumps(self.vocab, ensure_ascii=False, indent=2), encoding='utf-8')
        except Exception as e:
            print(f"[Vocab] Save failed: {e}")

    def learn_word(self, word: str, meaning: str, example: str = "", source: str = "user") -> Dict:
        """לומד מילה חדשה - אבל מחזיר הצעה לאישור"""
        word = word.strip()
        if len(word) < 2:
            return None
        
        # אם המילה כבר קיימת, רק עדכן שימוש
        if word in self.vocab["words"]:
            self.vocab["words"][word]["usage_count"] = self.vocab["words"][word].get("usage_count", 0) + 1
            self.vocab["words"][word]["last_seen"] = datetime.now().isoformat()
            self._save()
            return None  # אין צורך בהצעה, כבר יודעת

        # צור הצעה ללמידה חדשה
        proposal = {
            "id": f"vocab_{word}_{int(datetime.now().timestamp())}",
            "type": "vocabulary",
            "word": word,
            "meaning": meaning,
            "example": example,
            "source": source,
            "learned_at": datetime.now().isoformat(),
            "description_he": f"למדתי מילה חדשה: '{word}' = '{meaning}'. דוגמה: '{example}'. רוצה שאזכור את זה?",
            "risk": "low",
            "auto_approve": False
        }
        return proposal

    def approve_word(self, word: str, meaning: str, example: str = ""):
        """מאשר למידת מילה חדשה - רק אחרי אישור משתמש"""
        self.vocab["words"][word] = {
            "meaning": meaning,
            "example": example,
            "learned_at": datetime.now().isoformat(),
            "usage_count": 1,
            "approved": True
        }
        self.vocab["total_learned"] += 1
        self._save()
        print(f"[Vocab] ✅ למדתי מילה חדשה: {word} = {meaning}")

    def detect_unknown_words(self, text: str, known_words: set) -> List[str]:
        """מזהה מילים לא מוכרות - רק אם נראה כמו סלנג חדש, לא כל מילה"""
        # רק אם יש אינדיקציה לסלנג או בקשת למידה מפורשת
        # אל תזהה אוטומטית בכל שיחה כדי לא להציף
        if not any(kw in text for kw in ["תזכור", "תזכרי", "תלמד", "פירוש", "אומר", "סלנג"]):
            # אם אין בקשת למידה מפורשת, אל תחפש מילים לא מוכרות אוטומטית
            # רק אם המילה נראית כמו סלנג חריג (3+ פעמים אותה מילה חדשה?)
            return []
        
        tokens = re.findall(r'[\u0590-\u05FF]{2,}|[a-zA-Z]{3,}', text)
        unknown = []
        for tok in tokens:
            tok_lower = tok.lower()
            if tok_lower not in known_words and len(tok) > 3:  # רק מילים ארוכות יותר מ-3
                common_ignore = {"שלום", "תודה", "בבקשה", "אדיאל", "ג'וניור", "בוס", "תזכור", "תזכרי", "תלמד", "תלמדי", "פירוש", "אומר", "סלנג"}
                if tok not in common_ignore and tok not in self.vocab["words"]:
                    # בדוק אם זה לא פועל נפוץ
                    if tok not in ["קוראים", "עובד", "אוהב", "שונא", "גר", "עובדת", "אוהבת"]:
                        unknown.append(tok)
        return list(set(unknown))[:1]  # מקסימום 1 בכל פעם

    def get_known_words_set(self) -> set:
        """מחזיר סט מילים מוכרות - בסיס גדול כדי לא לזהות מילים נפוצות כלא מוכרות"""
        base_hebrew = set([
            # מילות יסוד
            "אני", "אתה", "את", "הוא", "היא", "אנחנו", "אתם", "אתן", "הם", "הן",
            "שלום", "תודה", "בבקשה", "סליחה", "כן", "לא", "מה", "מי", "איך", "למה", "כמה", "איפה", "מתי", "איזה",
            "בוס", "יאללה", "סגור", "על", "זה", "אחלה", "סבבה", "בקטנה", "תודה", "בכיף",
            # פעלים נפוצים
            "קוראים", "עובד", "עובדת", "אוהב", "אוהבת", "שונא", "שונאת", "גר", "גרה", "לומד", "לומדת",
            "עושה", "רוצה", "יכול", "צריך", "יש", "אין", "היה", "הייתה", "יהיה",
            "תזכור", "תזכרי", "תלמד", "תלמדי", "תפתח", "תפתחי", "תחפש", "תסגור",
            # עצמים
            "פרויקט", "עבודה", "בית", "משפחה", "חבר", "חברה", "כלב", "חתול", "ילד", "ילדה",
            "מחשב", "טלפון", "קוד", "שגיאה", "מסך", "חלון", "דפדפן", "גוגל", "כרום",
            "אדיאל", "ג'וניור", "גוניור", "זוכר", "זוכרת", "יודע", "יודעת", "עלי", "שלי",
            "פיצה", "אוכל", "מים", "קפה", "סרט", "מוזיקה", "שיר",
            "היום", "מחר", "אתמול", "עכשיו", "בוקר", "צהריים", "ערב", "לילה",
            "טוב", "רע", "גדול", "קטן", "חדש", "ישן", "יפה", "מגניב",
        ])
        learned = set(self.vocab["words"].keys())
        return base_hebrew.union(learned)


class UserProfile:
    """פרופיל משתמש שגדל ונהיה חכם יותר"""
    def __init__(self):
        self.profile = self._load_json(USER_PROFILE_FILE, {
            "name": None,
            "nickname": None,
            "language_style": "casual",  # casual / formal / slang-heavy
            "preferences": {},  # e.g., {"ide": "vscode", "music": "spotify"}
            "projects": [],  # פרויקטים שהבוס עובד עליהם
            "important_people": {},  # אנשים חשובים
            "facts": [],  # עובדות כלליות
            "interaction_count": 0,
            "first_seen": datetime.now().isoformat(),
            "last_seen": datetime.now().isoformat(),
            "learned_from_user": []  # מה למדנו ממנו
        })

    def _load_json(self, path: Path, default: Dict):
        if path.exists():
            try:
                return json.loads(path.read_text(encoding='utf-8'))
            except:
                return default
        return default

    def save(self):
        try:
            self.profile["last_seen"] = datetime.now().isoformat()
            USER_PROFILE_FILE.write_text(json.dumps(self.profile, ensure_ascii=False, indent=2), encoding='utf-8')
        except Exception as e:
            print(f"[Profile] Save failed: {e}")

    def extract_facts(self, user_text: str) -> List[Dict]:
        """מחלץ עובדות מהטקסט - משופר"""
        facts_found = []
        
        # דפוסים בעברית
        patterns = [
            (r"קוראים לי ([\u0590-\u05FF]+)", "name", "שם המשתמש"),
            (r"השם שלי ([\u0590-\u05FF]+)", "name", "שם המשתמש"),
            (r"אני גר ב([\u0590-\u05FFa-zA-Z ]+)", "location", "מקום מגורים"),
            (r"אני עובד ב([\u0590-\u05FFa-zA-Z ]+)", "work", "מקום עבודה"),
            (r"אני עובד כ([\u0590-\u05FFa-zA-Z ]+)", "job", "תפקיד"),
            (r"אני אוהב (?:את )?([\u0590-\u05FFa-zA-Z ]+)", "likes", "תחביב/אהבה"),
            (r"אני שונא (?:את )?([\u0590-\u05FFa-zA-Z ]+)", "dislikes", "שנאה"),
            (r"יש לי (?:כלב|חתול|ילד|ילדה) בשם ([\u0590-\u05FFa-zA-Z]+)", "pet_or_child", "חיית מחמד/ילד"),
            (r"הפרויקט שלי (?:הוא )?([\u0590-\u05FFa-zA-Z0-9 ]+)", "project", "פרויקט"),
            (r"אני עובד על ([\u0590-\u05FFa-zA-Z0-9 ]+)", "project", "פרויקט"),
        ]

        for pat, key, desc in patterns:
            m = re.search(pat, user_text)
            if m:
                value = m.group(1).strip()
                facts_found.append({
                    "key": key,
                    "value": value,
                    "description": desc,
                    "source_text": user_text,
                    "extracted_at": datetime.now().isoformat()
                })
        
        return facts_found

    def update_from_facts(self, facts: List[Dict], auto_approve=False) -> List[Dict]:
        """מעדכן פרופיל מעובדות, מחזיר הצעות לאישור אם לא auto"""
        proposals = []
        
        for fact in facts:
            key = fact["key"]
            value = fact["value"]
            
            # אם זה שם ויש כבר שם שונה, צור הצעה
            if key == "name":
                if self.profile.get("name") and self.profile["name"] != value:
                    proposals.append({
                        "id": f"profile_name_{int(datetime.now().timestamp())}",
                        "type": "profile_update",
                        "field": "name",
                        "old_value": self.profile["name"],
                        "new_value": value,
                        "description_he": f"זיהיתי שהשם שלך הוא '{value}', אבל זכרתי '{self.profile['name']}'. לעדכן?",
                        "risk": "low",
                        "fact": fact
                    })
                elif not self.profile.get("name"):
                    # שם חדש - הצעה
                    proposals.append({
                        "id": f"profile_name_new_{int(datetime.now().timestamp())}",
                        "type": "profile_new",
                        "field": "name",
                        "new_value": value,
                        "description_he": f"למדתי שקוראים לך '{value}'. לזכור את זה?",
                        "risk": "low",
                        "fact": fact
                    })
            else:
                # עובדה כללית אחרת
                proposals.append({
                    "id": f"fact_{key}_{int(datetime.now().timestamp())}",
                    "type": "fact",
                    "field": key,
                    "new_value": value,
                    "description_he": f"למדתי עליך: {fact['description']}: '{value}'. לשמור בזיכרון?",
                    "risk": "low",
                    "fact": fact
                })

        # אם auto_approve, שמור ישר
        if auto_approve:
            for prop in proposals:
                self.apply_profile_update(prop)
            return []
        
        return proposals

    def apply_profile_update(self, proposal: Dict):
        """מיישם עדכון פרופיל אחרי אישור"""
        field = proposal.get("field")
        new_val = proposal.get("new_value")
        
        if field == "name":
            self.profile["name"] = new_val
        elif field == "project":
            if new_val not in self.profile["projects"]:
                self.profile["projects"].append(new_val)
                if len(self.profile["projects"]) > 20:
                    self.profile["projects"] = self.profile["projects"][-20:]
        elif field in ["location", "work", "job"]:
            self.profile["preferences"][field] = new_val
        else:
            # עובדה כללית
            self.profile["facts"].append({
                "key": field,
                "value": new_val,
                "saved_at": datetime.now().isoformat()
            })
        
        self.profile["interaction_count"] += 1
        self.save()
        print(f"[Profile] ✅ עדכון פרופיל: {field} = {new_val}")


class ConversationIntelligence:
    """הופך כל שיחה לחכמה יותר"""
    def __init__(self):
        self.summaries = self._load_json(CONVERSATION_SUMMARY_FILE, {"summaries": []})
    
    def _load_json(self, path: Path, default: Dict):
        if path.exists():
            try:
                return json.loads(path.read_text(encoding='utf-8'))
            except:
                return default
        return default

    def should_create_proposal(self, user_text: str, assistant_text: str, intent_result: Dict) -> Optional[Dict]:
        """מחליט האם השיחה הזו צריכה ליצור הצעת למידה"""
        
        # אם confidence נמוך, הצע ללמוד intent חדש
        if intent_result["intent"] == "general_chat" and intent_result["confidence"] < 0.3:
            # המשתמש אומר משהו שאנחנו לא מבינים טוב
            if len(user_text.split()) > 2:
                return {
                    "id": f"intent_{int(datetime.now().timestamp())}",
                    "type": "intent_learn",
                    "user_text": user_text,
                    "description_he": f"לא הייתי בטוחה מה התכוונת ב-'{user_text}'. רוצה ללמד אותי מה זה אומר? אני יכולה לזכור את זה לפעם הבאה.",
                    "risk": "low",
                    "suggested_intent": "custom_user_phrase"
                }
        
        # אם המשתמש מתקן את אדיאל
        correction_keywords = ["לא", "טעית", "לא נכון", "לא הבנת", "זאת אומרת"]
        if any(kw in user_text for kw in correction_keywords) and len(user_text) < 100:
            return {
                "id": f"correction_{int(datetime.now().timestamp())}",
                "type": "correction",
                "user_text": user_text,
                "assistant_last": assistant_text,
                "description_he": f"קלטתי שתיקנת אותי: '{user_text}'. רוצה שאלמד מזה ואשתפר?",
                "risk": "low"
            }
        
        return None

    def summarize_conversation(self, messages: List[Dict]) -> str:
        """מסכם שיחה ארוכה לזיכרון"""
        if len(messages) < 4:
            return ""
        
        # פשוט - קח את המשפטים עם הכי הרבה מילים משמעותיות
        important = []
        for msg in messages[-10:]:
            if msg["role"] == "user" and len(msg["content"].split()) > 3:
                important.append(msg["content"])
        
        summary = " | ".join(important[-3:])
        return summary[:500]


class AdielLearningEngine:
    """
    מנוע למידה ראשי - מחבר הכל
    """
    def __init__(self):
        self.vocab = VocabularyManager()
        self.profile = UserProfile()
        self.intelligence = ConversationIntelligence()
        self.pending_proposals = []  # הצעות שממתינות לאישור
        
        print("[Learning] 🧠 מנוע למידה נטען - זוכר הכל, לומד כל שיחה")

    def process_interaction(self, user_text: str, assistant_text: str, intent_result: Dict) -> List[Dict]:
        """
        מעבד אינטראקציה ומחזיר רשימת הצעות ללמידה/עדכון
        כל הצעה דורשת אישור - אבל לא מציף, מקסימום 2 בכל שיחה
        """
        proposals = []
        
        # 1. חלץ עובדות - רק אם יש בקשת זיכרון מפורשת או עובדה חשובה חדשה
        facts = self.profile.extract_facts(user_text)
        if facts:
            # רק אם יש עובדה חדשה שלא קיימת כבר
            new_facts = []
            for f in facts:
                # בדוק אם כבר קיים
                exists = False
                if f["key"] == "name" and self.profile.profile.get("name") == f["value"]:
                    exists = True
                if not exists:
                    new_facts.append(f)
            
            if new_facts:
                profile_props = self.profile.update_from_facts(new_facts, auto_approve=False)
                # הגבל ל-1 הצעת פרופיל בכל פעם
                if profile_props:
                    proposals.append(profile_props[0])
        
        # 2. זהה מילים לא מוכרות - רק אם ביקשו ללמוד במפורש
        if any(kw in user_text for kw in ["תזכור", "תזכרי", "תלמד", "סלנג", "פירוש"]):
            known_words = self.vocab.get_known_words_set()
            unknown = self.vocab.detect_unknown_words(user_text, known_words)
            for word in unknown[:1]:  # מקסימום 1
                proposal = self.vocab.learn_word(word, meaning="לא ידוע עדיין", example=user_text, source="auto_detect")
                if proposal:
                    proposal["description_he"] = f"שמעתי מילה לא מוכרת: '{word}' במשפט '{user_text[:50]}...'. מה זה אומר? אם תסביר לי, אזכור לפעם הבאה."
                    proposals.append(proposal)
        
        # 3. בדוק אם צריך ללמוד intent חדש - רק אם confidence נמוך מאוד
        if intent_result["confidence"] < 0.25 and len(user_text.split()) > 3:
            intent_proposal = self.intelligence.should_create_proposal(user_text, assistant_text, intent_result)
            if intent_proposal and len(proposals) < 2:  # רק אם אין כבר 2 הצעות
                proposals.append(intent_proposal)
        
        # 4. אם המשתמש אמר "תזכור" במפורש וזו לא עובדה שזיהינו כבר
        if any(kw in user_text for kw in ["תזכור", "תזכרי"]) and not facts:
            # רק אם זה לא סתם "תזכור שאני..." שכבר טופל כ-fact
            if len(proposals) == 0:  # רק אם אין עדיין הצעה
                proposals.append({
                    "id": f"explicit_memory_{int(datetime.now().timestamp())}",
                    "type": "explicit_memory",
                    "user_text": user_text,
                    "description_he": f"ביקשת שאזכור: '{user_text}'. לשמור את זה בזיכרון הקבוע שלי?",
                    "risk": "low"
                })
        
        # הגבל ל-2 הצעות מקסימום בכל שיחה כדי לא להציף
        proposals = proposals[:2]
        
        # שמור הצעות ממתינות
        self.pending_proposals.extend(proposals)
        
        if proposals:
            print(f"[Learning] 📚 יצרתי {len(proposals)} הצעות למידה חדשות")
            for p in proposals:
                print(f"  - {p['type']}: {p['description_he'][:80]}...")
        
        return proposals

    def approve_proposal(self, proposal_id: str, extra_data: Dict = None) -> Dict:
        """מאשר הצעה - הופך אותה לסופית"""
        proposal = None
        for p in self.pending_proposals:
            if p["id"] == proposal_id:
                proposal = p
                break
        
        if not proposal:
            return {"success": False, "error": "Proposal not found"}
        
        # יישם לפי סוג
        try:
            if proposal["type"] in ["vocabulary", "vocab"]:
                word = proposal.get("word")
                meaning = extra_data.get("meaning") if extra_data else proposal.get("meaning")
                example = proposal.get("example", "")
                self.vocab.approve_word(word, meaning, example)
                
            elif proposal["type"] in ["profile_update", "profile_new", "fact", "explicit_memory"]:
                self.profile.apply_profile_update(proposal)
                
            elif proposal["type"] in ["intent_learn", "correction"]:
                # שמור intent חדש
                self.save_learned_intent(proposal, extra_data)
            
            # הסר מרשימת ממתינים
            self.pending_proposals = [p for p in self.pending_proposals if p["id"] != proposal_id]
            
            print(f"[Learning] ✅ הצעה אושרה: {proposal_id} -> {proposal['type']}")
            return {"success": True, "proposal": proposal, "message": f"למדתי: {proposal['description_he']}"}
            
        except Exception as e:
            print(f"[Learning] Approve failed: {e}")
            return {"success": False, "error": str(e)}

    def reject_proposal(self, proposal_id: str) -> Dict:
        """דוחה הצעה"""
        original_len = len(self.pending_proposals)
        self.pending_proposals = [p for p in self.pending_proposals if p["id"] != proposal_id]
        
        if len(self.pending_proposals) < original_len:
            print(f"[Learning] ❌ הצעה נדחתה: {proposal_id}")
            return {"success": True, "message": "הצעה נדחתה, לא אזכור את זה"}
        else:
            return {"success": False, "error": "Proposal not found"}

    def save_learned_intent(self, proposal: Dict, extra_data: Dict = None):
        """שומר intent חדש שנלמד"""
        try:
            data = {}
            if LEARNED_INTENTS_FILE.exists():
                data = json.loads(LEARNED_INTENTS_FILE.read_text(encoding='utf-8'))
            else:
                data = {"intents": []}
            
            data["intents"].append({
                "user_text": proposal.get("user_text"),
                "intent": extra_data.get("intent") if extra_data else proposal.get("suggested_intent", "custom"),
                "meaning": extra_data.get("meaning", "") if extra_data else "",
                "learned_at": datetime.now().isoformat(),
                "approved": True
            })
            
            # שמור רק 100 אחרונים
            if len(data["intents"]) > 100:
                data["intents"] = data["intents"][-100:]
            
            LEARNED_INTENTS_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
            print(f"[Learning] ✅ Intent חדש נשמר: {proposal.get('user_text')}")
        except Exception as e:
            print(f"[Learning] Save intent failed: {e}")

    def get_pending_proposals(self) -> List[Dict]:
        return self.pending_proposals
    
    def get_user_profile_summary(self) -> Dict:
        """מחזיר סיכום פרופיל לשימוש המוח"""
        return {
            "name": self.profile.profile.get("name"),
            "projects": self.profile.profile.get("projects", [])[-3:],
            "preferences": self.profile.profile.get("preferences", {}),
            "interaction_count": self.profile.profile.get("interaction_count", 0),
            "vocab_learned": len(self.vocab.vocab.get("words", {})),
            "facts_count": len(self.profile.profile.get("facts", []))
        }

    def get_smart_context(self) -> str:
        """בונה הקשר חכם לשיחה"""
        profile = self.profile.profile
        ctx = ""
        
        if profile.get("name"):
            ctx += f"הבוס קוראים לו {profile['name']}. "
        
        if profile.get("projects"):
            ctx += f"עובד על: {', '.join(profile['projects'][-2:])}. "
        
        if profile.get("preferences"):
            prefs = ", ".join(f"{k}:{v}" for k,v in list(profile["preferences"].items())[-3:])
            ctx += f"העדפות: {prefs}. "
        
        vocab_count = len(self.vocab.vocab.get("words", {}))
        if vocab_count > 0:
            ctx += f"למדתי {vocab_count} מילים חדשות ממנו. "
        
        ctx += f"דיברנו {profile.get('interaction_count', 0)} פעמים. "
        
        return ctx

# Singleton
_global_learning = None

def get_learning_engine() -> AdielLearningEngine:
    global _global_learning
    if _global_learning is None:
        _global_learning = AdielLearningEngine()
    return _global_learning
