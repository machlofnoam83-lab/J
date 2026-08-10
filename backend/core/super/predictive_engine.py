"""
Predictive Engine - חוזה מה תרצה לפני שאתה מבקש
כמו JARVIS שיודע שטוני צריך קפה
"""
from datetime import datetime, timedelta
from typing import Dict, List
from collections import Counter, defaultdict
import json
from pathlib import Path

class PredictiveEngine:
    def __init__(self):
        self.history_file = Path(__file__).parent.parent / "data" / "predictive_history.json"
        self.history_file.parent.mkdir(parents=True, exist_ok=True)
        self.patterns = self._load_patterns()
        self.daily_routine = {
            "08:00": ["בדוק מיילים", "מזג אוויר", "משימות היום"],
            "12:00": ["הפסקת צהריים", "חדשות"],
            "18:00": ["סיכום יום", "ארגון קבצים"],
            "22:00": ["תזכורת למחר", "גיבוי"],
        }
        print("[Predictive] 🔮 מנוע חיזוי נטען")

    def _load_patterns(self):
        if self.history_file.exists():
            try:
                return json.loads(self.history_file.read_text(encoding='utf-8'))
            except:
                pass
        return {"sequences": [], "time_patterns": defaultdict(list), "context_patterns": defaultdict(list)}

    def _save_patterns(self):
        try:
            data = {"sequences": self.patterns["sequences"][-100:], "time_patterns": dict(self.patterns["time_patterns"]), "context_patterns": dict(self.patterns["context_patterns"])}
            self.history_file.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
        except:
            pass

    def learn_from_action(self, user_text: str, context: Dict = None):
        context = context or {}
        now = datetime.now()
        hour_key = now.strftime("%H:00")
        self.patterns["sequences"].append({"text": user_text, "time": now.isoformat(), "hour": hour_key, "context": context})
        self.patterns["time_patterns"][hour_key].append(user_text)
        self._save_patterns()

    def predict_next_actions(self, current_text: str = "", current_time: datetime = None) -> List[Dict]:
        if current_time is None:
            current_time = datetime.now()
        hour_key = current_time.strftime("%H:00")
        predictions = []
        time_actions = self.patterns["time_patterns"].get(hour_key, [])
        if time_actions:
            most_common = Counter(time_actions).most_common(3)
            for action, count in most_common:
                if action != current_text:
                    predictions.append({"action": action, "reason": f"בדרך כלל ב-{hour_key} אתה מבקש '{action}'", "confidence": min(0.3 + count * 0.1, 0.85), "type": "time_based"})
        routine = self.daily_routine.get(hour_key)
        if routine:
            for task in routine:
                predictions.append({"action": task, "reason": f"שגרת יום: {hour_key} זמן ל-{task}", "confidence": 0.7, "type": "routine"})
        lower = current_text.lower()
        if "שגיאה" in lower:
            predictions.append({"action": "תתקן את השגיאה אוטומטית", "reason": "זיהיתי שגיאה", "confidence": 0.85, "type": "context"})
        if "קניות" in lower:
            predictions.append({"action": "תמלא פרטי משלוח", "reason": "אחרי קנייה", "confidence": 0.8, "type": "context"})
        predictions.sort(key=lambda x: x["confidence"], reverse=True)
        return predictions[:3]

    def get_proactive_suggestion(self) -> Dict:
        now = datetime.now()
        hour = now.hour
        if hour == 8:
            return {"title": "בוקר טוב בוס! ☀️", "message": "רוצה שאריץ שגרת בוקר?", "action": "תריץ שגרת בוקר", "type": "routine"}
        elif hour == 12:
            return {"title": "הפסקת צהריים? 🍔", "message": "רוצה מוזיקה?", "action": "תנגן מוזיקה רגועה", "type": "wellbeing"}
        elif hour == 18:
            return {"title": "סיכום יום 📊", "message": "רוצה סיכום?", "action": "תכין סיכום יום", "type": "routine"}
        predictions = self.predict_next_actions()
        if predictions:
            top = predictions[0]
            return {"title": "חזיתי שתרצה...", "message": f"{top['reason']}", "action": top["action"], "type": "predicted", "confidence": top["confidence"]}
        return {"title": "אני כאן, בוס", "message": "צריך משהו?", "action": "מה את יכולה לעשות?", "type": "general"}

_global_predictive = None
def get_predictive_engine():
    global _global_predictive
    if _global_predictive is None:
        _global_predictive = PredictiveEngine()
    return _global_predictive
