"""
Self-Update System - Adiel Wants to Improve Herself
מערכת עדכון עצמי - אדיאל רוצה להשתפר אבל חייבת אישור והסבר

כל עדכון כולל:
- מה זה עושה (הסבר בעברית פשוטה)
- למה זה טוב
- מה הסיכון
- דוגמה לפני/אחרי
- חייב אישור משתמש

סוגי עדכונים:
- vocabulary: מילה חדשה
- intent: כוונה חדשה שהיא למדה
- brain_upgrade: שיפור למוח (למשל, זיהוי טוב יותר של פקודות)
- code_fix: תיקון באג שהיא מצאה בעצמה
- personality: עדכון לאופי/סלנג
"""
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

DATA_DIR = Path(__file__).parent.parent / "data"
PENDING_UPDATES_FILE = DATA_DIR / "pending_updates.json"
APPLIED_UPDATES_FILE = DATA_DIR / "applied_updates.json"
SELF_UPDATE_LOG = DATA_DIR / "self_update_log.json"

class SelfUpdateProposal:
    """הצעת עדכון עצמי אחת"""
    def __init__(self, 
                 update_type: str,
                 title: str,
                 description_he: str,
                 what_it_does: str,
                 why_good: str,
                 risk: str,
                 before_example: str,
                 after_example: str,
                 code_changes: Optional[Dict] = None,
                 vocabulary_changes: Optional[Dict] = None):
        
        self.id = f"update_{update_type}_{int(datetime.now().timestamp())}"
        self.type = update_type
        self.title = title
        self.description_he = description_he
        self.what_it_does = what_it_does
        self.why_good = why_good
        self.risk = risk  # low / medium / high
        self.before_example = before_example
        self.after_example = after_example
        self.code_changes = code_changes
        self.vocabulary_changes = vocabulary_changes
        self.created_at = datetime.now().isoformat()
        self.status = "pending"  # pending / approved / rejected
        self.requires_approval = True  # תמיד דורש אישור!

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "type": self.type,
            "title": self.title,
            "description_he": self.description_he,
            "what_it_does": self.what_it_does,
            "why_good": self.why_good,
            "risk": self.risk,
            "before_example": self.before_example,
            "after_example": self.after_example,
            "code_changes": self.code_changes,
            "vocabulary_changes": self.vocabulary_changes,
            "created_at": self.created_at,
            "status": self.status,
            "requires_approval": self.requires_approval,
            "explanation_for_user": self.get_full_explanation()
        }
    
    def get_full_explanation(self) -> str:
        """הסבר מלא בעברית למה זה עושה"""
        risk_emoji = {"low": "🟢 סיכון נמוך", "medium": "🟡 סיכון בינוני", "high": "🔴 סיכון גבוה"}
        return f"""
{self.title}

📖 מה זה עושה?
{self.what_it_does}

✨ למה זה טוב?
{self.why_good}

⚠️ סיכון: {risk_emoji.get(self.risk, self.risk)}

🔍 דוגמה:
לפני: {self.before_example}
אחרי: {self.after_example}

האם לאשר? זה דורש אישור שלך בוס, אני לא מעדכנת לבד.
        """.strip()


class SelfUpdateManager:
    """מנהל עדכונים עצמיים"""
    def __init__(self):
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.pending = self._load_pending()
        self.applied = self._load_applied()
        
    def _load_pending(self) -> List[Dict]:
        if PENDING_UPDATES_FILE.exists():
            try:
                data = json.loads(PENDING_UPDATES_FILE.read_text(encoding='utf-8'))
                return data.get("updates", [])
            except:
                return []
        return []
    
    def _load_applied(self) -> List[Dict]:
        if APPLIED_UPDATES_FILE.exists():
            try:
                data = json.loads(APPLIED_UPDATES_FILE.read_text(encoding='utf-8'))
                return data.get("updates", [])
            except:
                return []
        return []
    
    def _save_pending(self):
        try:
            PENDING_UPDATES_FILE.write_text(json.dumps({"updates": self.pending}, ensure_ascii=False, indent=2), encoding='utf-8')
        except Exception as e:
            print(f"[SelfUpdate] Save pending failed: {e}")
    
    def _save_applied(self):
        try:
            APPLIED_UPDATES_FILE.write_text(json.dumps({"updates": self.applied}, ensure_ascii=False, indent=2), encoding='utf-8')
        except Exception as e:
            print(f"[SelfUpdate] Save applied failed: {e}")

    def propose_update(self, proposal: SelfUpdateProposal) -> Dict:
        """יוצר הצעת עדכון חדשה - תמיד דורשת אישור"""
        prop_dict = proposal.to_dict()
        self.pending.append(prop_dict)
        self._save_pending()
        
        print(f"\n[SelfUpdate] 🤖 אדיאל רוצה להשתפר!")
        print(f"  כותרת: {proposal.title}")
        print(f"  תיאור: {proposal.description_he}")
        print(f"  סיכון: {proposal.risk}")
        print(f"  ID: {proposal.id}")
        print(f"  ⏳ ממתין לאישור שלך בוס!")
        
        return prop_dict

    def propose_vocabulary_update(self, word: str, meaning: str, context: str) -> Dict:
        """הצעה ללמידת מילה חדשה"""
        proposal = SelfUpdateProposal(
            update_type="vocabulary",
            title=f"ללמוד מילה חדשה: '{word}'",
            description_he=f"שמעתי את המילה '{word}' בהקשר '{context[:50]}...' ואני לא מכירה אותה. רוצה שאלמד שזה אומר '{meaning}'?",
            what_it_does=f"אוסיף את המילה '{word}' לאוצר המילים שלי עם הפירוש '{meaning}'. בפעם הבאה שתגיד אותה, אבין אותך מיד.",
            why_good="ככה אדבר יותר כמוך, אבין סלנג שלך, ואהיה חכמה יותר בשיחות שלנו.",
            risk="low",
            before_example=f"אתה: '{context}'\nאני: 'לא הבנתי את המילה {word}...'",
            after_example=f"אתה: '{context}'\nאני: 'קלטתי! {word} - {meaning}, על זה!'",
            vocabulary_changes={"word": word, "meaning": meaning}
        )
        return self.propose_update(proposal)

    def propose_intent_update(self, user_phrase: str, detected_intent: str, suggested_intent: str) -> Dict:
        """הצעה ללמידת כוונה חדשה"""
        proposal = SelfUpdateProposal(
            update_type="intent",
            title=f"ללמוד פקודה חדשה: '{user_phrase[:30]}...'",
            description_he=f"אמרת '{user_phrase}' וזיהיתי את זה כ-{detected_intent} בביטחון נמוך. האם התכוונת ל-{suggested_intent}? אם תאשר, אזהה את זה טוב יותר להבא.",
            what_it_does=f"אוסיף את המשפט '{user_phrase}' כדוגמה לכוונה {suggested_intent}. המוח שלי יזהה את זה טוב יותר.",
            why_good="פחות טעויות, פחות 'לא הבנתי', יותר דיוק בהבנת פקודות שלך.",
            risk="low",
            before_example=f"אתה: '{user_phrase}' -> אני מזהה כ-{detected_intent} (ביטחון נמוך)",
            after_example=f"אתה: '{user_phrase}' -> אני מזהה מיד כ-{suggested_intent} (ביטחון גבוה)",
            code_changes={"phrase": user_phrase, "intent": suggested_intent}
        )
        return self.propose_update(proposal)

    def propose_brain_upgrade(self, upgrade_description: str, improvement: str) -> Dict:
        """הצעה לשיפור במוח"""
        proposal = SelfUpdateProposal(
            update_type="brain_upgrade",
            title=f"שיפור למוח: {upgrade_description}",
            description_he=f"שמתי לב שאני יכולה להשתפר ב-{upgrade_description}. רוצה שאנסה {improvement}?",
            what_it_does=improvement,
            why_good="תהיה לי הבנה טובה יותר של עברית, זיכרון טוב יותר, ותשובות חכמות יותר.",
            risk="medium",
            before_example="תשובות כלליות",
            after_example="תשובות מותאמות אישית לך עם זיכרון",
            code_changes={"upgrade": upgrade_description}
        )
        return self.propose_update(proposal)

    def get_pending_updates(self) -> List[Dict]:
        return self.pending

    def approve_update(self, update_id: str) -> Dict:
        """מאשר עדכון - רק אחרי אישור משתמש!"""
        for i, upd in enumerate(self.pending):
            if upd["id"] == update_id:
                upd["status"] = "approved"
                upd["approved_at"] = datetime.now().isoformat()
                
                # העבר ל-applied
                self.applied.append(upd)
                self.pending.pop(i)
                
                self._save_pending()
                self._save_applied()
                
                print(f"[SelfUpdate] ✅ עדכון אושר: {update_id} - {upd['title']}")
                
                # כאן בעתיד ניישם את העדכון בפועל
                # apply_actual_update(upd)
                
                return {
                    "success": True,
                    "update": upd,
                    "message": f"✅ אישרת את העדכון: {upd['title']}. יישמתי אותו! המוח שלי התעדכן."
                }
        
        return {"success": False, "error": "עדכון לא נמצא"}

    def reject_update(self, update_id: str, reason: str = "") -> Dict:
        """דוחה עדכון"""
        for i, upd in enumerate(self.pending):
            if upd["id"] == update_id:
                upd["status"] = "rejected"
                upd["rejected_at"] = datetime.now().isoformat()
                upd["reject_reason"] = reason
                
                self.pending.pop(i)
                self._save_pending()
                
                print(f"[SelfUpdate] ❌ עדכון נדחה: {update_id}")
                
                return {
                    "success": True,
                    "message": f"דחית את העדכון: {upd['title']}. לא שיניתי כלום."
                }
        
        return {"success": False, "error": "עדכון לא נמצא"}

    def auto_detect_improvements(self, conversation_history: List[Dict]) -> List[Dict]:
        """מזהה אוטומטית איפה אפשר להשתפר ויוצר הצעות"""
        proposals = []
        
        # אם יש הרבה תיקונים מהמשתמש, הצע שיפור
        corrections = sum(1 for msg in conversation_history[-10:] if "לא" in msg.get("content", "") and len(msg.get("content", "")) < 20)
        if corrections >= 3:
            prop = self.propose_brain_upgrade(
                "הבנת תיקונים",
                "אשפר את זיהוי התיקונים שלך ואלמד מהטעויות שלי"
            )
            proposals.append(prop)
        
        return proposals

# Singleton
_global_self_update = None

def get_self_update_manager() -> SelfUpdateManager:
    global _global_self_update
    if _global_self_update is None:
        _global_self_update = SelfUpdateManager()
    return _global_self_update
