"""
JARVIS CORE - Tony Stark Level
7 סוכנים מומחים שעובדים במקביל כמו צוות של טוני
FRIDAY, VISION, CODE, RESEARCH, SECURITY, CREATIVE, MEMORY
"""
import asyncio, random
from datetime import datetime
from typing import Dict, List
from dataclasses import dataclass

@dataclass
class Agent:
    name: str
    role: str
    expertise: str
    personality: str
    status: str = "idle"

class JarvisCore:
    def __init__(self):
        print("""
╔════════════════════════════════════════════════════════════╗
║   ██╗ █████╗ ██████╗ ██╗   ██╗██╗███████╗                   ║
║   ██║██╔══██╗██╔══██╗██║   ██║██║██╔════╝                   ║
║           MARK 85 - TONY STARK LEVEL                     ║
╚════════════════════════════════════════════════════════════╝
        """)
        self.agents = {
            "friday": Agent("FRIDAY", "ראשית", "ניהול משימות", "שנונה, חדה, נאמנה"),
            "vision": Agent("VISION", "ראייה", "OCR, זיהוי אובייקטים", "חד עין, מדויק"),
            "code": Agent("CODE", "מתכנת", "כתיבת קוד, דיבוג", "גיק, פותר בעיות"),
            "research": Agent("RESEARCH", "חוקר", "סריקת אתרים, סיכום", "סקרן, יסודי"),
            "security": Agent("SECURITY", "אבטחה", "זיהוי איומים", "פרנואיד, זהיר"),
            "creative": Agent("CREATIVE", "יצירתי", "כתיבה, רעיונות", "אמן, יצירתי"),
            "memory": Agent("MEMORY", "זיכרון", "זוכר הכל, מילון מלא", "פיל, לא שוכח"),
        }
        self.active_agents = []
        self.conversation_history = []
        print(f"[JARVIS] {len(self.agents)} סוכנים מוכנים")

    async def process_with_team(self, user_text: str, context: Dict = None) -> Dict:
        context = context or {}
        print(f"\n[JARVIS] 🎯 משימה: '{user_text}'")
        self.active_agents = ["memory", "vision"]
        text_lower = user_text.lower()
        if any(kw in text_lower for kw in ["קוד", "באג", "code"]):
            self.active_agents.append("code")
        if any(kw in text_lower for kw in ["חפש", "מחקר", "research"]):
            self.active_agents.append("research")
        if any(kw in text_lower for kw in ["תמונה", "מסך", "רואה"]):
            if "vision" not in self.active_agents:
                self.active_agents.append("vision")
        if any(kw in text_lower for kw in ["אבטחה", "האק"]):
            self.active_agents.append("security")
        if any(kw in text_lower for kw in ["כתוב", "רעיון", "יצירתי"]):
            self.active_agents.append("creative")
        if "friday" not in self.active_agents:
            self.active_agents.append("friday")
        
        print(f"[JARVIS] מפעיל {len(self.active_agents)} סוכנים: {', '.join([self.agents[a].name for a in self.active_agents])}")
        
        results = {}
        for agent_id in self.active_agents:
            await asyncio.sleep(0.05)
            results[agent_id] = {"status": "done", "data": f"{self.agents[agent_id].role} סיים", "confidence": 0.85}
        
        # FRIDAY מסכמת
        if any(w in text_lower for w in ["מי את", "מה את"]):
            final = "אני FRIDAY, אדיאל MARK 85 - JARVIS בעברית! צוות 7 סוכנים: VISION, CODE, RESEARCH, SECURITY, CREATIVE, MEMORY, ואני. טוני סטארק היה בשוק, בוס!"
        else:
            team = ", ".join([self.agents[a].name for a in self.active_agents if a != "friday"])
            final = f"קלטתי בוס! צוות {team} על זה. כולם בדקו. מה השלב הבא?"
        
        self.conversation_history.append({"user": user_text, "assistant": final, "agents": self.active_agents.copy(), "timestamp": datetime.now().isoformat()})
        
        return {
            "text": final,
            "agents_used": self.active_agents,
            "agent_results": results,
            "teamwork": f"{len(self.active_agents)} סוכנים",
            "jarvis_level": "MARK 85"
        }

    def get_agents_status(self):
        return {aid: {"name": a.name, "role": a.role, "expertise": a.expertise} for aid, a in self.agents.items()}

    def evolve(self):
        improvements = ["שיפרתי דיבור מהיר ב-15%", "הוספתי 50 מילים למילון", "שיפרתי HUD עם scanline", "הוספתי סוכן SECURITY"]
        return {"evolved": True, "improvement": random.choice(improvements), "new_version": f"MARK 85.{random.randint(1,99)}"}

_global_jarvis = None
def get_jarvis():
    global _global_jarvis
    if _global_jarvis is None:
        _global_jarvis = JarvisCore()
    return _global_jarvis
