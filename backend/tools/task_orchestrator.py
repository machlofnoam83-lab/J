"""
Task Orchestrator - המוח המרכזי שמנהל את כל המשימות
מקבל בקשה בעברית ומחליט איזה סוכן להפעיל
"""
import re
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime

from .shopping_agent import get_shopping_agent
from .travel_agent import get_travel_agent
from .research_agent import get_research_agent
from .form_filler import get_form_filler
from .email_manager import get_email_manager
from .calendar_manager import get_calendar_manager
from .cloud_file_manager import get_file_manager
from .report_generator import get_report_generator
from .computer_control import get_computer_control
from .routine_automation import get_routine_automation

@dataclass
class Task:
    id: str
    type: str
    description: str
    params: Dict
    status: str = "pending"
    created_at: datetime = None
    result: Optional[Dict] = None

class TaskOrchestrator:
    """
    אורקסטרטור - מנתב משימות בעברית לסוכן הנכון
    זה הלב של אדיאל כ-Agent אמיתי
    """
    def __init__(self):
        self.shopping = get_shopping_agent()
        self.travel = get_travel_agent()
        self.research = get_research_agent()
        self.form_filler = get_form_filler()
        self.email = get_email_manager()
        self.calendar = get_calendar_manager()
        self.file_manager = get_file_manager()
        self.reports = get_report_generator()
        self.computer = get_computer_control()
        self.routines = get_routine_automation()
        
        # היסטוריית משימות
        self.tasks: List[Task] = []

    def classify_task(self, text: str) -> Dict:
        """
        מסווג משימה בעברית - לאן לנתב
        """
        text_lower = text.lower()
        
        # מילות מפתח לכל קטגוריה
        patterns = {
            "shopping": [
                r"תקנה|קנייה|קנה לי|מחפש.*לקנות|זול.*ביותר|עגלה|משלוח",
                r"buy|purchase|shopping|cheapest|cart"
            ],
            "travel": [
                r"טיסה|מלון|חופשה|נופש|טיול|booking|flight|hotel",
                r"תזמין.*טיסה|תזמין.*מלון|משווה.*מחיר.*טיסה"
            ],
            "research": [
                r"תחקור|מחקר|תסרוק.*אתרים|תאסוף.*מידע|תסכם.*קובץ",
                r"research|scan.*sites|summarize"
            ],
            "form_filling": [
                r"תמלא.*טופס|טופס.*ממשלתי|הזנת.*נתונים",
                r"fill.*form|government.*form"
            ],
            "email": [
                r"מייל|אימייל|דואר|תנסח.*תשובה|סנן.*ספאם|דחיפות",
                r"email|inbox|spam|urgent"
            ],
            "calendar": [
                r"פגישה|יומן|לוח.*שנה|תיאום|זמן.*מושלם",
                r"meeting|calendar|schedule|find.*time"
            ],
            "file_management": [
                r"קבצים|תיקיות|דרייב|דרופבוקס|תארגן.*קבצים|תעביר",
                r"files|drive|dropbox|organize"
            ],
            "reports": [
                r"דוח|סיכום.*מנהלים|אקסל|CRM|תשלוף.*נתונים",
                r"report|excel|summary"
            ],
            "computer_control": [
                r"תפתח.*אפליקציה|תנגן.*ספוטיפיי|יוטיוב|תקליד|הכתבה|תכתוב.*מסמך",
                r"open.*app|spotify|youtube|type|dictate"
            ],
            "routine": [
                r"שגרה|רוטינה|כל.*בוקר|אוטומציה.*יום|בוקר.*טוב",
                r"routine|every.*morning|automation"
            ]
        }
        
        scores = {}
        for task_type, regex_list in patterns.items():
            score = 0
            for pattern in regex_list:
                if re.search(pattern, text_lower):
                    score += 1
            scores[task_type] = score
        
        best_type = max(scores, key=scores.get) if max(scores.values()) > 0 else "general_chat"
        confidence = scores.get(best_type, 0) / 2.0
        
        return {
            "type": best_type,
            "confidence": min(confidence, 0.95),
            "scores": scores,
            "text": text
        }

    async def execute_task(self, text: str, context: Dict = None) -> Dict:
        """
        מבצע משימה - הלב
        """
        classification = self.classify_task(text)
        task_type = classification["type"]
        print(f"[Orchestrator] 🎯 סיווג משימה: {task_type} (ביטחון {classification['confidence']:.2f}) - '{text}'")
        
        task = Task(
            id=f"task_{int(datetime.now().timestamp())}",
            type=task_type,
            description=text,
            params=context or {},
            created_at=datetime.now()
        )
        self.tasks.append(task)
        
        try:
            result = await self._route_task(task_type, text, context or {})
            task.status = "completed" if result.get("success") else "failed"
            task.result = result
            
            return {
                "success": result.get("success", False),
                "task_type": task_type,
                "classification": classification,
                "result": result,
                "task_id": task.id,
                "message": result.get("message", "בוצע")
            }
        except Exception as e:
            print(f"[Orchestrator] Task failed: {e}")
            import traceback; traceback.print_exc()
            task.status = "failed"
            return {
                "success": False,
                "task_type": task_type,
                "error": str(e),
                "task_id": task.id,
                "message": f"נכשלתי במשימה {task_type}: {e}"
            }

    async def _route_task(self, task_type: str, text: str, context: Dict) -> Dict:
        """ניתוב לסוכן המתאים"""
        
        # === קניות ===
        if task_type == "shopping":
            # חלץ מוצר מהטקסט
            # "תקנה לי אוזניות הכי זול" -> query="אוזניות"
            query_match = re.search(r"(?:תקנה|קנה|מחפש)\s+(?:לי\s+)?(.+?)(?:\s+הכי|\s+זול|$)", text)
            query = query_match.group(1).strip() if query_match else text
            
            # פרטי משלוח מדמו / פרופיל
            shipping = context.get("shipping", {
                "name": "נועם לוי",
                "email": "noam@example.com",
                "phone": "050-1234567",
                "address": "הרצל 1",
                "city": "תל אביב",
                "zip": "61000"
            })
            
            return await self.shopping.full_purchase_flow(query=query, shipping_info=shipping)

        # === חופשות ===
        elif task_type == "travel":
            # חלץ ערים
            # "תזמין טיסה מתל אביב ללונדון" -> from=תל אביב, to=לונדון
            from_match = re.search(r"מ(.+?)\s+ל", text)
            to_match = re.search(r"ל([א-תa-zA-Z]+)", text)
            
            from_city = from_match.group(1).strip() if from_match else context.get("from", "תל אביב")
            to_city = to_match.group(1).strip() if to_match else context.get("to", "לונדון")
            
            # תאריכים - ברירת מחדל שבוע הבא
            from datetime import timedelta
            check_in = (datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d")
            check_out = (datetime.now() + timedelta(days=10)).strftime("%Y-%m-%d")
            
            return await self.travel.find_best_deal(
                from_city=from_city,
                to_city=to_city,
                check_in=context.get("check_in", check_in),
                check_out=context.get("check_out", check_out),
                budget=context.get("budget")
            )

        # === מחקר ===
        elif task_type == "research":
            # "תחקור על בינה מלאכותית" -> topic="בינה מלאכותית"
            topic_match = re.search(r"(?:תחקור|מחקר)\s+(?:על\s+)?(.+)", text)
            topic = topic_match.group(1).strip() if topic_match else text
            
            return await self.research.research_topic(topic=topic, depth=context.get("depth", 5))

        # === טפסים ===
        elif task_type == "form_filling":
            url = context.get("url", "https://example.com/form")
            profile = context.get("profile", {
                "first_name": "נועם",
                "last_name": "לוי",
                "id": "123456789",
                "phone": "050-1234567",
                "email": "noam@example.com"
            })
            return await self.form_filler.fill_government_form(url=url, profile=profile)

        # === מייל ===
        elif task_type == "email":
            if "דחוף" in text or "דחיפות" in text:
                return {
                    "success": True,
                    "result": await self.email.get_urgent_emails(),
                    "message": "סיננתי מיילים דחופים"
                }
            elif "ספאם" in text:
                return await self.email.filter_spam()
            elif "תנסח" in text or "תשובה" in text:
                # צריך email_id - לוקח ראשון
                emails = await self.email.fetch_emails(limit=1)
                if emails:
                    return await self.email.draft_reply(email_id=emails[0].id, instructions=text)
                return {"success": False, "error": "לא נמצאו מיילים"}
            else:
                result = await self.email.categorize_inbox()
                return result

        # === יומן ===
        elif task_type == "calendar":
            # "תמצא זמן לפגישה עם דני וגל מחר"
            attendees_match = re.search(r"עם\s+(.+)", text)
            attendees = []
            if attendees_match:
                attendees = [a.strip() for a in attendees_match.group(1).split("ו")]
            
            if not attendees:
                attendees = context.get("attendees", ["dani@example.com", "gal@example.com"])
            
            return await self.calendar.find_perfect_meeting_time(
                attendees=attendees,
                duration=context.get("duration", 60)
            )

        # === קבצים ===
        elif task_type == "file_management":
            source = context.get("source", "~/Downloads")
            if "ארגן" in text or "תארגן" in text:
                return self.file_manager.organize_files(source_dir=source)
            elif "דרייב" in text or "drive" in text.lower():
                return self.file_manager.sync_to_cloud(local_dir=source, cloud_provider="drive")
            elif "כפילויות" in text:
                return self.file_manager.find_duplicates(directory=source)
            else:
                files = self.file_manager.search_files(query=text, directory=source)
                return {"success": True, "files": files[:10], "count": len(files), "message": f"מצאתי {len(files)} קבצים"}

        # === דוחות ===
        elif task_type == "reports":
            files = context.get("files", [])
            crm_query = context.get("crm_query", "")
            return await self.reports.full_report_flow(files=files, crm_query=crm_query)

        # === שליטה במחשב ===
        elif task_type == "computer_control":
            if "ספוטיפיי" in text or "spotify" in text.lower():
                query_match = re.search(r"ספוטיפיי\s+(.+)|spotify\s+(.+)", text, re.IGNORECASE)
                query = (query_match.group(1) or query_match.group(2)).strip() if query_match else ""
                return self.computer.play_spotify(query=query)
            
            elif "יוטיוב" in text or "youtube" in text.lower():
                query_match = re.search(r"יוטיוב\s+(.+)|youtube\s+(.+)", text, re.IGNORECASE)
                query = (query_match.group(1) or query_match.group(2)).strip() if query_match else text
                return self.computer.play_youtube(query=query)
            
            elif "תפתח" in text:
                app_match = re.search(r"תפתח\s+(?:את\s+)?(.+)", text)
                app_name = app_match.group(1).strip() if app_match else "chrome"
                return self.computer.open_app(app_name)
            
            elif "תקליד" in text or "תכתוב" in text:
                text_match = re.search(r"תקליד\s+(.+)|תכתוב\s+(.+)", text)
                to_type = (text_match.group(1) or text_match.group(2)).strip() if text_match else text
                return self.computer.type_text(to_type)
            
            else:
                return {"success": False, "message": f"לא זיהיתי פקודת מחשב ב-{text}"}

        # === רוטינות ===
        elif task_type == "routine":
            if "בוקר" in text:
                return await self.routines.run_routine("morning_routine")
            elif "ערב" in text:
                return await self.routines.run_routine("evening_routine")
            elif "פוקוס" in text:
                return await self.routines.run_routine("focus_mode")
            else:
                routines = self.routines.list_routines()
                return {"success": True, "routines": routines, "message": f"יש {len(routines)} רוטינות: {', '.join([r['name'] for r in routines])}"}

        else:
            return {"success": False, "message": f"לא זיהיתי משימה ב-{text}, נסה: 'תקנה', 'תזמין טיסה', 'תחקור', 'תמלא טופס', 'תבדוק מיילים', 'תמצא זמן לפגישה'"}

    def get_task_history(self) -> List[Dict]:
        return [
            {
                "id": t.id,
                "type": t.type,
                "description": t.description,
                "status": t.status,
                "created_at": t.created_at.isoformat() if t.created_at else None
            } for t in self.tasks[-20:]
        ]

def get_task_orchestrator():
    return TaskOrchestrator()
