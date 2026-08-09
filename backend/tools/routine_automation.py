"""
Routine Automation - אוטומציה של שגרת יום
מבצע רצף פעולות קבוע כל בוקר/ערב
"""
import asyncio
from datetime import datetime, time
from typing import List, Dict, Optional
from dataclasses import dataclass
from pathlib import Path
import json

@dataclass
class RoutineStep:
    id: str
    name: str
    action: str  # open_app, check_email, weather, todo, etc
    params: Dict
    order: int
    enabled: bool = True

@dataclass
class Routine:
    id: str
    name: str
    description: str
    schedule: str  # "08:00", "every_morning", "custom"
    steps: List[RoutineStep]
    enabled: bool = True
    last_run: Optional[datetime] = None

class RoutineAutomation:
    """
    אוטומציית שגרה - רצפים אוטומטיים
    """
    def __init__(self):
        self.data_file = Path(__file__).parent.parent / "data" / "routines.json"
        self.data_file.parent.mkdir(parents=True, exist_ok=True)
        self.routines = self._load_routines()
        
        # רוטינות ברירת מחדל
        if not self.routines:
            self.routines = self._create_default_routines()
            self._save()

    def _load_routines(self) -> List[Routine]:
        if self.data_file.exists():
            try:
                data = json.loads(self.data_file.read_text(encoding='utf-8'))
                routines = []
                for r in data.get("routines", []):
                    steps = [RoutineStep(**s) for s in r.get("steps", [])]
                    routines.append(Routine(
                        id=r["id"],
                        name=r["name"],
                        description=r["description"],
                        schedule=r["schedule"],
                        steps=steps,
                        enabled=r.get("enabled", True),
                        last_run=datetime.fromisoformat(r["last_run"]) if r.get("last_run") else None
                    ))
                return routines
            except Exception as e:
                print(f"[Routine] Load failed: {e}")
        return []

    def _save(self):
        try:
            data = {
                "routines": [
                    {
                        "id": r.id,
                        "name": r.name,
                        "description": r.description,
                        "schedule": r.schedule,
                        "enabled": r.enabled,
                        "last_run": r.last_run.isoformat() if r.last_run else None,
                        "steps": [
                            {"id": s.id, "name": s.name, "action": s.action, "params": s.params, "order": s.order, "enabled": s.enabled}
                            for s in r.steps
                        ]
                    } for r in self.routines
                ]
            }
            self.data_file.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
        except Exception as e:
            print(f"[Routine] Save failed: {e}")

    def _create_default_routines(self) -> List[Routine]:
        """רוטינות ברירת מחדל - בוקר טוב של אדיאל"""
        morning = Routine(
            id="morning_routine",
            name="בוקר טוב - שגרת בוקר",
            description="כל בוקר ב-08:00: מייל, מזג אוויר, משימות, חדשות",
            schedule="08:00",
            steps=[
                RoutineStep("m1", "פתיחת מייל וסינון דחופים", "check_email", {"filter": "urgent", "limit": 10}, 1),
                RoutineStep("m2", "מזג אוויר היום", "weather", {"city": "תל אביב"}, 2),
                RoutineStep("m3", "משימות להיום מלוח שנה", "calendar_today", {}, 3),
                RoutineStep("m4", "חדשות טק חשובות", "research", {"topic": "AI news today", "depth": 3}, 4),
                RoutineStep("m5", "פתיחת אפליקציות עבודה", "open_apps", {"apps": ["chrome", "code", "spotify"]}, 5),
            ]
        )
        
        evening = Routine(
            id="evening_routine",
            name="סיכום יום - שגרת ערב",
            description="כל ערב ב-18:00: סיכום משימות, ארגון קבצים, דוח יומי",
            schedule="18:00",
            steps=[
                RoutineStep("e1", "סיכום מיילים שלא נקראו", "email_summary", {}, 1),
                RoutineStep("e2", "ארגון קבצי הורדות", "organize_files", {"source": "~/Downloads"}, 2),
                RoutineStep("e3", "הפקת דוח יומי", "daily_report", {}, 3),
            ]
        )
        
        focus = Routine(
            id="focus_mode",
            name="מצב פוקוס - Deep Work",
            description="כשצריך להתרכז: סגירת התראות, מוזיקה, טיימר",
            schedule="manual",
            steps=[
                RoutineStep("f1", "השתקת התראות", "mute_notifications", {}, 1),
                RoutineStep("f2", "ניגון מוזיקת פוקוס בספוטיפיי", "play_spotify", {"query": "focus music", "action": "play"}, 2),
                RoutineStep("f3", "פתיחת VS Code + טרמינל", "open_apps", {"apps": ["code", "terminal"]}, 3),
                RoutineStep("f4", "התחלת טיימר 50 דקות", "timer", {"minutes": 50}, 4),
            ]
        )
        
        return [morning, evening, focus]

    async def run_routine(self, routine_id: str) -> Dict:
        """הרצת רוטינה"""
        routine = next((r for r in self.routines if r.id == routine_id), None)
        
        if not routine:
            return {"success": False, "error": f"Routine {routine_id} not found"}
        
        if not routine.enabled:
            return {"success": False, "error": f"Routine {routine_id} disabled"}
        
        print(f"[Routine] 🚀 מריץ רוטינה: {routine.name}")
        
        results = []
        for step in sorted(routine.steps, key=lambda s: s.order):
            if not step.enabled:
                continue
            
            print(f"[Routine] Step {step.order}: {step.name} ({step.action})")
            
            try:
                result = await self._execute_step(step)
                results.append({
                    "step_id": step.id,
                    "name": step.name,
                    "action": step.action,
                    "success": result.get("success", False),
                    "result": result
                })
                await asyncio.sleep(0.5)  # הפסקה קטנה בין צעדים
            except Exception as e:
                results.append({
                    "step_id": step.id,
                    "name": step.name,
                    "success": False,
                    "error": str(e)
                })
        
        routine.last_run = datetime.now()
        self._save()
        
        success_count = sum(1 for r in results if r["success"])
        
        return {
            "success": True,
            "routine_id": routine.id,
            "routine_name": routine.name,
            "total_steps": len(routine.steps),
            "successful_steps": success_count,
            "failed_steps": len(results) - success_count,
            "results": results,
            "message": f"הרצתי רוטינת '{routine.name}' - {success_count}/{len(results)} צעדים הצליחו",
            "next_scheduled": routine.schedule
        }

    async def _execute_step(self, step: RoutineStep) -> Dict:
        """ביצוע צעד בודד"""
        action = step.action
        params = step.params
        
        # Email
        if action == "check_email":
            from .email_manager import get_email_manager
            email_mgr = get_email_manager()
            if params.get("filter") == "urgent":
                emails = await email_mgr.get_urgent_emails()
                return {"success": True, "urgent_count": len(emails), "emails": [e.subject for e in emails[:3]]}
            else:
                emails = await email_mgr.fetch_emails(limit=params.get("limit", 10))
                return {"success": True, "count": len(emails)}
        
        # Calendar
        elif action == "calendar_today":
            from .calendar_manager import get_calendar_manager
            cal = get_calendar_manager()
            events = await cal.get_events(days_ahead=1)
            return {"success": True, "events_today": len(events), "events": [e.title for e in events[:5]]}
        
        # Weather (דמו)
        elif action == "weather":
            city = params.get("city", "תל אביב")
            import random
            temp = random.randint(20, 32)
            return {"success": True, "city": city, "temp": temp, "condition": "בהיר", "message": f"{city}: {temp}° בהיר"}
        
        # Research
        elif action == "research":
            from .research_agent import get_research_agent
            agent = get_research_agent()
            result = await agent.research_topic(params.get("topic", "AI news"), depth=params.get("depth", 3))
            return result
        
        # Open apps
        elif action == "open_apps":
            from .computer_control import get_computer_control
            ctrl = get_computer_control()
            results = []
            for app_name in params.get("apps", []):
                res = ctrl.open_app(app_name)
                results.append(res)
            return {"success": True, "opened": results}
        
        # File organization
        elif action == "organize_files":
            from .cloud_file_manager import get_file_manager
            fm = get_file_manager()
            result = fm.organize_files(params.get("source", "~/Downloads"))
            return result
        
        # Spotify
        elif action == "play_spotify":
            from .computer_control import get_computer_control
            ctrl = get_computer_control()
            return ctrl.play_spotify(query=params.get("query", ""), action=params.get("action", "play"))
        
        # Timer
        elif action == "timer":
            minutes = params.get("minutes", 25)
            return {"success": True, "minutes": minutes, "message": f"טיימר {minutes} דקות התחיל (דמו)"}
        
        # Mute notifications (דמו)
        elif action == "mute_notifications":
            return {"success": True, "message": "התראות הושתקו (דמו - ב-Windows אמיתי היה משנה Focus Assist)"}
        
        # Email summary
        elif action == "email_summary":
            from .email_manager import get_email_manager
            mgr = get_email_manager()
            result = await mgr.categorize_inbox()
            return result
        
        # Daily report
        elif action == "daily_report":
            from .report_generator import get_report_generator
            gen = get_report_generator()
            result = await gen.full_report_flow()
            return result
        
        else:
            return {"success": False, "error": f"Unknown action {action}"}

    def list_routines(self) -> List[Dict]:
        return [
            {
                "id": r.id,
                "name": r.name,
                "description": r.description,
                "schedule": r.schedule,
                "enabled": r.enabled,
                "steps_count": len(r.steps),
                "last_run": r.last_run.isoformat() if r.last_run else None
            } for r in self.routines
        ]

    def create_routine(self, name: str, description: str, schedule: str, steps: List[Dict]) -> Dict:
        routine_id = f"routine_{int(datetime.now().timestamp())}"
        routine_steps = [
            RoutineStep(
                id=f"step_{i}_{routine_id}",
                name=s["name"],
                action=s["action"],
                params=s.get("params", {}),
                order=s.get("order", i),
                enabled=s.get("enabled", True)
            ) for i, s in enumerate(steps)
        ]
        
        routine = Routine(
            id=routine_id,
            name=name,
            description=description,
            schedule=schedule,
            steps=routine_steps
        )
        
        self.routines.append(routine)
        self._save()
        
        return {"success": True, "routine_id": routine_id, "routine": routine}

def get_routine_automation():
    return RoutineAutomation()
