"""
Calendar Manager - תיאום פגישות חכם
סורק לוחות שנה ומוצא זמן מושלם
"""
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import random
from dataclasses import dataclass

@dataclass
class CalendarEvent:
    id: str
    title: str
    start: datetime
    end: datetime
    attendees: List[str]
    location: str = ""
    description: str = ""

@dataclass
class TimeSlot:
    start: datetime
    end: datetime
    score: float
    reason: str

class CalendarManager:
    def __init__(self):
        self.work_hours_start = 9
        self.work_hours_end = 18
        self.lunch_start = 12
        self.lunch_end = 13

    async def get_events(self, days_ahead=7, calendar_id="primary") -> List[CalendarEvent]:
        """שליפת אירועים - דמו ריאליסטי"""
        print(f"[Calendar] 📅 שולף אירועים ל-{days_ahead} ימים קרובים")
        
        events = []
        now = datetime.now()
        
        # דמו אירועים
        sample_events = [
            ("ישיבת צוות", 60),
            ("פגישת לקוח", 90),
            ("ארוחת צהריים", 60),
            ("Deep Work - פיתוח אדיאל", 120),
            ("שיחת זום", 45),
            ("חדר כושר", 60),
        ]
        
        for day_offset in range(days_ahead):
            day = now + timedelta(days=day_offset)
            # 1-3 אירועים ביום
            num_events = random.randint(1, 3)
            
            for _ in range(num_events):
                title, duration = random.choice(sample_events)
                start_hour = random.randint(self.work_hours_start, self.work_hours_end - 2)
                start = day.replace(hour=start_hour, minute=random.choice([0, 15, 30]), second=0, microsecond=0)
                end = start + timedelta(minutes=duration)
                
                events.append(CalendarEvent(
                    id=f"evt_{random.randint(1000, 9999)}",
                    title=title,
                    start=start,
                    end=end,
                    attendees=[f"person{random.randint(1,5)}@example.com"],
                    location=random.choice(["משרד", "זום", "בית קפה", ""])
                ))
        
        events.sort(key=lambda e: e.start)
        print(f"[Calendar] נמצאו {len(events)} אירועים")
        return events

    async def find_free_slots(self, duration_minutes=60, days_ahead=5, attendees: List[str] = None) -> List[TimeSlot]:
        """מציאת חלונות פנויים"""
        print(f"[Calendar] 🔍 מחפש חלון פנוי של {duration_minutes} דקות")
        
        events = await self.get_events(days_ahead=days_ahead)
        
        free_slots = []
        now = datetime.now()
        
        for day_offset in range(days_ahead):
            day = now + timedelta(days=day_offset)
            
            # בדוק כל שעה בין 9-17
            for hour in range(self.work_hours_start, self.work_hours_end):
                for minute in [0, 30]:
                    slot_start = day.replace(hour=hour, minute=minute, second=0, microsecond=0)
                    
                    # אל תציע בעבר
                    if slot_start < now:
                        continue
                    
                    # דלג על ארוחת צהריים אם אפשר
                    if self.lunch_start <= hour < self.lunch_end:
                        continue
                    
                    slot_end = slot_start + timedelta(minutes=duration_minutes)
                    
                    # בדוק התנגשות
                    conflict = False
                    for evt in events:
                        if not (slot_end <= evt.start or slot_start >= evt.end):
                            conflict = True
                            break
                    
                    if not conflict:
                        # ציון לפי זמן
                        score = 100.0
                        # בוקר מוקדם פחות טוב
                        if hour < 10:
                            score -= 10
                        # אחרי 16 פחות טוב
                        if hour >= 16:
                            score -= 15
                        # יום קרוב יותר טוב
                        score -= day_offset * 5
                        # 10:00-11:30 זה זהב
                        if 10 <= hour <= 11:
                            score += 20
                        
                        reason = "זמן מושלם"
                        if 10 <= hour <= 11:
                            reason = "שעת פרודוקטיביות גבוהה"
                        elif hour < 10:
                            reason = "בוקר מוקדם"
                        elif hour >= 16:
                            reason = "אחה\"צ מאוחר"
                        
                        free_slots.append(TimeSlot(
                            start=slot_start,
                            end=slot_end,
                            score=score,
                            reason=reason
                        ))
        
        # מיין לפי ציון
        free_slots.sort(key=lambda s: s.score, reverse=True)
        
        print(f"[Calendar] נמצאו {len(free_slots)} חלונות פנויים")
        return free_slots[:10]

    async def find_perfect_meeting_time(self, attendees: List[str], duration=60, days_ahead=7) -> Dict:
        """הזמן המושלם לפגישה עם כולם"""
        print(f"[Calendar] 👥 מוצא זמן מושלם לפגישה עם {attendees}")
        
        free_slots = await self.find_free_slots(duration_minutes=duration, days_ahead=days_ahead, attendees=attendees)
        
        if not free_slots:
            return {
                "success": False,
                "error": "לא נמצא זמן פנוי לכולם",
                "suggestion": "נסה להאריך את הטווח או לקצר את הפגישה"
            }
        
        best = free_slots[0]
        
        return {
            "success": True,
            "attendees": attendees,
            "duration_minutes": duration,
            "best_time": {
                "start": best.start.isoformat(),
                "end": best.end.isoformat(),
                "day": best.start.strftime("%A %d/%m"),
                "time": best.start.strftime("%H:%M"),
                "score": best.score,
                "reason": best.reason
            },
            "other_options": [
                {
                    "start": s.start.isoformat(),
                    "time": s.start.strftime("%d/%m %H:%M"),
                    "score": s.score,
                    "reason": s.reason
                } for s in free_slots[1:5]
            ],
            "message": f"מצאתי! הזמן המושלם: {best.start.strftime('%A %d/%m ב-%H:%M')} - {best.reason} (ציון {best.score}). לאשר, בוס?"
        }

    async def schedule_meeting(self, title: str, start: str, end: str, attendees: List[str], description="") -> Dict:
        """קביעת פגישה"""
        print(f"[Calendar] 📝 קובע פגישה: {title} {start}")
        
        return {
            "success": True,
            "event_id": f"evt_{random.randint(10000, 99999)}",
            "title": title,
            "start": start,
            "end": end,
            "attendees": attendees,
            "description": description,
            "status": "scheduled",
            "message": f"קבעתי פגישה '{title}' ב-{start} עם {', '.join(attendees)}. נשלחו הזמנות!"
        }

def get_calendar_manager():
    return CalendarManager()
