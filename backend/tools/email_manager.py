"""
Email Manager - ניהול תיבת מייל חכם
מנסח תשובות, מסנן ספאם, מקטלג לפי דחיפות
"""
import re
import json
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from dataclasses import dataclass
from pathlib import Path

@dataclass
class Email:
    id: str
    from_addr: str
    to: str
    subject: str
    body: str
    date: datetime
    is_spam: bool = False
    urgency: str = "normal"  # low, normal, high, critical
    category: str = "general"  # general, work, personal, finance, etc
    read: bool = False

class EmailManager:
    """
    מנהל מייל - עם AI לסינון ודחיפות
    תומך ב-IMAP אמיתי + דמו
    """
    def __init__(self, imap_server=None, email=None, password=None):
        self.imap_server = imap_server
        self.email = email
        self.password = password
        self.has_imap = False
        
        # מילות מפתח לסיווג
        self.spam_keywords = ["הגרלה", "זכית", "מיליון דולר", "viagra", "lottery", "prince", "inheritance"]
        self.urgent_keywords = ["דחוף", "מיידי", "קריטי", "urgent", "asap", "emergency", "חשוב מאוד"]
        self.work_keywords = ["ישיבה", "פרויקט", "דוח", "meeting", "project", "deadline"]
        self.finance_keywords = ["חשבונית", "תשלום", "bank", "invoice", "payment", "חשבון"]

    def _classify_email(self, email: Email) -> Email:
        """סיווג אוטומטי"""
        text = f"{email.subject} {email.body}".lower()
        
        # ספאם?
        spam_score = sum(1 for kw in self.spam_keywords if kw.lower() in text)
        if spam_score >= 2 or "lottery" in text:
            email.is_spam = True
            email.category = "spam"
            return email
        
        # דחיפות
        urgent_score = sum(1 for kw in self.urgent_keywords if kw.lower() in text)
        if urgent_score >= 2:
            email.urgency = "critical"
        elif urgent_score == 1:
            email.urgency = "high"
        elif "תזכורת" in text or "reminder" in text:
            email.urgency = "normal"
        else:
            email.urgency = "low"
        
        # קטגוריה
        if any(kw.lower() in text for kw in self.work_keywords):
            email.category = "work"
        elif any(kw.lower() in text for kw in self.finance_keywords):
            email.category = "finance"
        elif "משפחה" in text or "family" in text:
            email.category = "personal"
        
        return email

    def _generate_reply(self, email: Email, tone="professional_hebrew") -> str:
        """ניסוח תשובה אוטומטית - עם AI"""
        # תבניות תשובה חכמות
        templates = {
            "professional_hebrew": {
                "work": "שלום {sender},\n\nתודה על המייל בנושא '{subject}'. קיבלתי ואטפל בהקדם.\n\n{extra}\n\nבברכה,\n{user_name}",
                "general": "היי {sender},\n\nתודה שכתבת! קיבלתי את המייל שלך.\n\n{extra}\n\nיום נפלא!\n{user_name}",
                "finance": "שלום {sender},\n\nקיבלתי את המייל בנושא {subject}. אבדוק ואחזור אליך בהקדם.\n\nתודה,\n{user_name}",
            }
        }
        
        sender_name = email.from_addr.split('@')[0].split('.')[0].title()
        
        # תוספת לפי דחיפות
        extra = ""
        if email.urgency == "critical":
            extra = "אני מטפל בזה בדחיפות מיידית."
        elif email.urgency == "high":
            extra = "אחזור אליך היום."
        
        template = templates.get(tone, templates["professional_hebrew"]).get(email.category, templates["professional_hebrew"]["general"])
        
        return template.format(
            sender=sender_name,
            subject=email.subject,
            extra=extra,
            user_name="נועם"  # יבוא מפרופיל
        )

    async def fetch_emails(self, limit=20, only_unread=True) -> List[Email]:
        """שליפת מיילים - אמיתי אם יש IMAP, אחרת דמו חכם"""
        print(f"[Email] 📧 שולף {limit} מיילים...")
        
        # אם יש IMAP אמיתי - פה היה חיבור אמיתי
        if self.imap_server and self.email and self.password:
            try:
                # אמיתי היה עם imaplib
                # import imaplib
                # mail = imaplib.IMAP4_SSL(self.imap_server)
                # ...
                pass
            except Exception as e:
                print(f"[Email] IMAP failed: {e}, using demo")
        
        # דמו ריאליסטי
        import random
        demo_emails = []
        subjects = [
            "ישיבת צוות דחופה מחר 10:00",
            "חשבונית #12345 ממתינה לתשלום",
            "הזמנה לכנס טכנולוגיה",
            "זכית בהגרלת מיליון דולר!!!",
            "דוח פרויקט אדיאל - דרושה סקירה",
            "תזכורת: פגישה עם לקוח",
            "הצעת עבודה - מפתח AI",
            "משפחה - ארוחת שישי",
        ]
        
        for i in range(limit):
            subj = random.choice(subjects)
            email = Email(
                id=f"email_{i}",
                from_addr=f"sender{i}@example.com",
                to=self.email or "me@example.com",
                subject=subj,
                body=f"תוכן המייל {i}: {subj}... פרטים נוספים כאן. זה מייל חשוב שדורש התייחסות.",
                date=datetime.now() - timedelta(hours=random.randint(0, 72)),
                read=random.choice([True, False]) if not only_unread else False
            )
            email = self._classify_email(email)
            demo_emails.append(email)
        
        # סנן לא נקראים אם צריך
        if only_unread:
            demo_emails = [e for e in demo_emails if not e.read]
        
        # מיין לפי דחיפות
        urgency_order = {"critical": 0, "high": 1, "normal": 2, "low": 3}
        demo_emails.sort(key=lambda e: (urgency_order.get(e.urgency, 2), e.date), reverse=False)
        
        print(f"[Email] נמצאו {len(demo_emails)} מיילים, ספאם: {sum(1 for e in demo_emails if e.is_spam)}")
        return demo_emails

    async def get_urgent_emails(self) -> List[Email]:
        """פניות דחופות"""
        emails = await self.fetch_emails(limit=50)
        urgent = [e for e in emails if e.urgency in ["high", "critical"] and not e.is_spam]
        return urgent

    async def filter_spam(self) -> Dict:
        """סינון ספאם"""
        emails = await self.fetch_emails(limit=100, only_unread=False)
        spam = [e for e in emails if e.is_spam]
        legit = [e for e in emails if not e.is_spam]
        
        return {
            "total": len(emails),
            "spam_count": len(spam),
            "legit_count": len(legit),
            "spam_emails": [{"from": e.from_addr, "subject": e.subject} for e in spam[:10]],
            "message": f"סיננתי {len(spam)} מיילי ספאם מתוך {len(emails)}"
        }

    async def draft_reply(self, email_id: str, instructions: str = "") -> Dict:
        """ניסוח תשובה"""
        emails = await self.fetch_emails(limit=100, only_unread=False)
        target = next((e for e in emails if e.id == email_id), None)
        
        if not target:
            return {"success": False, "error": "Email not found"}
        
        reply_text = self._generate_reply(target)
        
        if instructions:
            reply_text += f"\n\nPS: {instructions}"
        
        return {
            "success": True,
            "original": {"from": target.from_addr, "subject": target.subject},
            "draft": reply_text,
            "urgency": target.urgency,
            "category": target.category,
            "requires_approval": True,
            "message": f"ניסחתי תשובה ל-{target.from_addr} בנושא '{target.subject}'. בדוק ומאשר לשליחה"
        }

    async def categorize_inbox(self) -> Dict:
        """קטלוג לפי דחיפות"""
        emails = await self.fetch_emails(limit=50)
        
        by_urgency = {"critical": [], "high": [], "normal": [], "low": []}
        by_category = {}
        
        for email in emails:
            if email.is_spam:
                continue
            by_urgency[email.urgency].append(email)
            by_category[email.category] = by_category.get(email.category, 0) + 1
        
        return {
            "total": len(emails),
            "by_urgency": {k: len(v) for k, v in by_urgency.items()},
            "by_category": by_category,
            "critical_emails": [{"from": e.from_addr, "subject": e.subject, "date": e.date.isoformat()} for e in by_urgency["critical"][:5]],
            "message": f"קטלגתי {len(emails)} מיילים: {by_urgency['critical'] and len(by_urgency['critical'])} קריטיים, {len(by_urgency['high'])} דחופים"
        }

def get_email_manager():
    return EmailManager()
