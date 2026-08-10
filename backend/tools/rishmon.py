"""
רישמון - Rishmon - מערכת רישום וזהויות
מנהל רשימות, זהויות, הרשאות, ורישום מרכזי

מחובר לאדיאל ג'וניור + אגרון
"""
import json
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
from collections import defaultdict
import re
import uuid

DATA_DIR = Path(__file__).parent.parent / "data"
RISHMON_FILE = DATA_DIR / "rishmon_registry.json"
ZEHUTON_FILE = DATA_DIR / "zehuton_identities.json"

class Zehuton:
    """
    זהותון - ניהול זהויות
    כמו Identity Token
    """
    def __init__(self, user_id: str = None, name: str = None):
        self.id = user_id or f"zehut_{uuid.uuid4().hex[:8]}"
        self.name = name or "אנונימי"
        self.created_at = datetime.now().isoformat()
        self.permissions = []
        self.attributes = {}
        self.verified = False

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "created_at": self.created_at,
            "permissions": self.permissions,
            "attributes": self.attributes,
            "verified": self.verified
        }

class Rishmon:
    """
    רישמון - מערכת רישום מרכזית
    רושם הכל: משתמשים, משימות, קבצים, פרויקטים, זהויות
    
    מחובר לאדיאל + אגרון
    """
    def __init__(self):
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.registry = self._load_registry()
        self.zehutonim = self._load_zehutonim()
        
        print(f"[Rishmon] 📋 רישמון נטען: {len(self.registry.get('entries', []))} רשומות")
        print(f"[Rishmon] 🆔 זהותון: {len(self.zehutonim.get('identities', []))} זהויות")

    def _load_registry(self) -> Dict:
        if RISHMON_FILE.exists():
            try:
                return json.loads(RISHMON_FILE.read_text(encoding='utf-8'))
            except:
                pass
        return {
            "entries": [],
            "by_type": defaultdict(list),
            "created_at": datetime.now().isoformat(),
            "version": "1.0"
        }

    def _load_zehutonim(self) -> Dict:
        if ZEHUTON_FILE.exists():
            try:
                data = json.loads(ZEHUTON_FILE.read_text(encoding='utf-8'))
                # המרה מ-defaultdict אם צריך
                return data
            except:
                pass
        return {
            "identities": [],
            "by_name": {},
            "created_at": datetime.now().isoformat()
        }

    def _save_registry(self):
        try:
            # המר defaultdict ל-dict רגיל לשמירה
            data_to_save = {
                "entries": self.registry["entries"][-500:],  # 500 אחרונים
                "by_type": dict(self.registry.get("by_type", {})),
                "created_at": self.registry.get("created_at"),
                "last_update": datetime.now().isoformat(),
                "version": "1.0",
                "total": len(self.registry["entries"])
            }
            RISHMON_FILE.write_text(json.dumps(data_to_save, ensure_ascii=False, indent=2), encoding='utf-8')
        except Exception as e:
            print(f"[Rishmon] Save registry failed: {e}")

    def _save_zehutonim(self):
        try:
            ZEHUTON_FILE.write_text(json.dumps(self.zehutonim, ensure_ascii=False, indent=2), encoding='utf-8')
        except Exception as e:
            print(f"[Rishmon] Save zehutonim failed: {e}")

    def register(self, entry_type: str, title: str, data: Dict = None, owner_zehut: str = None) -> Dict:
        """
        רושם רשומה חדשה ברישמון
        entry_type: user, task, file, project, email, etc.
        """
        data = data or {}
        
        entry = {
            "id": f"rishmon_{entry_type}_{uuid.uuid4().hex[:8]}",
            "type": entry_type,
            "title": title,
            "data": data,
            "owner_zehut": owner_zehut,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "hash": self._hash_entry(title, data),
            "status": "active"
        }
        
        self.registry["entries"].append(entry)
        
        # אינדקס לפי type
        if isinstance(self.registry["by_type"], dict):
            if entry_type not in self.registry["by_type"]:
                self.registry["by_type"][entry_type] = []
            self.registry["by_type"][entry_type].append(entry["id"])
        else:
            # אם זה defaultdict שנטען כ-dict, המרה
            self.registry["by_type"] = defaultdict(list, self.registry["by_type"])
            self.registry["by_type"][entry_type].append(entry["id"])
        
        self._save_registry()
        
        print(f"[Rishmon] + {entry_type}: {title[:40]}... ({entry['id']})")
        return entry

    def _hash_entry(self, title: str, data: Dict) -> str:
        """יוצר hash לרשומה"""
        content = f"{title}{json.dumps(data, sort_keys=True)}"
        return hashlib.md5(content.encode('utf-8')).hexdigest()[:12]

    def search(self, query: str, entry_type: str = None, limit=20) -> List[Dict]:
        """חיפוש ברישמון"""
        print(f"[Rishmon] 🔍 מחפש '{query}' type={entry_type}")
        
        query_lower = query.lower()
        results = []
        
        for entry in reversed(self.registry["entries"]):  # מהחדש לישן
            if entry_type and entry["type"] != entry_type:
                continue
            
            text = f"{entry['title']} {json.dumps(entry['data'], ensure_ascii=False)}".lower()
            if query_lower in text or any(word in text for word in query_lower.split() if len(word) > 2):
                results.append(entry)
                if len(results) >= limit:
                    break
        
        print(f"[Rishmon] נמצאו {len(results)} תוצאות")
        return results

    def list_by_type(self, entry_type: str, limit=50) -> List[Dict]:
        """רשימת רשומות לפי סוג"""
        entries = []
        for entry in reversed(self.registry["entries"]):
            if entry["type"] == entry_type:
                entries.append(entry)
                if len(entries) >= limit:
                    break
        return entries

    # === זהותון - ניהול זהויות ===

    def create_zehuton(self, name: str, attributes: Dict = None, permissions: List[str] = None) -> Zehuton:
        """יוצר זהות חדשה - זהותון"""
        zehut = Zehuton(name=name)
        zehut.attributes = attributes or {}
        zehut.permissions = permissions or ["basic"]
        
        self.zehutonim["identities"].append(zehut.to_dict())
        self.zehutonim["by_name"][name] = zehut.id
        
        self._save_zehutonim()
        
        # גם רושם ברישמון
        self.register(
            entry_type="identity",
            title=f"זהותון חדש: {name}",
            data=zehut.to_dict(),
            owner_zehut=zehut.id
        )
        
        print(f"[Rishmon] 🆔 זהותון חדש: {name} ({zehut.id})")
        return zehut

    def get_zehuton(self, identifier: str) -> Optional[Dict]:
        """שליפת זהות לפי ID או שם"""
        # חיפוש לפי ID
        for ident in self.zehutonim["identities"]:
            if ident["id"] == identifier or ident["name"] == identifier:
                return ident
        
        # חיפוש לפי שם ב-by_name
        zehut_id = self.zehutonim.get("by_name", {}).get(identifier)
        if zehut_id:
            for ident in self.zehutonim["identities"]:
                if ident["id"] == zehut_id:
                    return ident
        
        return None

    def verify_zehuton(self, zehut_id: str) -> bool:
        """אימות זהות"""
        zehut = self.get_zehuton(zehut_id)
        if not zehut:
            return False
        
        # במציאות היה בדיקת חתימה, תעודה, וכו'
        # כאן פשוט מסמן כמאומת
        for ident in self.zehutonim["identities"]:
            if ident["id"] == zehut_id:
                ident["verified"] = True
                ident["verified_at"] = datetime.now().isoformat()
                break
        
        self._save_zehutonim()
        print(f"[Rishmon] ✅ זהותון אומת: {zehut_id}")
        return True

    def list_zehutonim(self) -> List[Dict]:
        """רשימת כל הזהויות"""
        return self.zehutonim["identities"]

    # === חיבור לאדיאל ===

    def get_context_for_adiel(self, query: str = "") -> str:
        """הקשר לאדיאל - מחובר לאדיאל ג'וניור!"""
        recent = self.registry["entries"][-5:] if self.registry["entries"] else []
        identities = self.zehutonim["identities"][-3:] if self.zehutonim["identities"] else []
        
        context = f"רישמון: {len(self.registry['entries'])} רשומות, {len(identities)} זהויות. "
        
        if recent:
            context += f"אחרון: {recent[-1]['title'][:30]}... "
        
        if query:
            results = self.search(query, limit=3)
            if results:
                context += f"חיפוש '{query}': {len(results)} תוצאות. "
        
        return context

    def connect_to_adiel(self):
        """חיבור מלא לאדיאל - אגרון + רישמון + זהותון"""
        print("[Rishmon] 🔗 מחבר אגרון + רישמון + זהותון לאדיאל...")
        
        # 1. תן לאגרון לאגור גם מרישמון
        try:
            from .agron import get_agron
            agron = get_agron()
            
            # אגור רשומות חשובות מרישמון
            for entry in self.registry["entries"][-10:]:
                agron.ingest(
                    source="rishmon",
                    title=entry["title"],
                    content=f"Type: {entry['type']}, Data: {str(entry['data'])[:200]}",
                    metadata={"rishmon_id": entry["id"], "type": entry["type"]}
                )
            
            print(f"[Rishmon] ✓ חובר לאגרון - {len(self.registry['entries'])} רשומות אוגרו")
        except Exception as e:
            print(f"[Rishmon] Agron connect failed: {e}")
        
        # 2. צור זהותון ברירת מחדל לבוס
        try:
            existing = self.get_zehuton("בוס")
            if not existing:
                boss_zehut = self.create_zehuton(
                    name="בוס",
                    attributes={"role": "owner", "is_boss": True},
                    permissions=["all", "admin", "boss"]
                )
                self.verify_zehuton(boss_zehut.id)
                print(f"[Rishmon] ✓ זהותון בוס נוצר: {boss_zehut.id}")
        except Exception as e:
            print(f"[Rishmon] Zehuton boss creation failed: {e}")
        
        print("[Rishmon] ✅ חיבור לאדיאל הושלם - אגרון + רישמון + זהותון מחוברים!")

# Singleton
_global_rishmon = None

def get_rishmon() -> Rishmon:
    global _global_rishmon
    if _global_rishmon is None:
        _global_rishmon = Rishmon()
    return _global_rishmon

def get_zehuton_manager():
    return get_rishmon()
