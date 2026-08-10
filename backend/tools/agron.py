"""
אגרון - Agron - מערכת אגירת ואיגום מידע
מאגד מידע מכל המקורות: מיילים, יומן, קבצים, מחקר, זיכרון, מילון
והופך למאגר ידע אחד חכם לאדיאל

כמו ש-Datalake אוגר הכל
"""
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
from collections import defaultdict, Counter
import re

DATA_DIR = Path(__file__).parent.parent / "data"
AGRON_FILE = DATA_DIR / "agron_knowledge.json"
AGRON_INDEX = DATA_DIR / "agron_index.json"

class Agron:
    """
    אגרון - מאגר הידע המרכזי של אדיאל
    אוגר הכל ויוצר מאגר אחד חכם
    """
    def __init__(self):
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.knowledge = self._load_knowledge()
        self.index = defaultdict(list)  # מילה -> רשימת מקורות
        self._build_index()
        print(f"[Agron] 📦 אגרון נטען: {len(self.knowledge.get('items', []))} פריטים, {len(self.index)} מילות אינדקס")

    def _load_knowledge(self) -> Dict:
        if AGRON_FILE.exists():
            try:
                return json.loads(AGRON_FILE.read_text(encoding='utf-8'))
            except:
                pass
        return {
            "items": [],  # כל פריטי המידע
            "sources": Counter(),  # ספירת מקורות
            "created_at": datetime.now().isoformat(),
            "last_update": None
        }

    def _save(self):
        try:
            self.knowledge["last_update"] = datetime.now().isoformat()
            AGRON_FILE.write_text(json.dumps(self.knowledge, ensure_ascii=False, indent=2), encoding='utf-8')
            
            # שמור אינדקס
            index_data = {k: v[-20:] for k, v in self.index.items()}  # 20 אחרונים לכל מילה
            AGRON_INDEX.write_text(json.dumps(index_data, ensure_ascii=False, indent=2), encoding='utf-8')
        except Exception as e:
            print(f"[Agron] Save failed: {e}")

    def _build_index(self):
        """בונה אינדקס חיפוש מהיר"""
        self.index.clear()
        for idx, item in enumerate(self.knowledge.get("items", [])[-500:]):  # 500 אחרונים
            text = f"{item.get('title','')} {item.get('content','')} {item.get('source','')}"
            words = re.findall(r'[\u0590-\u05FF]{2,}|[a-zA-Z]{3,}', text.lower())
            for word in set(words):
                if len(word) > 2:
                    self.index[word].append(idx)

    def ingest(self, source: str, title: str, content: str, metadata: Dict = None) -> Dict:
        """
        אוגר פריט מידע חדש
        source: email, calendar, file, research, memory, dictionary, etc.
        """
        metadata = metadata or {}
        
        item = {
            "id": f"agron_{len(self.knowledge['items'])}_{int(datetime.now().timestamp())}",
            "source": source,
            "title": title,
            "content": content[:2000],  # הגבל אורך
            "metadata": metadata,
            "created_at": datetime.now().isoformat(),
            "tags": self._extract_tags(content),
            "importance": self._calc_importance(source, content)
        }
        
        self.knowledge["items"].append(item)
        self.knowledge["sources"][source] += 1
        
        # עדכן אינדקס
        words = re.findall(r'[\u0590-\u05FF]{2,}|[a-zA-Z]{3,}', f"{title} {content}".lower())
        for word in set(words):
            if len(word) > 2:
                self.index[word].append(len(self.knowledge["items"])-1)
        
        # שמור כל 10 פריטים
        if len(self.knowledge["items"]) % 10 == 0:
            self._save()
        
        print(f"[Agron] + {source}: {title[:40]}... (importance {item['importance']})")
        return item

    def _extract_tags(self, text: str) -> List[str]:
        """חילוץ תגיות אוטומטי"""
        tags = []
        text_lower = text.lower()
        
        # קטגוריות
        if any(w in text_lower for w in ["פגישה", "ישיבה", "meeting"]):
            tags.append("פגישה")
        if any(w in text_lower for w in ["פרויקט", "project", "אדיאל"]):
            tags.append("פרויקט")
        if any(w in text_lower for w in ["קנייה", "קנה", "shopping"]):
            tags.append("קניות")
        if any(w in text_lower for w in ["טיסה", "מלון", "חופשה"]):
            tags.append("חופשה")
        if any(w in text_lower for w in ["שגיאה", "באג", "error"]):
            tags.append("באג")
        if any(w in text_lower for w in ["קוד", "תכנות", "code"]):
            tags.append("קוד")
        
        return tags[:5]

    def _calc_importance(self, source: str, content: str) -> int:
        """חישוב חשיבות 1-10"""
        score = 5  # ברירת מחדל
        
        # מקור חשוב
        if source in ["memory", "profile"]:
            score += 2
        if source in ["email"] and any(w in content.lower() for w in ["דחוף", "קריטי", "urgent"]):
            score += 3
        
        # תוכן חשוב
        if len(content) > 500:
            score += 1
        if any(w in content.lower() for w in ["אדיאל", "בוס", "חשוב"]):
            score += 1
        
        return min(score, 10)

    def search(self, query: str, limit=10, min_importance=0) -> List[Dict]:
        """חיפוש במאגר הידע"""
        print(f"[Agron] 🔍 מחפש '{query}'")
        
        query_words = re.findall(r'[\u0590-\u05FF]{2,}|[a-zA-Z]{3,}', query.lower())
        if not query_words:
            return []
        
        # מצא פריטים עם המילים
        candidate_indices = set()
        for word in query_words:
            if word in self.index:
                candidate_indices.update(self.index[word])
        
        # דרג לפי רלוונטיות
        scored = []
        for idx in candidate_indices:
            if idx >= len(self.knowledge["items"]):
                continue
            item = self.knowledge["items"][idx]
            
            if item.get("importance", 0) < min_importance:
                continue
            
            # חשב score
            text = f"{item['title']} {item['content']}".lower()
            score = sum(1 for qw in query_words if qw in text)
            score += item.get("importance", 0) * 0.5
            
            scored.append((score, item))
        
        scored.sort(key=lambda x: x[0], reverse=True)
        
        results = [item for score, item in scored[:limit]]
        print(f"[Agron] נמצאו {len(results)} תוצאות ל-'{query}'")
        return results

    def aggregate_all_sources(self) -> Dict:
        """איגום מכל המקורות - כמו שביקשת לחבר לאדיאל"""
        print(f"[Agron] 📦 מאגד מכל המקורות לאדיאל...")
        
        aggregated = {
            "total_items": len(self.knowledge["items"]),
            "by_source": dict(self.knowledge["sources"]),
            "by_tag": Counter(),
            "recent": [],
            "important": []
        }
        
        # ספירת תגיות
        for item in self.knowledge["items"]:
            for tag in item.get("tags", []):
                aggregated["by_tag"][tag] += 1
        
        # פריטים אחרונים
        aggregated["recent"] = self.knowledge["items"][-10:]
        
        # פריטים חשובים
        sorted_by_importance = sorted(self.knowledge["items"], key=lambda x: x.get("importance", 0), reverse=True)
        aggregated["important"] = sorted_by_importance[:10]
        
        return aggregated

    def get_context_for_brain(self, query: str = "", limit=5) -> str:
        """מחזיר הקשר למוח של אדיאל - חיבור לאדיאל!"""
        if not query:
            # החזר סיכום כללי
            agg = self.aggregate_all_sources()
            context = f"מאגר אגרון: {agg['total_items']} פריטים. "
            context += f"מקורות: {', '.join([f'{k}:{v}' for k,v in list(agg['by_source'].items())[:3]])}. "
            if agg["important"]:
                context += f"הכי חשוב: {agg['important'][0]['title'][:50]}..."
            return context
        
        # חיפוש רלוונטי לשאלה
        results = self.search(query, limit=limit)
        if not results:
            return ""
        
        context = f"מידע מאגרון על '{query}':\n"
        for i, item in enumerate(results[:3], 1):
            context += f"{i}. [{item['source']}] {item['title']}: {item['content'][:150]}...\n"
        
        return context

    def ingest_from_all_tools(self):
        """אוגר מכל הכלים הקיימים - חיבור מלא לאדיאל"""
        print("[Agron] 🔄 אוגר מכל כלי אדיאל...")
        
        # 1. מזיכרון
        try:
            from ..core.memory import AdielMemory
            mem = AdielMemory()
            for conv in mem.long_term.get("conversations", [])[-20:]:
                self.ingest(
                    source="memory",
                    title=f"שיחה: {conv.get('user','')[:30]}",
                    content=f"User: {conv.get('user','')}\nAssistant: {conv.get('assistant','')}",
                    metadata={"type": "conversation"}
                )
        except Exception as e:
            print(f"[Agron] Memory ingest failed: {e}")
        
        # 2. מפרופיל
        try:
            from ..core.learning_engine import get_learning_engine
            learning = get_learning_engine()
            profile = learning.get_user_profile_summary()
            if profile.get("name"):
                self.ingest(
                    source="profile",
                    title=f"פרופיל: {profile['name']}",
                    content=f"שם: {profile['name']}, פרויקטים: {profile.get('projects', [])}, שיחות: {profile.get('interaction_count',0)}",
                    metadata={"type": "profile"}
                )
        except Exception as e:
            print(f"[Agron] Profile ingest failed: {e}")
        
        # 3. ממילון
        try:
            from ..core.hebrew_dictionary import get_hebrew_dictionary
            heb_dict = get_hebrew_dictionary()
            stats = heb_dict.get_stats()
            self.ingest(
                source="dictionary",
                title=f"מילון עברי: {stats['total_all']} מילים",
                content=f"מילון עם {stats['total_words']} מילים, {stats['total_slang']} סלנג, {stats['total_tech']} טכני",
                metadata={"type": "dictionary", "stats": stats}
            )
        except Exception as e:
            print(f"[Agron] Dictionary ingest failed: {e}")
        
        self._save()
        print(f"[Agron] ✅ איגום הושלם: {len(self.knowledge['items'])} פריטים")

# Singleton
_global_agron = None

def get_agron() -> Agron:
    global _global_agron
    if _global_agron is None:
        _global_agron = Agron()
    return _global_agron
