"""
Memory Engine - Built from scratch
מערכת זיכרון פרטית לאדיאל - ללא תלות ב-vector DB חיצוני
כולל TF-IDF + Cosine similarity ממומש מאפס
"""
import json
import os
import re
import math
from collections import Counter, defaultdict
from datetime import datetime
from typing import List, Dict, Any, Tuple

MEMORY_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "memory.json")
SHORT_TERM_LIMIT = 12

class HebrewTokenizer:
    """טוקנייזר עברי פשוט - ממומש מאפס"""
    @staticmethod
    def tokenize(text: str) -> List[str]:
        # ניקוי בסיסי, תמיכה בעברית + אנגלית
        text = text.lower()
        # שמור עברית, אנגלית, מספרים
        tokens = re.findall(r'[\u0590-\u05FF]+|[a-zA-Z]+|\d+', text)
        # הסר stopwords עבריות בסיסיות
        stopwords = {"את", "על", "עם", "של", "זה", "זאת", "אני", "אתה", "הוא", "היא", "אנחנו", "אתם", "הם", "מה", "מי", "איפה", "כמה", "למה", "כי", "אם", "אז", "גם", "כן", "לא", "את", "את", "של", "כל"}
        return [t for t in tokens if t not in stopwords and len(t) > 1]

class LocalEmbeddingEngine:
    """מנוע Embedding מקומי - TF-IDF ממומש מאפס"""
    def __init__(self):
        self.doc_freq = Counter()
        self.total_docs = 0
        self.vocab = set()

    def fit(self, documents: List[str]):
        """למד את ה-IDF מהמסמכים"""
        self.total_docs = len(documents)
        self.doc_freq = Counter()
        for doc in documents:
            tokens = set(HebrewTokenizer.tokenize(doc))
            self.vocab.update(tokens)
            for tok in tokens:
                self.doc_freq[tok] += 1

    def tfidf_vector(self, text: str) -> Dict[str, float]:
        tokens = HebrewTokenizer.tokenize(text)
        if not tokens:
            return {}
        tf = Counter(tokens)
        total = len(tokens)
        vec = {}
        for tok, count in tf.items():
            tf_val = count / total
            # IDF with smoothing
            idf = math.log((self.total_docs + 1) / (self.doc_freq.get(tok, 0) + 1)) + 1
            vec[tok] = tf_val * idf
        return vec

    @staticmethod
    def cosine_similarity(v1: Dict[str, float], v2: Dict[str, float]) -> float:
        if not v1 or not v2:
            return 0.0
        # dot product
        dot = sum(v1.get(k, 0) * v2.get(k, 0) for k in set(v1) | set(v2))
        norm1 = math.sqrt(sum(x*x for x in v1.values()))
        norm2 = math.sqrt(sum(x*x for x in v2.values()))
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot / (norm1 * norm2)

class AdielMemory:
    def __init__(self):
        os.makedirs(os.path.dirname(MEMORY_FILE), exist_ok=True)
        self.short_term: List[Dict[str, str]] = []  # [{role, content, timestamp}]
        self.long_term: Dict[str, Any] = self._load_long_term()
        self.embedding_engine = LocalEmbeddingEngine()
        self._rebuild_embeddings()

    def _load_long_term(self) -> Dict[str, Any]:
        if os.path.exists(MEMORY_FILE):
            try:
                with open(MEMORY_FILE, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                pass
        return {
            "user_facts": {},  # שם, עבודה, העדפות
            "conversations": [],  # היסטוריית שיחות חשובות
            "facts": [],  # עובדות כלליות שהבוס לימד
            "created_at": datetime.now().isoformat()
        }

    def _save_long_term(self):
        try:
            with open(MEMORY_FILE, 'w', encoding='utf-8') as f:
                json.dump(self.long_term, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[Memory] Save failed: {e}")

    def _rebuild_embeddings(self):
        docs = []
        # בנה מאגר מסמכים מהזיכרון
        for conv in self.long_term.get("conversations", [])[-100:]:
            docs.append(conv.get("user", "") + " " + conv.get("assistant", ""))
        for fact in self.long_term.get("facts", []):
            docs.append(fact)
        if docs:
            self.embedding_engine.fit(docs)

    def add_short_term(self, role: str, content: str):
        entry = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        }
        self.short_term.append(entry)
        if len(self.short_term) > SHORT_TERM_LIMIT:
            self.short_term = self.short_term[-SHORT_TERM_LIMIT:]

    def add_conversation(self, user_text: str, assistant_text: str, save_important: bool = False):
        self.add_short_term("user", user_text)
        self.add_short_term("assistant", assistant_text)
        
        # אם חשוב - שמור לטווח ארוך
        if save_important or self._is_important(user_text):
            self.long_term["conversations"].append({
                "user": user_text,
                "assistant": assistant_text,
                "timestamp": datetime.now().isoformat()
            })
            # שמור רק 200 אחרונות
            if len(self.long_term["conversations"]) > 200:
                self.long_term["conversations"] = self.long_term["conversations"][-200:]
            self._rebuild_embeddings()
            self._save_long_term()

    def _is_important(self, text: str) -> bool:
        keywords = ["תזכור", "תזכרי", "קוראים לי", "אני עובד", "אני גר", "תשמרי", "חשוב"]
        return any(k in text for k in keywords)

    def extract_and_save_facts(self, user_text: str):
        """חלץ עובדות בסיסיות מהטקסט ושמור"""
        # דוגמה פשוטה - ממומש מאפס
        patterns = {
            "name": r"קוראים לי ([\u0590-\u05FFa-zA-Z]+)",
            "location": r"אני גר ב([\u0590-\u05FFa-zA-Z ]+)",
            "work": r"אני עובד (?:ב|כ)?([\u0590-\u05FFa-zA-Z ]+)",
        }
        for key, pat in patterns.items():
            m = re.search(pat, user_text)
            if m:
                self.long_term["user_facts"][key] = m.group(1).strip()
        self._save_long_term()

    def search_relevant_memories(self, query: str, top_k: int = 3) -> List[Dict]:
        """חיפוש סמנטי מקומי בזיכרון"""
        if not self.long_term.get("conversations"):
            return []
        
        query_vec = self.embedding_engine.tfidf_vector(query)
        scored = []
        for conv in self.long_term["conversations"][-50:]:  # חפש ב-50 האחרונות
            doc_text = conv.get("user", "") + " " + conv.get("assistant", "")
            doc_vec = self.embedding_engine.tfidf_vector(doc_text)
            score = self.embedding_engine.cosine_similarity(query_vec, doc_vec)
            scored.append((score, conv))
        
        scored.sort(key=lambda x: x[0], reverse=True)
        return [conv for score, conv in scored[:top_k] if score > 0.15]

    def get_context_string(self) -> str:
        """בנה string של context ל-LLM"""
        ctx = ""
        if self.long_term["user_facts"]:
            ctx += "עובדות על הבוס: " + ", ".join(f"{k}:{v}" for k,v in self.long_term["user_facts"].items()) + "\n"
        
        # שיחות רלוונטיות אחרונות
        if self.short_term:
            ctx += "\nשיחה נוכחית:\n"
            for msg in self.short_term[-6:]:
                ctx += f"{msg['role']}: {msg['content']}\n"
        return ctx

    def clear_short_term(self):
        self.short_term = []
