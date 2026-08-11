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

        # BM25 index (v2.1) - tokenizer + tf לכל שיחה + אורך ממוצע
        self._bm25_docs = []  # [(tf Counter, length, conv_ref)]
        for conv in self.long_term.get("conversations", [])[-100:]:
            text = conv.get("user", "") + " " + conv.get("assistant", "")
            tokens = HebrewTokenizer.tokenize(text)
            self._bm25_docs.append((Counter(tokens), len(tokens), conv))
        self._bm25_avgdl = (
            sum(l for _, l, _ in self._bm25_docs) / max(len(self._bm25_docs), 1)
        )

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
        # v2.4: כל שיחה סורקת עובדות על הבוס (שם/מיקום/עבודה) - כך אדיאל באמת מכירה אותך
        try:
            self.extract_and_save_facts(user_text)
        except Exception:
            pass
        self.add_short_term("user", user_text)
        self.add_short_term("assistant", assistant_text)

        # v2.3: כל שיחה נשמרת לטווח ארוך - "זוכרת הכל" באמת, לא רק "חשוב".
        # בלי זה חיפוש ה-BM25 היה תמיד ריק ומונה ה-"דיברנו N פעמים" היה 0 לנצח.
        self.long_term["conversations"].append({
            "user": user_text,
            "assistant": assistant_text,
            "timestamp": datetime.now().isoformat()
        })
        # שמור רק 200 אחרונות
        if len(self.long_term["conversations"]) > 200:
            self.long_term["conversations"] = self.long_term["conversations"][-200:]
        important = save_important or self._is_important(user_text)
        # כל מספר שיחות: שמור לדיסק + בנה אינדקס BM25 מחדש (זול - עד 200 רשומות)
        if important or len(self.long_term["conversations"]) % 3 == 0:
            self._rebuild_embeddings()
            self._save_long_term()

    def _is_important(self, text: str) -> bool:
        keywords = ["תזכור", "תזכרי", "קוראים לי", "אני עובד", "אני גר", "תשמרי", "חשוב"]
        return any(k in text for k in keywords)

    def extract_and_save_facts(self, user_text: str) -> bool:
        """חלץ עובדות בסיסיות מהטקסט ושמור. שומר לדיסק רק אם משהו באמת השתנה (v2.4)"""
        patterns = {
            "name": r"קוראים לי ([\u0590-\u05FFa-zA-Z]+)",
            "location": r"אני גר ב([\u0590-\u05FFa-zA-Z ]+)",
            "work": r"אני עובד (?:ב|כ)?([\u0590-\u05FFa-zA-Z ]+)",
        }
        changed = False
        for key, pat in patterns.items():
            m = re.search(pat, user_text)
            if m:
                val = m.group(1).strip()
                if val and self.long_term["user_facts"].get(key) != val:
                    self.long_term["user_facts"][key] = val
                    changed = True
        if changed:
            self._save_long_term()
        return changed

    def search_relevant_memories(self, query: str, top_k: int = 3) -> List[Dict]:
        """חיפוש BM25 (v2.1 - מתקדם מ-TF-IDF cosine) + בוסט עדכניות"""
        if not self.long_term.get("conversations"):
            return []

        bm25_docs = getattr(self, "_bm25_docs", None)
        if not bm25_docs:
            return []

        q_tokens = set(HebrewTokenizer.tokenize(query))
        if not q_tokens:
            return []

        N = len(bm25_docs)
        avgdl = max(self._bm25_avgdl, 1e-6)
        k1, b = 1.5, 0.75
        scored = []
        for idx, (tf, dl, conv) in enumerate(bm25_docs):
            if dl == 0:
                continue
            score = 0.0
            for t in q_tokens:
                f = tf.get(t, 0)
                if f == 0:
                    continue
                n_t = self.embedding_engine.doc_freq.get(t, 0)
                idf = math.log(1 + (N - n_t + 0.5) / (n_t + 0.5))
                score += idf * (f * (k1 + 1)) / (f + k1 * (1 - b + b * dl / avgdl))
            # שיחות אחרונות מקבלות עדיפות קלה
            recency = idx / max(N - 1, 1)
            score *= (0.85 + 0.3 * recency)
            if score > 1.0:
                scored.append((score, conv))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [conv for score, conv in scored[:top_k]]

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
