"""
Intent Classifier - Built from scratch
מסווג כוונות בעברית - ממומש מאפס עם כללי אצבע + לוגיסטיק regression פשוט
"""
import re
import math
from typing import Dict, List, Tuple
from collections import Counter

class HebrewIntentClassifier:
    def __init__(self):
        # הגדרת כוונות עם מילות מפתח בעברית
        self.intents_keywords = {
            "hud_dock_side": [
                "שים בצד", "זוזי הצידה", "תזוזי", "צד המסך", "dock", "בצד",
                "תפני לי מקום", "את מסתירה", "לך הצידה"
            ],
            "hud_center": [
                "חזור לאמצע", "תחזרי לאמצע", "מרכז המסך", "תבואי לאמצע",
                "אמצע", "גדולה", "תגדלי"
            ],
            "hud_hide": [
                "הסתר", "תסתתרי", "תיעלמי", "תתעופפי", "תירדמי", "שקט",
                "הסתר חלון", "תנוחי", "hide"
            ],
            "hud_show": [
                "תופיעי", "תחזרי", "תתעוררי", "איפה את", "תראי אותך", "show"
            ],
            "screen_analysis": [
                "מה אתה רואה", "מה את רואה", "מה יש על המסך", "תסרוק מסך",
                "מה פתוח", "תבדקי מסך", "יש שגיאה", "תעזרי לי עם הקוד",
                "מה השגיאה", "למה לא עובד", "תנתח", "תסתכל",
                "תקרא", "מה כתוב", "תקרא טקסט"
            ],
            "dictionary_lookup": [
                "מה זה", "מה הפירוש", "פירוש המילה", "מה המשמעות",
                "תסביר", "מה אומר", "מילון", "הגדרה", "מה הפרוש"
            ],
            "system_open": [
                "תפתח", "תפתחי", "open", "תריץ", "תפעיל", "launch"
            ],
            "system_volume": [
                "ווליום", "קול", "תגביר", "תנמיך", "השתק", "mute", "volume"
            ],
            "system_search": [
                "תחפש", "חפש", "תחפשי", "תמצא", "search", "גוגל"
            ],
            "time_date": [
                "מה השעה", "כמה השעה", "איזה יום", "תאריך", "שעה"
            ],
            "memory_save": [
                "תזכור", "תזכרי", "תשמור", "תשמרי", "אל תשכח", "זכור"
            ],
            "goodbye": [
                "ביי", "להתראות", "יום טוב", "לילה טוב", "סיימנו"
            ],
            "greeting": [
                "היי", "שלום", "הלו", "בוקר טוב", "ערב טוב", "מה נשמע", "אהלן", "hi", "hello", "hey"
            ],
            "how_are_you": [
                "מה קורה", "מה העניינים", "איך את", "איך אתה", "איך מרגישה", "מה שלומך", "how are you"
            ],
            "thanks": [
                "תודה", "תודה רבה", "אחלה תודה", "תודה לך", "thanks", "thank you", "תודה על", "אלופה", "כל הכבוד"
            ],
            "identity": [
                "מי את", "מה את", "מי אתה", "מה אתה", "איזה מודל", "מי יצר אותך", "who are you"
            ]
        }
        
        # ממפה מילות מפתח ל-intents - לביצועים מהירים
        self.keyword_to_intent = {}
        for intent, keywords in self.intents_keywords.items():
            for kw in keywords:
                self.keyword_to_intent[kw.lower()] = intent

        # בניית מודל TF-IDF פשוט לאימון (מאפס)
        self._train_simple_model()

    def _train_simple_model(self):
        """אימון מודל לוגיסטי פשוט מאפס - תבניות דוגמה"""
        # דוגמאות אימון בעברית
        self.training_data = [
            ("שים את החלון בצד", "hud_dock_side"),
            ("תזוזי הצידה בבקשה", "hud_dock_side"),
            ("חזרי לאמצע המסך", "hud_center"),
            ("תבואי למרכז", "hud_center"),
            ("תיעלמי רגע", "hud_hide"),
            ("מה אתה רואה במסך", "screen_analysis"),
            ("תקרא מה כתוב", "screen_analysis"),
            ("מה זה פאנן", "dictionary_lookup"),
            ("מה הפירוש של סגור", "dictionary_lookup"),
            ("תסביר מה זה יאללה", "dictionary_lookup"),
            ("תפתח את כרום", "system_open"),
            ("תגביר ווליום", "system_volume"),
            ("תחפש בגוגל", "system_search"),
            ("מה השעה", "time_date"),
            ("תזכור שאני אוהב", "memory_save"),
            ("ביי להתראות", "goodbye"),
            ("היי אדיאל", "greeting"),
            ("שלום בוקר טוב", "greeting"),
            ("מה קורה מה נשמע", "how_are_you"),
            ("איך את מרגישה היום", "how_are_you"),
            ("תודה רבה לך", "thanks"),
            ("אחלה תודה על העזרה", "thanks"),
            ("מי את אדיאל", "identity"),
            ("מה את בדיוק", "identity"),
        ]
        
        # חישוב centroid TF לכל intent
        self.intent_centroids = {}
        intent_docs = {}
        for text, intent in self.training_data:
            intent_docs.setdefault(intent, []).append(text)
        
        # פשוט - נבנה מילון מילים לכל intent
        for intent, docs in intent_docs.items():
            all_tokens = []
            for doc in docs:
                all_tokens.extend(self._tokenize(doc))
            self.intent_centroids[intent] = Counter(all_tokens)

    # מילות קישור נטולות משמעות - מסוננות מה-centroids כדי למנוע התאמות כוזבות
    _STOPWORDS = {
        "את", "אתה", "אתם", "אני", "הוא", "היא", "של", "על", "אל", "עם", "לי", "לך",
        "זה", "זאת", "הם", "אנחנו", "כל", "או", "גם", "לא", "כן", "אם", "כי", "יש",
        "the", "a", "is", "to", "of", "and", "in", "on", "it"
    }

    def _tokenize(self, text: str) -> List[str]:
        text = text.lower()
        tokens = re.findall(r'[\u0590-\u05FF]+|[a-z]+', text)
        return [t for t in tokens if t not in self._STOPWORDS]

    def _keyword_match(self, text: str) -> Tuple[str, float]:
        """חיפוש ישיר במילות מפתח - דיוק גבוה"""
        text_lower = text.lower()
        best_intent = None
        best_score = 0
        
        for keyword, intent in self.keyword_to_intent.items():
            if keyword in text_lower:
                # ציון לפי אורך מילת מפתח - התאמה ארוכה יותר = ציון גבוה יותר
                score = len(keyword) / len(text_lower) * 2 + 0.5
                if score > best_score:
                    best_score = score
                    best_intent = intent
        
        return best_intent, min(best_score, 0.95)

    def _centroid_match(self, text: str) -> Tuple[str, float]:
        """התאמת centroid - ממומש מאפס"""
        tokens = self._tokenize(text)
        if not tokens:
            return None, 0.0
        
        token_counter = Counter(tokens)
        best_intent = None
        best_score = 0.0
        
        for intent, centroid in self.intent_centroids.items():
            # cosine similarity פשוט
            dot = sum(token_counter[t] * centroid[t] for t in token_counter)
            norm1 = math.sqrt(sum(v*v for v in token_counter.values()))
            norm2 = math.sqrt(sum(v*v for v in centroid.values()))
            if norm1 == 0 or norm2 == 0:
                continue
            sim = dot / (norm1 * norm2)
            if sim > best_score:
                best_score = sim
                best_intent = intent
        
        return best_intent, best_score

    def classify(self, text: str) -> Dict:
        """
        מסווג כוונה - מחזיר intent + confidence + entities
        """
        text = text.strip()
        if not text:
            return {"intent": "general_chat", "confidence": 0.0, "entities": {}}

        # שלב 1: Keyword exact match
        kw_intent, kw_score = self._keyword_match(text)
        
        # שלב 2: Centroid model
        cent_intent, cent_score = self._centroid_match(text)
        
        # בחירת המנצח
        if kw_intent and kw_score >= 0.4:
            final_intent = kw_intent
            confidence = kw_score
        elif cent_intent and cent_score >= 0.35:
            final_intent = cent_intent
            confidence = cent_score
        else:
            final_intent = "general_chat"
            confidence = 0.2

        # חילוץ entities בסיסי
        entities = self._extract_entities(text, final_intent)

        return {
            "intent": final_intent,
            "confidence": confidence,
            "entities": entities,
            "raw_text": text
        }

    def _extract_entities(self, text: str, intent: str) -> Dict:
        entities = {}
        text_lower = text.lower()
        
        if intent == "system_open":
            # נסה לזהות מה לפתוח
            apps_map = {
                "כרום": "chrome", "chrome": "chrome",
                "פיירפוקס": "firefox",
                "קוד": "code", "vs code": "code", "vscode": "code",
                "דיסקורד": "discord",
                "ספוטיפיי": "spotify", "spotify": "spotify",
                "נוטפד": "notepad", "notepad": "notepad",
                "מחשבון": "calc", "calculator": "calc",
                "אקספלורר": "explorer", "תיקייה": "explorer",
            }
            for he_name, en_id in apps_map.items():
                if he_name in text_lower:
                    entities["app"] = en_id
                    entities["app_he"] = he_name
                    break
            # אם לא זוהה, קח את המילה שאחרי "תפתח"
            if "app" not in entities:
                m = re.search(r"(?:תפתח|תפתחי|open)\s+(?:את\s+)?(.+)", text_lower)
                if m:
                    entities["app_query"] = m.group(1).strip()

        elif intent == "system_search":
            m = re.search(r"(?:תחפש|חפש|search)\s+(?:בגוגל\s+)?(.+)", text_lower)
            if m:
                entities["query"] = m.group(1).strip()

        elif intent == "screen_analysis":
            entities["needs_screen"] = True

        elif intent == "dictionary_lookup":
            patterns = [
                r"מה זה ([\u0590-\u05FF]+)",
                r"מה הפירוש של ([\u0590-\u05FF]+)",
                r"פירוש המילה ([\u0590-\u05FF]+)",
            ]
            for pat in patterns:
                m = re.search(pat, text_lower)
                if m:
                    entities["word"] = m.group(1).strip().split()[0]
                    break
            if "word" not in entities:
                words = re.findall(r'[\u0590-\u05FF]{2,}', text)
                if words:
                    entities["word"] = words[-1]

        return entities
