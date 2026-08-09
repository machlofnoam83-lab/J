"""
Adiel Junior - Brain Engine
המוח המרכזי - ממומש מאפס, פרטי לגמרי
מנהל שיחה, זיכרון, כוונות, וכלים
"""
import os
import random
import datetime
import asyncio
from typing import Dict, Any, Optional, Tuple

from .personality import ADIEL_SYSTEM_PROMPT, get_random_response
from .memory import AdielMemory
from .intents import HebrewIntentClassifier

# אופציונלי - LLM חיצוני כ-power up
try:
    import ollama
    HAS_OLLAMA = True
except:
    HAS_OLLAMA = False

try:
    from openai import OpenAI
    HAS_OPENAI = True
except:
    HAS_OPENAI = False


class AdielBrain:
    """
    מוח פרטי לאדיאל - עובד גם בלי אינטרנט
    """
    def __init__(self):
        self.memory = AdielMemory()
        self.intent_classifier = HebrewIntentClassifier()
        
        # קונפיגורציה
        self.use_cloud_llm = os.getenv("ALLOW_CLOUD_LLM", "false").lower() == "true"
        self.ollama_model = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
        self.openai_api_key = os.getenv("OPENAI_API_KEY", "")
        
        # מצב נוכחי
        self.conversation_turns = 0
        
        print("[Brain] אדיאל ג'וניור התעוררה - המוח הפרטי נטען")

    async def process(self, user_text: str, screen_context: Optional[str] = None) -> Dict[str, Any]:
        """
        עיבוד ראשי - מקבל טקסט + הקשר מסך ומחזיר תשובה + פעולות
        """
        self.conversation_turns += 1
        print(f"[Brain] מעבד: '{user_text}' | intent checking...")

        # 1. סיווג כוונה
        intent_result = self.intent_classifier.classify(user_text)
        intent = intent_result["intent"]
        confidence = intent_result["confidence"]
        entities = intent_result["entities"]

        print(f"[Brain] Intent: {intent} ({confidence:.2f}) | Entities: {entities}")

        # 2. חיפוש בזיכרון רלוונטי
        relevant_memories = self.memory.search_relevant_memories(user_text, top_k=2)

        # 3. שמירת עובדות אם צריך
        if intent == "memory_save" or "תזכור" in user_text or "תזכרי" in user_text:
            self.memory.extract_and_save_facts(user_text)

        # 4. טיפול בכוונות ספציפיות (ללא צורך ב-LLM)
        action_result = await self._handle_intent(intent, entities, user_text, screen_context)
        
        if action_result.get("handled_locally"):
            response_text = action_result["response"]
            hud_command = action_result.get("hud_command")
            system_action = action_result.get("system_action")
        else:
            # 5. יצירת תשובה - נסה LLM, אחרת תבנית מקומית
            response_text = await self._generate_response(
                user_text, intent, entities, screen_context, relevant_memories
            )
            hud_command = action_result.get("hud_command")
            system_action = action_result.get("system_action")

        # 6. שמירה בזיכרון
        self.memory.add_conversation(user_text, response_text)
        self.memory.extract_and_save_facts(user_text)

        return {
            "text": response_text,
            "intent": intent,
            "confidence": confidence,
            "hud_command": hud_command,
            "system_action": system_action,
            "screen_context_used": screen_context is not None,
            "memory_count": len(self.memory.long_term.get("conversations", []))
        }

    async def _handle_intent(self, intent: str, entities: Dict, user_text: str, screen_context: Optional[str]) -> Dict:
        """טיפול בכוונות שמתבצעות לוקלית לגמרי - ללא LLM"""
        
        # --- HUD CONTROL ---
        if intent == "hud_dock_side":
            return {
                "handled_locally": True,
                "response": get_random_response("hud_dock"),
                "hud_command": {"action": "dock", "position": "right"}
            }
        elif intent == "hud_center":
            return {
                "handled_locally": True,
                "response": get_random_response("hud_center"),
                "hud_command": {"action": "center"}
            }
        elif intent == "hud_hide":
            return {
                "handled_locally": True,
                "response": get_random_response("hud_hide"),
                "hud_command": {"action": "hide"}
            }
        elif intent == "hud_show":
            return {
                "handled_locally": True,
                "response": "אני כאן, בוס! מה צריך?",
                "hud_command": {"action": "show"}
            }

        # --- TIME ---
        elif intent == "time_date":
            now = datetime.datetime.now()
            hebrew_days = ["שני", "שלישי", "רביעי", "חמישי", "שישי", "שבת", "ראשון"]
            day = hebrew_days[now.weekday()]
            response = f"עכשיו {now.strftime('%H:%M')}, יום {day}, {now.strftime('%d/%m/%Y')}, בוס."
            return {"handled_locally": True, "response": response}

        # --- SCREEN ANALYSIS (local part) ---
        elif intent == "screen_analysis":
            if not screen_context:
                return {
                    "handled_locally": False,  # צריך לעבור ל-LLM עם מסך
                    "response": None,
                    "hud_command": None
                }
            # אם יש הקשר מסך, ניתן ל-LLM לנתח, אבל נוסיף prefix מקומי
            # נחזיר handled=False כדי שיגיע ל-_generate_response

        # --- SYSTEM ACTIONS ---
        elif intent == "system_open":
            app = entities.get("app") or entities.get("app_query", "unknown")
            return {
                "handled_locally": True,
                "response": f"על זה, פותחת {entities.get('app_he', app)}.",
                "system_action": {"type": "open_app", "app": app, "query": entities.get("app_query")}
            }
        elif intent == "system_volume":
            # ניתוח כוונת ווליום
            if "גביר" in user_text or "להגביר" in user_text or "יותר חזק" in user_text:
                vol_action = "up"
                resp = "מגבירה, בוס."
            elif "נמיך" in user_text or "להנמיך" in user_text or "חלש" in user_text:
                vol_action = "down"
                resp = "מנמיכה."
            elif "השתק" in user_text or "mute" in user_text or "שקט" in user_text:
                vol_action = "mute"
                resp = "סגור, משתיקה."
            else:
                vol_action = "toggle"
                resp = "מטפלת בווליום."
            return {
                "handled_locally": True,
                "response": resp,
                "system_action": {"type": "volume", "action": vol_action}
            }
        elif intent == "system_search":
            query = entities.get("query", user_text)
            return {
                "handled_locally": True,
                "response": f"מחפשת בגוגל: {query}",
                "system_action": {"type": "search", "query": query}
            }

        # --- MEMORY SAVE ---
        elif intent == "memory_save":
            return {
                "handled_locally": True,
                "response": "קלטתי, שמרתי את זה בזיכרון, בוס. לא אשכח."
            }

        # --- GOODBYE ---
        elif intent == "goodbye":
            return {
                "handled_locally": True,
                "response": random.choice(["יאללה ביי בוס, אני כאן אם צריך.", "סגור בוס, היה כיף. תקרא לי כשצריך.", "ביי בוס, שמה את עצמי על שקט."])
            }

        # לא טופל לוקלית - צריך LLM / NLG כללי
        return {"handled_locally": False}

    async def _generate_response(self, user_text: str, intent: str, entities: Dict, 
                                 screen_context: Optional[str], relevant_memories: list) -> str:
        """יצירת תשובה - היררכיה: Ollama Local -> OpenAI -> Local Templates"""

        # בנה context
        memory_context = self.memory.get_context_string()
        
        # נסה Ollama מקומי קודם (פרטי לגמרי)
        if HAS_OLLAMA and not self.use_cloud_llm:
            try:
                ollama_response = await self._try_ollama(user_text, screen_context, memory_context, intent)
                if ollama_response:
                    return ollama_response
            except Exception as e:
                print(f"[Brain] Ollama failed: {e}")

        # נסה OpenAI אם מורשה
        if self.use_cloud_llm and HAS_OPENAI and self.openai_api_key:
            try:
                openai_resp = await self._try_openai(user_text, screen_context, memory_context)
                if openai_resp:
                    return openai_resp
            except Exception as e:
                print(f"[Brain] OpenAI failed: {e}")

        # Fallback - תבניות מקומיות חכמות (המוח הפרטי האמיתי)
        return self._generate_local_response(user_text, intent, screen_context, relevant_memories)

    async def _try_ollama(self, user_text: str, screen_context: Optional[str], memory_ctx: str, intent: str) -> Optional[str]:
        """נסה Ollama מקומי"""
        try:
            prompt = f"""{ADIEL_SYSTEM_PROMPT}

הקשר זיכרון:
{memory_ctx}

הקשר מסך נוכחי:
{screen_context or 'אין מידע מסך'}

הודעת המשתמש: {user_text}
כוונה: {intent}

עני בעברית, קצר, בסגנון אדיאל ג'וניור. אם יש מסך, נתח אותו ספציפית.
"""

            # קריאה סינכרונית ב-thread נפרד למניעת חסימה
            def call_ollama():
                try:
                    response = ollama.chat(model=self.ollama_model, messages=[
                        {"role": "system", "content": ADIEL_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt}
                    ])
                    return response['message']['content']
                except:
                    return None

            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, call_ollama)
            return result
        except Exception as e:
            print(f"[Ollama] Error: {e}")
            return None

    async def _try_openai(self, user_text: str, screen_context: Optional[str], memory_ctx: str) -> Optional[str]:
        if not self.openai_api_key:
            return None
        try:
            client = OpenAI(api_key=self.openai_api_key)
            
            messages = [
                {"role": "system", "content": ADIEL_SYSTEM_PROMPT},
            ]
            if memory_ctx:
                messages.append({"role": "system", "content": f"זיכרון:\n{memory_ctx}"})
            if screen_context:
                messages.append({"role": "system", "content": f"מה רואים במסך כרגע: {screen_context}"})
            messages.append({"role": "user", "content": user_text})

            def call_openai():
                resp = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=messages,
                    max_tokens=300,
                    temperature=0.8
                )
                return resp.choices[0].message.content

            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, call_openai)
            return result
        except Exception as e:
            print(f"[OpenAI] Error: {e}")
            return None

    def _generate_local_response(self, user_text: str, intent: str, screen_context: Optional[str], relevant_memories: list) -> str:
        """המוח הפרטי האמיתי - NLG מקומי ממומש מאפס"""

        # אם יש הקשר מסך
        if screen_context and intent == "screen_analysis":
            templates = [
                f"בוס, אני רואה על המסך: {screen_context[:300]}. ",
                f"קלטתי את המסך - {screen_context[:300]}. ",
                f"סרקתי: {screen_context[:200]}. ",
            ]
            base = random.choice(templates)
            
            # ניתוח חכם מקומי
            lower_ctx = screen_context.lower()
            analysis = ""
            if "error" in lower_ctx or "שגיאה" in lower_ctx or "exception" in lower_ctx:
                analysis += "נראה שיש כאן שגיאה. תן לי לנחש - כנראה שכחת משהו קטן בסינטקס או יש בעיית import. רוצה שאפרט?"
            elif "code" in lower_ctx or "def " in lower_ctx or "import" in lower_ctx:
                analysis += "זה נראה כמו קוד. אם תגיד לי מה הבעיה, אעזור לדבג."
            elif "browser" in lower_ctx or "chrome" in lower_ctx or "כרום" in lower_ctx:
                analysis += "זה דפדפן פתוח. מה אתה רוצה שאבדוק שם?"
            else:
                analysis += "מה בדיוק אתה רוצה שאעשה עם זה?"

            return base + analysis

        # שיחה כללית
        if relevant_memories:
            mem_hint = relevant_memories[0].get("user", "")[:80]
            return f"זה מזכיר לי שדיברנו על '{mem_hint}...' - {get_random_response('fallback_chat')}"

        # ברירת מחדל - תשובות כלליות חכמות לפי מילות מפתח
        lower = user_text.lower()

        if any(w in lower for w in ["איך אתה", "איך את", "מה שלומך"]):
            return random.choice([
                "אחלה, בוס! רצה על Full Power. מה איתך?",
                "מצוין, מוכנה לפעולה. מה קורה אצלך?",
                "על הגל, בוס. המערכות ירוקות."
            ])
        if any(w in lower for w in ["תודה", "אלופה", "מלכה"]):
            return random.choice([
                "בכיף בוס, תמיד כאן.",
                "יאללה, זה התפקיד שלי. מה עוד?",
                "על לא דבר. אני פה."
            ])
        if any(w in lower for w in ["עזרה", "לא מצליח", "לא עובד"]):
            return "קלטתי שיש בעיה. תספר לי בדיוק מה לא עובד, ואם אפשר - תגיד 'מה את רואה במסך' ואסרוק לך."

        # ברירת מחדל כללית
        return get_random_response("fallback_chat")
