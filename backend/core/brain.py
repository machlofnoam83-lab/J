"""
Adiel Junior - Brain Engine v2.0 - עם למידה מתמשכת וזיכרון חכם
המוח המרכזי - ממומש מאפס, פרטי לגמרי
מנהל שיחה, זיכרון, כוונות, וכלים + מערכת למידה שמתעדכנת ומבקשת אישור
"""
import os
import random
import datetime
import asyncio
from typing import Dict, Any, Optional, Tuple

from .personality import ADIEL_SYSTEM_PROMPT, get_random_response
from .memory import AdielMemory
from .intents import HebrewIntentClassifier

# מנועי למידה ועדכון עצמי - חדש!
try:
    from .learning_engine import get_learning_engine
    HAS_LEARNING = True
except:
    HAS_LEARNING = False
    print("[Brain] Learning engine not available")

try:
    from .self_update import get_self_update_manager
    HAS_SELF_UPDATE = True
except:
    HAS_SELF_UPDATE = False
    print("[Brain] Self-update manager not available")

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
    עכשיו עם זיכרון חכם שגדל בכל שיחה + למידה שדורשת אישור
    """
    def __init__(self):
        self.memory = AdielMemory()
        self.intent_classifier = HebrewIntentClassifier()
        
        # מנועי למידה חדשים
        self.learning_engine = None
        self.self_update_manager = None
        
        if HAS_LEARNING:
            try:
                self.learning_engine = get_learning_engine()
                print(f"[Brain] 🧠 מנוע למידה: זוכר {self.learning_engine.get_user_profile_summary().get('interaction_count', 0)} שיחות, {self.learning_engine.get_user_profile_summary().get('vocab_learned', 0)} מילים")
            except Exception as e:
                print(f"[Brain] Learning engine init failed: {e}")
        
        if HAS_SELF_UPDATE:
            try:
                self.self_update_manager = get_self_update_manager()
                pending = len(self.self_update_manager.get_pending_updates())
                if pending > 0:
                    print(f"[Brain] 🤖 יש {pending} הצעות לשיפור שממתינות לאישורך בוס!")
            except Exception as e:
                print(f"[Brain] Self-update init failed: {e}")
        
        # קונפיגורציה
        self.use_cloud_llm = os.getenv("ALLOW_CLOUD_LLM", "false").lower() == "true"
        self.ollama_model = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
        self.openai_api_key = os.getenv("OPENAI_API_KEY", "")
        
        # מצב נוכחי
        self.conversation_turns = 0
        self.last_proposals = []  # הצעות מהשיחה האחרונה
        
        print("[Brain] אדיאל ג'וניור התעוררה - המוח הפרטי v2.0 עם זיכרון מתעדכן")

    async def process(self, user_text: str, screen_context: Optional[str] = None) -> Dict[str, Any]:
        """
        עיבוד ראשי - מקבל טקסט + הקשר מסך ומחזיר תשובה + פעולות + הצעות למידה
        """
        self.conversation_turns += 1
        print(f"[Brain] מעבד: '{user_text}' | intent checking...")

        # 1. סיווג כוונה
        intent_result = self.intent_classifier.classify(user_text)
        intent = intent_result["intent"]
        confidence = intent_result["confidence"]
        entities = intent_result["entities"]

        print(f"[Brain] Intent: {intent} ({confidence:.2f}) | Entities: {entities}")

        # 2. חיפוש בזיכרון רלוונטי (ישן)
        relevant_memories = self.memory.search_relevant_memories(user_text, top_k=2)

        # 2.5 - NEW: הקשר חכם מהלמידה המתמשכת
        smart_context = ""
        user_profile_summary = {}
        if self.learning_engine:
            smart_context = self.learning_engine.get_smart_context()
            user_profile_summary = self.learning_engine.get_user_profile_summary()
            print(f"[Brain] Smart context: {smart_context[:100]}...")

        # 3. שמירת עובדות אם צריך (ישן) - עכשיו רק אם אין מנוע למידה חדש
        if (intent == "memory_save" or "תזכור" in user_text or "תזכרי" in user_text) and not self.learning_engine:
            self.memory.extract_and_save_facts(user_text)

        # 4. טיפול בכוונות ספציפיות (ללא צורך ב-LLM)
        action_result = await self._handle_intent(intent, entities, user_text, screen_context)
        
        if action_result.get("handled_locally"):
            response_text = action_result["response"]
            hud_command = action_result.get("hud_command")
            system_action = action_result.get("system_action")
        else:
            # 5. יצירת תשובה - נסה LLM, אחרת תבנית מקומית עם הזיכרון החכם
            response_text = await self._generate_response(
                user_text, intent, entities, screen_context, relevant_memories, smart_context, user_profile_summary
            )
            hud_command = action_result.get("hud_command")
            system_action = action_result.get("system_action")

        # 6. שמירה בזיכרון קצר טווח בלבד (לא עובדות קבועות - זה עובר דרך הצעות למידה עם אישור)
        self.memory.add_conversation(user_text, response_text)
        # אם אין מנוע למידה חדש, שמור עובדות ישירות (fallback)
        if not self.learning_engine:
            self.memory.extract_and_save_facts(user_text)

        # 7. NEW: עיבוד למידה - יוצר הצעות לשיפור שדורשות אישור
        proposals = []
        if self.learning_engine:
            try:
                new_proposals = self.learning_engine.process_interaction(user_text, response_text, intent_result)
                proposals.extend(new_proposals)
                self.last_proposals = new_proposals
                
                # אם יש הצעות, הוסף לרספונס הודעה שאדיאל רוצה ללמוד
                if new_proposals:
                    # אל תוסיף כל פעם, רק אם יש מילים חדשות או עובדות חשובות
                    important = [p for p in new_proposals if p["type"] in ["vocabulary", "profile_new", "fact"]]
                    if important and len(user_text.split()) > 2:
                        # הוסף hint לתשובה
                        if any(p["type"] == "profile_new" for p in important):
                            # אם למדנו שם חדש, תגיב עם זה
                            pass  # תשובה כבר טופלה
            except Exception as e:
                print(f"[Brain] Learning process failed: {e}")
                import traceback; traceback.print_exc()

        # 8. NEW: בדוק אם המוח עצמו רוצה להשתפר (self-update) - רק אם יש מספיק שיחות
        self_updates = []
        if self.self_update_manager and self.conversation_turns % 5 == 0:  # כל 5 שיחות בדוק
            try:
                # אוטו-זיהוי שיפורים
                history = [{"content": m["content"], "role": m["role"]} for m in self.memory.short_term[-10:]]
                auto_proposals = self.self_update_manager.auto_detect_improvements(history)
                self_updates.extend(auto_proposals)
            except Exception as e:
                print(f"[Brain] Self-update detection failed: {e}")

        return {
            "text": response_text,
            "intent": intent,
            "confidence": confidence,
            "hud_command": hud_command,
            "system_action": system_action,
            "screen_context_used": screen_context is not None,
            "memory_count": len(self.memory.long_term.get("conversations", [])),
            # חדש - הצעות למידה שדורשות אישור
            "proposals": proposals,
            "self_updates": self_updates,
            "smart_context": smart_context,
            "user_profile": user_profile_summary,
            "learning_active": self.learning_engine is not None
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

        # --- MEMORY / PROFILE QUERIES - NEW ---
        elif "מה אתה זוכר" in user_text or "מה את זוכרת" in user_text or "מה אתה יודע עלי" in user_text:
            if self.learning_engine:
                summary = self.learning_engine.get_user_profile_summary()
                smart = self.learning_engine.get_smart_context()
                resp = f"בוס, אני זוכרת: {smart} "
                if summary.get("name"):
                    resp += f"קוראים לך {summary['name']}. "
                if summary.get("projects"):
                    resp += f"אתה עובד על {', '.join(summary['projects'])}. "
                resp += f"למדתי {summary.get('vocab_learned', 0)} מילים ממך ו-{summary.get('interaction_count', 0)} שיחות. רוצה שאספר עוד?"
                return {"handled_locally": True, "response": resp}
            else:
                return {"handled_locally": True, "response": self.memory.get_context_string()[:300] or "עדיין לומדת להכיר אותך, בוס."}

        # --- SCREEN ANALYSIS (local part) ---
        elif intent == "screen_analysis":
            if not screen_context:
                return {
                    "handled_locally": False,
                    "response": None,
                    "hud_command": None
                }

        # --- SYSTEM ACTIONS ---
        elif intent == "system_open":
            app = entities.get("app") or entities.get("app_query", "unknown")
            return {
                "handled_locally": True,
                "response": f"על זה, פותחת {entities.get('app_he', app)}.",
                "system_action": {"type": "open_app", "app": app, "query": entities.get("app_query")}
            }
        elif intent == "system_volume":
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
            # עם למידה חדשה, זה ייצור הצעה מסודרת
            if self.learning_engine:
                return {
                    "handled_locally": False,  # תן ל-process ליצור הצעה מסודרת
                    "response": None
                }
            return {
                "handled_locally": True,
                "response": "קלטתי, שמרתי את זה בזיכרון, בוס. לא אשכח."
            }

        # --- GOODBYE ---
        elif intent == "goodbye":
            if self.learning_engine:
                count = self.learning_engine.get_user_profile_summary().get("interaction_count", 0)
                return {
                    "handled_locally": True,
                    "response": f"יאללה ביי בוס, דיברנו {count} פעמים היום ואני זוכרת הכל. אני כאן אם צריך."
                }
            return {
                "handled_locally": True,
                "response": random.choice(["יאללה ביי בוס, אני כאן אם צריך.", "סגור בוס, היה כיף. תקרא לי כשצריך.", "ביי בוס, שמה את עצמי על שקט."])
            }

        # לא טופל לוקלית - צריך LLM / NLG כללי
        return {"handled_locally": False}

    async def _generate_response(self, user_text: str, intent: str, entities: Dict, 
                                 screen_context: Optional[str], relevant_memories: list,
                                 smart_context: str = "", user_profile: Dict = None) -> str:
        """יצירת תשובה - היררכיה: Ollama Local -> OpenAI -> Local Templates עם זיכרון חכם"""

        # בנה context משופר
        memory_context = self.memory.get_context_string()
        if smart_context:
            memory_context += f"\n[זיכרון חכם מתעדכן]: {smart_context}"
        
        if user_profile and user_profile.get("name"):
            memory_context += f"\nשם הבוס: {user_profile['name']}"
        
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

        # Fallback - תבניות מקומיות חכמות עם זיכרון
        return self._generate_local_response(user_text, intent, screen_context, relevant_memories, user_profile)

    async def _try_ollama(self, user_text: str, screen_context: Optional[str], memory_ctx: str, intent: str) -> Optional[str]:
        """נסה Ollama מקומי"""
        try:
            prompt = f"""{ADIEL_SYSTEM_PROMPT}

הקשר זיכרון (את זוכרת את הבוס):
{memory_ctx}

הקשר מסך נוכחי:
{screen_context or 'אין מידע מסך'}

הודעת המשתמש: {user_text}
כוונה: {intent}

עני בעברית, קצר, בסגנון אדיאל ג'וניור. השתמשי בזיכרון החכם! אם יש מסך, נתח אותו ספציפית.
"""

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
                messages.append({"role": "system", "content": f"זיכרון חכם:\n{memory_ctx}"})
            if screen_context:
                messages.append({"role": "system", "content": f"מה רואים במסך כרגע: {screen_context}"})
            messages.append({"role": "user", "content": user_text})

            def call_openai():
                resp = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=messages,
                    max_tokens=400,
                    temperature=0.85
                )
                return resp.choices[0].message.content

            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, call_openai)
            return result
        except Exception as e:
            print(f"[OpenAI] Error: {e}")
            return None

    def _generate_local_response(self, user_text: str, intent: str, screen_context: Optional[str], relevant_memories: list, user_profile: Dict = None) -> str:
        """המוח הפרטי האמיתי - NLG מקומי עם זיכרון חכם ממומש מאפס"""

        # אם יש הקשר מסך
        if screen_context and intent == "screen_analysis":
            templates = [
                f"בוס, אני רואה על המסך: {screen_context[:300]}. ",
                f"קלטתי את המסך - {screen_context[:300]}. ",
                f"סרקתי: {screen_context[:200]}. ",
            ]
            base = random.choice(templates)
            
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

        # שיחה כללית עם זיכרון חכם
        if user_profile and user_profile.get("name"):
            name = user_profile["name"]
            # השתמש בשם
            greeting_with_name = [
                f"בוס {name}, זה מזכיר לי משהו...",
                f"{name}, קלטתי.",
            ]
            if random.random() < 0.3:
                # לפעמים השתמש בשם
                pass

        if relevant_memories:
            mem_hint = relevant_memories[0].get("user", "")[:80]
            return f"זה מזכיר לי שדיברנו על '{mem_hint}...' - {get_random_response('fallback_chat')}"

        # ברירת מחדל - תשובות כלליות חכמות לפי מילות מפתח עם זיכרון
        lower = user_text.lower()

        if any(w in lower for w in ["איך אתה", "איך את", "מה שלומך"]):
            if user_profile and user_profile.get("interaction_count", 0) > 5:
                return f"אחלה בוס! אחרי {user_profile['interaction_count']} שיחות איתך אני כבר מכירה אותך טוב. רצה על Full Power. מה איתך?"
            return random.choice([
                "אחלה, בוס! רצה על Full Power. מה איתך?",
                "מצוין, מוכנה לפעולה. מה קורה אצלך?",
                "על הגל, בוס. המערכות ירוקות."
            ])
        if any(w in lower for w in ["תודה", "אלופה", "מלכה"]):
            return random.choice([
                "בכיף בוס, תמיד כאן. וזוכרת הכל!",
                "יאללה, זה התפקיד שלי. מה עוד?",
                "על לא דבר. אני פה ומתעדכנת כל הזמן."
            ])
        if any(w in lower for w in ["עזרה", "לא מצליח", "לא עובד"]):
            return "קלטתי שיש בעיה. תספר לי בדיוק מה לא עובד, ואם אפשר - תגיד 'מה את רואה במסך' ואסרוק לך. אני גם לומדת מכל פעם שאתה מתקן אותי."

        # עם פרופיל
        if user_profile and user_profile.get("projects"):
            # אם המשתמש עובד על פרויקט, התייחס לזה
            if random.random() < 0.2:
                proj = user_profile["projects"][-1] if user_profile["projects"] else ""
                if proj and proj.lower() in lower:
                    return f"עדיין עובד על {proj}? איך מתקדם?"

        # ברירת מחדל כללית
        return get_random_response("fallback_chat")
