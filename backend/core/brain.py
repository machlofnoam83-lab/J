"""
Adiel Junior - Brain Engine v3.0 - עם LLM 6GB + דיבור מהיר + קריאת טקסט
- זיכרון חכם
- למידה עם אישור
- True AI מאפס
- LLM 6GB RAM (Phi-3, Llama 3.1 Q4) - מתאים ל-6GB
- מבין דיבור מהיר
- קורא טקסט
- מייצר קול לכל תשובה
"""
import os
import random
import datetime
import asyncio
from typing import Dict, Any, Optional

from .personality import ADIEL_SYSTEM_PROMPT, get_random_response
from .memory import AdielMemory
from .intents import HebrewIntentClassifier

# למידה
try:
    from .learning_engine import get_learning_engine
    HAS_LEARNING = True
except:
    HAS_LEARNING = False

try:
    from .self_update import get_self_update_manager
    HAS_SELF_UPDATE = True
except:
    HAS_SELF_UPDATE = False

# True AI מאפס
try:
    from .true_ai_model import get_true_ai
    HAS_TRUE_AI = True
except:
    HAS_TRUE_AI = False

# LLM 6GB - חדש!
try:
    from .llm_6gb import get_llm_6gb
    HAS_LLM_6GB = True
except:
    HAS_LLM_6GB = False

# דיבור מהיר
try:
    from ..audio.fast_stt import FastSpeechProcessor
    HAS_FAST_STT = True
except:
    HAS_FAST_STT = False

# קריאת טקסט
try:
    from ..vision.advanced_reader import get_advanced_reader
    HAS_ADV_READER = True
except:
    HAS_ADV_READER = False

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
    def __init__(self):
        self.memory = AdielMemory()
        self.intent_classifier = HebrewIntentClassifier()
        
        self.learning_engine = None
        self.self_update_manager = None
        self.true_ai = None
        self.llm_6gb = None
        self.fast_processor = None
        self.advanced_reader = None
        
        if HAS_LEARNING:
            try:
                self.learning_engine = get_learning_engine()
                print(f"[Brain] 🧠 למידה: {self.learning_engine.get_user_profile_summary().get('interaction_count',0)} שיחות")
            except Exception as e:
                print(f"[Brain] Learning failed: {e}")
        
        if HAS_SELF_UPDATE:
            try:
                self.self_update_manager = get_self_update_manager()
            except:
                pass

        if HAS_TRUE_AI:
            try:
                self.true_ai = get_true_ai()
                print(f"[Brain] ✨ True AI: Vocab {self.true_ai.tokenizer.vocab_size}")
            except Exception as e:
                print(f"[Brain] True AI failed: {e}")

        if HAS_LLM_6GB:
            try:
                self.llm_6gb = get_llm_6gb(model_preference="auto")
                info = self.llm_6gb.get_info()
                print(f"[Brain] 🚀 LLM 6GB: {info['model_name']} backend={info['backend']} RAM={info['available_ram_gb']}GB")
            except Exception as e:
                print(f"[Brain] LLM 6GB failed: {e}")

        if HAS_FAST_STT:
            try:
                self.fast_processor = FastSpeechProcessor()
                print(f"[Brain] ⚡ Fast Speech Processor - מבין דיבור מהיר!")
            except:
                pass

        if HAS_ADV_READER:
            try:
                self.advanced_reader = get_advanced_reader()
                print(f"[Brain] 📖 Advanced Reader - קורא טקסט חכם")
            except:
                pass
        
        self.use_cloud_llm = os.getenv("ALLOW_CLOUD_LLM", "false").lower() == "true"
        self.ollama_model = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
        self.openai_api_key = os.getenv("OPENAI_API_KEY", "")
        self.conversation_turns = 0
        
        print("[Brain] אדיאל v3.0 - LLM 6GB + דיבור מהיר + קריאת טקסט + קול לכל תשובה")

    async def process(self, user_text: str, screen_context: Optional[str] = None) -> Dict[str, Any]:
        self.conversation_turns += 1

        # --- שיפור לדיבור מהיר: ניקוי ותיקון ---
        original_text = user_text
        if self.fast_processor:
            user_text = self.fast_processor.postprocess_text(user_text)
            if user_text != original_text:
                print(f"[Brain] Fast speech fix: '{original_text}' -> '{user_text}'")

        print(f"[Brain] מעבד: '{user_text}'")
        
        intent_result = self.intent_classifier.classify(user_text)
        intent = intent_result["intent"]
        confidence = intent_result["confidence"]
        entities = intent_result["entities"]
        print(f"[Brain] Intent: {intent} ({confidence:.2f})")

        relevant_memories = self.memory.search_relevant_memories(user_text, top_k=2)
        
        smart_context = ""
        user_profile_summary = {}
        if self.learning_engine:
            smart_context = self.learning_engine.get_smart_context()
            user_profile_summary = self.learning_engine.get_user_profile_summary()

        # אם זה בקשת קריאת טקסט
        if any(kw in user_text.lower() for kw in ["תקרא", "קורא", "טקסט", "מה כתוב", "read text"]):
            if self.advanced_reader and screen_context:
                # screen_context כבר מכיל טקסט, אבל נוסיף הבנה
                understanding = self.advanced_reader.understand_text(screen_context, question=user_text)
                if understanding["success"]:
                    # הוסף לסיכום המסך
                    screen_context += f"\n\n[קריאת טקסט חכמה]: {understanding['summary']}"
                    if understanding["keywords"]:
                        screen_context += f"\nמילות מפתח: {', '.join(understanding['keywords'][:5])}"
                    if understanding["answer"]:
                        screen_context += f"\nתשובה לשאלה: {understanding['answer']}"

        action_result = await self._handle_intent(intent, entities, user_text, screen_context)
        
        if action_result.get("handled_locally"):
            response_text = action_result["response"]
            hud_command = action_result.get("hud_command")
            system_action = action_result.get("system_action")
        else:
            response_text = await self._generate_response(
                user_text, intent, entities, screen_context, relevant_memories, smart_context, user_profile_summary
            )
            hud_command = action_result.get("hud_command")
            system_action = action_result.get("system_action")

        self.memory.add_conversation(user_text, response_text)
        if not self.learning_engine:
            self.memory.extract_and_save_facts(user_text)

        proposals = []
        if self.learning_engine:
            try:
                new_proposals = self.learning_engine.process_interaction(user_text, response_text, intent_result)
                proposals.extend(new_proposals)
            except Exception as e:
                print(f"[Brain] Learning failed: {e}")

        self_updates = []
        if self.self_update_manager and self.conversation_turns % 5 == 0:
            try:
                history = [{"content": m["content"], "role": m["role"]} for m in self.memory.short_term[-10:]]
                auto_proposals = self.self_update_manager.auto_detect_improvements(history)
                self_updates.extend(auto_proposals)
            except:
                pass

        return {
            "text": response_text,
            "intent": intent,
            "confidence": confidence,
            "hud_command": hud_command,
            "system_action": system_action,
            "screen_context_used": screen_context is not None,
            "memory_count": len(self.memory.long_term.get("conversations", [])),
            "proposals": proposals,
            "self_updates": self_updates,
            "smart_context": smart_context,
            "user_profile": user_profile_summary,
            "learning_active": self.learning_engine is not None,
            "voice_enabled": True,  # תמיד עם קול!
            "fast_speech_fixed": original_text != user_text
        }

    async def _handle_intent(self, intent: str, entities: Dict, user_text: str, screen_context: Optional[str]) -> Dict:
        if intent == "hud_dock_side":
            return {"handled_locally": True, "response": get_random_response("hud_dock"), "hud_command": {"action": "dock", "position": "right"}}
        elif intent == "hud_center":
            return {"handled_locally": True, "response": get_random_response("hud_center"), "hud_command": {"action": "center"}}
        elif intent == "hud_hide":
            return {"handled_locally": True, "response": get_random_response("hud_hide"), "hud_command": {"action": "hide"}}
        elif intent == "hud_show":
            return {"handled_locally": True, "response": "אני כאן, בוס! מה צריך?", "hud_command": {"action": "show"}}

        elif intent == "time_date":
            now = datetime.datetime.now()
            hebrew_days = ["שני", "שלישי", "רביעי", "חמישי", "שישי", "שבת", "ראשון"]
            day = hebrew_days[now.weekday()]
            return {"handled_locally": True, "response": f"עכשיו {now.strftime('%H:%M')}, יום {day}, {now.strftime('%d/%m/%Y')}, בוס."}

        elif "מה אתה זוכר" in user_text or "מה את זוכרת" in user_text:
            if self.learning_engine:
                summary = self.learning_engine.get_user_profile_summary()
                smart = self.learning_engine.get_smart_context()
                resp = f"בוס, אני זוכרת: {smart} "
                if summary.get("name"):
                    resp += f"קוראים לך {summary['name']}. "
                resp += f"למדתי {summary.get('vocab_learned',0)} מילים ו-{summary.get('interaction_count',0)} שיחות."
                return {"handled_locally": True, "response": resp}
            else:
                return {"handled_locally": True, "response": self.memory.get_context_string()[:300] or "עדיין לומדת להכיר אותך, בוס."}

        elif intent == "screen_analysis":
            if not screen_context:
                return {"handled_locally": False, "response": None}

        elif intent == "dictionary_lookup":
            word = entities.get("word", "").strip()
            if not word:
                import re
                words = re.findall(r'[\u0590-\u05FF]{2,}', user_text)
                if words:
                    word = words[-1]
            if word:
                try:
                    from .hebrew_dictionary import get_hebrew_dictionary
                    heb_dict = get_hebrew_dictionary()
                    result = heb_dict.lookup(word)
                    if result.get("found"):
                        meaning = result.get("meaning", "")
                        example = result.get("example", "")
                        synonyms = result.get("synonyms", [])
                        resp = f"📚 {word}: {meaning}"
                        if example:
                            resp += f"\nדוגמה: {example}"
                        if synonyms:
                            resp += f"\nנרדפות: {', '.join(synonyms[:3])}"
                        if result.get("english"):
                            resp += f"\nEnglish: {result['english']}"
                        return {"handled_locally": True, "response": resp}
                    else:
                        suggestions = result.get("suggestion", [])
                        sug_text = f" התכוונת ל: {', '.join(suggestions)}?" if suggestions else ""
                        stats = heb_dict.get_stats()
                        resp = f"לא מצאתי את '{word}' במילון (יש לי {stats['total_all']} מילים).{sug_text} רוצה שאלמד? תגיד: תלמד את המילה {word}"
                        return {"handled_locally": True, "response": resp}
                except Exception as e:
                    print(f"[Brain] Dict failed: {e}")
                    return {"handled_locally": True, "response": f"ניסיתי לחפש '{word}' אבל הייתה שגיאה."}
            else:
                return {"handled_locally": True, "response": "איזו מילה לחפש, בוס? תגיד: מה זה [מילה]"}

        elif intent == "system_open":
            app = entities.get("app") or entities.get("app_query", "unknown")
            return {"handled_locally": True, "response": f"על זה, פותחת {entities.get('app_he', app)}.", "system_action": {"type": "open_app", "app": app}}
        elif intent == "system_volume":
            if "גביר" in user_text:
                return {"handled_locally": True, "response": "מגבירה, בוס.", "system_action": {"type": "volume", "action": "up"}}
            elif "נמיך" in user_text:
                return {"handled_locally": True, "response": "מנמיכה.", "system_action": {"type": "volume", "action": "down"}}
            elif "השתק" in user_text:
                return {"handled_locally": True, "response": "משתיקה.", "system_action": {"type": "volume", "action": "mute"}}
            else:
                return {"handled_locally": True, "response": "מטפלת בווליום.", "system_action": {"type": "volume", "action": "toggle"}}
        elif intent == "system_search":
            query = entities.get("query", user_text)
            return {"handled_locally": True, "response": f"מחפשת בגוגל: {query}", "system_action": {"type": "search", "query": query}}

        elif intent == "memory_save":
            if self.learning_engine:
                return {"handled_locally": False, "response": None}
            return {"handled_locally": True, "response": "קלטתי, שמרתי בזיכרון, בוס."}

        elif intent == "goodbye":
            if self.learning_engine:
                count = self.learning_engine.get_user_profile_summary().get("interaction_count", 0)
                return {"handled_locally": True, "response": f"יאללה ביי בוס, דיברנו {count} פעמים היום ואני זוכרת הכל. אני כאן אם צריך."}
            return {"handled_locally": True, "response": random.choice(["יאללה ביי בוס, אני כאן אם צריך.", "סגור בוס, היה כיף."])} 

        return {"handled_locally": False}

    async def _generate_response(self, user_text: str, intent: str, entities: Dict, 
                                 screen_context: Optional[str], relevant_memories: list,
                                 smart_context: str = "", user_profile: Dict = None) -> str:
        """היררכיה: LLM 6GB -> Ollama -> OpenAI -> True AI -> Templates"""
        memory_context = self.memory.get_context_string()
        if smart_context:
            memory_context += f"\n[זיכרון חכם]: {smart_context}"
        if user_profile and user_profile.get("name"):
            memory_context += f"\nשם: {user_profile['name']}"

        # 1. LLM 6GB - חדש! 6GB RAM
        if hasattr(self, 'llm_6gb') and self.llm_6gb:
            try:
                system_with_memory = ADIEL_SYSTEM_PROMPT
                if memory_context:
                    system_with_memory += f"\n\nזיכרון:\n{memory_context}"
                if screen_context:
                    system_with_memory += f"\n\nמסך: {screen_context[:1000]}"
                
                # הוסף הקשר לדיבור מהיר
                system_with_memory += "\n\nהמשתמש לפעמים מדבר מהר, תבין גם אם מילים ממוזגות או סלנג מהיר."
                
                response = self.llm_6gb.generate(
                    prompt=user_text,
                    system_prompt=system_with_memory,
                    max_tokens=350,
                    temperature=0.85
                )
                if response and len(response.strip()) > 10:
                    print(f"[Brain] 🚀 LLM 6GB: {response[:80]}...")
                    if self.true_ai:
                        self.true_ai.learn_from_interaction(user_text, response)
                    return response.strip()
            except Exception as e:
                print(f"[Brain] LLM 6GB failed: {e}")

        # 2. Ollama
        if HAS_OLLAMA and not self.use_cloud_llm:
            try:
                ollama_response = await self._try_ollama(user_text, screen_context, memory_context, intent)
                if ollama_response:
                    return ollama_response
            except Exception as e:
                print(f"[Brain] Ollama failed: {e}")

        # 3. OpenAI
        if self.use_cloud_llm and HAS_OPENAI and self.openai_api_key:
            try:
                openai_resp = await self._try_openai(user_text, screen_context, memory_context)
                if openai_resp:
                    return openai_resp
            except Exception as e:
                print(f"[Brain] OpenAI failed: {e}")

        # 4. True AI + קריאת טקסט חכמה
        return self._generate_local_response(user_text, intent, screen_context, relevant_memories, user_profile)

    async def _try_ollama(self, user_text: str, screen_context: Optional[str], memory_ctx: str, intent: str) -> Optional[str]:
        try:
            prompt = f"{ADIEL_SYSTEM_PROMPT}\n\nזיכרון:\n{memory_ctx}\n\nמסך:\n{screen_context or 'אין'}\n\nמשתמש: {user_text}\nכוונה: {intent}\n\nעני בעברית קצר, אדיאל ג'וניור:"
            def call_ollama():
                try:
                    import ollama
                    response = ollama.chat(model=self.ollama_model, messages=[
                        {"role": "system", "content": ADIEL_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt}
                    ])
                    return response['message']['content']
                except:
                    return None
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, call_ollama)
        except:
            return None

    async def _try_openai(self, user_text: str, screen_context: Optional[str], memory_ctx: str) -> Optional[str]:
        if not self.openai_api_key:
            return None
        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.openai_api_key)
            messages = [
                {"role": "system", "content": ADIEL_SYSTEM_PROMPT},
            ]
            if memory_ctx:
                messages.append({"role": "system", "content": f"זיכרון:\n{memory_ctx}"})
            if screen_context:
                messages.append({"role": "system", "content": f"מסך: {screen_context[:1000]}"})
            messages.append({"role": "user", "content": user_text})

            def call_openai():
                resp = client.chat.completions.create(model="gpt-4o-mini", messages=messages, max_tokens=350, temperature=0.8)
                return resp.choices[0].message.content

            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, call_openai)
        except:
            return None

    def _generate_local_response(self, user_text: str, intent: str, screen_context: Optional[str], relevant_memories: list, user_profile: Dict = None) -> str:
        # True AI
        if self.true_ai:
            try:
                context = {}
                if user_profile:
                    context = user_profile
                if self.learning_engine:
                    context.update(self.learning_engine.get_user_profile_summary())
                true_response = self.true_ai.generate_response(user_text, context=context)
                if true_response and len(true_response) > 5:
                    self.true_ai.learn_from_interaction(user_text, true_response)
                    print(f"[Brain] ✨ True AI: {true_response[:60]}...")
                    return true_response
            except Exception as e:
                print(f"[Brain] True AI failed: {e}")

        # מסך עם קריאת טקסט חכמה
        if screen_context and intent == "screen_analysis":
            base = random.choice([
                f"בוס, אני רואה: {screen_context[:300]}. ",
                f"קלטתי מסך: {screen_context[:300]}. ",
            ])
            lower = screen_context.lower()
            if "error" in lower or "שגיאה" in lower:
                return base + "נראה שיש שגיאה. רוצה שאסביר ואתקן?"
            elif "code" in lower or "def " in lower:
                return base + "זה קוד. תגיד מה הבעיה ונדבג יחד?"
            else:
                return base + "מה לעשות עם זה?"

        if relevant_memories:
            mem_hint = relevant_memories[0].get("user", "")[:80]
            return f"זה מזכיר לי '{mem_hint}...' - {get_random_response('fallback_chat')}"

        lower = user_text.lower()
        if any(w in lower for w in ["איך אתה", "מה שלומך"]):
            if user_profile and user_profile.get("interaction_count", 0) > 5:
                return f"אחלה בוס! אחרי {user_profile['interaction_count']} שיחות אני כבר מכירה אותך טוב. מה איתך?"
            return random.choice(["אחלה בוס! רצה על Full Power. מה איתך?", "מצוין, מוכנה לפעולה."])

        if any(w in lower for w in ["תודה", "אלופה"]):
            return random.choice(["בכיף בוס, תמיד כאן. וזוכרת הכל!", "יאללה, זה התפקיד שלי."])

        if any(w in lower for w in ["עזרה", "לא מצליח"]):
            return "קלטתי שיש בעיה. תספר בדיוק מה לא עובד, תגיד 'מה את רואה במסך' ואסרוק."

        return get_random_response("fallback_chat")
