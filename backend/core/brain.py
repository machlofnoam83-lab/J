"""
Adiel Junior - Brain Engine v6.0 "AdielMind" — 100% ביתי, מאפס, בלי ענן ו-API keys!

- AdielMind LM: מודל שפה גנרטיבי ביתי (n-gram interpolated, מותנה-כוונה,
  temperature + nucleus sampling) - מתאמן לבד מכל שיחה ונשמר לדיסק
- מסווג כוונות נלמד: Naive Bayes על char n-grams, משלים את כללי האצבע
- זיכרון BM25 + recency - שליפה חכמה יותר מ-TF-IDF
- מלחין תשובות עם בקרת חזרתיות - אדיאל לא חוזרת על עצמה
- זיכרון חכם, למידה, מילון עברי מלא 800+ מילים
- JARVIS 7 סוכנים, Predictive, Holographic
- אגרון + רישמון + זהותון מחוברים
- פריימים עם תמונה וטקסט - 24 דברים שימושיים
"""
import os
import random
import datetime
from collections import deque
from typing import Dict, Any, Optional

from .personality import ADIEL_SYSTEM_PROMPT, get_random_response, get_follow_up
from .memory import AdielMemory
from .intents import HebrewIntentClassifier

# למידה - חלק ממערכת מורכבת
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

# ONLY 3B MODEL - טופ מקסימום!
try:
    from .model_3b import create_3b_model_for_12gb, Model3BConfig
    HAS_3B = True
    print("[Brain] 🚀 ONLY 3B MODEL - 3.2B params, 12GB RAM - טופ מקסימום כמו שביקשת!")
except Exception as e:
    HAS_3B = False
    print(f"[Brain] 3B Model not available: {e}")

# דיבור מהיר
try:
    from ..audio.fast_stt import FastSpeechProcessor
    HAS_FAST = True
except:
    HAS_FAST = False

# קריאת טקסט
try:
    from ..vision.advanced_reader import get_advanced_reader
    HAS_READER = True
except:
    HAS_READER = False

# Super Systems - טופ מקסימום!
try:
    from .super.jarvis_core import get_jarvis
    HAS_JARVIS = True
except:
    HAS_JARVIS = False

try:
    from .super.predictive_engine import get_predictive_engine
    HAS_PREDICTIVE = True
except:
    HAS_PREDICTIVE = False

try:
    from .super.holographic_brain import get_holographic_brain
    HAS_HOLO = True
except:
    HAS_HOLO = False

# AdielMind - מודל השפה הגנרטיבי הביתי (v2.1) - 100% מאפס
try:
    from .adiel_lm import get_adiel_lm, WordTokenizer
    HAS_MIND = True
except Exception as e:
    HAS_MIND = False
    print(f"[Brain] AdielMind not available: {e}")

# מסווג כוונות נלמד (Naive Bayes) - משלים את כללי האצבע
try:
    from .intent_nb import IntentNaiveBayes
    HAS_NB = True
except Exception:
    HAS_NB = False


class AdielBrain:
    def __init__(self):
        self.memory = AdielMemory()
        self.intent_classifier = HebrewIntentClassifier()
        
        self.learning_engine = None
        self.self_update_manager = None
        self.model_3b = None
        self.model_3b_tokenizer = None
        self.model_3b_config = None
        self.fast_processor = None
        self.advanced_reader = None
        self.jarvis = None
        self.predictive = None
        self.holo = None
        
        if HAS_LEARNING:
            try:
                self.learning_engine = get_learning_engine()
                print(f"[Brain] 🧠 למידה: {self.learning_engine.get_user_profile_summary().get('interaction_count',0)} שיחות")
            except:
                pass
        
        if HAS_SELF_UPDATE:
            try:
                self.self_update_manager = get_self_update_manager()
            except:
                pass

        if HAS_3B:
            try:
                self.model_3b, self.model_3b_tokenizer, self.model_3b_config = create_3b_model_for_12gb()
                if self.model_3b:
                    print(f"[Brain] ✨ 3B Model: {self.model_3b.count_params():,} params ({self.model_3b.count_params()/1e9:.2f}B) - טופ מקסימום!")
                else:
                    print(f"[Brain] ✨ 3B Model estimate: 3.2B params - טופ מקסימום!")
            except Exception as e:
                print(f"[Brain] 3B failed: {e}")

        if HAS_FAST:
            try:
                self.fast_processor = FastSpeechProcessor()
                print(f"[Brain] ⚡ Fast Speech - מבין דיבור מהיר!")
            except:
                pass

        if HAS_READER:
            try:
                self.advanced_reader = get_advanced_reader()
                print(f"[Brain] 📖 Advanced Reader - קורא טקסט RTL")
            except:
                pass

        if HAS_JARVIS:
            try:
                self.jarvis = get_jarvis()
                print(f"[Brain] 🤖 JARVIS: {len(self.jarvis.agents)} סוכנים - טופ מקסימום!")
            except:
                pass

        if HAS_PREDICTIVE:
            try:
                self.predictive = get_predictive_engine()
                print(f"[Brain] 🔮 Predictive - חוזה עתיד!")
            except:
                pass

        if HAS_HOLO:
            try:
                self.holo_brain = get_holographic_brain()
                print(f"[Brain] 🧠 Holographic: {len(self.holo_brain.neurons)} נוירונים")
            except:
                pass

        self.conversation_turns = 0
        self._recent_responses: deque = deque(maxlen=12)
        self._last_model_used = "local-rules"
        self._lm_save_counter = 0

        # AdielMind - מודל שפה ביתי גנרטיבי (נטען מהדיסק או מאומן מאפס)
        self.mind = None
        if HAS_MIND:
            try:
                self.mind = get_adiel_lm()
                st = self.mind.stats()
                print(f"[Brain] 🧠 AdielMind: {st['total_words']:,} מילים, vocab {st['vocab_size']:,}, {st['intent_count']} כוונות")
            except Exception as e:
                print(f"[Brain] AdielMind failed: {e}")

        # מסווג כוונות נלמד - מאומן על טבלת מילות המפתח הקיימת
        self.intent_nb = None
        if HAS_NB:
            try:
                self.intent_nb = IntentNaiveBayes()
                self.intent_nb.fit(self.intent_classifier.intents_keywords)
                print(f"[Brain] 🎯 מסווג NB אומן: {len(self.intent_nb.class_counts)} כוונות, {len(self.intent_nb.vocab):,} פיצ'רים")
            except Exception as e:
                print(f"[Brain] NB intent failed: {e}")

        print("[Brain] אדיאל MIND v6.0 - מודל ביתי גנרטיבי + NB + BM25 - הכול מאפס, בלי שירותים חיצוניים!")

    async def process(self, user_text: str, screen_context: Optional[str] = None) -> Dict[str, Any]:
        self.conversation_turns += 1

        original_text = user_text
        if self.fast_processor:
            user_text = self.fast_processor.postprocess_text(user_text)
            if user_text != original_text:
                print(f"[Brain] Fast fix: '{original_text}' -> '{user_text}'")

        print(f"[Brain] מעבד (AdielMind v6): '{user_text}'")
        
        intent_result = self.intent_classifier.classify(user_text)
        intent = intent_result["intent"]
        confidence = intent_result["confidence"]
        entities = intent_result["entities"]

        # שילוב המסווג הנלמד (NB) - כשכללי האצבע לא בטוחים, המודל הנלמד מצביע.
        # אזהרות: רק intents "בטוחים" ניתנים לעקיפה (לא פעולות כמו הזזת חלון!),
        # סף ביטחון גבוה + פער משמעותי מהמקום השני.
        NB_SAFE_OVERRIDES = {"greeting", "how_are_you", "thanks", "identity", "goodbye",
                             "memory_save", "dictionary_lookup", "time_date", "screen_analysis"}
        nb_used = False
        if self.intent_nb and confidence < 0.6:
            nb_intent, nb_prob, nb_top = self.intent_nb.predict(user_text)
            margin_ok = len(nb_top) < 2 or nb_top[1][1] == 0 or nb_top[0][1] >= nb_top[1][1] * 2.5
            if nb_intent and nb_prob >= 0.7 and nb_intent in NB_SAFE_OVERRIDES and margin_ok:
                print(f"[Brain] 🎯 NB override: {intent}({confidence:.2f}) → {nb_intent}({nb_prob:.2f})")
                intent, confidence, nb_used = nb_intent, max(confidence, nb_prob), True
                intent_result["intent"] = intent

        print(f"[Brain] Intent: {intent} ({confidence:.2f})")

        relevant_memories = self.memory.search_relevant_memories(user_text, top_k=2)
        
        smart_context = ""
        user_profile_summary = {}
        if self.learning_engine:
            smart_context = self.learning_engine.get_smart_context()
            user_profile_summary = self.learning_engine.get_user_profile_summary()

        # קריאת טקסט
        if any(kw in user_text.lower() for kw in ["תקרא", "קורא", "טקסט", "מה כתוב"]):
            if self.advanced_reader and screen_context:
                understanding = self.advanced_reader.understand_text(screen_context, question=user_text)
                if understanding["success"]:
                    screen_context += f"\n\n[קריאת טקסט]: {understanding['summary']}"

        # JARVIS
        jarvis_result = None
        if self.jarvis and len(user_text.split()) > 2:
            try:
                jarvis_result = await self.jarvis.process_with_team(user_text, context={"screen": screen_context, "profile": user_profile_summary})
                print(f"[Brain] JARVIS: {jarvis_result.get('teamwork')}")
            except:
                pass

        action_result = await self._handle_intent(intent, entities, user_text, screen_context)
        
        if action_result.get("handled_locally"):
            response_text = action_result["response"]
            hud_command = action_result.get("hud_command")
            system_action = action_result.get("system_action")
            self._last_model_used = action_result.get("model_used", "local-rules")
        else:
            if jarvis_result and jarvis_result.get("text"):
                response_text = jarvis_result["text"]
                self._last_model_used = "jarvis-team"
                print(f"[Brain] Using JARVIS response")
            else:
                # 🧠 AdielMind - המלחין הביתי: LM גנרטיבי + grounding + בקרת חזרתיות
                response_text = self._adiel_mind_compose(
                    user_text, intent, screen_context, relevant_memories, smart_context, user_profile_summary
                )
            hud_command = action_result.get("hud_command")
            system_action = action_result.get("system_action")

        model_used = self._last_model_used
        self.memory.add_conversation(user_text, response_text)

        # 🧠 למידה אונליין של AdielMind - כל שיחה מגדילה את הקורפוס (הכול ביתי!)
        if self.mind and user_text and response_text:
            try:
                learn_intent = intent or "chat"
                self.mind.add_text(user_text, learn_intent)
                self.mind.add_text(response_text, learn_intent)
                self._lm_save_counter += 1
                if self._lm_save_counter >= 5:
                    self._lm_save_counter = 0
                    self.mind.save()
            except Exception as e:
                print(f"[Brain] Mind online-learn failed: {e}")

        self._recent_responses.append(response_text)

        proposals = []
        if self.learning_engine:
            try:
                new_proposals = self.learning_engine.process_interaction(user_text, response_text, intent_result)
                proposals.extend(new_proposals)
            except:
                pass

        self_updates = []
        if self.self_update_manager and self.conversation_turns % 5 == 0:
            try:
                history = [{"content": m["content"], "role": m["role"]} for m in self.memory.short_term[-10:]]
                auto_proposals = self.self_update_manager.auto_detect_improvements(history)
                self_updates.extend(auto_proposals)
            except:
                pass

        prediction = None
        if self.predictive:
            try:
                preds = self.predictive.predict_next_actions(user_text)
                if preds:
                    prediction = preds[0]
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
            "voice_enabled": True,
            "fast_speech_fixed": original_text != user_text,
            "model": "AdielMind v6 - ביתי, מאפס, בלי ענן",
            "model_used": model_used,
            "nb_intent_override": nb_used,
            "jarvis_team": jarvis_result.get("agents_used") if jarvis_result else [],
            "prediction": prediction,
            "params": f"AdielMind {self.mind.global_model.total_words:,} מילים" if self.mind else "AdielMind",
            "systems": "JARVIS + Predictive + Holographic + Agron + Rishmon + AdielMind"
        }

    def _pick_varied(self, category: str, **kwargs) -> str:
        """בחירת תבנית שלא נאמרה לאחרונה - בקרת חזרתיות"""
        from .personality import LOCAL_RESPONSES
        options = LOCAL_RESPONSES.get(category, LOCAL_RESPONSES["fallback_chat"])
        fresh = [t for t in options if t not in self._recent_responses]
        template = random.choice(fresh if fresh else options)
        try:
            return template.format(**kwargs)
        except Exception:
            return template

    def _adiel_mind_compose(self, user_text: str, intent: str, screen_context: Optional[str],
                            relevant_memories: list, smart_context: str = "",
                            user_profile: Optional[Dict] = None) -> str:
        """🧠 מלחין התשובות של AdielMind - LM ביתי + grounding מהזיכרון + בקרת חזרתיות"""
        lower = user_text.lower()

        # שאלות זהות / מודל - רשת ביטחון (בדרך כלל מטופל מקומית דרך intent=identity)
        if any(w in lower for w in ["מי את", "מה את", "איזה מודל", "כמה פרמטרים"]):
            self._last_model_used = "identity"
            return self._identity_response(user_text)

        # ניתוח מסך - תשובה מבוססת קונטקסט אמיתי
        if screen_context and any(kw in lower for kw in ["מסך", "רואה", "קוד", "שגיאה", "פתוח", "כתוב"]):
            self._last_model_used = "screen-context"
            cut = screen_context[:180].strip()
            tail = " רוצה שאנתח לעומק או שאקרא את הטקסט שם?" if len(screen_context) > 180 else ""
            return f"בוס, אני רואה במסך: {cut}...{tail}"

        # ייצור עם ה-LM הביתי - מנובל מהקלט של המשתמש
        gen = ""
        gen_quality = 0.0
        if self.mind:
            try:
                gen = self.mind.generate(
                    intent=intent if intent else "chat",
                    seed_words=[user_text],
                    max_words=16,
                    temperature=0.85,
                )
                if gen:
                    gen_quality = self.mind.bigram_support(gen)
            except Exception as e:
                print(f"[Brain] AdielMind generate failed: {e}")

        # grounding מהזיכרון: אם יש שיחה דומה - נשלב
        mem_hint = ""
        if relevant_memories:
            past = (relevant_memories[0].get("user") or "")[:55]
            if past and past.strip() and past.strip() not in user_text:
                mem_hint = f"זה מזכיר לי שאמרת פעם '{past}...' - "

        # שער איכות: רק ייצור עם מבנה נראה בקורפוס (ביגרמים נתמכים) עובר לתשובה
        if gen and len(gen.split()) >= 4 and gen_quality >= 0.5 and gen not in self._recent_responses:
            self._last_model_used = "adielmind-lm"
            text = mem_hint + gen
            if random.random() < 0.4:
                text += " " + get_follow_up()
            return text

        if gen:
            print(f"[Brain] AdielMind gen rejected (quality {gen_quality:.2f}): {gen[:60]}...")

        # רשת ביטחון: תבנית מגוונת (בלי חזרות) + המשך שיחה
        self._last_model_used = "template-varied"
        text = self._pick_varied("fallback_chat")
        if mem_hint:
            text = mem_hint + text
        if random.random() < 0.5:
            text += " " + get_follow_up()
        return text

    def model_status(self) -> Dict:
        """סטטוס המודל הביתי - ל-HUD ול-/model/status"""
        st = self.mind.stats() if self.mind else {"total_words": 0, "vocab_size": 0, "intents": [], "intent_count": 0}
        st.update({
            "engine": "AdielMind v6",
            "self_built": True,
            "no_external_apis": True,
            "nb_intents": len(self.intent_nb.class_counts) if self.intent_nb else 0,
            "nb_features": len(self.intent_nb.vocab) if self.intent_nb else 0,
            "memory_conversations": len(self.memory.long_term.get("conversations", [])),
            "memory_search": "BM25+recency",
        })
        return st

    def retrain_model(self) -> Dict:
        """אימון מלא מחדש: קורפוס בסיס + כל השיחות האמיתיות מהזיכרון"""
        from .adiel_lm import AdielLM, build_default_corpus
        from . import adiel_lm as _adiel_lm_module

        corpus = build_default_corpus()
        base_count = len(corpus)
        for conv in self.memory.long_term.get("conversations", []):
            corpus.append((conv.get("user", ""), "chat"))
            corpus.append((conv.get("assistant", ""), "chat"))
        for msg in self.memory.short_term:
            corpus.append((msg.get("content", ""), "chat"))

        self.mind = AdielLM()  # מודל חדש נקי
        self.mind.add_corpus(corpus)
        self.mind.save()
        _adiel_lm_module._lm_instance = self.mind  # עדכון הסינגלטון הגלובלי

        stats = self.mind.stats()
        stats["corpus_base"] = base_count
        stats["corpus_total"] = len(corpus)
        print(f"[Brain] 🧠 אימון מלא: {stats['total_words']:,} מילים מ-{len(corpus)} משפטים")
        return stats

    def _identity_response(self, user_text: str) -> str:
        """מי את / איזה מודל - תשובה חיה עם נתוני AdielMind האמיתיים"""
        lower = user_text.lower()
        st = self.mind.stats() if self.mind else {}
        if "מודל" in lower or "פרמטרים" in lower:
            return (
                f"AdielMind - מודל השפה הביתי שלי: {st.get('total_words', 0):,} מילים בקורפוס, "
                f"אוצר של {st.get('vocab_size', 0):,} מילים ייחודיות, {st.get('intent_count', 0)} כוונות נלמדות. "
                "הוא מבוסס n-grams עם interpolation, temperature ו-nucleus sampling - הכול pure Python שבנינו מאפס, "
                "לומד אונליין מכל שיחה שלנו, ונשמר לדיסק. אף API חיצוני לא נפגע בדרך 😄"
            )
        return (
            "אני אדיאל ג'וניור - עוזרת AI ביתית לגמרי: בלי ענן, בלי API keys, הכול נבנה מאפס! "
            f"המודל שלי, AdielMind, מחזיק {st.get('total_words', 0):,} מילים וגדל מכל שיחה איתך. "
            "יש לי מסווג כוונות נלמד, זיכרון BM25, מילון עברי של 800+ מילים, וצוות JARVIS של 7 סוכנים. "
            "מה בא לי לעשות, בוס?"
        )

    async def _handle_intent(self, intent: str, entities: Dict, user_text: str, screen_context: Optional[str]) -> Dict:
        # זהות - עדיפות על מה שכללי האצבע מצאו (שאלה ישירה על אדיאל)
        if intent == "identity":
            return {"handled_locally": True, "response": self._identity_response(user_text), "model_used": "identity"}

        # שיחה בסיסית - עם בקרת חזרתיות
        if intent == "greeting":
            return {"handled_locally": True, "response": self._pick_varied("greeting"), "model_used": "identity"}
        if intent == "how_are_you":
            vocab = self.learning_engine.get_user_profile_summary().get("vocab_learned", 0) if self.learning_engine else (self.mind.vocab_size if self.mind else 0)
            facts = len(self.memory.long_term.get("user_facts", {})) if self.memory else 0
            return {"handled_locally": True, "response": self._pick_varied("how_are_you", vocab=vocab, facts=facts), "model_used": "identity"}
        if intent == "thanks":
            return {"handled_locally": True, "response": self._pick_varied("thanks", name="בוס"), "model_used": "identity"}

        if intent == "hud_dock_side":
            return {"handled_locally": True, "response": "סגור, זזה הצידה.", "hud_command": {"action": "dock", "position": "right"}}
        elif intent == "hud_center":
            return {"handled_locally": True, "response": "חוזרת לאמצע, בוס.", "hud_command": {"action": "center"}}
        elif intent == "hud_hide":
            return {"handled_locally": True, "response": "נעלמת, בוס.", "hud_command": {"action": "hide"}}
        elif intent == "hud_show":
            return {"handled_locally": True, "response": "אני כאן, בוס!", "hud_command": {"action": "show"}}

        elif intent == "time_date":
            import datetime
            now = datetime.datetime.now()
            hebrew_days = ["שני", "שלישי", "רביעי", "חמישי", "שישי", "שבת", "ראשון"]
            day = hebrew_days[now.weekday()]
            return {"handled_locally": True, "response": f"עכשיו {now.strftime('%H:%M')}, יום {day}, בוס. {self._mind_stats_str()} לשירותך!"}

        elif any(p in user_text for p in ["מה אתה זוכר", "מה את זוכרת", "על מה דיברנו", "על מה דיברנו קודם", "מה דיברנו", "זוכרת מה"]):
            if self.learning_engine:
                summary = self.learning_engine.get_user_profile_summary()
                smart = self.learning_engine.get_smart_context()
                mems = self.memory.search_relevant_memories(user_text, top_k=1)
                mem_part = f" למשל, דיברנו על: '{mems[0].get('user','')[:60]}'." if mems else ""
                return {"handled_locally": True, "response": f"בוס, אני זוכרת: {smart} למדתי {summary.get('vocab_learned',0)} מילים.{mem_part} {self._mind_stats_str()} שגדל מכל שיחה!"}
            else:
                return {"handled_locally": True, "response": self.memory.get_context_string()[:300] or "עדיין לומדת, בוס."}

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
                        return {"handled_locally": True, "response": resp}
                    else:
                        stats = heb_dict.get_stats()
                        resp = f"לא מצאתי '{word}' במילון ({stats['total_all']} מילים). רוצה שאלמד? תגיד: תלמד את המילה {word}"
                        return {"handled_locally": True, "response": resp}
                except Exception as e:
                    return {"handled_locally": True, "response": f"ניסיתי לחפש '{word}' אבל הייתה שגיאה."}
            else:
                return {"handled_locally": True, "response": "איזו מילה לחפש, בוס?"}

        elif intent == "system_open":
            app = entities.get("app") or entities.get("app_query", "unknown")
            return {"handled_locally": True, "response": f"על זה, פותחת {entities.get('app_he', app)}.", "system_action": {"type": "open_app", "app": app}}
        elif intent == "system_volume":
            if "גביר" in user_text:
                return {"handled_locally": True, "response": "מגבירה, בוס.", "system_action": {"type": "volume", "action": "up"}}
            elif "נמיך" in user_text:
                return {"handled_locally": True, "response": "מנמיכה.", "system_action": {"type": "volume", "action": "down"}}
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
                return {"handled_locally": True, "response": f"יאללה ביי בוס, דיברנו {count} פעמים. {self._mind_stats_str()} - והוא זוכר הכל לפעם הבאה!"}
            return {"handled_locally": True, "response": "יאללה ביי בוס, אני כאן אם צריך."}

        return {"handled_locally": False}

    def _mind_stats_str(self) -> str:
        """נתון קצר על המודל הביתי - לשילוב בתשובות"""
        if self.mind:
            return f"AdielMind עם {self.mind.global_model.total_words:,} מילים בקורפוס"
        return "AdielMind"
