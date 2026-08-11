"""
Continuous Conversation - שיחה קולית רציפה אמיתית
לא רק להעיר עם קול, גם לדבר עם קול באופן רציף!

זה מה שהופך אותה לשיחה אמיתית, לא דמו
"""
import asyncio
import time
from datetime import datetime, timedelta
from typing import Optional, Callable
from pathlib import Path

class ContinuousConversation:
    """
    מנהל שיחה קולית רציפה - כמו שיחה אמיתית עם חבר
    """
    def __init__(self, brain, stt_engine, tts_engine, wake_detector=None):
        self.brain = brain
        self.stt = stt_engine
        self.tts = tts_engine
        self.wake_detector = wake_detector
        
        # מצב שיחה
        self.is_active = False
        self.conversation_start = None
        self.last_interaction = None
        self.turn_count = 0
        self.silence_limit = 30  # שניות של שקט עד סיום שיחה
        self.max_duration = 300  # 5 דקות מקסימום שיחה רציפה
        
        # היסטוריה
        self.history = []
        
        print("[ContinuousConv] 🎙️ שיחה רציפה מוכנה - לא רק הערה, גם דיבור רציף!")

    def start_conversation(self, trigger_text: str = "היי"):
        """מתחיל שיחה רציפה"""
        self.is_active = True
        self.conversation_start = datetime.now()
        self.last_interaction = datetime.now()
        self.turn_count = 0
        print(f"[ContinuousConv] 🟢 שיחה רציפה התחילה! טריגר: '{trigger_text}' - תדבר חופשי, אני מקשיבה!")
        return True

    def end_conversation(self, reason="timeout"):
        """מסיים שיחה"""
        if self.is_active:
            duration = (datetime.now() - self.conversation_start).total_seconds() if self.conversation_start else 0
            print(f"[ContinuousConv] 🔴 שיחה הסתיימה אחרי {duration:.0f}s, {self.turn_count} תורות, סיבה: {reason}")
            self.is_active = False
            self.conversation_start = None
            return True
        return False

    def should_continue(self) -> bool:
        """האם להמשיך שיחה רציפה?"""
        if not self.is_active:
            return False
        
        now = datetime.now()
        
        # בדוק זמן מקסימום
        if self.conversation_start:
            elapsed = (now - self.conversation_start).total_seconds()
            if elapsed > self.max_duration:
                print(f"[ContinuousConv] ⏰ זמן מקסימום {self.max_duration}s עבר")
                return False
        
        # בדוק שקט
        if self.last_interaction:
            silence = (now - self.last_interaction).total_seconds()
            if silence > self.silence_limit:
                print(f"[ContinuousConv] 🤫 שקט {silence:.0f}s > {self.silence_limit}s - מסיים שיחה")
                return False
        
        return True

    async def listen_continuous(self, max_turns=20) -> bool:
        """
        לולאת האזנה רציפה - לב השיחה האמיתית!
        מקשיבה שוב ושוב בלי צורך במילת הפעלה
        """
        if not self.stt:
            print("[ContinuousConv] אין STT - לא יכול להקשיב רציף")
            return False
        
        print(f"[ContinuousConv] 👂 מתחיל לולאת האזנה רציפה - עד {max_turns} תורות או {self.silence_limit}s שקט")
        
        turn = 0
        while self.should_continue() and turn < max_turns:
            turn += 1
            print(f"\n[ContinuousConv] תור {turn}/{max_turns} - מקשיב... (דבר עכשיו! גם מהר!)")
            
            try:
                # הקלטה - עם fast STT שמבין דיבור מהיר!
                if hasattr(self.stt, 'listen_fast'):
                    # השתמש ב-fast אם יש
                    user_text = await asyncio.get_event_loop().run_in_executor(
                        None, lambda: self.stt.listen_fast(max_seconds=10)
                    )
                else:
                    user_text = await asyncio.get_event_loop().run_in_executor(
                        None, lambda: self.stt.listen_and_transcribe(max_seconds=8)
                    )
                
                if not user_text or len(user_text.strip()) < 2:
                    print(f"[ContinuousConv] לא שמעתי כלום בתור {turn}, ממשיך...")
                    # אל תסיים מיד, תן עוד צ'אנס
                    await asyncio.sleep(0.5)
                    continue
                
                print(f"[ContinuousConv] 🎤 שמעתי (תור {turn}): '{user_text}'")
                
                # בדוק אם המשתמש רוצה לסיים
                if any(word in user_text.lower() for word in ["ביי", "להתראות", "סגור", "סיימנו", "תודה ביי"]):
                    print(f"[ContinuousConv] המשתמש אמר ביי - מסיים שיחה")
                    await self.process_and_speak(user_text)
                    self.end_conversation(reason="user said bye")
                    break
                
                # עבד והגב עם קול!
                await self.process_and_speak(user_text)
                
                self.last_interaction = datetime.now()
                self.turn_count += 1
                
                # הפסקה קטנה לפני התור הבא
                await asyncio.sleep(0.3)
                
            except Exception as e:
                print(f"[ContinuousConv] שגיאה בתור {turn}: {e}")
                import traceback; traceback.print_exc()
                await asyncio.sleep(1)
        
        print(f"[ContinuousConv] לולאה הסתיימה אחרי {turn} תורות")
        self.end_conversation(reason="loop ended")
        return True

    async def process_and_speak(self, user_text: str):
        """מעבד טקסט ומדבר עם קול - תמיד עם קול!"""
        if not self.brain:
            print("[ContinuousConv] אין מוח")
            return
        
        try:
            # עבד עם המוח - עם 3B model, מילון מלא, דיבור מהיר
            result = await self.brain.process(user_text, screen_context=None)
            response_text = result.get("text", "...")
            
            print(f"[ContinuousConv] 🧠 תשובה: '{response_text[:80]}...'")
            
            # תמיד עם קול! - לא משנה איזו שאלה
            if self.tts:
                try:
                    # נסה AI Voice Generator קודם - קול אמיתי לכל שאלה
                    from ..audio.ai_voice import get_ai_voice
                    try:
                        ai_voice = get_ai_voice()
                        path, b64 = await ai_voice.generate_voice_for_any_question(response_text, play=True)
                        if path or b64:
                            print(f"[ContinuousConv] 🔊 AI Voice - מדברת עם קול AI אמיתי לכל שאלה!")
                        else:
                            # Fallback ל-TTS רגיל
                            await self.tts.synthesize(response_text, play=True)
                    except:
                        # Fallback
                        await self.tts.synthesize(response_text, play=True)
                        print(f"[ContinuousConv] 🔊 TTS - מדברת!")
                except Exception as e:
                    print(f"[ContinuousConv] TTS failed: {e}")
                    # גם אם TTS נכשל, לפחות נדפיס
                    print(f"[ContinuousConv] 💬 (בלי קול): {response_text}")
            else:
                print(f"[ContinuousConv] 💬 (אין TTS): {response_text}")
            
            # שמור בהיסטוריה
            self.history.append({
                "user": user_text,
                "assistant": response_text,
                "time": datetime.now().isoformat()
            })
            
            return response_text
            
        except Exception as e:
            print(f"[ContinuousConv] Process and speak failed: {e}")
            import traceback; traceback.print_exc()
            return None

# Singleton
_global_continuous = None

def get_continuous_conversation(brain=None, stt=None, tts=None, wake_detector=None):
    global _global_continuous
    if _global_continuous is None:
        _global_continuous = ContinuousConversation(brain, stt, tts, wake_detector)
    else:
        # עדכן מנועים אם השתנו
        if brain:
            _global_continuous.brain = brain
        if stt:
            _global_continuous.stt = stt
        if tts:
            _global_continuous.tts = tts
    return _global_continuous
