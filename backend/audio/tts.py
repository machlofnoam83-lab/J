"""
Hebrew TTS Engine - Natural & Fast
דיבור עברי טבעי עם edge-tts + pygame playback
"""
import os
import asyncio
import tempfile
import threading
from typing import Optional
import uuid

try:
    import edge_tts
    HAS_EDGE = True
except:
    HAS_EDGE = False
    print("[TTS] edge-tts not available")

try:
    import pygame
    HAS_PYGAME = True
except:
    HAS_PYGAME = False

# קולות עבריים מומלצים - נבדקו לאיכות טבעית
HEBREW_VOICES = {
    "avigail": "he-IL-AvigailNeural",  # נשי, צעיר, טבעי מאוד - ברירת מחדל לג'וניור
    "hila": "he-IL-HilaNeural",        # נשי, בוגר יותר
    "asaf": "he-IL-AsafNeural",        # גברי
}

class HebrewTTS:
    def __init__(self, voice="avigail", rate="+0%", pitch="+0Hz"):
        self.voice_id = HEBREW_VOICES.get(voice, HEBREW_VOICES["avigail"])
        self.rate = rate
        self.pitch = pitch
        self.temp_dir = os.path.join(tempfile.gettempdir(), "adiel_tts")
        os.makedirs(self.temp_dir, exist_ok=True)
        
        # אתחול pygame למיקסר
        if HAS_PYGAME:
            try:
                pygame.mixer.init(frequency=24000)
            except Exception as e:
                print(f"[TTS] Pygame mixer init failed: {e}")

        print(f"[TTS] Initialized with voice {self.voice_id}")

    async def _synthesize_edge(self, text: str, output_path: str) -> bool:
        """סינתזה עם edge-tts"""
        if not HAS_EDGE:
            return False
        
        try:
            # הוספת SSML קל לשיפור טבעיות
            # edge-tts מקבל rate/pitch כ-%
            communicate = edge_tts.Communicate(
                text,
                self.voice_id,
                rate=self.rate,
                pitch=self.pitch
            )
            
            await communicate.save(output_path)
            return True
        except Exception as e:
            print(f"[TTS] edge-tts failed: {e}")
            return False

    async def synthesize(self, text: str, play=True) -> Optional[str]:
        """
        מסנתז טקסט ומנגן (או שומר)
        מחזיר נתיב לקובץ
        """
        if not text or not text.strip():
            return None

        # ניקוי טקסט ל-TTS - הסר אימוג'ים, תקן
        clean_text = self._prepare_text_for_tts(text)
        if not clean_text:
            return None

        file_id = str(uuid.uuid4())[:8]
        output_path = os.path.join(self.temp_dir, f"adiel_{file_id}.mp3")
        
        success = await self._synthesize_edge(clean_text, output_path)
        
        if not success:
            print("[TTS] Synthesis failed, no fallback yet")
            return None
            
        print(f"[TTS] Synthesized: '{clean_text[:50]}...' -> {output_path}")

        if play and HAS_PYGAME:
            self._play_audio(output_path)
        
        return output_path

    def _prepare_text_for_tts(self, text: str) -> str:
        """הכנת טקסט ל-TTS - תיקון סלנג וסימנים"""
        import re
        
        # הסר markdown, קוד, אימוג'ים
        text = re.sub(r'```.*?```', ' ', text, flags=re.DOTALL)
        text = re.sub(r'`[^`]+`', ' ', text)
        text = re.sub(r'[#*_\[\]]', ' ', text)
        
        # תיקון סלנג לכתיב ש-TTS יבין טוב יותר
        slang_fixes = {
            "יאללה": "יאללה",
            "סגור": "סגור",
            "בוס": "בוס,",
            "...": ". ",
        }
        for k, v in slang_fixes.items():
            text = text.replace(k, v)
        
        # הגבל אורך - edge-tts מוגבל
        if len(text) > 800:
            text = text[:800] + "."
            
        # הוסף הפסקות טבעיות
        text = text.replace("?", "? ")
        text = text.replace("!", "! ")
        text = text.replace(",", ", ")
        
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def _play_audio(self, path: str):
        """ניגון לא חוסם"""
        def play_thread():
            try:
                if not os.path.exists(path):
                    return
                pygame.mixer.music.load(path)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    pygame.time.wait(100)
                # נקה קובץ אחרי ניגון
                try:
                    os.remove(path)
                except:
                    pass
            except Exception as e:
                print(f"[TTS] Playback failed: {e}")

        t = threading.Thread(target=play_thread, daemon=True)
        t.start()

    def synthesize_sync(self, text: str, play=True) -> Optional[str]:
        """גרסה סינכרונית"""
        try:
            # נסה לקבל loop קיים, אם לא - צור חדש
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # אם ה-loop רץ (FastAPI), השתמש ב-run_in_executor
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as pool:
                        future = pool.submit(asyncio.run, self.synthesize(text, play))
                        return future.result(timeout=15)
                else:
                    return loop.run_until_complete(self.synthesize(text, play))
            except RuntimeError:
                return asyncio.run(self.synthesize(text, play))
        except Exception as e:
            print(f"[TTS] Sync synthesize failed: {e}")
            return None

    def stop(self):
        if HAS_PYGAME:
            try:
                pygame.mixer.music.stop()
            except:
                pass

# פונקציית wrapper פשוטה ל-backend main
_global_tts = None

def get_tts_engine():
    global _global_tts
    if _global_tts is None:
        _global_tts = HebrewTTS(voice="avigail")
    return _global_tts

async def speak_hebrew(text: str, play=True) -> Optional[str]:
    engine = get_tts_engine()
    return await engine.synthesize(text, play=play)

# Test
if __name__ == "__main__":
    async def test():
        tts = HebrewTTS()
        await tts.synthesize("שלום בוס! אני אדיאל ג'וניור, העוזרת האישית שלך. מוכנה לפעולה!", play=True)
        await asyncio.sleep(5)

    asyncio.run(test())
