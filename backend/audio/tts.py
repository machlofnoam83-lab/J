"""
Hebrew TTS Engine - Natural & Fast v2.1 with Auto-Fix — 🔓 בלי API keys!
דיבור עברי טבעי - עם מערכת fallback אוטומטית שמתקנת

Chain: קולות מקוריים -> edge-tts (חינמי, מדולג אם ADIEL_OFFLINE=1) -> pyttsx3 (אופליין) -> win32 SAPI
מחזיר גם base64 ל-frontend playback
"""
import os
import asyncio
import tempfile
import threading
import base64
from typing import Optional, Tuple
import uuid

# מצב אופליין מלא - שום דבר לא יוצא לרשת (ADIEL_OFFLINE=1)
OFFLINE_MODE = os.getenv("ADIEL_OFFLINE", "").lower() in ("1", "true", "yes", "on")

try:
    import edge_tts
    HAS_EDGE = True
except:
    HAS_EDGE = False
    print("[TTS] edge-tts not available - will use fallback")

try:
    import pygame
    HAS_PYGAME = True
except:
    HAS_PYGAME = False
    print("[TTS] pygame-ce not available - no local playback")

# Fallbacks
try:
    import pyttsx3
    HAS_PYTTSX3 = True
except:
    HAS_PYTTSX3 = False

try:
    import win32com.client
    HAS_WIN32 = True
except:
    HAS_WIN32 = False

# קולות עבריים מומלצים
HEBREW_VOICES = {
    "avigail": "he-IL-AvigailNeural",
    "hila": "he-IL-HilaNeural",
    "asaf": "he-IL-AsafNeural",
}

class HebrewTTS:
    def __init__(self, voice="avigail", rate="+0%", pitch="+0Hz"):
        self.voice_id = HEBREW_VOICES.get(voice, HEBREW_VOICES["avigail"])
        self.rate = rate
        self.pitch = pitch
        self.temp_dir = os.path.join(tempfile.gettempdir(), "adiel_tts")
        os.makedirs(self.temp_dir, exist_ok=True)
        
        # אתחול pygame
        if HAS_PYGAME:
            try:
                pygame.mixer.quit()
                pygame.mixer.init(frequency=24000, size=-16, channels=2, buffer=512)
                print(f"[TTS] Pygame mixer ready")
            except Exception as e:
                print(f"[TTS] Pygame mixer init failed: {e}")

        # אתחול pyttsx3 fallback
        self.pyttsx3_engine = None
        if HAS_PYTTSX3:
            try:
                self.pyttsx3_engine = pyttsx3.init()
                self.pyttsx3_engine.setProperty('rate', 180)
                print(f"[TTS] pyttsx3 fallback ready")
            except Exception as e:
                print(f"[TTS] pyttsx3 init failed: {e}")

        print(f"[TTS] Initialized - Edge:{HAS_EDGE and not OFFLINE_MODE} Pygame:{HAS_PYGAME} pyttsx3:{HAS_PYTTSX3} SAPI:{HAS_WIN32} voice:{self.voice_id} OfflineMode:{OFFLINE_MODE} (🔓 בלי מפתחות)")

    async def _synthesize_edge(self, text: str, output_path: str) -> bool:
        """סינתזה עם edge-tts - הכי טבעי לעברית - עם fallback קולות"""
        if not HAS_EDGE:
            return False
        
        # רשימת קולות לנסות לפי סדר עדיפות
        voices_to_try = [self.voice_id, "he-IL-HilaNeural", "he-IL-AsafNeural", "he-IL-AvigailNeural"]
        # הסר כפילויות תוך שמירת סדר
        voices_to_try = list(dict.fromkeys(voices_to_try))
        
        for voice in voices_to_try:
            try:
                # נסיון 1: בלי rate/pitch (הכי יציב)
                communicate = edge_tts.Communicate(text, voice)
                await communicate.save(output_path)
                
                if os.path.exists(output_path) and os.path.getsize(output_path) > 100:
                    if voice != self.voice_id:
                        print(f"[TTS] ✓ edge-tts success with fallback voice {voice}")
                    return True
                    
            except Exception as e:
                print(f"[TTS] edge-tts voice {voice} failed (no params): {e}")
                
                # נסיון 2: עם rate/pitch רק אם בלי נכשל? בד"כ בלי עדיף
                try:
                    communicate = edge_tts.Communicate(text, voice, rate=self.rate, pitch=self.pitch)
                    await communicate.save(output_path)
                    if os.path.exists(output_path) and os.path.getsize(output_path) > 100:
                        return True
                except Exception as e2:
                    print(f"[TTS] edge-tts voice {voice} failed (with params): {e2}")
                    continue
        
        print(f"[TTS] All edge-tts voices failed for text: {text[:30]}...")
        return False

    def _synthesize_pyttsx3(self, text: str, output_path: str) -> bool:
        """Fallback 1: pyttsx3 אופליין"""
        if not HAS_PYTTSX3 or not self.pyttsx3_engine:
            return False
        
        try:
            # pyttsx3 רוצה wav, לא mp3
            wav_path = output_path.replace('.mp3', '.wav')
            self.pyttsx3_engine.save_to_file(text, wav_path)
            self.pyttsx3_engine.runAndWait()
            
            if os.path.exists(wav_path) and os.path.getsize(wav_path) > 100:
                # אם ביקשו mp3, נשאיר wav וגם נעתיק
                if output_path != wav_path:
                    import shutil
                    shutil.copy(wav_path, output_path.replace('.mp3', '_pyttsx3.wav'))
                return True
            return False
        except Exception as e:
            print(f"[TTS] pyttsx3 failed: {e}")
            return False

    def _synthesize_sapi(self, text: str) -> bool:
        """Fallback 2: Windows SAPI - מדבר ישירות, בלי קובץ"""
        if not HAS_WIN32:
            return False
        try:
            speaker = win32com.client.Dispatch("SAPI.SpVoice")
            speaker.Speak(text)
            return True
        except Exception as e:
            print(f"[TTS] SAPI failed: {e}")
            return False

    def _synthesize_pyttsx3_direct(self, text: str) -> bool:
        """pyttsx3 דיבור ישיר"""
        if not HAS_PYTTSX3 or not self.pyttsx3_engine:
            return False
        try:
            self.pyttsx3_engine.say(text)
            self.pyttsx3_engine.runAndWait()
            return True
        except Exception as e:
            print(f"[TTS] pyttsx3 direct failed: {e}")
            return False

    def _file_to_base64(self, path: str) -> Optional[str]:
        """ממיר קובץ קול ל-base64 ל-frontend"""
        try:
            if not os.path.exists(path):
                # נסה wav
                wav = path.replace('.mp3', '.wav')
                if os.path.exists(wav):
                    path = wav
                else:
                    wav2 = path.replace('.mp3', '_pyttsx3.wav')
                    if os.path.exists(wav2):
                        path = wav2
                    else:
                        return None
            
            with open(path, 'rb') as f:
                data = f.read()
                if len(data) < 100:
                    return None
                b64 = base64.b64encode(data).decode('utf-8')
                # זהה סוג
                if path.endswith('.mp3'):
                    return f"data:audio/mpeg;base64,{b64}"
                else:
                    return f"data:audio/wav;base64,{b64}"
        except Exception as e:
            print(f"[TTS] base64 failed: {e}")
            return None

    async def synthesize(self, text: str, play=True) -> Optional[Tuple[str, Optional[str]]]:
        """
        מסנתז טקסט - עם קול מקורי חדש + fallback chain אוטומטי
        מחזיר (path, base64) - base64 ל-frontend playback
        """
        if not text or not text.strip():
            return None

        clean_text = self._prepare_text_for_tts(text)
        if not clean_text:
            return None

        file_id = str(uuid.uuid4())[:8]
        output_path = os.path.join(self.temp_dir, f"adiel_{file_id}.mp3")
        
        print(f"[TTS] Synthesizing: '{clean_text[:50]}...'")

        # נסיון 0: קול מקורי חדש שיצרנו במיוחד - אם יש התאמה למשפט נפוץ
        try:
            from .custom_voice import get_custom_voice_for_text
            custom_path = get_custom_voice_for_text(clean_text)
            if custom_path and custom_path.exists():
                print(f"[TTS] ✓ Using ORIGINAL custom voice: {custom_path.name} for '{clean_text[:30]}'")
                b64 = self._file_to_base64(str(custom_path))
                if play and HAS_PYGAME:
                    self._play_audio(str(custom_path))
                # גם frontend ינגן את ה-base64
                return str(custom_path), b64
        except Exception as e:
            print(f"[TTS] Custom voice check failed: {e}")
        
        # נסיון 1: edge-tts (הכי טוב לעברית; מדולג במצב אופליין)
        if HAS_EDGE and not OFFLINE_MODE:
            success = await self._synthesize_edge(clean_text, output_path)
            if success:
                print(f"[TTS] ✓ edge-tts success -> {output_path}")
                b64 = self._file_to_base64(output_path)
                if play and HAS_PYGAME:
                    self._play_audio(output_path)
                return output_path, b64
            else:
                print(f"[TTS] edge-tts failed, trying fallback...")
        
        # נסיון 2: pyttsx3 עם קובץ + ניגון
        if HAS_PYTTSX3:
            wav_path = output_path.replace('.mp3', '.wav')
            success = self._synthesize_pyttsx3(clean_text, wav_path)
            if success:
                print(f"[TTS] ✓ pyttsx3 file success")
                b64 = self._file_to_base64(wav_path)
                if play:
                    # pyttsx3 כבר ניגן בקובץ? ננגן שוב ישירות
                    self._synthesize_pyttsx3_direct(clean_text)
                return wav_path, b64
        
        # נסיון 3: Windows SAPI ישיר (בלי קובץ, רק ניגון)
        if HAS_WIN32 and play:
            success = self._synthesize_sapi(clean_text)
            if success:
                print(f"[TTS] ✓ SAPI direct success")
                return None, None  # אין קובץ, אבל ניגן
        
        # נסיון 4: pyttsx3 ישיר (אחרון)
        if HAS_PYTTSX3 and play:
            success = self._synthesize_pyttsx3_direct(clean_text)
            if success:
                print(f"[TTS] ✓ pyttsx3 direct success")
                return None, None
        
        print(f"[TTS] ✗ All TTS methods failed for: {clean_text[:30]}")
        print(f"[TTS] בדוק: pip install edge-tts pyttsx3 pygame-ce")
        print(f"[TTS] ובדוק רמקולים + ווליום Windows")
        return None

    def _prepare_text_for_tts(self, text: str) -> str:
        """הכנת טקסט ל-TTS"""
        import re
        text = re.sub(r'```.*?```', ' ', text, flags=re.DOTALL)
        text = re.sub(r'`[^`]+`', ' ', text)
        text = re.sub(r'[#*_\[\]]', ' ', text)
        
        slang_fixes = {
            "יאללה": "יאללה",
            "סגור": "סגור",
            "בוס": "בוס,",
            "...": ". ",
        }
        for k, v in slang_fixes.items():
            text = text.replace(k, v)
        
        if len(text) > 800:
            text = text[:800] + "."
            
        text = text.replace("?", "? ")
        text = text.replace("!", "! ")
        text = text.replace(",", ", ")
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def _play_audio(self, playback_path: str):
        """ניגון לא חוסם עם pygame - תוקן באג UnboundLocal"""
        def play_thread():
            try:
                current_path = playback_path
                if not os.path.exists(current_path):
                    alt = current_path.replace('.mp3', '.wav')
                    if os.path.exists(alt):
                        current_path = alt
                    else:
                        alt2 = current_path.replace('.mp3', '_pyttsx3.wav')
                        if os.path.exists(alt2):
                            current_path = alt2
                        else:
                            print(f"[TTS] File not found: {playback_path}")
                            return
                
                if not HAS_PYGAME:
                    return
                
                pygame.mixer.music.load(current_path)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    pygame.time.wait(100)
                try:
                    # נקה רק קבצי temp adiel_, לא קולות מקוריים
                    if "adiel_" in current_path and os.path.exists(current_path):
                        os.remove(current_path)
                        w = current_path.replace('.mp3', '.wav')
                        if os.path.exists(w) and "adiel_" in w:
                            os.remove(w)
                except:
                    pass
            except Exception as e:
                print(f"[TTS] Playback failed: {e}")
                import traceback; traceback.print_exc()

        t = threading.Thread(target=play_thread, daemon=True)
        t.start()

    def synthesize_sync(self, text: str, play=True):
        """גרסה סינכרונית"""
        try:
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as pool:
                        future = pool.submit(asyncio.run, self.synthesize(text, play))
                        return future.result(timeout=20)
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

# Singleton
_global_tts = None

def get_tts_engine():
    global _global_tts
    if _global_tts is None:
        _global_tts = HebrewTTS(voice="avigail")
    return _global_tts

async def speak_hebrew(text: str, play=True):
    engine = get_tts_engine()
    result = await engine.synthesize(text, play=play)
    if result:
        if isinstance(result, tuple):
            return result[0]
        return result
    return None

# Test
if __name__ == "__main__":
    async def test():
        tts = HebrewTTS()
        print("Testing TTS chain...")
        result = await tts.synthesize("שלום בוס! אני אדיאל ג'וניור, בדיקה. אם אתה שומע אותי, הרמקולים עובדים!", play=True)
        print(f"Result: {result}")
        await asyncio.sleep(6)

    asyncio.run(test())
