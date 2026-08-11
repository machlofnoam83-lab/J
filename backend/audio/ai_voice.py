"""
AI Voice Generator - קול AI אמיתי שמייצר קול לכל שאלה
לא משנה איזו שאלה - תמיד עונה עם קול

🔓 100% ביתי - בלי API keys, בלי ענן! שרשרת קול מלאה:
1. קולות מקוריים שהקלטנו (voice_*.mp3) - הכי "אנחנו"
2. Edge-TTS עברית חינמית (בלי מפתח; מדולגת אם ADIEL_OFFLINE=1)
3. pyttsx3 לאופליין מלא - גם שומרת קובץ WAV ל-base64!
4. Windows SAPI fallback

תמיד מחזיר base64 כדי שה-frontend ינגן
"""
import os
import base64
import tempfile
from pathlib import Path
from typing import Optional, Tuple
import uuid

# מצב אופליין מלא - שום דבר לא יוצא לרשת (ADIEL_OFFLINE=1)
OFFLINE_MODE = os.getenv("ADIEL_OFFLINE", "").lower() in ("1", "true", "yes", "on")

# קולות מקוריים שיצרנו
ASSETS_DIR = Path(__file__).parent.parent.parent / "frontend" / "src" / "assets"

try:
    import edge_tts
    HAS_EDGE = True
except:
    HAS_EDGE = False

try:
    import pygame
    HAS_PYGAME = True
except:
    HAS_PYGAME = False

class AIVoiceGenerator:
    """
    מחולל קול AI - תמיד עונה עם קול, לא משנה השאלה
    """
    def __init__(self):
        self.temp_dir = Path(tempfile.gettempdir()) / "adiel_ai_voice"
        self.temp_dir.mkdir(exist_ok=True)
        
        # קולות מקוריים
        self.custom_voices = {
            "hello": ASSETS_DIR / "voice_hello.mp3",
            "wake": ASSETS_DIR / "voice_wake.mp3",
            "on_it": ASSETS_DIR / "voice_on_it.mp3",
            "how_are_you": ASSETS_DIR / "voice_how_are_you.mp3",
            "understand": ASSETS_DIR / "voice_i_understand.mp3",
            "what_screen": ASSETS_DIR / "voice_what_screen.mp3",
            "bye": ASSETS_DIR / "voice_bye.mp3",
            "fun": ASSETS_DIR / "voice_fun.mp3",
            "thinking": ASSETS_DIR / "voice_thinking.mp3",
            "proposal": ASSETS_DIR / "voice_proposal.mp3",
        }
        
        # Pygame
        if HAS_PYGAME:
            try:
                pygame.mixer.quit()
                pygame.mixer.init(frequency=24000, size=-16, channels=2, buffer=512)
            except:
                pass

        print(f"[AIVoice] Initialized - Custom:{len([p for p in self.custom_voices.values() if p.exists()])} Edge:{HAS_EDGE and not OFFLINE_MODE} OfflineMode:{OFFLINE_MODE} (🔓 בלי מפתחות!)")

    def _get_custom_voice_match(self, text: str) -> Optional[Path]:
        """בדוק אם יש קול מקורי מתאים למשפט"""
        text_lower = text.lower().strip()
        
        mapping = {
            "כן בוס": "wake",
            "אני כאן": "wake",
            "שלום בוס": "hello",
            "היי בוס": "hello",
            "על זה": "on_it",
            "קלטתי": "understand",
            "מה את רואה": "what_screen",
            "ביי בוס": "bye",
            "אחלה בוס": "how_are_you",
            "תודה": "fun",
        }
        
        for phrase, voice_key in mapping.items():
            if phrase in text_lower or text_lower in phrase:
                path = self.custom_voices.get(voice_key)
                if path and path.exists():
                    return path
        
        return None

    def _file_to_base64(self, path: Path) -> Optional[str]:
        try:
            if not path.exists():
                return None
            data = path.read_bytes()
            if len(data) < 100:
                return None
            b64 = base64.b64encode(data).decode()
            if path.suffix == ".mp3":
                return f"data:audio/mpeg;base64,{b64}"
            else:
                return f"data:audio/wav;base64,{b64}"
        except Exception as e:
            print(f"[AIVoice] base64 failed: {e}")
            return None

    async def generate_edge(self, text: str) -> Optional[Tuple[Path, str]]:
        """Edge-TTS עם קול מקורי - fallback"""
        if not HAS_EDGE:
            return None
        
        try:
            # נסה קולות עבריים עם סדר עדיפות - Hila הכי יציב לפי לוג שלך
            voices = ["he-IL-HilaNeural", "he-IL-AvigailNeural", "he-IL-AsafNeural"]
            
            for voice in voices:
                try:
                    file_path = self.temp_dir / f"edge_{uuid.uuid4().hex[:8]}.mp3"
                    communicate = edge_tts.Communicate(text, voice)
                    await communicate.save(str(file_path))
                    
                    if file_path.exists() and file_path.stat().st_size > 100:
                        b64 = self._file_to_base64(file_path)
                        print(f"[AIVoice] ✓ Edge-TTS {voice} success")
                        return file_path, b64
                except Exception as ve:
                    print(f"[AIVoice] Edge voice {voice} failed: {ve}")
                    continue
            
        except Exception as e:
            print(f"[AIVoice] Edge-TTS failed: {e}")
        
        return None

    async def generate_voice_for_any_question(self, text: str, play=True) -> Tuple[Optional[Path], Optional[str]]:
        """
        מייצר קול לכל שאלה - לא משנה מה שואלים!
        תמיד מחזיר קול
        """
        if not text or not text.strip():
            return None, None
        
        # ניקוי
        clean_text = text.strip()[:800]
        print(f"[AIVoice] Generating voice for ANY question: '{clean_text[:50]}...'")
        
        # 1. קול מקורי אם יש התאמה
        custom_path = self._get_custom_voice_match(clean_text)
        if custom_path:
            print(f"[AIVoice] Using custom original voice: {custom_path.name}")
            b64 = self._file_to_base64(custom_path)
            if play:
                self._play_file(custom_path)
            return custom_path, b64
        
        # 2. Edge-TTS חינמי - לכל שאלה (מדולג במצב אופליין)
        if not OFFLINE_MODE:
            edge_result = await self.generate_edge(clean_text)
            if edge_result:
                path, b64 = edge_result
                if play:
                    self._play_file(path)
                return path, b64

        # 3. אופליין מלא: pyttsx3 שומרת קובץ WAV → base64 (עובד בלי רשת כלל!)
        try:
            import pyttsx3
            engine = pyttsx3.init()
            wav_path = self.temp_dir / f"offline_{uuid.uuid4().hex[:8]}.wav"
            engine.save_to_file(clean_text, str(wav_path))
            engine.runAndWait()
            if wav_path.exists() and wav_path.stat().st_size > 100:
                b64 = self._file_to_base64(wav_path)
                print(f"[AIVoice] ✓ pyttsx3 OFFLINE wav success ({wav_path.stat().st_size} bytes)")
                if play:
                    self._play_file(wav_path)
                return wav_path, b64
        except Exception as e:
            print(f"[AIVoice] pyttsx3 offline wav failed: {e}")

        # 4. SAPI / pyttsx3 דיבור ישיר - בלי קובץ אבל שומעים
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.say(clean_text)
            engine.runAndWait()
            print("[AIVoice] ✓ pyttsx3 spoke")
            return None, None
        except:
            pass

        print("[AIVoice] ✗ All methods failed - no voice for question")
        return None, None

    def _play_file(self, path: Path):
        """ניגון"""
        try:
            if not HAS_PYGAME:
                return
            if not path.exists():
                return
            
            def play_thread():
                try:
                    import pygame
                    pygame.mixer.music.load(str(path))
                    pygame.mixer.music.play()
                    while pygame.mixer.music.get_busy():
                        pygame.time.wait(100)
                    # נקה רק temp files
                    if "adiel_" in str(path) or "edge_" in str(path) or "eleven_" in str(path):
                        try:
                            path.unlink(missing_ok=True)
                        except:
                            pass
                except Exception as e:
                    print(f"[AIVoice] Play failed: {e}")
            
            import threading
            threading.Thread(target=play_thread, daemon=True).start()
        except Exception as e:
            print(f"[AIVoice] Play file failed: {e}")

# Singleton
_global_ai_voice = None

def get_ai_voice() -> AIVoiceGenerator:
    global _global_ai_voice
    if _global_ai_voice is None:
        _global_ai_voice = AIVoiceGenerator()
    return _global_ai_voice

async def generate_voice_for_question(text: str, play=True):
    """פונקציית wrapper - מייצרת קול לכל שאלה"""
    generator = get_ai_voice()
    return await generator.generate_voice_for_any_question(text, play=play)
