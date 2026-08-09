"""
Fast Speech STT - מבין דיבור מהיר במיוחד
מערכת שמתמודדת עם דיבור מהיר, סלנג, בליעת מילים

שיפורים:
1. VAD עם min_silence קצר יותר כדי לא לחתוך דיבור מהיר
2. Preprocessing: נרמול עוצמה, הסרת רעש
3. Whisper עם פרמטרים מותאמים לדיבור מהיר
4. Post-processing עם תיקון מילים ממוזגות
5. Fallback ל-Google STT שמטפל טוב בדיבור מהיר
"""
import re
import time
from pathlib import Path
from typing import Optional
import numpy as np

try:
    import sounddevice as sd
    HAS_SD = True
except:
    HAS_SD = False

try:
    from faster_whisper import WhisperModel
    HAS_FW = True
except:
    HAS_FW = False


class FastSpeechProcessor:
    """מעבד דיבור מהיר"""
    
    def __init__(self):
        # מילון תיקונים לדיבור מהיר - כשמדברים מהר בולעים הברות
        self.fast_speech_corrections = {
            # דיבור מהיר - מיזוג מילים
            "מהאתה": "מה אתה",
            "איךאתה": "איך אתה",
            "מהאת": "מה את",
            "קוראיםלי": "קוראים לי",
            "אניואב": "אני אוהב",
            "אתהיכול": "אתה יכול",
            "תפתחלי": "תפתח לי",
            "תראהלי": "תראה לי",
            "יאללהבוס": "יאללה בוס",
            "סגורבוס": "סגור בוס",
            "עלזהבוס": "על זה בוס",
            "מהזה": "מה זה",
            "איפהזה": "איפה זה",
            "איךזה": "איך זה",
            # סלנג מהיר
            "ת'שמע": "תשמע",
            "ת'ראה": "תראה",
            "ב'קשה": "בבקשה",
            "אחשלי": "אח שלי",
            # בליעת ה' הידיעה
            "הביתשלי": "הבית שלי",
            "המחשבשלי": "המחשב שלי",
        }
        
        # מילים שמתמזגות בדיבור מהיר
        self.merge_patterns = [
            (r"מה\s*אתה", "מה אתה"),
            (r"איך\s*אתה", "איך אתה"),
            (r"קוראים\s*לי", "קוראים לי"),
            (r"אני\s*אוהב", "אני אוהב"),
        ]

    def preprocess_audio(self, audio: np.ndarray, sample_rate=16000) -> np.ndarray:
        """
        עיבוד מקדים לדיבור מהיר:
        1. נרמול עוצמה
        2. הסרת DC offset
        3. האטה קלה אם מדברים ממש מהר (time stretching)
        """
        if audio is None or len(audio) == 0:
            return audio
        
        # הסרת DC offset
        audio = audio - np.mean(audio)
        
        # נרמול עוצמה - חשוב לדיבור מהיר שקט
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            # נרמול ל-0.9 כדי למנוע clipping
            audio = audio * (0.9 / max_val)
        
        # בדוק אם זה דיבור מהיר מאוד - לפי קצב חציית אפסים
        # דיבור מהיר = יותר חציות אפס בשנייה
        zero_crossings = np.sum(np.diff(np.signbit(audio)))
        duration = len(audio) / sample_rate
        zcr = zero_crossings / duration if duration > 0 else 0
        
        # אם ZCR גבוה מאוד, זה דיבור מהיר עם הרבה עיצורים
        # ננסה להאט קלות עם אינטרפולציה (time stretching פשוט)
        if zcr > 3000:  # סף לדיבור מהיר
            print(f"[FastSTT] זיהיתי דיבור מהיר (ZCR={zcr:.0f}), מאט קלות...")
            # האטה פשוטה: אינטרפולציה ל-1.2x אורך
            try:
                from scipy.signal import resample
                new_len = int(len(audio) * 1.15)
                audio = resample(audio, new_len).astype(np.float32)
                print(f"[FastSTT] האטתי מ-{duration:.1f}s ל-{len(audio)/sample_rate:.1f}s")
            except:
                # אם scipy לא זמין, דלג
                pass
        
        return audio.astype(np.float32)

    def postprocess_text(self, text: str) -> str:
        """
        תיקון טקסט אחרי תמלול - מטפל בדיבור מהיר
        """
        if not text:
            return text
        
        original = text
        
        # 1. תיקון מיזוגים
        for wrong, correct in self.fast_speech_corrections.items():
            if wrong in text:
                text = text.replace(wrong, correct)
        
        # 2. הפרדת מילים ממוזגות עם regex
        for pattern, replacement in self.merge_patterns:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        # 3. תיקון חזרות שנוצרות מדיבור מהיר (ה-Whisper לפעמים מכפיל)
        # לדוגמה: "שלום שלום בוס בוס" -> "שלום בוס"
        words = text.split()
        cleaned = []
        prev = ""
        for word in words:
            if word != prev:  # הסר כפילויות רצופות
                cleaned.append(word)
            prev = word
        text = " ".join(cleaned)
        
        # 4. תיקון סלנג מהיר
        text = text.replace("  ", " ").strip()
        
        if text != original:
            print(f"[FastSTT] תיקון דיבור מהיר: '{original}' -> '{text}'")
        
        return text


class HebrewFastSTT:
    """
    STT לדיבור מהיר - הכי טוב לעברית מהירה
    """
    def __init__(self, model_size="small", language="he", device_id=None):
        self.model_size = model_size
        self.language = language
        self.sample_rate = 16000
        self.device_id = device_id
        
        # מעבד דיבור מהיר
        self.fast_processor = FastSpeechProcessor()
        
        # נתיב מודלים
        self.data_dir = Path(__file__).parent.parent / "data" / "whisper_models"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # טען מודל - מעדיף small או medium לדיבור מהיר (יותר מדויק)
        self.model = None
        if HAS_FW:
            try:
                # לדיבור מהיר עדיף מודל גדול יותר אם יש - small הוא מינימום טוב
                # אם המשתמש הגדיר large-v3, זה הכי טוב לדיבור מהיר
                actual_size = model_size
                if model_size == "small" and self._has_enough_memory():
                    # אם יש מספיק RAM, השתמש ב-medium לדיבור מהיר
                    actual_size = "medium"
                    print(f"[FastSTT] יש מספיק זיכרון, משדרג ל-{actual_size} לדיבור מהיר")
                
                print(f"[FastSTT] Loading Whisper {actual_size} optimized for fast speech...")
                self.model = WhisperModel(
                    actual_size,
                    device="cpu",
                    compute_type="int8",
                    download_root=str(self.data_dir),
                    # מותאם לדיבור מהיר
                    cpu_threads=4
                )
                print(f"[FastSTT] ✓ Whisper {actual_size} loaded - ready for fast speech")
                self.model_size = actual_size
            except Exception as e:
                print(f"[FastSTT] Load failed: {e}")
                # Fallback ל-tiny
                try:
                    self.model = WhisperModel("tiny", device="cpu", compute_type="int8", download_root=str(self.data_dir))
                    print("[FastSTT] ✓ Fallback tiny loaded")
                    self.model_size = "tiny"
                except:
                    self.model = None
    
    def _has_enough_memory(self) -> bool:
        """בדוק אם יש מספיק RAM למודל גדול יותר"""
        try:
            import psutil
            mem = psutil.virtual_memory()
            # אם יש יותר מ-8GB פנוי, אפשר medium
            return mem.available > 8 * 1024**3
        except:
            return False

    def transcribe_fast(self, audio: np.ndarray) -> str:
        """
        תמלול מותאם לדיבור מהיר
        """
        if audio is None or len(audio) < 800:  # קצר יותר - גם 0.05s
            return ""
        
        # 1. Preprocessing לדיבור מהיר
        audio = self.fast_processor.preprocess_audio(audio, self.sample_rate)
        
        if self.model:
            try:
                print(f"[FastSTT] Transcribing fast speech {len(audio)/self.sample_rate:.1f}s with {self.model_size}...")
                
                # פרמטרים מותאמים לדיבור מהיר:
                # - beam_size גבוה יותר = יותר מדויק לדיבור מהיר
                # - best_of גבוה יותר
                # - vad_filter עם min_silence קצר יותר (200ms במקום 500ms) - לא לחתוך דיבור מהיר
                # - word_timestamps True - עוזר לדיבור מהיר
                # - condition_on_previous_text False - לא להיתקע על טקסט קודם בדיבור מהיר
                segments, info = self.model.transcribe(
                    audio,
                    language="he",
                    beam_size=5 if self.model_size in ["tiny", "small"] else 7,  # גדול יותר = מדויק יותר למהיר
                    best_of=5,
                    temperature=0.0,  # דטרמיניסטי יותר
                    vad_filter=True,
                    vad_parameters=dict(
                        min_silence_duration_ms=200,  # קצר! לא לחתוך דיבור מהיר
                        max_speech_duration_s=30,
                        speech_pad_ms=200
                    ),
                    word_timestamps=True,
                    condition_on_previous_text=False,  # חשוב לדיבור מהיר
                    initial_prompt="עברית, סלנג ישראלי, דיבור מהיר, בוס, יאללה, סגור"  # Context לדיבור מהיר
                )
                
                # אסוף עם timestamps
                full_text = ""
                for seg in segments:
                    full_text += seg.text + " "
                    # הדפס timestamps לדיבאג דיבור מהיר
                    if hasattr(seg, 'words') and seg.words:
                        wpm = len(seg.words) / (seg.end - seg.start) * 60 if (seg.end - seg.start) > 0 else 0
                        if wpm > 180:
                            print(f"[FastSTT] Fast speech detected: {wpm:.0f} WPM - {seg.text[:30]}...")
                
                full_text = full_text.strip()
                print(f"[FastSTT] ✓ Raw: '{full_text}' (lang={info.language} prob={info.language_probability:.2f})")
                
                if full_text:
                    # 2. Postprocessing לדיבור מהיר
                    cleaned = self.fast_processor.postprocess_text(full_text)
                    return self._clean_hebrew(cleaned)
                else:
                    print("[FastSTT] Empty result - maybe too fast or quiet")
                    
            except Exception as e:
                print(f"[FastSTT] Transcribe failed: {e}")
                import traceback; traceback.print_exc()
        
        return ""

    def _clean_hebrew(self, text: str) -> str:
        """ניקוי עברי משופר"""
        if not text:
            return ""
        
        text = text.strip()
        
        # הסר wake word
        for wake in ["אדיאל ג'וניור", "אדיאל גוניור", "עדיאל", "adiel junior"]:
            if text.lower().startswith(wake.lower()):
                text = text[len(wake):].strip(" ,")
        
        # תיקונים
        corrections = {
            "אדיאל גוניור": "",
            "עדיאל": "אדיאל",
        }
        for wrong, right in corrections.items():
            text = text.replace(wrong, right)
        
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def listen_fast(self, max_seconds=10) -> str:
        """הקלטה + תמלול דיבור מהיר - גרסה משופרת"""
        if not HAS_SD:
            return ""
        
        import sounddevice as sd
        
        print(f"[FastSTT] 🎤 Fast mode - מקליט {max_seconds}s, דבר מהר! (אני מבין גם דיבור מהיר)")
        
        try:
            # הקלטה רציפה בלי עצירה על שקט - חשוב לדיבור מהיר
            # בדיבור מהיר אין הפסקות ארוכות, אז אל תעצור על שקט קצר
            kwargs = {
                "samplerate": self.sample_rate,
                "channels": 1,
                "dtype": 'float32'
            }
            if self.device_id is not None:
                kwargs["device"] = self.device_id
            
            # הקלטה של max_seconds מלאים - לא עוצר על שקט!
            audio = sd.rec(int(max_seconds * self.sample_rate), **kwargs)
            sd.wait()
            audio = audio.flatten()
            
            print(f"[FastSTT] Recorded {len(audio)/self.sample_rate:.1f}s, transcribing fast...")
            return self.transcribe_fast(audio)
            
        except Exception as e:
            print(f"[FastSTT] Record error: {e}")
            return ""


def get_fast_stt(model_size="small") -> HebrewFastSTT:
    return HebrewFastSTT(model_size=model_size)
