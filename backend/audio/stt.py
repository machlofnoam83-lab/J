"""
Hebrew STT Engine - Fast
זיהוי דיבור עברי מהיר עם faster-whisper
"""
import os
import time
import tempfile
import threading
from typing import Optional, Callable
import numpy as np

try:
    import sounddevice as sd
    import soundfile as sf
    HAS_SD = True
except:
    HAS_SD = False

try:
    from faster_whisper import WhisperModel
    HAS_FW = True
except:
    HAS_FW = False

try:
    from vosk import Model, KaldiRecognizer
    import json
    HAS_VOSK = True
except:
    HAS_VOSK = False


class HebrewSTT:
    """
    מנוע STT עברי - הקלטה + תמלול
    """
    def __init__(self, model_size="small", language="he"):
        self.model_size = model_size
        self.language = language
        self.sample_rate = 16000
        
        # טען faster-whisper
        self.model = None
        if HAS_FW:
            try:
                # small is good balance for Hebrew, large-v3 best but heavy
                # נשתמש ב-small כברירת מחדל, large-v3 אם יש GPU
                print(f"[STT] Loading Whisper model {model_size}...")
                self.model = WhisperModel(model_size, device="cpu", compute_type="int8", download_root=os.path.join(os.path.dirname(__file__), "..", "data", "whisper_models"))
                print(f"[STT] Whisper {model_size} loaded")
            except Exception as e:
                print(f"[STT] Whisper load failed: {e}")
                self.model = None
        
        # Fallback Vosk
        self.vosk_model = None
        if HAS_VOSK and not self.model:
            try:
                import os
                model_path = os.path.join(os.path.dirname(__file__), "..", "data", "vosk-model-small-he-0.22")
                if os.path.exists(model_path):
                    self.vosk_model = Model(model_path)
                    print("[STT] Vosk Hebrew loaded as primary")
            except Exception as e:
                print(f"[STT] Vosk Hebrew failed: {e}")

        self.is_recording = False
        self.audio_data = []

    def _record_audio(self, max_seconds=10, silence_threshold=0.01, silence_duration=1.2) -> Optional[np.ndarray]:
        """
        הקלטה עם VAD פשוט - עוצר בשקט
        """
        if not HAS_SD:
            print("[STT] No audio device, cannot record")
            return None

        print(f"[STT] Recording... (max {max_seconds}s, speak now)")
        self.audio_data = []
        self.is_recording = True
        
        silence_counter = 0
        silence_frames_needed = int(self.sample_rate * silence_duration / 1024)
        
        def callback(indata, frames, time_info, status):
            if not self.is_recording:
                return
            self.audio_data.append(indata.copy())
            
        try:
            with sd.InputStream(samplerate=self.sample_rate, channels=1, dtype='float32', blocksize=1024, callback=callback):
                start = time.time()
                while self.is_recording:
                    if time.time() - start > max_seconds:
                        print("[STT] Max time reached")
                        break
                    
                    if len(self.audio_data) > 10:
                        # בדוק שקט ב-0.5 שניה אחרונה
                        recent = np.concatenate(self.audio_data[-10:])
                        energy = np.sqrt(np.mean(recent**2))
                        if energy < silence_threshold:
                            silence_counter += 1
                            if silence_counter > silence_frames_needed and (time.time() - start) > 1.0:
                                print("[STT] Silence detected, stopping")
                                break
                        else:
                            silence_counter = 0
                    
                    time.sleep(0.05)
                    
        except Exception as e:
            print(f"[STT] Recording error: {e}")
            return None
        finally:
            self.is_recording = False

        if not self.audio_data:
            return None
            
        audio = np.concatenate(self.audio_data, axis=0).flatten()
        print(f"[STT] Recorded {len(audio)/self.sample_rate:.2f}s")
        return audio

    def transcribe_audio(self, audio: np.ndarray) -> str:
        """תמלול אודיו קיים"""
        if audio is None or len(audio) < 1000:
            return ""

        # נסה Whisper
        if self.model:
            try:
                # Faster-whisper רוצה float32 mono
                segments, info = self.model.transcribe(
                    audio,
                    language="he" if self.language == "he" else self.language,
                    beam_size=5,
                    best_of=5,
                    vad_filter=True,
                    vad_parameters=dict(min_silence_duration_ms=300)
                )
                text = " ".join([seg.text for seg in segments]).strip()
                print(f"[STT] Whisper: '{text}' (lang={info.language} prob={info.language_probability:.2f})")
                return self._clean_hebrew_text(text)
            except Exception as e:
                print(f"[STT] Whisper transcribe failed: {e}")

        # Fallback Vosk
        if self.vosk_model:
            try:
                import io
                rec = KaldiRecognizer(self.vosk_model, self.sample_rate)
                # המרה ל-int16
                audio_i16 = (audio * 32767).astype(np.int16).tobytes()
                rec.AcceptWaveform(audio_i16)
                result = json.loads(rec.FinalResult())
                text = result.get("text", "")
                print(f"[STT] Vosk: '{text}'")
                return self._clean_hebrew_text(text)
            except Exception as e:
                print(f"[STT] Vosk transcribe failed: {e}")

        return ""

    def listen_and_transcribe(self, max_seconds=8) -> str:
        """הקלטה + תמלול בבת אחת"""
        audio = self._record_audio(max_seconds=max_seconds)
        if audio is None:
            return ""
        return self.transcribe_audio(audio)

    def _clean_hebrew_text(self, text: str) -> str:
        """ניקוי ותיקון טקסט עברי"""
        if not text:
            return ""
        
        # ניקוי בסיסי
        text = text.strip()
        
        # תיקוני שגיאות נפוצות ב-Whisper עברי
        corrections = {
            "אדיאל גוניור": "אדיאל ג'וניור",  # אחידות
            "אדיאל ג'וניור": "",  # הסר wake word מהפקודה עצמה
            "עדיאל": "אדיאל",
            "גونیור": "ג'וניור",
        }
        
        for wrong, right in corrections.items():
            if wrong in text and right == "":
                text = text.replace(wrong, "").strip()
            elif wrong != "אדיאל ג'וניור":  # אל תתקן את ה-wake אם הוא המילה עצמה
                text = text.replace(wrong, right)

        # הסר wake word מתחילת משפט אם נשאר
        for wake in ["אדיאל ג'וניור", "אדיאל גוניור", "עדיאל ג'וניור", "adiel junior"]:
            if text.lower().startswith(wake.lower()):
                text = text[len(wake):].strip(" ,")
                
        # תקן רווחים כפולים
        import re
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()

    def stop_recording(self):
        self.is_recording = False

# Test
if __name__ == "__main__":
    stt = HebrewSTT(model_size="small")
    print("דבר עכשיו בעברית...")
    text = stt.listen_and_transcribe()
    print(f"זיהיתי: {text}")
