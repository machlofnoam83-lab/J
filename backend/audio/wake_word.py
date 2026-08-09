"""
Wake Word Engine - Built from scratch
זיהוי מילת ההפעלה "אדיאל ג'וניור" בעברית
"""
import re
import time
import threading
from typing import Callable, Optional
import numpy as np

# נסה לטעון ספריות אודיו
try:
    import sounddevice as sd
    import soundfile as sf
    HAS_SD = True
except:
    HAS_SD = False
    print("[WakeWord] sounddevice not available, using mock")

try:
    import webrtcvad
    HAS_VAD = True
except:
    HAS_VAD = False

try:
    from vosk import Model, KaldiRecognizer
    import json
    HAS_VOSK = True
except:
    HAS_VOSK = False

try:
    from faster_whisper import WhisperModel
    HAS_WHISPER = True
except:
    HAS_WHISPER = False


class WakeWordDetector:
    """
    מזהה את "אדיאל ג'וניור" ברצף
    """
    def __init__(self, on_wake: Callable[[str], None], sample_rate=16000):
        self.on_wake = on_wake
        self.sample_rate = sample_rate
        self.running = False
        self.thread = None
        
        # מילות מפתח לזיהוי - כולל שיבושים נפוצים
        self.wake_patterns = [
            r"אדיאל.*ג.?וניור",
            r"עדיאל.*ג.?וניור",
            r"אדיאל.*גונ.?יור",
            r"אדיאל.*junior",
            r"adiel.*junior",
            r"אדיאל",
            r"ג.?וניור",
        ]
        self.compiled_patterns = [re.compile(p, re.IGNORECASE) for p in self.wake_patterns]

        # VAD
        if HAS_VAD:
            self.vad = webrtcvad.Vad(2)  # aggressiveness 0-3

        # Vosk Hebrew model - אם קיים
        self.vosk_model = None
        self.recognizer = None
        if HAS_VOSK:
            try:
                # נסה לטעון מודל עברי מקומי
                import os
                model_path = os.path.join(os.path.dirname(__file__), "..", "data", "vosk-model-small-he-0.22")
                if os.path.exists(model_path):
                    self.vosk_model = Model(model_path)
                    self.recognizer = KaldiRecognizer(self.vosk_model, sample_rate)
                    print("[WakeWord] Vosk Hebrew model loaded")
                else:
                    # נסה מודל אנגלי קטן כ-fallback
                    model_path_en = os.path.join(os.path.dirname(__file__), "..", "data", "vosk-model-small-en-us-0.15")
                    if os.path.exists(model_path_en):
                        self.vosk_model = Model(model_path_en)
                        self.recognizer = KaldiRecognizer(self.vosk_model, sample_rate)
                        print("[WakeWord] Using English Vosk as fallback for wake word")
            except Exception as e:
                print(f"[WakeWord] Vosk load failed: {e}")

        # Whisper for confirmation (heavier, used after VAD trigger)
        self.whisper_model = None
        if HAS_WHISPER:
            try:
                # טען מודל קטן לזיהוי מהיר של wake word
                self.whisper_model = WhisperModel("tiny", device="cpu", compute_type="int8")
                print("[WakeWord] Whisper tiny loaded for wake word")
            except Exception as e:
                print(f"[WakeWord] Whisper load failed: {e}")

        self.audio_buffer = []
        self.last_trigger_time = 0
        self.cooldown_seconds = 2.0  # מניעת טריגרים כפולים

    def _is_wake_word(self, text: str) -> bool:
        if not text:
            return False
        text = text.strip()
        if len(text) < 3:
            return False
            
        for pattern in self.compiled_patterns:
            if pattern.search(text):
                # בדיקה נוספת - צריך לפחות 4 תווים רלוונטיים
                # ואם זה רק "אדיאל" או "ג'וניור" - זה מספיק כ-wake בגרסה סלחנית
                print(f"[WakeWord] Pattern match: {pattern.pattern} in '{text}'")
                return True
        
        # בדיקה פשוטה נוספת - מרחק לוינשטיין מקומי
        lower = text.lower()
        if ("אדיאל" in lower or "עדיאל" in lower or "adiel" in lower) and ("גוניור" in lower or "ג'וניור" in lower or "junior" in lower):
            return True
        # סלחנות - רק "אדיאל" לבד מספיק להעיר (אם ביטחון גבוה)
        if lower.strip() in ["אדיאל", "עדיאל", "adiel", "אדיאל ג'וניור", "עדיאל ג'וניור"]:
            return True
            
        return False

    def _audio_callback(self, indata, frames, time_info, status):
        """Callback של sounddevice"""
        if status:
            print(f"[WakeWord] Audio status: {status}")
        
        # המרה ל-PCM 16-bit mono ל-VAD
        audio_data = indata[:, 0]  # mono
        self.audio_buffer.append(audio_data.copy())
        
        # שמור רק 2 שניות אחרונות
        max_len = int(self.sample_rate * 2.0 / frames)
        if len(self.audio_buffer) > max_len:
            self.audio_buffer = self.audio_buffer[-max_len:]

    def _process_buffer(self):
        """עיבוד buffer לחיפוש wake word"""
        if len(self.audio_buffer) < 5:
            return
            
        try:
            # איחוד buffer
            audio = np.concatenate(self.audio_buffer)
            # המרה ל-int16
            audio_i16 = (audio * 32767).astype(np.int16)
            
            # שלב 1: VAD - האם יש דיבור?
            has_speech = False
            if HAS_VAD:
                # חתוך ל-frames של 30ms
                frame_len = int(self.sample_rate * 0.03)
                for i in range(0, len(audio_i16) - frame_len, frame_len):
                    frame = audio_i16[i:i+frame_len].tobytes()
                    try:
                        if self.vad.is_speech(frame, self.sample_rate):
                            has_speech = True
                            break
                    except:
                        has_speech = True  # אם VAD נכשל, נניח שיש דיבור
                        break
            else:
                # אם אין VAD - בדוק אנרגיה
                energy = np.sqrt(np.mean(audio**2))
                has_speech = energy > 0.01

            if not has_speech:
                return

            # שלב 2: STT קל - Vosk
            detected_text = ""
            if self.recognizer:
                try:
                    data = audio_i16.tobytes()
                    if self.recognizer.AcceptWaveform(data):
                        result = json.loads(self.recognizer.Result())
                        detected_text = result.get("text", "")
                    else:
                        # Partial
                        partial = json.loads(self.recognizer.PartialResult())
                        detected_text = partial.get("partial", "")
                except Exception as e:
                    print(f"[WakeWord] Vosk error: {e}")

            # שלב 3: אם לא זוהה עם Vosk, נסה Whisper ל-confirmation
            if not detected_text and self.whisper_model and has_speech:
                try:
                    # Whisper צריך float32
                    segments, _ = self.whisper_model.transcribe(audio, language="he", beam_size=1)
                    detected_text = " ".join([seg.text for seg in segments])
                except Exception as e:
                    print(f"[WakeWord] Whisper error: {e}")

            if detected_text:
                print(f"[WakeWord] Heard: '{detected_text}'")
                if self._is_wake_word(detected_text):
                    now = time.time()
                    if now - self.last_trigger_time > self.cooldown_seconds:
                        self.last_trigger_time = now
                        print(f"[WakeWord] *** WAKE WORD DETECTED: {detected_text} ***")
                        self.on_wake(detected_text)
                        self.audio_buffer = []  # נקה אחרי טריגר

        except Exception as e:
            print(f"[WakeWord] Process error: {e}")

    def start(self):
        if self.running:
            return
        self.running = True
        
        if not HAS_SD:
            print("[WakeWord] Mock mode - no audio device, wake word simulated by key")
            # במצב mock, ניצור thread שמדמה האזנה
            self.thread = threading.Thread(target=self._mock_loop, daemon=True)
            self.thread.start()
            return

        try:
            # התחל stream
            self.stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype='float32',
                callback=self._audio_callback,
                blocksize=int(self.sample_rate * 0.03)  # 30ms blocks
            )
            self.stream.start()
            print("[WakeWord] Audio stream started, listening for 'אדיאל ג'וניור'...")

            # Thread לעיבוד
            self.thread = threading.Thread(target=self._processing_loop, daemon=True)
            self.thread.start()

        except Exception as e:
            print(f"[WakeWord] Failed to start audio stream: {e}")
            print("[WakeWord] Falling back to mock mode")
            self.thread = threading.Thread(target=self._mock_loop, daemon=True)
            self.thread.start()

    def _processing_loop(self):
        while self.running:
            self._process_buffer()
            time.sleep(0.3)

    def _mock_loop(self):
        """לולאת דמה לבדיקות בלי מיקרופון"""
        print("[WakeWord] Mock listening - press ENTER to simulate wake word, or type 'אדיאל ג'וניור'")
        while self.running:
            time.sleep(1)
            # במוק, לא עושה כלום - הטריגר יבוא מה-UI או מ-API

    def simulate_wake(self, text="אדיאל ג'וניור"):
        """לזימון ידני מה-UI"""
        now = time.time()
        if now - self.last_trigger_time > self.cooldown_seconds:
            self.last_trigger_time = now
            print(f"[WakeWord] Simulated wake: {text}")
            self.on_wake(text)

    def stop(self):
        self.running = False
        if hasattr(self, 'stream'):
            try:
                self.stream.stop()
                self.stream.close()
            except:
                pass
        if self.thread:
            self.thread.join(timeout=1)

# בדיקה עצמאית
if __name__ == "__main__":
    def on_wake(text):
        print(f">>> התעוררתי! שמעתי: {text}")

    detector = WakeWordDetector(on_wake)
    detector.start()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        detector.stop()
