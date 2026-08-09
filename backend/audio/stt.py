"""
Hebrew STT Engine - Fast v2.1 Fixed
זיהוי דיבור עברי מהיר עם faster-whisper - תיקון באג os + שיפור הקלטה
"""
import os
import time
import tempfile
import threading
from pathlib import Path
from typing import Optional, Callable
import numpy as np

try:
    import sounddevice as sd
    import soundfile as sf
    HAS_SD = True
except:
    HAS_SD = False
    print("[STT] sounddevice not available")

try:
    from faster_whisper import WhisperModel
    HAS_FW = True
except:
    HAS_FW = False
    print("[STT] faster-whisper not available")

try:
    from vosk import Model, KaldiRecognizer
    import json
    HAS_VOSK = True
except:
    HAS_VOSK = False


class HebrewSTT:
    """
    מנוע STT עברי - הקלטה + תמלול - גרסה מתוקנת
    עכשיו עם בחירת מיקרופון
    """
    def __init__(self, model_size="small", language="he", device_id=None):
        self.model_size = model_size
        self.language = language
        self.sample_rate = 16000
        self.device_id = device_id
        
        # נסה לטעון device manager
        if device_id is None:
            try:
                from .device_manager import get_device_manager
                dm = get_device_manager()
                self.device_id = dm.get_selected_input()
                if self.device_id is not None:
                    print(f"[STT] Using selected mic: {self.device_id}")
            except:
                pass
        
        # נתיב מודלים - משתמש ב-Path כדי לא להיתקל בבאג os
        self.data_dir = Path(__file__).parent.parent / "data" / "whisper_models"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # טען faster-whisper
        self.model = None
        if HAS_FW:
            try:
                print(f"[STT] Loading Whisper model {model_size} from {self.data_dir}...")
                # Fix: אל תשתמש ב-os בתוך פונקציה עם import os מקומי
                self.model = WhisperModel(
                    model_size, 
                    device="cpu", 
                    compute_type="int8",
                    download_root=str(self.data_dir)
                )
                print(f"[STT] ✓ Whisper {model_size} loaded")
            except Exception as e:
                print(f"[STT] Whisper load failed: {e}")
                import traceback
                traceback.print_exc()
                self.model = None
                # נסה tiny כ-fallback אם small נכשל (tiny יותר קל)
                if model_size != "tiny":
                    try:
                        print(f"[STT] Trying fallback tiny model...")
                        self.model = WhisperModel(
                            "tiny",
                            device="cpu",
                            compute_type="int8",
                            download_root=str(self.data_dir)
                        )
                        print(f"[STT] ✓ Whisper tiny fallback loaded")
                    except Exception as e2:
                        print(f"[STT] Tiny fallback also failed: {e2}")
        
        # Fallback Vosk - בלי import os מקומי שגורם לבאג!
        self.vosk_model = None
        if HAS_VOSK and not self.model:
            try:
                vosk_path = Path(__file__).parent.parent / "data" / "vosk-model-small-he-0.22"
                if vosk_path.exists():
                    self.vosk_model = Model(str(vosk_path))
                    print("[STT] Vosk Hebrew loaded as primary")
                else:
                    print(f"[STT] Vosk model not found at {vosk_path}, skipping")
            except Exception as e:
                print(f"[STT] Vosk Hebrew failed: {e}")

        self.is_recording = False
        self.audio_data = []

    def _record_audio(self, max_seconds=8, silence_threshold=0.005, silence_duration=2.0) -> Optional[np.ndarray]:
        """
        הקלטה משופרת - פחות רגישה לשקט
        - silence_threshold נמוך יותר (0.005 במקום 0.01) - קולט גם כשקט
        - silence_duration ארוך יותר (2 שניות במקום 1.2) - לא עוצר מהר
        - נותן 8 שניות מלאות לדבר
        """
        if not HAS_SD:
            print("[STT] No audio device, cannot record")
            return None

        print(f"[STT] 🎤 Recording... (max {max_seconds}s, דבר עכשיו! אל תשתוק)")
        self.audio_data = []
        self.is_recording = True
        
        silence_counter = 0
        # כמה frames של שקט צריך כדי לעצור - עכשיו 2 שניות
        silence_frames_needed = int(self.sample_rate * silence_duration / 1024)
        
        def callback(indata, frames, time_info, status):
            if not self.is_recording:
                return
            if status:
                print(f"[STT] Audio status: {status}")
            self.audio_data.append(indata.copy())
            
        try:
            # השתמש במיקרופון הנבחר אם יש
            stream_kwargs = {
                "samplerate": self.sample_rate,
                "channels": 1,
                "dtype": 'float32',
                "blocksize": 1024,
                "callback": callback
            }
            if self.device_id is not None:
                stream_kwargs["device"] = self.device_id
            
            with sd.InputStream(**stream_kwargs):
                start = time.time()
                has_speech = False
                
                while self.is_recording:
                    elapsed = time.time() - start
                    if elapsed > max_seconds:
                        print(f"[STT] Max time {max_seconds}s reached")
                        break
                    
                    if len(self.audio_data) > 5:
                        # בדוק אנרגיה ב-0.5 שניה אחרונה
                        recent = np.concatenate(self.audio_data[-5:])
                        energy = np.sqrt(np.mean(recent**2))
                        
                        # דיבאג - הדפס אנרגיה כל שניה
                        if int(elapsed * 10) % 10 == 0:
                            print(f"[STT] Energy: {energy:.4f} (threshold {silence_threshold}) - has_speech:{has_speech}")
                        
                        if energy > silence_threshold:
                            has_speech = True
                            silence_counter = 0
                        else:
                            if has_speech:
                                # רק אם כבר היה דיבור, התחל לספור שקט
                                silence_counter += 1
                                if silence_counter > silence_frames_needed:
                                    print(f"[STT] Silence {silence_duration}s after speech, stopping")
                                    break
                            # אם עוד לא היה דיבור בכלל, אל תעצור - תן זמן להתחיל לדבר
                            elif elapsed > 2.0 and not has_speech:
                                # אם אחרי 2 שניות עדיין אין דיבור, המשך לחכות
                                pass
                    
                    time.sleep(0.05)
                    
        except Exception as e:
            print(f"[STT] Recording error: {e}")
            import traceback; traceback.print_exc()
            return None
        finally:
            self.is_recording = False

        if not self.audio_data:
            print("[STT] No audio data recorded")
            return None
            
        audio = np.concatenate(self.audio_data, axis=0).flatten()
        duration = len(audio)/self.sample_rate
        print(f"[STT] ✓ Recorded {duration:.2f}s")
        
        # בדוק אם זה שקט מוחלט
        max_amp = np.max(np.abs(audio))
        if max_amp < 0.005:
            print(f"[STT] ⚠ Recording very quiet (max {max_amp:.4f}) - mic maybe muted?")
            print(f"[STT] בדוק: 1. מיקרופון לא על Mute 2. עוצמת קלט ב-Windows 3. דבר חזק יותר")
        
        return audio

    def transcribe_audio(self, audio: np.ndarray) -> str:
        """תמלול אודיו קיים"""
        if audio is None or len(audio) < 1000:
            print("[STT] Audio too short or None")
            return ""

        # נסה Whisper
        if self.model:
            try:
                print(f"[STT] Transcribing {len(audio)/self.sample_rate:.1f}s with Whisper...")
                segments, info = self.model.transcribe(
                    audio,
                    language="he" if self.language == "he" else self.language,
                    beam_size=3,  # קטן יותר = מהיר יותר
                    best_of=3,
                    vad_filter=True,
                    vad_parameters=dict(min_silence_duration_ms=500)
                )
                text = " ".join([seg.text for seg in segments]).strip()
                print(f"[STT] ✓ Whisper: '{text}' (lang={info.language} prob={info.language_probability:.2f})")
                if text:
                    return self._clean_hebrew_text(text)
                else:
                    print("[STT] Whisper returned empty, audio maybe too quiet or not Hebrew?")
            except Exception as e:
                print(f"[STT] Whisper transcribe failed: {e}")
                import traceback; traceback.print_exc()

        # Fallback Vosk
        if self.vosk_model:
            try:
                print("[STT] Trying Vosk...")
                rec = KaldiRecognizer(self.vosk_model, self.sample_rate)
                audio_i16 = (audio * 32767).astype(np.int16).tobytes()
                rec.AcceptWaveform(audio_i16)
                result = json.loads(rec.FinalResult())
                text = result.get("text", "")
                print(f"[STT] Vosk: '{text}'")
                return self._clean_hebrew_text(text)
            except Exception as e:
                print(f"[STT] Vosk transcribe failed: {e}")

        print("[STT] No transcription returned")
        return ""

    def listen_and_transcribe(self, max_seconds=8) -> str:
        """הקלטה + תמלול בבת אחת"""
        print(f"[STT] Starting listen_and_transcribe max {max_seconds}s")
        audio = self._record_audio(max_seconds=max_seconds)
        if audio is None:
            print("[STT] Recording returned None")
            return ""
        
        # אם ההקלטה קצרה מדי, אל תנסה לתמלל
        if len(audio) / self.sample_rate < 0.5:
            print("[STT] Recording too short (<0.5s), ignoring")
            return ""
            
        return self.transcribe_audio(audio)

    def _clean_hebrew_text(self, text: str) -> str:
        """ניקוי ותיקון טקסט עברי"""
        if not text:
            return ""
        
        text = text.strip()
        
        corrections = {
            "אדיאל גוניור": "אדיאל ג'וניור",
            "אדיאל ג'וניור": "",
            "עדיאל": "אדיאל",
            "גونیור": "ג'וניור",
        }
        
        for wrong, right in corrections.items():
            if wrong in text and right == "":
                text = text.replace(wrong, "").strip()
            elif wrong != "אדיאל ג'וניור":
                text = text.replace(wrong, right)

        for wake in ["אדיאל ג'וניור", "אדיאל גוניור", "עדיאל ג'וניור", "adiel junior"]:
            if text.lower().startswith(wake.lower()):
                text = text[len(wake):].strip(" ,")
                
        import re
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()

    def stop_recording(self):
        self.is_recording = False
