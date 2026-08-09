#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Audio - בודק מיקרופון ורמקולים של אדיאל

מריץ בדיקות ומתקן אוטומטית אם אפשר
"""
import sys
import os
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "backend"))

def print_header(text):
    print("\n" + "="*60)
    print(f"  {text}")
    print("="*60)

def test_imports():
    print_header("בודק ספריות קול...")
    checks = {}
    
    try:
        import sounddevice
        print("  ✓ sounddevice - מיקרופון/רמקולים")
        checks["sounddevice"] = True
        # List devices
        try:
            devices = sounddevice.query_devices()
            print(f"    נמצאו {len(devices)} התקני קול:")
            for i, dev in enumerate(devices[:5]):
                print(f"      [{i}] {dev['name']} - in:{dev['max_input_channels']} out:{dev['max_output_channels']}")
        except Exception as e:
            print(f"    ⚠ לא מצליח לרשום התקנים: {e}")
    except ImportError as e:
        print(f"  ✗ sounddevice חסר: {e}")
        checks["sounddevice"] = False
    
    try:
        import soundfile
        print("  ✓ soundfile")
        checks["soundfile"] = True
    except ImportError:
        print("  ✗ soundfile חסר")
        checks["soundfile"] = False
    
    try:
        import edge_tts
        print("  ✓ edge-tts (קול עברי טבעי)")
        checks["edge_tts"] = True
    except ImportError:
        print("  ✗ edge-tts חסר")
        checks["edge_tts"] = False
    
    try:
        import pygame
        print(f"  ✓ pygame-ce {pygame.__version__} (ניגון קול)")
        checks["pygame"] = True
    except ImportError:
        print("  ✗ pygame-ce חסר")
        checks["pygame"] = False
    
    try:
        from faster_whisper import WhisperModel
        print("  ✓ faster-whisper (זיהוי דיבור עברי)")
        checks["whisper"] = True
    except ImportError:
        print("  ✗ faster-whisper חסר")
        checks["whisper"] = False
    
    try:
        import webrtcvad
        print("  ✓ webrtcvad-wheels (זיהוי דיבור)")
        checks["webrtcvad"] = True
    except ImportError:
        try:
            import webrtcvad_wheels
            print("  ✓ webrtcvad-wheels (alt import)")
            checks["webrtcvad"] = True
        except:
            print("  ✗ webrtcvad חסר")
            checks["webrtcvad"] = False
    
    # Windows specific
    if os.name == 'nt':
        try:
            import win32com.client
            print("  ✓ win32com (TTS חלופי Windows)")
            checks["win32com"] = True
        except:
            print("  ⚠ win32com לא מותקן (לא קריטי, יש fallback)")
            checks["win32com"] = False
        
        try:
            import pyttsx3
            print("  ✓ pyttsx3 (TTS אופליין)")
            checks["pyttsx3"] = True
        except:
            print("  ⚠ pyttsx3 לא מותקן - נתקין אוטומטית אם צריך")
            checks["pyttsx3"] = False
    
    return checks

def test_speakers():
    print_header("בודק רמקולים - אדיאל מנסה לדבר...")
    
    try:
        import asyncio
        sys.path.insert(0, str(ROOT / "backend"))
        
        # נסה edge-tts
        try:
            from audio.tts import HebrewTTS
            print("  מנסה edge-tts עם קול עברי...")
            
            async def test_edge():
                tts = HebrewTTS(voice="avigail")
                path = await tts.synthesize("שלום בוס! אני אדיאל ג'וניור, בדיקת רמקולים. אם אתה שומע אותי, הרמקולים עובדים.", play=True)
                return path
            
            import asyncio
            result = asyncio.run(test_edge())
            if result:
                print("  ✓ edge-tts ניגן! אם שמעת - הרמקולים תקינים")
                return True
            else:
                print("  ✗ edge-tts נכשל")
        except Exception as e:
            print(f"  ✗ edge-tts נכשל: {e}")
            import traceback; traceback.print_exc()
        
        # Fallback pyttsx3
        print("\n  מנסה pyttsx3 (אופליין)...")
        try:
            import pyttsx3
            engine = pyttsx3.init()
            # נסה למצוא קול עברי אם יש
            voices = engine.getProperty('voices')
            print(f"    נמצאו {len(voices)} קולות:")
            for v in voices[:3]:
                print(f"      {v.name} - {v.languages}")
            
            engine.setProperty('rate', 180)
            engine.say("שלום בוס, בדיקת רמקולים עם pyttsx3")
            engine.runAndWait()
            print("  ✓ pyttsx3 ניגן")
            return True
        except Exception as e:
            print(f"  ✗ pyttsx3 נכשל: {e}")
        
        # Fallback Windows SAPI
        if os.name == 'nt':
            print("\n  מנסה Windows SAPI...")
            try:
                import win32com.client
                speaker = win32com.client.Dispatch("SAPI.SpVoice")
                speaker.Speak("שלום בוס, בדיקת רמקולים")
                print("  ✓ SAPI ניגן")
                return True
            except Exception as e:
                print(f"  ✗ SAPI נכשל: {e}")
        
        print("  ✗ כל ניסיונות הרמקולים נכשלו")
        return False
        
    except Exception as e:
        print(f"  ✗ שגיאה כללית בבדיקת רמקולים: {e}")
        import traceback; traceback.print_exc()
        return False

def test_microphone():
    print_header("בודק מיקרופון...")
    
    try:
        import sounddevice as sd
        import numpy as np
        
        print("  התקני קלט זמינים:")
        devices = sd.query_devices()
        input_devices = [d for d in devices if d['max_input_channels'] > 0]
        
        if not input_devices:
            print("  ✗ אין מיקרופון מזוהה!")
            print("  פתרונות:")
            print("  1. חבר מיקרופון")
            print("  2. בדוק הרשאות: Windows Settings -> Privacy -> Microphone -> Allow apps")
            print("  3. אפשר להשתמש במקלדת בינתיים - ה-HUD עובד גם בלי מיקרופון")
            return False
        
        for dev in input_devices:
            print(f"    - {dev['name']} (ערוצים: {dev['max_input_channels']})")
        
        print("\n  מקליט 3 שניות... דבר עכשיו!")
        print("  (אם אתה רואה רמות קול, המיקרופון עובד)")
        
        duration = 3
        samplerate = 16000
        recording = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='float32')
        sd.wait()
        
        # בדוק אם יש קול
        max_vol = np.max(np.abs(recording))
        mean_vol = np.mean(np.abs(recording))
        
        print(f"    עוצמה מקסימלית: {max_vol:.4f}")
        print(f"    עוצמה ממוצעת: {mean_vol:.4f}")
        
        if max_vol < 0.01:
            print("  ⚠ שקט מוחלט - המיקרופון אולי מושתק או לא מחובר")
            print("  בדוק:")
            print("  - המיקרופון לא על Mute")
            print("  - עוצמת קלט ב-Windows Sound Settings")
            return False
        else:
            print("  ✓ זיהה קול! המיקרופון עובד")
            
            # נסה לתמלל עם whisper אם יש
            try:
                from audio.stt import HebrewSTT
                print("\n  מנסה לתמלל את ההקלטה...")
                stt = HebrewSTT(model_size="tiny")
                text = stt.transcribe_audio(recording.flatten())
                print(f"    זיהה: '{text}'")
                if text:
                    print("  ✓ STT עובד!")
                else:
                    print("  ⚠ STT לא זיהה טקסט, אבל המיקרופון הקליט")
            except Exception as e:
                print(f"  ⚠ STT לא זמין: {e}")
            
            return True
            
    except ImportError as e:
        print(f"  ✗ sounddevice לא מותקן: {e}")
        print("  הרץ: pip install sounddevice soundfile")
        return False
    except Exception as e:
        print(f"  ✗ שגיאה במיקרופון: {e}")
        import traceback; traceback.print_exc()
        
        if "PortAudio" in str(e):
            print("\n  פתרון: התקן PortAudio או חבר מיקרופון")
        elif "permission" in str(e).lower() or "access" in str(e).lower():
            print("\n  פתרון: תן הרשאת מיקרופון ב-Windows Settings -> Privacy -> Microphone")
        
        return False

def fix_common_issues():
    print_header("מתקן בעיות נפוצות אוטומטית...")
    
    # התקן חבילות חסרות
    missing = []
    try:
        import pygame
    except:
        missing.append("pygame-ce")
    
    try:
        import sounddevice
    except:
        missing.append("sounddevice")
        missing.append("soundfile")
    
    try:
        import edge_tts
    except:
        missing.append("edge-tts")
    
    if missing:
        print(f"  מתקין חסרים: {missing}")
        import subprocess
        cmd = [sys.executable, "-m", "pip", "install"] + missing
        result = subprocess.run(cmd)
        if result.returncode == 0:
            print("  ✓ תיקון הצליח")
        else:
            print("  ✗ תיקון נכשל")
    else:
        print("  ✓ כל הספריות מותקנות")

if __name__ == "__main__":
    print("Adiel Junior - Audio Test")
    print(f"Python: {sys.version}")
    print(f"Platform: {os.name} - {sys.platform}")
    
    checks = test_imports()
    
    if "--fix" in sys.argv:
        fix_common_issues()
    
    # בדיקת רמקולים
    if "--no-speaker" not in sys.argv:
        test_speakers()
    
    # בדיקת מיקרופון
    if "--no-mic" not in sys.argv:
        test_microphone()
    
    print_header("סיכום")
    print(f"אם הרמקולים לא עובדים:")
    print(f"  1. הרץ: python scripts/test_audio.py --fix")
    print(f"  2. בדוק ווליום Windows")
    print(f"  3. נסה: pip install pyttsx3 pygame-ce edge-tts")
    print()
    print(f"אם המיקרופון לא עובד:")
    print(f"  1. חבר מיקרופון + תן הרשאות ב-Settings -> Privacy -> Microphone")
    print(f"  2. אפשר להשתמש במקלדת! ה-HUD עובד מצוין עם טקסט בעברית")
    print(f"  3. לחץ 🎤 אדיאל ג'וניור ב-HUD במקום לדבר")
    print()
    print(f"ה-HUD עובד גם בלי קול! פשוט תכתוב בעברית.")
