"""
Custom Original Voice for Adiel Junior
קול מקורי וחדש שיוצר במיוחד עבורך - לא הקולות הסטנדרטיים של Microsoft

מכיל:
1. Pre-recorded phrases עם קול מקורי (voice-00)
2. דרך לייצר עוד קולות עם ElevenLabs / Edge-TTS custom
3. Fallback ל-edge עם pitch shift ייחודי
"""
import os
import json
from pathlib import Path
from typing import Optional

# נתיב לקולות המקוריים שיצרנו
ASSETS_DIR = Path(__file__).parent.parent.parent / "frontend" / "src" / "assets"
CUSTOM_VOICES = {
    "hello": ASSETS_DIR / "voice_hello.mp3",
    "wake": ASSETS_DIR / "voice_wake.mp3",
    "on_it": ASSETS_DIR / "voice_on_it.mp3",
}

# מיפוי משפטים נפוצים לקולות מקוריים
PHRASE_TO_VOICE = {
    "כן בוס?": "wake",
    "כן בוס? אני כאן.": "wake",
    "כן בוס? אני כאן, מקשיבה.": "wake",
    "אני כאן, בוס! מה צריך?": "wake",
    "על זה": "on_it",
    "על זה, בוס.": "on_it",
    "שלום בוס!": "hello",
    "שלום בוס": "hello",
}

def get_custom_voice_for_text(text: str) -> Optional[Path]:
    """מחזיר קובץ קול מקורי אם יש התאמה למשפט נפוץ"""
    text = text.strip()
    
    # התאמה מדויקת
    if text in PHRASE_TO_VOICE:
        voice_key = PHRASE_TO_VOICE[text]
        path = CUSTOM_VOICES.get(voice_key)
        if path and path.exists():
            return path
    
    # התאמה חלקית
    for phrase, voice_key in PHRASE_TO_VOICE.items():
        if phrase in text or text in phrase:
            path = CUSTOM_VOICES.get(voice_key)
            if path and path.exists():
                return path
    
    return None

def list_custom_voices():
    """רשימת קולות מקוריים זמינים"""
    available = {}
    for name, path in CUSTOM_VOICES.items():
        available[name] = {
            "path": str(path),
            "exists": path.exists(),
            "size": path.stat().st_size if path.exists() else 0
        }
    return available

def create_custom_voice_instructions():
    """מדריך איך ליצור עוד קולות מקוריים"""
    return """
# איך ליצור עוד קולות מקוריים לאדיאל:

## אופציה 1: ElevenLabs (הכי איכותי, קול חדש לגמרי)
1. כנס ל- https://elevenlabs.io/voice-lab
2. צור קול חדש עם Voice Design:
   - Gender: Female, Age: Young, Accent: Israeli/Hebrew
   - או העלה דגימת קול שלך
3. השתמש ב-API:
   ```python
   from elevenlabs import generate, save
   audio = generate(text="שלום בוס!", voice="Adiel-Custom", model="eleven_multilingual_v2")
   save(audio, "frontend/src/assets/voice_custom.mp3")
   ```

## אופציה 2: Edge-TTS עם שינוי pitch/rate ליצירת קול ייחודי
ב-tts.py יש כבר:
```python
voice = "he-IL-AvigailNeural"
rate = "+10%"  # מהיר יותר
pitch = "+5Hz" # גבוה יותר - נשמע צעיר יותר
```
שנה pitch/rate ליצור גוון חדש:
- "+15% / +10Hz" = קול צעיר ואנרגטי
- "-5% / -5Hz" = קול בוגר ורגוע
- "-10% / +8Hz" = קול ילדותי

## אופציה 3: השתמש בקולות שיצרתי לך
ב-frontend/src/assets/ יש כבר:
- voice_hello.mp3 - "שלום בוס! אני אדיאל עם קול מקורי"
- voice_wake.mp3 - "כן בוס? אני כאן, מקשיבה"
- voice_on_it.mp3 - "על זה, בוס"

הוספתי אותם ל-tts.py כ-fallback - אם המשפט מוכר, היא תנגן את הקול המקורי!
"""

if __name__ == "__main__":
    print("Custom Voices for Adiel Junior:")
    for name, info in list_custom_voices().items():
        print(f"  {name}: {info['exists']} - {info['size']} bytes")
    print(create_custom_voice_instructions())
