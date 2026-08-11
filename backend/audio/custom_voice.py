"""
Custom Original Voice for Adiel Junior
קול מקורי וחדש שיוצר במיוחד עבורך - לא הקולות הסטנדרטיים של Microsoft

מכיל (🔓 הכול בלי מפתחות!):
1. Pre-recorded phrases עם קול מקורי (voice-00)
2. דרך לייצר עוד קולות - הקלטה עצמית / Edge-TTS חינמי עם pitch custom
3. Fallback אופליין עם pyttsx3 (ADIEL_OFFLINE=1)
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
    "how_are_you": ASSETS_DIR / "voice_how_are_you.mp3",
    "understand": ASSETS_DIR / "voice_i_understand.mp3",
    "what_screen": ASSETS_DIR / "voice_what_screen.mp3",
    "bye": ASSETS_DIR / "voice_bye.mp3",
    "fun": ASSETS_DIR / "voice_fun.mp3",
    "thinking": ASSETS_DIR / "voice_thinking.mp3",
    "proposal": ASSETS_DIR / "voice_proposal.mp3",
}

# מיפוי משפטים נפוצים לקולות מקוריים - מורחב
PHRASE_TO_VOICE = {
    # Wake
    "כן בוס?": "wake",
    "כן בוס? אני כאן.": "wake",
    "כן בוס? אני כאן, מקשיבה.": "wake",
    "אני כאן, בוס! מה צריך?": "wake",
    "כן בוס? אני ערה": "wake",
    
    # On it
    "על זה": "on_it",
    "על זה, בוס.": "on_it",
    "קלטתי, בוס.": "understand",
    "קלטתי": "understand",
    
    # Hello
    "שלום בוס!": "hello",
    "שלום בוס": "hello",
    "היי בוס": "hello",
    
    # How are you
    "אחלה, בוס! רצה על Full Power. מה איתך?": "how_are_you",
    "אחלה, בוס! רצה על פול פאוור": "how_are_you",
    
    # Screen
    "בוא נראה מה יש לך על המסך": "what_screen",
    
    # Bye
    "יאללה ביי בוס": "bye",
    "ביי בוס": "bye",
    
    # Fun
    "אתה אלוף": "fun",
    "על לא דבר": "fun",
    
    # Thinking
    "רגע, חושבת": "thinking",
    
    # Proposal
    "יש לי הצעה": "proposal",
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
    """מדריך איך ליצור עוד קולות מקוריים - 🔓 100% בלי מפתחות וענן"""
    return """
# איך ליצור עוד קולות מקוריים לאדיאל - הכול ביתי, בלי API!

## אופציה 1: הקלטה עצמית (הכי "אנחנו" 🎤)
1. פתח "מקליט קול" ב-Windows (Voice Recorder) או כל אפליקציית הקלטה
2. הקלט משפטים קצרים: "שלום בוס!", "כן בוס?", "על זה!", "יאללה ביי בוס"
3. שמור כ-MP3/WAV בתיקייה: frontend/src/assets/
4. תן שם לפי התבנית: voice_<שם>.mp3 (למשל voice_laugh.mp3)
5. הוסף מיפוי ב-ai_voice.py: {"שלום בוס": "laugh"}
זהו - הקול שלך (או של מישהי שאתה מכיר) הוא הקול של אדיאל!

## אופציה 2: Edge-TTS חינמי (בלי מפתח!) עם שינוי pitch/rate
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
לייצר קובץ: python -m backend.audio.tts (או דרך /speak endpoint)

## אופציה 3: אופליין מלא (ADIEL_OFFLINE=1)
pyttsx3 + קולות Windows המותקנים - עובד גם בלי אינטרנט בכלל.

## אופציה 4: השתמש בקולות שיצרתי לך
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
