#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train AI - אימון המוח בשעה האחרונה
כמו שביקשת: בשעה האחרונה מאמן את ה-AI

מאמן את True AI Model + Learning Engine על כל השיחות הקיימות
"""
import sys
from pathlib import Path
import time

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "backend"))

print("="*70)
print("  🧠 Adiel Junior - AI Training - אימון המוח")
print("  כמו שביקשת: שעה אחרונה = אימון")
print("="*70)

try:
    from core.true_ai_model import get_true_ai
    from core.learning_engine import get_learning_engine
    from core.memory import AdielMemory
    
    print("\n[1/4] טוען מודל AI אמיתי...")
    true_ai = get_true_ai()
    print(f"  ✓ Vocab: {true_ai.tokenizer.vocab_size}, Markov: {len(true_ai.markov.chain)}")
    
    print("\n[2/4] טוען זיכרון ולמידה...")
    learning = get_learning_engine()
    memory = AdielMemory()
    
    print(f"  ✓ פרופיל: {learning.get_user_profile_summary()}")
    print(f"  ✓ שיחות בזיכרון: {len(memory.long_term.get('conversations', []))}")
    
    # אסוף כל השיחות
    all_texts = []
    
    # משיחות בזיכרון
    for conv in memory.long_term.get("conversations", [])[-50:]:
        all_texts.append(conv.get("user", ""))
        all_texts.append(conv.get("assistant", ""))
    
    # משפה מפרופיל
    for fact in learning.profile.profile.get("facts", []):
        all_texts.append(fact.get("value", ""))
    
    # הוסף דאטה אימון עברי בסיסי
    training_data = [
        "שלום בוס מה קורה",
        "קוראים לי נועם ואני עובד על פרויקט אדיאל",
        "תפתח לי את כרום",
        "מה אתה רואה במסך",
        "יש לי שגיאה בקוד",
        "תזכור שאני אוהב פיצה",
        "מה השעה עכשיו",
        "תודה אלופה",
        "שים את החלון בצד",
        "חזור לאמצע",
        "תעזור לי עם הקוד הזה",
        "איך אתה עוזר לי?",
        "תקנה לי אוזניות הכי זול",
        "תזמין טיסה מתל אביב ללונדון",
        "תחקור על בינה מלאכותית",
        "תמלא טופס ממשלתי",
        "תבדוק מיילים דחופים",
        "תמצא זמן לפגישה עם דני",
        "תארגן את הקבצים בהורדות",
        "תכין דוח מנהלים",
        "תפתח ספוטיפיי ותנגן מוזיקה",
        "תכתוב מסמך מהקלטה",
    ]
    
    all_texts.extend(training_data)
    
    print(f"\n[3/4] מאמן על {len(all_texts)} משפטים...")
    print(f"  זה ייקח כדקה...")
    
    start = time.time()
    
    # אמן Markov
    true_ai.markov.train(all_texts)
    
    # אמן Retrieval
    for i in range(0, len(all_texts)-1, 2):
        user = all_texts[i] if i < len(all_texts) else ""
        assistant = all_texts[i+1] if i+1 < len(all_texts) else "קלטתי בוס"
        if user and assistant:
            true_ai.retrieval.add_conversation(user, assistant)
            # גם טוקנייזר
            for word in user.split():
                if len(word) > 2:
                    true_ai.tokenizer.add_word(word)
    
    elapsed = time.time() - start
    
    print(f"\n  ✓ אימון הושלם ב-{elapsed:.1f} שניות!")
    print(f"  - Vocab חדש: {true_ai.tokenizer.vocab_size}")
    print(f"  - Markov states: {len(true_ai.markov.chain)}")
    print(f"  - Conversations in retrieval: {len(true_ai.retrieval.conversations)}")
    
    print("\n[4/4] בדיקת מודל אחרי אימון...")
    test_queries = ["היי", "מי את", "מה את יודעת לעשות", "תעזור לי"]
    for q in test_queries:
        response = true_ai.generate_response(q, context={"name": "בוס"})
        print(f"  Q: {q}")
        print(f"  A: {response}")
        print()
    
    print("="*70)
    print("  ✅ אימון הושלם! המוח מוכן לשיחה אמיתית")
    print("  עכשיו תריץ: scripts/run_with_autofix.bat")
    print("="*70)
    
except Exception as e:
    print(f"\n❌ שגיאה באימון: {e}")
    import traceback; traceback.print_exc()
    sys.exit(1)
