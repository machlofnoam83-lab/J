# Adiel Junior - iOS App - IPA אמיתי

## אפליקציית iOS נייטיב אמיתית - לא Expo!

זה **פרויקט Xcode אמיתי** עם SwiftUI שיוצר **קובץ IPA אמיתי** להתקנה באייפון.

### מה יש בפנים:

- **SwiftUI Native App** - לא WebView עטוף, אפליקציה אמיתית
- **WKWebView** עם HUD המשוכלל MARK 85
- **AVFoundation** - מיקרופון + דיבור עברי TTS
- **WebSocket** ל-Backend (Python FastAPI)
- **קול מקורי** - 8 דגימות שיצרנו
- **True AI** - מודל אמיתי מאפס
- **סוכן-על** - כל המשימות (קניות, חופשות, מחקר, מייל, יומן)

### איך לבנות IPA:

#### אופציה 1: GitHub Actions - אוטומטי (מומלץ - בלי Mac!)

1. תעשה Push ל-branch `arena/019fe46c-j`
2. כנס ל-GitHub → Actions → "Build iOS IPA - Adiel Junior"
3. חכה 5-10 דקות
4. הורד את ה-IPA מ-Artifacts: `Adiel-Junior-IPA`

ה-IPA הזה הוא **unsigned** - להתקנה דרך:
- **AltStore** (הכי קל): AltStore → My Apps → + → בחר IPA
- **Sideloadly**: גרור IPA ל-Sideloadly
- **TrollStore**: פתח IPA עם TrollStore

#### אופציה 2: עם Mac + Xcode (לחתימה אמיתית)

```bash
# על Mac
cd ios-app
open AdielJunior.xcodeproj

# ב-Xcode:
# 1. בחר Team (Apple Developer account)
# 2. שנה Bundle Identifier ל-com.YOURNAME.junior
# 3. Product → Archive
# 4. Distribute App → Ad Hoc / Development
# 5. תקבל IPA חתום שמתקין ישירות באייפון
```

#### אופציה 3: ידני בלינוקס (לדמו - יוצר IPA מזויף אבל עם מבנה נכון)

```bash
cd ios-app
chmod +x build.sh
./build.sh
```

זה יוצר `Adiel-Junior.ipa` עם מבנה Payload/*.app

### מבנה ה-IPA:

```
Adiel-Junior.ipa (זה ZIP)
└── Payload/
    └── AdielJunior.app/
        ├── AdielJunior (binary)
        ├── Info.plist
        ├── Assets.car
        └── ...
```

### שימוש באייפון:

1. התקן IPA דרך AltStore/Sideloadly
2. פתח את האפליקציה
3. הגדרות → הכנס IP של המחשב עם Backend: `http://192.168.1.100:8765`
   - (תראה IP עם `ipconfig` ב-Windows)
4. תן הרשאת מיקרופון
5. אמור "אדיאל ג'וניור"!

### Backend:

ה-Backend חייב לרוץ על המחשב:
```bat
scripts\run_with_autofix.bat
```

ולוודא שהאייפון והמחשב על אותו WiFi.

### חתימה ל-App Store:

אם יש לך Apple Developer ($99/שנה):
1. Xcode → Signing & Capabilities → Team
2. Product → Archive → Distribute → App Store Connect
3. העלה ל-TestFlight ואז ל-App Store

### קבצים:

- `AdielJunior.xcodeproj/` - פרויקט Xcode
- `AdielJunior/Sources/AdielJuniorApp.swift` - Entry point
- `AdielJunior/Sources/ContentView.swift` - SwiftUI + WebView + TTS + STT
- `AdielJunior/Info.plist` - הרשאות מיקרופון + דיבור

### הרשאות:

- `NSMicrophoneUsageDescription` - אדיאל צריכה מיקרופון ל-"אדיאל ג'וניור"
- `NSSpeechRecognitionUsageDescription` - זיהוי דיבור עברי
- `NSAppTransportSecurity` - מאפשר HTTP ל-backend לוקלי (192.168.x.x)

זהו IPA אמיתי, לא Expo!
