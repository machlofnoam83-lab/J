"""
Screen Vision Engine - Live Screen Awareness
צילום מסך חי + ניתוח מקומי
"""
import os
import io
import base64
import time
from typing import Optional, Dict, Tuple
from datetime import datetime

try:
    import mss
    HAS_MSS = True
except:
    HAS_MSS = False
    print("[Vision] mss not available")

try:
    from PIL import Image
    HAS_PIL = True
except:
    HAS_PIL = False

try:
    import pytesseract
    HAS_TESSERACT = True
except:
    HAS_TESSERACT = False

try:
    import pygetwindow as gw
    HAS_WINDOW = True
except:
    HAS_WINDOW = False


class ScreenVision:
    def __init__(self, downscale_width=1280):
        self.downscale_width = downscale_width
        self.last_capture = None
        self.last_capture_time = 0
        self.captures_dir = os.path.join(os.path.dirname(__file__), "..", "data", "screenshots")
        os.makedirs(self.captures_dir, exist_ok=True)

    def capture_screen(self, save=False, base64_encode=True) -> Optional[Dict]:
        """
        מצלם מסך ומחזיר מידע
        """
        if not HAS_MSS or not HAS_PIL:
            # Mock capture for testing without display
            print("[Vision] Mock capture (no mss/PIL)")
            return {
                "image": None,
                "base64": None,
                "width": 1920,
                "height": 1080,
                "timestamp": datetime.now().isoformat(),
                "active_window": "Mock Window - Visual Studio Code",
                "text_preview": "Mock screen content - code editor with Python"
            }

        try:
            with mss.mss() as sct:
                # צלם מסך ראשי
                monitor = sct.monitors[1]  # Primary monitor
                raw = sct.grab(monitor)
                
                img = Image.frombytes("RGB", raw.size, raw.bgra, "raw", "BGRX")
                original_size = img.size
                
                # Downscale לחסכון
                if img.width > self.downscale_width:
                    ratio = self.downscale_width / img.width
                    new_height = int(img.height * ratio)
                    img = img.resize((self.downscale_width, new_height), Image.LANCZOS)

                self.last_capture = img
                self.last_capture_time = time.time()

                b64_str = None
                if base64_encode:
                    buffer = io.BytesIO()
                    img.save(buffer, format="JPEG", quality=80, optimize=True)
                    b64_str = base64.b64encode(buffer.getvalue()).decode('utf-8')

                filepath = None
                if save:
                    filepath = os.path.join(self.captures_dir, f"screen_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
                    img.save(filepath, "JPEG", quality=85)

                # חלון פעיל
                active_window_title = "Unknown"
                if HAS_WINDOW:
                    try:
                        win = gw.getActiveWindow()
                        if win:
                            active_window_title = win.title
                    except Exception as e:
                        # Linux fallback - try other method
                        pass

                return {
                    "image": img,
                    "base64": b64_str,
                    "width": original_size[0],
                    "height": original_size[1],
                    "downscaled_width": img.width,
                    "downscaled_height": img.height,
                    "timestamp": datetime.now().isoformat(),
                    "active_window": active_window_title,
                    "filepath": filepath
                }

        except Exception as e:
            print(f"[Vision] Capture failed: {e}")
            # ב-Linux ללא DISPLAY, זה ייכשל - החזר mock
            return {
                "image": None,
                "base64": None,
                "width": 1920,
                "height": 1080,
                "timestamp": datetime.now().isoformat(),
                "active_window": f"Capture failed: {e}",
                "text_preview": ""
            }

    def extract_text_ocr(self, image=None, lang="heb+eng") -> str:
        """
        OCR מקומי - חילוץ טקסט מהמסך
        """
        if not HAS_TESSERACT:
            return ""

        try:
            img = image or self.last_capture
            if img is None:
                # נסה לצלם
                result = self.capture_screen(base64_encode=False)
                img = result.get("image") if result else None
            
            if img is None:
                return ""

            # OCR
            # הגדרות לדיוק טוב יותר
            custom_config = r'--oem 3 --psm 6'
            text = pytesseract.image_to_string(img, lang=lang, config=custom_config)
            return text.strip()[:2000]  # הגבל אורך
        except Exception as e:
            print(f"[Vision] OCR failed: {e}")
            return ""

    def analyze_screen_context(self, include_ocr=True) -> str:
        """
        בנה תיאור הקשר מסך לשליחה למוח
        """
        capture = self.capture_screen(base64_encode=False)
        if not capture:
            return "לא הצלחתי לצלם מסך"

        active = capture.get("active_window", "לא ידוע")
        width = capture.get("width", 0)
        height = capture.get("height", 0)

        context = f"חלון פעיל: {active}\nרזולוציה: {width}x{height}\n"

        # אם זה VS Code / IDE, ציין
        lower_active = active.lower()
        if "code" in lower_active or "visual studio" in lower_active:
            context += "המשתמש עובד בקוד (IDE).\n"
        elif "chrome" in lower_active or "firefox" in lower_active or "edge" in lower_active or "browser" in lower_active:
            context += "דפדפן פתוח.\n"
        elif "word" in lower_active or "docs" in lower_active or "notion" in lower_active:
            context += "מסמך/עורך טקסט פתוח.\n"
        elif "game" in lower_active or "steam" in lower_active:
            context += "משחק פתוח.\n"

        if include_ocr:
            ocr_text = self.extract_text_ocr(capture.get("image"))
            if ocr_text:
                # נקה OCR וקח שורות משמעותיות
                lines = [l.strip() for l in ocr_text.split('\n') if len(l.strip()) > 3][:20]
                context += f"\nטקסט שזוהה על המסך:\n" + "\n".join(lines[:15])
                if len(ocr_text) > 500:
                    context += f"\n...ועוד {len(lines)} שורות"
            else:
                context += "\n(לא זוהה טקסט ברור ב-OCR, ייתכן שצריך Vision API מתקדם)"

        return context[:2000]  # הגבל

    def get_base64_for_api(self, max_size=1024) -> Optional[str]:
        """קבל base64 מוכן ל-API Vision"""
        capture = self.capture_screen(base64_encode=True)
        if not capture:
            return None
        
        # אם התמונה גדולה מדי, הקטן עוד
        img = capture.get("image")
        if img and img.width > max_size:
            ratio = max_size / img.width
            img = img.resize((max_size, int(img.height * ratio)), Image.LANCZOS)
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=75)
            return base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        return capture.get("base64")

# Singleton
_global_vision = None

def get_vision_engine():
    global _global_vision
    if _global_vision is None:
        _global_vision = ScreenVision()
    return _global_vision

# Test
if __name__ == "__main__":
    vis = ScreenVision()
    ctx = vis.analyze_screen_context()
    print("=== Screen Context ===")
    print(ctx)
