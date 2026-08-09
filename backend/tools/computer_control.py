"""
Computer Control - שליטה מלאה במחשב ובאפליקציות
פתיחת תוכנות, Spotify, YouTube, הקלדה, אוטומציה
"""
import os
import sys
import time
import webbrowser
import subprocess
from typing import Dict, List, Optional
from pathlib import Path
import json

class ComputerControl:
    """
    שליטה מלאה במחשב
    """
    def __init__(self):
        self.has_pyautogui = False
        self.has_spotipy = False
        
        try:
            import pyautogui
            self.has_pyautogui = True
        except:
            pass
        
        try:
            import spotipy
            self.has_spotipy = True
        except:
            pass

    # === אפליקציות ===

    def open_app(self, app_name: str) -> Dict:
        """פתיחת אפליקציה לפי שם"""
        app_name_lower = app_name.lower()
        
        apps = {
            "chrome": ["chrome", "google-chrome", "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe"],
            "firefox": ["firefox"],
            "code": ["code", "codium", "C:\\Users\\{user}\\AppData\\Local\\Programs\\Microsoft VS Code\\Code.exe"],
            "vscode": ["code", "codium"],
            "spotify": ["spotify", "C:\\Users\\{user}\\AppData\\Roaming\\Spotify\\Spotify.exe"],
            "notepad": ["notepad", "notepad.exe", "gedit"],
            "explorer": ["explorer", "nautilus", "dolphin"],
            "calc": ["calc", "gnome-calculator", "kcalc"],
            "word": ["winword", "libreoffice --writer"],
            "excel": ["excel", "libreoffice --calc"],
            "terminal": ["cmd", "powershell", "gnome-terminal", "konsole"],
            "discord": ["discord", "C:\\Users\\{user}\\AppData\\Local\\Discord\\Update.exe"],
            "youtube": ["chrome https://youtube.com", "firefox https://youtube.com"],
        }
        
        # חפש התאמה
        for key, commands in apps.items():
            if key in app_name_lower or app_name_lower in key:
                for cmd in commands:
                    try:
                        cmd_expanded = cmd.format(user=os.getenv("USERNAME", os.getenv("USER", "")))
                        if os.path.exists(cmd_expanded):
                            subprocess.Popen([cmd_expanded], shell=False)
                        else:
                            subprocess.Popen(cmd_expanded, shell=True)
                        return {"success": True, "app": key, "command": cmd, "message": f"פותחת {key}..."}
                    except FileNotFoundError:
                        continue
                    except Exception as e:
                        print(f"[ComputerControl] Open {key} failed: {e}")
        
        # נסה ישירות
        try:
            if sys.platform == "win32":
                os.startfile(app_name)
            else:
                subprocess.Popen([app_name], shell=False)
            return {"success": True, "app": app_name, "message": f"פותחת {app_name}"}
        except Exception as e:
            return {"success": False, "error": str(e), "message": f"לא הצלחתי לפתוח {app_name}: {e}"}

    def close_app(self, app_name: str) -> Dict:
        """סגירת אפליקציה"""
        try:
            import psutil
            closed = []
            for proc in psutil.process_iter(['name']):
                try:
                    if app_name.lower() in proc.info['name'].lower():
                        proc.terminate()
                        closed.append(proc.info['name'])
                except:
                    pass
            return {"success": True, "closed": closed, "message": f"סגרתי {len(closed)} תהליכים של {app_name}"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    # === מדיה ===

    def play_spotify(self, query: str = "", action="play") -> Dict:
        """שליטה בספוטיפיי"""
        print(f"[ComputerControl] 🎵 Spotify: {action} {query}")
        
        # נסה עם spotipy API אם יש
        if self.has_spotipy:
            try:
                # כאן היה קוד אמיתי עם Spotify API
                pass
            except:
                pass
        
        # Fallback - פתח ספוטיפיי + חפש
        try:
            if query:
                url = f"https://open.spotify.com/search/{query.replace(' ', '%20')}"
                webbrowser.open(url)
                return {"success": True, "action": "search", "query": query, "url": url, "message": f"מחפש ב-Spotify: {query}"}
            else:
                # פתח ספוטיפיי
                self.open_app("spotify")
                if self.has_pyautogui:
                    import pyautogui
                    time.sleep(1)
                    pyautogui.press('playpause' if action == "play" else 'nexttrack')
                return {"success": True, "action": action, "message": f"Spotify {action}"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def play_youtube(self, query: str) -> Dict:
        """ניגון יוטיוב"""
        print(f"[ComputerControl] ▶️ YouTube: {query}")
        try:
            url = f"https://www.youtube.com/results?search_query={query.replace(' ', '+')}"
            webbrowser.open(url)
            # במציאות היה לוחץ על הסרטון הראשון עם Playwright
            return {"success": True, "query": query, "url": url, "message": f"פותחת YouTube: {query}"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    # === הקלדה וכתיבה ===

    def type_text(self, text: str, interval=0.02) -> Dict:
        """הקלדה אוטומטית במקום שהסמן נמצא"""
        if not self.has_pyautogui:
            return {"success": False, "error": "pyautogui not installed"}
        
        try:
            import pyautogui
            print(f"[ComputerControl] ⌨️ מקליד {len(text)} תווים...")
            pyautogui.typewrite(text, interval=interval)
            return {"success": True, "chars": len(text), "message": f"הקלדתי {len(text)} תווים"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def dictate_document(self, spoken_text: str, format_type="document") -> Dict:
        """כתיבת מסמך ארוך מהקלטה קולית"""
        print(f"[ComputerControl] 📝 מכתיב מסמך: {len(spoken_text)} תווים")
        
        # נקה טקסט דיבור
        # הסר "אדיאל ג'וניור" וכו'
        cleaned = spoken_text.replace("אדיאל ג'וניור", "").strip()
        
        # הוסף פיסוק חכם
        # אם המשתמש אמר "נקודה" -> .
        cleaned = cleaned.replace(" נקודה", ".").replace(" פסיק", ",").replace(" סימן שאלה", "?").replace(" שורה חדשה", "\n")
        
        # שמור לקובץ
        docs_dir = Path.home() / "Documents" / "AdielDictations"
        docs_dir.mkdir(parents=True, exist_ok=True)
        
        filename = f"dictation_{int(time.time())}.txt"
        file_path = docs_dir / filename
        
        try:
            # עיצוב לפי סוג
            if format_type == "email":
                content = f"נושא: {cleaned[:50]}...\n\n{cleaned}\n\nנשלח ע\"י אדיאל ג'וניור"
            elif format_type == "report":
                content = f"# דוח - {time.strftime('%d/%m/%Y')}\n\n{cleaned}\n\n---\nנוצר ע\"י אדיאל"
            else:
                content = cleaned
            
            file_path.write_text(content, encoding='utf-8')
            
            return {
                "success": True,
                "file_path": str(file_path),
                "chars": len(cleaned),
                "format": format_type,
                "message": f"כתבתי מסמך {len(cleaned)} תווים ושמרתי ב-{file_path}"
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def rewrite_text(self, text: str, style="professional") -> Dict:
        """שכתוב טקסט"""
        print(f"[ComputerControl] ✏️ משכתב טקסט בסגנון {style}")
        
        # דמו - במציאות היה משתמש ב-LLM
        styles = {
            "professional": f"{text}\n\n(נוסח מקצועי - שוכתב ע\"י אדיאל)",
            "casual": f"יאללה, אז ככה: {text} - מה דעתך?",
            "formal": f"בהמשך לפנייתך, {text} בברכה,",
            "short": text[:100] + "..." if len(text) > 100 else text,
        }
        
        rewritten = styles.get(style, styles["professional"])
        
        return {
            "success": True,
            "original_length": len(text),
            "rewritten_length": len(rewritten),
            "style": style,
            "rewritten": rewritten,
            "message": f"שכתבתי בסגנון {style}"
        }

    # === תמונות מסך ושליטה ===

    def take_screenshot(self) -> Optional[str]:
        """צילום מסך מלא"""
        try:
            import mss
            from PIL import Image
            import tempfile
            
            with mss.mss() as sct:
                monitor = sct.monitors[1]
                img = sct.grab(monitor)
                pil_img = Image.frombytes("RGB", img.size, img.bgra, "raw", "BGRX")
                
                path = tempfile.mktemp(suffix=".png")
                pil_img.save(path)
                return path
        except Exception as e:
            print(f"[ComputerControl] Screenshot failed: {e}")
            return None

# Singleton
_global_control = None

def get_computer_control():
    global _global_control
    if _global_control is None:
        _global_control = ComputerControl()
    return _global_control
