"""
System Tools - Plugin Architecture
כלים שהמוח יכול להפעיל - פתיחת אפליקציות, חיפוש, ווליום וכו'
"""
import os
import subprocess
import webbrowser
import sys
from typing import Dict, Any

try:
    import pyautogui
    HAS_PYAUTO = True
except:
    HAS_PYAUTO = False

try:
    import psutil
    HAS_PSUTIL = True
except:
    HAS_PSUTIL = False


class SystemTools:
    def __init__(self):
        self.app_map = {
            # Windows paths - will be adjusted
            "chrome": {
                "win": r"C:\Program Files\Google\Chrome\Application\chrome.exe",
                "cmd": "chrome",
                "alt": ["chrome", "google-chrome"]
            },
            "code": {
                "win": r"C:\Users\{user}\AppData\Local\Programs\Microsoft VS Code\Code.exe",
                "cmd": "code",
                "alt": ["code", "vscode", "codium"]
            },
            "notepad": {
                "win": "notepad.exe",
                "cmd": "notepad",
                "alt": ["notepad", "gedit", "nano"]
            },
            "calc": {
                "win": "calc.exe",
                "cmd": "gnome-calculator",
                "alt": ["calc", "kcalc"]
            },
            "explorer": {
                "win": "explorer.exe",
                "cmd": "nautilus",
                "alt": ["explorer", "nautilus", "dolphin"]
            },
            "spotify": {
                "win": r"C:\Users\{user}\AppData\Roaming\Spotify\Spotify.exe",
                "cmd": "spotify",
                "alt": ["spotify"]
            },
            "discord": {
                "win": r"C:\Users\{user}\AppData\Local\Discord\Update.exe",
                "cmd": "discord",
                "alt": ["discord"]
            },
        }

    def open_app(self, app_id: str, query: str = None) -> Dict[str, Any]:
        """פתיחת אפליקציה"""
        app_id = app_id.lower().strip()
        
        print(f"[SystemTools] Opening app: {app_id} query={query}")

        # אם query ולא app מזוהה, נסה לפתוח את ה-query ישירות
        if query and app_id == "unknown":
            app_id = query.lower().split()[0]

        # נסה לפי מפה
        if app_id in self.app_map:
            config = self.app_map[app_id]
            try:
                if sys.platform == "win32":
                    path = config["win"].format(user=os.getenv("USERNAME", ""))
                    if os.path.exists(path):
                        subprocess.Popen([path], shell=False)
                    else:
                        # נסה דרך PATH
                        subprocess.Popen([config["cmd"]], shell=True)
                else:
                    # Linux/Mac
                    for alt_cmd in config["alt"]:
                        try:
                            subprocess.Popen([alt_cmd], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                            break
                        except FileNotFoundError:
                            continue
                    else:
                        # נסה xdg-open fallback
                        subprocess.Popen([config["cmd"]], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, shell=False)

                return {"success": True, "app": app_id}
            except Exception as e:
                print(f"[SystemTools] Open failed: {e}")
                return {"success": False, "error": str(e)}

        # אם לא במפה - נסה לפתוח כפקודה ישירה
        else:
            try:
                if sys.platform == "win32":
                    subprocess.Popen(f'start "" "{app_id}"', shell=True)
                else:
                    subprocess.Popen([app_id], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return {"success": True, "app": app_id}
            except Exception as e:
                return {"success": False, "error": str(e)}

    def search_google(self, query: str) -> Dict[str, Any]:
        """חיפוש בגוגל"""
        try:
            url = f"https://www.google.com/search?q={query.replace(' ', '+')}"
            webbrowser.open(url)
            return {"success": True, "url": url}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def control_volume(self, action: str) -> Dict[str, Any]:
        """שליטה בווליום - Windows + Linux"""
        try:
            if not HAS_PYAUTO:
                return {"success": False, "error": "pyautogui not available"}

            if sys.platform == "win32":
                # Windows - שימוש בלחצני מדיה
                if action == "up":
                    pyautogui.press("volumeup", presses=3)
                elif action == "down":
                    pyautogui.press("volumedown", presses=3)
                elif action == "mute":
                    pyautogui.press("volumemute")
                elif action == "toggle":
                    pyautogui.press("volumemute")
            else:
                # Linux - amixer / pactl
                if action == "up":
                    subprocess.run(["amixer", "-D", "pulse", "sset", "Master", "5%+"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                elif action == "down":
                    subprocess.run(["amixer", "-D", "pulse", "sset", "Master", "5%-"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                elif action == "mute":
                    subprocess.run(["amixer", "-D", "pulse", "sset", "Master", "toggle"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            return {"success": True, "action": action}
        except Exception as e:
            print(f"[SystemTools] Volume error: {e}")
            return {"success": False, "error": str(e)}

    def get_system_info(self) -> Dict[str, Any]:
        if not HAS_PSUTIL:
            return {"error": "psutil not available"}
        
        try:
            cpu = psutil.cpu_percent(interval=0.5)
            mem = psutil.virtual_memory()
            return {
                "cpu_percent": cpu,
                "memory_percent": mem.percent,
                "memory_used_gb": round(mem.used / (1024**3), 2),
            }
        except Exception as e:
            return {"error": str(e)}

    def execute_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """כלי ראשי - מבצע action מהמוח"""
        if not action:
            return {"success": False}

        action_type = action.get("type")
        
        if action_type == "open_app":
            return self.open_app(action.get("app", ""), action.get("query"))
        elif action_type == "search":
            return self.search_google(action.get("query", ""))
        elif action_type == "volume":
            return self.control_volume(action.get("action", "toggle"))
        else:
            return {"success": False, "error": f"Unknown action type {action_type}"}

# Singleton
_global_tools = None
def get_system_tools():
    global _global_tools
    if _global_tools is None:
        _global_tools = SystemTools()
    return _global_tools
