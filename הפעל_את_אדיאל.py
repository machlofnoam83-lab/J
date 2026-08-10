#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
אדיאל ג'וניור - קובץ אחד שפותח הכל!
לא צריך מיליון קבצים - רק דאבל קליק על זה!
"""
import os, sys, subprocess, time, webbrowser
from pathlib import Path
import threading

ROOT = Path(__file__).parent
BACKEND_DIR = ROOT / "backend"
FRONTEND_DIR = ROOT / "frontend" / "src"
VENV_DIR = ROOT / "venv"
VENV_PYTHON = VENV_DIR / "Scripts" / "python.exe" if os.name == 'nt' else VENV_DIR / "bin" / "python"

def banner():
    print("""
╔════════════════════════════════════════════════════════════╗
║  █████╗ ██████╗ ██╗███████╗██╗      ██╗██╗   ██╗███╗   ██╗██╗  ║
║  ██╔══██╗██╔══██╗██║██╔════╝██║      ██║██║   ██║████╗  ██║██║  ║
║  ███████║██║  ██║██║█████╗  ██║      ██║██║   ██║██╔██╗ ██║██║  ║
║  ██╔══██║██║  ██║██║██╔══╝  ██║      ██║██║   ██║██║╚██╗██║██║  ║
║  ██║  ██║██████╔╝██║███████╗███████╗ ██║╚██████╔╝██║ ╚████║██║  ║
║  ULTIMATE • קול AI לכל שאלה • מילון מלא • דיבור מהיר • 6GB LLM ║
║  קובץ אחד שפותח הכל!                                      ║
╚════════════════════════════════════════════════════════════╝
    """)

def get_python():
    return str(VENV_PYTHON) if VENV_PYTHON.exists() else sys.executable

def main():
    banner()
    print("כולם בקובץ אחד - לא צריך מיליון קבצים!\n")
    
    # venv
    if not VENV_DIR.exists():
        print("📦 יוצר venv...")
        subprocess.run([sys.executable, "-m", "venv", str(VENV_DIR)], check=False)
    
    py_exe = get_python()
    
    # deps
    print("\n📥 מתקין תלויות עם תיקון אוטומטי...")
    auto = ROOT / "scripts" / "auto_installer.py"
    if auto.exists():
        subprocess.run([py_exe, str(auto)], check=False)
    
    req = BACKEND_DIR / "requirements.txt"
    if req.exists():
        subprocess.run([py_exe, "-m", "pip", "install", "--upgrade", "pip"], check=False)
        subprocess.run([py_exe, "-m", "pip", "install", "-r", str(req)], check=False)
    
    # backend
    print("\n🚀 מריץ Backend (LLM 6GB + קול AI לכל שאלה + מילון מלא + דיבור מהיר)...")
    main_py = BACKEND_DIR / "main.py"
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    proc = subprocess.Popen([py_exe, str(main_py)], cwd=str(BACKEND_DIR), env=env,
                            creationflags=subprocess.CREATE_NEW_CONSOLE if os.name=='nt' else 0)
    print(f"  ✓ Backend PID {proc.pid} - http://localhost:8765")
    time.sleep(5)

    # frontend
    print("\n🎨 מריץ HUD אולטימטיבי (UI חדש + מילון עברי מלא)...")
    
    # נסה Electron
    electron_bin = ROOT / "frontend" / "node_modules" / ".bin" / "electron"
    electron_cmd = str(electron_bin) + (".cmd" if os.name=='nt' else "")
    if os.path.exists(electron_cmd) or os.path.exists(str(electron_bin)):
        try:
            subprocess.Popen(["npx", "electron", "."], cwd=str(ROOT / "frontend"), 
                             creationflags=subprocess.CREATE_NEW_CONSOLE if os.name=='nt' else 0)
            print("  ✓ Electron ULTIMATE HUD!")
        except:
            pass
    else:
        # Fallback - http server + browser
        def serve():
            os.chdir(str(FRONTEND_DIR))
            import http.server, socketserver
            handler = http.server.SimpleHTTPRequestHandler
            with socketserver.TCPServer(("127.0.0.1", 3000), handler) as httpd:
                print("  Frontend http://127.0.0.1:3000")
                httpd.serve_forever()
        threading.Thread(target=serve, daemon=True).start()
        time.sleep(1)
        webbrowser.open("http://127.0.0.1:3000/index.html")
        print("  ✓ דפדפן עם ULTIMATE HUD!")
    
    print("\n" + "="*60)
    print("  ✅ אדיאל ULTIMATE רצה!")
    print("  Backend: http://localhost:8765/status")
    print("  HUD: ULTIMATE עם מילון מלא 500+ מילים")
    print("  קול: AI לכל שאלה + קול מקורי")
    print("  מוח: LLM 6GB + True AI + דיבור מהיר + קריאת טקסט")
    print("  מילון: עברי מלא עם פירושים")
    print("  תגיד: אדיאל ג'וניור")
    print("="*60)
    try:
        input("\nלחץ Enter לסגירה...\n")
    except:
        pass
    try:
        proc.terminate()
    except:
        pass

if __name__ == "__main__":
    main()
