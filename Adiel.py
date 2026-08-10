#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Adiel Junior - ONE FILE TO RULE THEM ALL - ENGLISH VERSION
Double click to start everything - No million files!

This is same as הפעל_את_אדיאל.py but with English name to avoid Windows encoding issues
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
║  ADIEL JUNIOR - ULTIMATE - ONE FILE!                       ║
║  MARK 85 • Original Voice • Real AI • Full Hebrew Dict    ║
║  6GB LLM • Fast Speech • Text Reader • Voice for Every Q   ║
║  ONE FILE THAT OPENS EVERYTHING!                           ║
╚════════════════════════════════════════════════════════════╝
    """)

def get_python():
    return str(VENV_PYTHON) if VENV_PYTHON.exists() else sys.executable

def main():
    banner()
    print("One file - opens everything! No million files!\n")
    
    if not VENV_DIR.exists():
        print("📦 Creating venv...")
        subprocess.run([sys.executable, "-m", "venv", str(VENV_DIR)], check=False)
    
    py_exe = get_python()
    
    print("\n📥 Installing deps with auto-fix...")
    auto = ROOT / "scripts" / "auto_installer.py"
    if auto.exists():
        subprocess.run([py_exe, str(auto)], check=False)
    
    req = BACKEND_DIR / "requirements.txt"
    if req.exists():
        subprocess.run([py_exe, "-m", "pip", "install", "--upgrade", "pip"], check=False)
        subprocess.run([py_exe, "-m", "pip", "install", "-r", str(req)], check=False)
    
    print("\n🚀 Starting Backend (LLM 6GB + AI Voice + Full Dict + Fast Speech)...")
    main_py = BACKEND_DIR / "main.py"
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    proc = subprocess.Popen([py_exe, str(main_py)], cwd=str(BACKEND_DIR), env=env,
                            creationflags=subprocess.CREATE_NEW_CONSOLE if os.name=='nt' else 0)
    print(f"  ✓ Backend PID {proc.pid} - http://localhost:8765")
    time.sleep(5)

    print("\n🎨 Starting ULTIMATE HUD...")
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
        print("  ✓ Browser with ULTIMATE HUD!")
    
    print("\n" + "="*60)
    print("  ✅ Adiel ULTIMATE Running!")
    print("  Backend: http://localhost:8765/status")
    print("  HUD: ULTIMATE with 500+ Hebrew Dict")
    print("  Voice: AI for every question + Original voice")
    print("  Brain: LLM 6GB + True AI + Fast Speech + Text Reader")
    print("  Say: 'Adiel Junior'")
    print("="*60)
    try:
        input("\nPress Enter to close...\n")
    except:
        pass
    try:
        proc.terminate()
    except:
        pass

if __name__ == "__main__":
    main()
