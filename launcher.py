"""
Adiel Junior - Main Launcher
מריץ Backend + Frontend (Electron או GUI fallback)
זהו קובץ ה-EXE הראשי ש-PyInstaller יבנה

Usage:
    python launcher.py
    python launcher.py --backend-only
    python launcher.py --gui-only
"""
import os
import sys
import subprocess
import threading
import time
import webbrowser
from pathlib import Path

ROOT = Path(__file__).parent
BACKEND_DIR = ROOT / "backend"
FRONTEND_DIR = ROOT / "frontend"

def check_env():
    print("="*60)
    print("  Adiel Junior - Launcher")
    print("  אדיאל ג'וניור - משגר ראשי")
    print("="*60)
    print(f"Root: {ROOT}")
    print(f"Python: {sys.version}")
    print(f"Platform: {sys.platform}")

def start_backend():
    print("\n[Launcher] Starting Backend...")
    backend_main = BACKEND_DIR / "main.py"
    backend_exe = BACKEND_DIR / "dist" / ("adiel_backend.exe" if os.name == 'nt' else "adiel_backend")

    if backend_exe.exists() and not "--dev" in sys.argv:
        print(f"[Launcher] Using compiled backend: {backend_exe}")
        proc = subprocess.Popen([str(backend_exe)], cwd=str(BACKEND_DIR / "dist"))
    else:
        if not backend_main.exists():
            print(f"[Launcher] Backend main not found at {backend_main}")
            return None
        env = os.environ.copy()
        env["PORT"] = "8765"
        print(f"[Launcher] Running backend via python: {backend_main}")
        proc = subprocess.Popen([sys.executable, str(backend_main)], cwd=str(BACKEND_DIR), env=env)

    # Wait for backend to be ready
    print("[Launcher] Waiting for backend (http://localhost:8765/status)...")
    import requests
    for i in range(30):
        try:
            r = requests.get("http://localhost:8765/status", timeout=1)
            if r.status_code == 200:
                print(f"[Launcher] Backend ready after {i}s: {r.json()}")
                break
        except:
            pass
        time.sleep(1)
        if i % 5 == 0:
            print(f"[Launcher] Still waiting... {i}s")
    else:
        print("[Launcher] Backend didn't become ready in 30s, continuing anyway")

    return proc

def start_frontend_electron():
    print("\n[Launcher] Trying Electron frontend...")
    package_json = FRONTEND_DIR / "package.json"
    if not package_json.exists():
        print("[Launcher] frontend/package.json not found - Electron not available")
        return None

    # Check if electron available
    electron_node_modules = FRONTEND_DIR / "node_modules" / "electron"
    if not electron_node_modules.exists():
        print("[Launcher] Electron not installed, running npm install...")
        try:
            subprocess.run(["npm", "install"], cwd=str(FRONTEND_DIR), check=True, shell=os.name=='nt')
        except Exception as e:
            print(f"[Launcher] npm install failed: {e}")
            return None

    try:
        # Start electron
        # We use npx electron .
        env = os.environ.copy()
        print("[Launcher] Starting Electron...")
        proc = subprocess.Popen(["npx", "electron", "."], cwd=str(FRONTEND_DIR), shell=os.name=='nt')
        return proc
    except Exception as e:
        print(f"[Launcher] Electron start failed: {e}")
        return None

def start_fallback_gui():
    print("\n[Launcher] Starting Fallback Python GUI...")
    gui_path = BACKEND_DIR / "gui_fallback.py"
    if not gui_path.exists():
        print(f"[Launcher] GUI fallback not found at {gui_path}")
        return None
    proc = subprocess.Popen([sys.executable, str(gui_path)], cwd=str(BACKEND_DIR))
    return proc

def start_web_fallback():
    print("\n[Launcher] Opening web HUD (frontend/src/index.html) in browser as last fallback...")
    html_path = FRONTEND_DIR / "src" / "index.html"
    if html_path.exists():
        webbrowser.open(f"file://{html_path}")
        print(f"[Launcher] Opened {html_path} in browser - note: backend connection still needed")
    else:
        print("[Launcher] Web HUD not found")

def main():
    check_env()

    backend_only = "--backend-only" in sys.argv
    gui_only = "--gui-only" in sys.argv
    dev = "--dev" in sys.argv

    backend_proc = None
    frontend_proc = None

    try:
        if not gui_only:
            backend_proc = start_backend()

        if backend_only:
            print("\n[Launcher] Backend-only mode, keeping alive...")
            print("Backend running at http://localhost:8765")
            print("Frontend WebSocket at ws://localhost:8765/ws")
            print("Press Ctrl+C to exit")
            while True:
                time.sleep(1)
            return

        # Frontend
        time.sleep(1)
        frontend_proc = start_frontend_electron()

        if not frontend_proc:
            print("[Launcher] Electron failed, trying Python fallback GUI...")
            frontend_proc = start_fallback_gui()

        if not frontend_proc:
            start_web_fallback()
            print("\n[Launcher] All GUI methods failed, running backend only")
            print("You can still access backend status at http://localhost:8765/status")
            print("And connect via frontend/src/index.html opened in browser")

        print("\n[Launcher] Both processes started, waiting...")
        print("Press Ctrl+C to exit both")

        # Keep launcher alive
        while True:
            time.sleep(1)
            # Check if procs still alive
            if backend_proc and backend_proc.poll() is not None:
                print(f"[Launcher] Backend exited with code {backend_proc.returncode}")
                break
            if frontend_proc and frontend_proc.poll() is not None:
                print(f"[Launcher] Frontend exited with code {frontend_proc.returncode}")
                # Don't break - backend may still run
                # break

    except KeyboardInterrupt:
        print("\n[Launcher] Shutting down...")
    finally:
        if frontend_proc:
            try:
                frontend_proc.terminate()
                frontend_proc.wait(timeout=3)
            except:
                try:
                    frontend_proc.kill()
                except:
                    pass

        if backend_proc:
            try:
                backend_proc.terminate()
                backend_proc.wait(timeout=3)
            except:
                try:
                    backend_proc.kill()
                except:
                    pass

        print("[Launcher] Clean exit")

if __name__ == "__main__":
    main()
