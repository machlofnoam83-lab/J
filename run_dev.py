"""
Dev Runner - מריץ Backend + Frontend Web Server לדמו
לשימוש בסביבת פיתוח ללא Electron
"""
import os
import sys
import subprocess
import time
import threading
import http.server
import socketserver
from pathlib import Path

ROOT = Path(__file__).parent
BACKEND = ROOT / "backend"
FRONTEND = ROOT / "frontend" / "src"

def start_backend():
    print("[Dev] Starting backend...")
    env = os.environ.copy()
    env["PORT"] = "8765"
    proc = subprocess.Popen([sys.executable, "main.py"], cwd=str(BACKEND), env=env)
    return proc

def start_frontend_server(port=3000):
    print(f"[Dev] Starting frontend static server at http://localhost:{port}")
    os.chdir(str(FRONTEND))
    handler = http.server.SimpleHTTPRequestHandler
    
    # Allow CORS for WS
    class CORSHandler(handler):
        def end_headers(self):
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
            self.send_header('Access-Control-Allow-Headers', '*')
            super().end_headers()
    
    with socketserver.TCPServer(("0.0.0.0", port), CORSHandler) as httpd:
        print(f"[Dev] Frontend serving at 0.0.0.0:{port} -> {FRONTEND}")
        print(f"[Dev] Open: http://localhost:{port}/index.html")
        httpd.serve_forever()

if __name__ == "__main__":
    backend_proc = start_backend()
    time.sleep(2)
    
    try:
        # Start frontend server in thread
        frontend_thread = threading.Thread(target=start_frontend_server, args=(3000,), daemon=True)
        frontend_thread.start()
        
        print("\n" + "="*60)
        print("  Adiel Junior - Dev Mode Running")
        print("="*60)
        print("Backend: http://localhost:8765/status")
        print("Frontend: http://localhost:3000/index.html")
        print("WS: ws://localhost:8765/ws")
        print("\nPress Ctrl+C to exit")
        print("="*60 + "\n")
        
        while True:
            time.sleep(1)
            if backend_proc.poll() is not None:
                print(f"Backend exited {backend_proc.returncode}")
                break
                
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        if backend_proc:
            backend_proc.terminate()
            try:
                backend_proc.wait(timeout=3)
            except:
                backend_proc.kill()
