#!/usr/bin/env python3
"""JARVIS launcher — cross-platform.

Starts the brain server, then either the Electron shell (desktop app) or your
browser (fallback / development). Everything stays on loopback.

    python run_jarvis.py                 # electron if installed, else browser
    python run_jarvis.py --browser       # force browser mode
    python run_jarvis.py --port 9000     # custom port
"""

from __future__ import annotations

import argparse
import os
import shutil
import socket
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) != 0


def wait_up(port: int, timeout: float = 120.0) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout:
        if not port_free(port):
            return True
        time.sleep(0.5)
    return False


def electron_binary() -> str | None:
    local = ROOT / "node_modules" / ".bin" / ("electron.cmd" if os.name == "nt" else "electron")
    if local.exists():
        return str(local)
    dist = ROOT / "node_modules" / "electron" / "dist" / ("electron.exe" if os.name == "nt" else "electron")
    if dist.exists():
        return str(dist)
    found = shutil.which("electron")
    return found


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8756)
    ap.add_argument("--browser", action="store_true", help="skip Electron")
    ap.add_argument("--no-open", action="store_true", help="do not open a browser window")
    args = ap.parse_args()

    env = dict(os.environ, JARVIS_PORT=str(args.port), PYTHONIOENCODING="utf-8")

    if not args.browser:
        exe = electron_binary()
        if exe:
            print(f"[jarvis] launching desktop shell via {exe}")
            print("[jarvis] the shell owns the brain process — close it to stop JARVIS")
            env["JARVIS_PORT"] = str(args.port)
            return subprocess.call([exe, str(ROOT)], cwd=str(ROOT), env=env)
        print("[jarvis] Electron not installed (npm install) — using browser mode")

    if not port_free(args.port):
        print(f"[jarvis] something already listens on {args.port} — assuming the brain is up")
    else:
        print(f"[jarvis] starting brain server on 127.0.0.1:{args.port}")
        subprocess.Popen([sys.executable, "-m", "core.server", "--port", str(args.port)],
                         cwd=str(ROOT), env=env)
        if not wait_up(args.port):
            print("[jarvis] brain did not open the port in time; check logs above", file=sys.stderr)
            return 1

    url = f"http://127.0.0.1:{args.port}/"
    if not args.no_open:
        print(f"[jarvis] opening {url}")
        try:
            webbrowser.open(url)
        except Exception:
            pass
    else:
        print(f"[jarvis] HUD available at {url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
