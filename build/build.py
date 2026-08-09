"""
Build Script - Executable Builder for Adiel Junior
בונה EXE יחיד ל-Windows

- Backend -> PyInstaller -> adiel_backend.exe
- Frontend + Backend -> Electron Builder -> Setup.exe

Usage:
    python build/build.py --all        # הכל
    python build/build.py --backend    # רק backend
    python build/build.py --frontend   # רק frontend
"""
import os
import sys
import subprocess
import shutil
import argparse
from pathlib import Path

ROOT = Path(__file__).parent.parent
BACKEND = ROOT / "backend"
FRONTEND = ROOT / "frontend"
DIST = ROOT / "dist"

def run(cmd, cwd=None):
    print(f"> {cmd}")
    result = subprocess.run(cmd, shell=True, cwd=cwd)
    if result.returncode != 0:
        print(f"Command failed with {result.returncode}")
        sys.exit(result.returncode)

def build_backend():
    print("\n=== Building Backend with PyInstaller ===")
    os.chdir(BACKEND)
    
    # Install pyinstaller if needed
    try:
        import PyInstaller
    except:
        run(f"{sys.executable} -m pip install pyinstaller")

    # PyInstaller spec
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--onefile",
        "--noconsole",
        "--name", "adiel_backend",
        "--distpath", str(BACKEND / "dist"),
        "--workpath", str(ROOT / "build" / "pyinstaller_build"),
        "--specpath", str(ROOT / "build"),
        "--add-data", f"{BACKEND / 'core'}:core",
        "--hidden-import", "engineio.async_drivers.threading",
        "--hidden-import", "uvicorn.logging",
        "--hidden-import", "uvicorn.loops.auto",
        "--collect-all", "faster_whisper",
        "--collect-all", "vosk",
        "main.py"
    ]

    # Windows path separator fix
    if os.name == 'nt':
        cmd = [
            sys.executable, "-m", "PyInstaller",
            "--onefile",
            "--noconsole",
            "--name", "adiel_backend",
            f"--distpath={BACKEND / 'dist'}",
            f"--workpath={ROOT / 'build' / 'pyinstaller_build'}",
            f"--specpath={ROOT / 'build'}",
            "main.py"
        ]

    run(" ".join([f'"{c}"' if ' ' in c else c for c in cmd]), cwd=BACKEND)

    exe_path = BACKEND / "dist" / ("adiel_backend.exe" if os.name == 'nt' else "adiel_backend")
    if exe_path.exists():
        print(f"✅ Backend built: {exe_path} ({exe_path.stat().st_size / 1024 / 1024:.1f} MB)")
        return True
    else:
        print("❌ Backend build failed")
        return False

def build_frontend():
    print("\n=== Building Frontend with Electron Builder ===")
    os.chdir(FRONTEND)

    # Check npm
    if not shutil.which("npm"):
        print("npm not found, please install Node.js")
        return False

    # Install deps
    if not (FRONTEND / "node_modules").exists():
        run("npm install", cwd=FRONTEND)

    # Build
    run("npm run build", cwd=FRONTEND)

    # Find output
    dist_files = list((FRONTEND / "dist").glob("*.exe")) if (FRONTEND / "dist").exists() else []
    if dist_files:
        for f in dist_files:
            print(f"✅ Frontend built: {f}")
            # Copy to root dist
            DIST.mkdir(exist_ok=True)
            shutil.copy(f, DIST / f.name)
        return True
    else:
        print("❌ Frontend build may have failed - check frontend/dist")
        return False

def main():
    parser = argparse.ArgumentParser(description="Build Adiel Junior Executable")
    parser.add_argument("--backend", action="store_true", help="Build backend only")
    parser.add_argument("--frontend", action="store_true", help="Build frontend only")
    parser.add_argument("--all", action="store_true", help="Build all (default)")
    args = parser.parse_args()

    if not args.backend and not args.frontend:
        args.all = True

    DIST.mkdir(exist_ok=True)

    if args.all or args.backend:
        build_backend()

    if args.all or args.frontend:
        build_frontend()

    print("\n=== Build Finished ===")
    if DIST.exists():
        print(f"Artifacts in {DIST}:")
        for f in DIST.iterdir():
            print(f" - {f.name} ({f.stat().st_size / 1024 / 1024:.1f} MB)")

if __name__ == "__main__":
    main()
