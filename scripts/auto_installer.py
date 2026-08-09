#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Adiel Junior - Auto Fix Installer
מערכת התקנה חכמה שמתקנת שגיאות אוטומטית

היא:
1. מזהה שגיאות pip / npm בזמן אמת
2. מתקנת requirements.txt אוטומטית
3. מנסה שוב עד שמצליח
4. עובדת ב-Python 3.10 / 3.11 / 3.12 / 3.13

Usage:
    python scripts/auto_installer.py
    python scripts/auto_installer.py --fix-only
    python scripts/auto_installer.py --check
"""
import os
import sys
import re
import subprocess
import shutil
import time
from pathlib import Path
from typing import Tuple, List, Dict

ROOT = Path(__file__).parent.parent
BACKEND = ROOT / "backend"
FRONTEND = ROOT / "frontend"
REQUIREMENTS = BACKEND / "requirements.txt"

# מפת תיקונים אוטומטית - מזהה שגיאה -> תיקון
ERROR_FIXES = {
    # Python packages
    "pygame": {
        "patterns": [r"pygame", r"Failed.*pygame", r"pygame.*failed"],
        "fix": "replace_pygame",
        "description": "מחליף pygame הישן ב-pygame-ce המודרני"
    },
    "pillow_version": {
        "patterns": [r"Pillow.*__version__", r"KeyError.*__version__", r"Pillow==10\.2\.0"],
        "fix": "upgrade_pillow",
        "description": "משדרג Pillow ל-10.4.0+"
    },
    "webrtcvad": {
        "patterns": [r"webrtcvad", r"Failed.*webrtcvad", r"Microsoft Visual C\+\+.*webrtcvad"],
        "fix": "replace_webrtcvad",
        "description": "מחליף webrtcvad ב-webrtcvad-wheels עם wheels מוכנים"
    },
    "opencv": {
        "patterns": [r"opencv-python", r"Failed.*opencv", r"cv2"],
        "fix": "fix_opencv",
        "description": "מנסה opencv עם headless או גרסה חדשה יותר"
    },
    "numpy": {
        "patterns": [r"numpy.*failed", r"numpy.*compilation"],
        "fix": "fix_numpy",
        "description": "מתאים numpy לפייתון הנוכחי"
    },
    "fastapi_pydantic": {
        "patterns": [r"fastapi.*pydantic", r"pydantic.*fastapi", r"pydantic-core"],
        "fix": "upgrade_fastapi_pydantic",
        "description": "משדרג fastapi+pydantic לגרסאות תואמות"
    },
    # NPM errors
    "npm_wrong_dir": {
        "patterns": [r"Could not read package\.json", r"ENOENT.*package\.json.*scripts"],
        "fix": "fix_npm_dir",
        "description": "מזהה שהרצת npm בתיקייה הלא נכונה, עובר ל-frontend"
    },
    "npm_peer_deps": {
        "patterns": [r"ERESOLVE.*peer", r"peer dep", r"Could not resolve dependency"],
        "fix": "npm_legacy",
        "description": "מנסה npm עם --legacy-peer-deps ל-Node 24"
    },
}

class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

def log(msg: str, color=Colors.RESET, bold=False):
    prefix = Colors.BOLD if bold else ""
    print(f"{prefix}{color}{msg}{Colors.RESET}")

def run_cmd(cmd: List[str], cwd: Path = None, capture=True) -> Tuple[int, str, str]:
    """מריץ פקודה ומחזיר קוד, stdout, stderr"""
    try:
        result = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            capture_output=capture,
            text=True,
            shell=(os.name == 'nt'),
            timeout=300
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return 1, "", "Timeout"
    except Exception as e:
        return 1, "", str(e)

def read_requirements() -> str:
    if REQUIREMENTS.exists():
        return REQUIREMENTS.read_text(encoding='utf-8')
    return ""

def write_requirements(content: str):
    REQUIREMENTS.write_text(content, encoding='utf-8')
    log(f"  ✓ עדכנתי {REQUIREMENTS}", Colors.GREEN)

def fix_pygame(content: str) -> str:
    log("  🔧 מתקן pygame -> pygame-ce...", Colors.YELLOW)
    content = re.sub(r"pygame==[^\n]+", "pygame-ce>=2.4.1", content)
    content = re.sub(r"^pygame\s*$", "pygame-ce>=2.4.1", content, flags=re.MULTILINE)
    if "pygame-ce" not in content:
        content += "\npygame-ce>=2.4.1\n"
    # הסר שורות pygame ישנות כפולות
    lines = []
    seen_ce = False
    for line in content.splitlines():
        if "pygame-ce" in line:
            if not seen_ce:
                lines.append(line)
                seen_ce = True
        elif re.match(r"^\s*pygame\s*(==|>=)?.*", line):
            # דלג על pygame ישן
            if not seen_ce:
                lines.append("pygame-ce>=2.4.1")
                seen_ce = True
        else:
            lines.append(line)
    return "\n".join(lines) + "\n"

def fix_pillow(content: str) -> str:
    log("  🔧 משדרג Pillow ל-10.4.0+...", Colors.YELLOW)
    content = re.sub(r"Pillow==[^\n]+", "Pillow>=10.4.0", content)
    content = re.sub(r"Pillow>=10\.2\.0", "Pillow>=10.4.0", content)
    if "Pillow" not in content:
        content += "\nPillow>=10.4.0\n"
    return content

def fix_webrtcvad(content: str) -> str:
    log("  🔧 מחליף webrtcvad -> webrtcvad-wheels...", Colors.YELLOW)
    content = re.sub(r"webrtcvad==[^\n]+", "webrtcvad-wheels>=0.2.13", content)
    content = re.sub(r"^webrtcvad\s*$", "webrtcvad-wheels>=0.2.13", content, flags=re.MULTILINE)
    if "webrtcvad-wheels" not in content:
        content = content.replace("webrtcvad", "webrtcvad-wheels")
        if "webrtcvad-wheels" not in content:
            content += "\nwebrtcvad-wheels>=0.2.13\n"
    return content

def fix_opencv(content: str) -> str:
    log("  🔧 מתקן opencv-python...", Colors.YELLOW)
    # נסה headless אם הרגיל נכשל
    if "opencv-python-headless" not in content:
        content = re.sub(r"opencv-python==[^\n]+", "opencv-python>=4.10.0.84", content)
        content = re.sub(r"opencv-python>=.*", "opencv-python>=4.10.0.84", content)
    return content

def fix_numpy(content: str) -> str:
    log("  🔧 מתאים numpy לפייתון...", Colors.YELLOW)
    # Python 3.12+ צריך numpy 1.26+
    content = re.sub(r"numpy==[^\n]+", "numpy>=1.26.4", content)
    return content

def upgrade_fastapi_pydantic(content: str) -> str:
    log("  🔧 משדרג fastapi+pydantic...", Colors.YELLOW)
    content = re.sub(r"fastapi==[^\n]+", "fastapi>=0.115.0", content)
    content = re.sub(r"pydantic==[^\n]+", "pydantic>=2.9.0", content)
    content = re.sub(r"uvicorn\[standard\]==[^\n]+", "uvicorn[standard]>=0.30.0", content)
    return content

FIX_FUNCTIONS = {
    "replace_pygame": fix_pygame,
    "upgrade_pillow": fix_pillow,
    "replace_webrtcvad": fix_webrtcvad,
    "fix_opencv": fix_opencv,
    "fix_numpy": fix_numpy,
    "upgrade_fastapi_pydantic": upgrade_fastapi_pydantic,
}

def detect_errors(output: str) -> List[Dict]:
    """מזהה שגיאות בפלט ומחזיר רשימת תיקונים"""
    detected = []
    output_lower = output.lower()
    for error_id, config in ERROR_FIXES.items():
        for pattern in config["patterns"]:
            if re.search(pattern, output, re.IGNORECASE):
                detected.append({"id": error_id, **config})
                break
    return detected

def attempt_pip_install(cwd=ROOT) -> Tuple[bool, str]:
    """מנסה להתקין requirements, מחזיר הצלחה ופלט"""
    log(f"\n📦 מנסה pip install -r {REQUIREMENTS}...", Colors.BLUE)
    
    # בדוק אם יש venv
    venv_python = ROOT / "venv" / "Scripts" / "python.exe" if os.name == 'nt' else ROOT / "venv" / "bin" / "python"
    python_exe = str(venv_python) if venv_python.exists() else sys.executable
    
    cmd = [python_exe, "-m", "pip", "install", "-r", str(REQUIREMENTS)]
    
    code, stdout, stderr = run_cmd(cmd, cwd=cwd)
    full_output = stdout + "\n" + stderr
    
    if code == 0:
        log("  ✓ pip install הצליח!", Colors.GREEN, bold=True)
        return True, full_output
    else:
        log(f"  ✗ pip install נכשל (code {code})", Colors.RED)
        # הדפס 20 שורות אחרונות
        last_lines = full_output.strip().splitlines()[-20:]
        for line in last_lines:
            if line.strip():
                print(f"    {line}")
        return False, full_output

def attempt_npm_install() -> Tuple[bool, str]:
    """מנסה npm install עם תיקון אוטומטי"""
    log(f"\n📦 מנסה npm install ב-frontend...", Colors.BLUE)
    
    if not (FRONTEND / "package.json").exists():
        log(f"  ✗ לא נמצא {FRONTEND / 'package.json'}", Colors.RED)
        return False, "package.json not found"
    
    # ניסיון 1: רגיל
    code, stdout, stderr = run_cmd(["npm", "install"], cwd=FRONTEND)
    if code == 0:
        log("  ✓ npm install הצליח!", Colors.GREEN, bold=True)
        return True, stdout
    
    full_output = stdout + stderr
    
    # מזהה שגיאת תיקייה לא נכונה
    if "scripts" in full_output and "package.json" in full_output:
        log("  🔧 זיהיתי הרצה בתיקייה הלא נכונה, מתקן...", Colors.YELLOW)
        # כבר ב-frontend, אז זה לא הבעיה, נסה legacy
        pass
    
    # ניסיון 2: --legacy-peer-deps (ל-Node 24)
    log("  🔧 מנסה עם --legacy-peer-deps (ל-Node 24+)...", Colors.YELLOW)
    code2, stdout2, stderr2 = run_cmd(["npm", "install", "--legacy-peer-deps"], cwd=FRONTEND)
    if code2 == 0:
        log("  ✓ הצליח עם --legacy-peer-deps!", Colors.GREEN, bold=True)
        return True, stdout2
    
    log(f"  ✗ גם עם legacy נכשל", Colors.RED)
    return False, stdout2 + stderr2

def auto_fix_loop(max_retries=5) -> bool:
    """לולאת תיקון אוטומטית ראשית"""
    log("="*60, Colors.CYAN)
    log("  Adiel Junior - Auto Fix Installer", Colors.CYAN, bold=True)
    log("  מערכת התקנה חכמה שמתקנת שגיאות לבד", Colors.CYAN)
    log("="*60, Colors.CYAN)
    log(f"Python: {sys.version}", Colors.BLUE)
    log(f"Root: {ROOT}", Colors.BLUE)
    log(f"Requirements: {REQUIREMENTS}", Colors.BLUE)
    
    # בדיקת פייתון
    if sys.version_info < (3, 10):
        log("\n✗ Python ישן מדי! צריך 3.10+ ", Colors.RED, bold=True)
        log("הורד מ- https://python.org", Colors.YELLOW)
        return False
    
    # יצירת venv אם לא קיים
    venv_path = ROOT / "venv"
    if not venv_path.exists():
        log(f"\n📁 יוצר venv ב-{venv_path}...", Colors.BLUE)
        code, out, err = run_cmd([sys.executable, "-m", "venv", str(venv_path)], cwd=ROOT)
        if code == 0:
            log("  ✓ venv נוצר", Colors.GREEN)
        else:
            log(f"  ✗ venv נכשל: {err}", Colors.RED)
    
    # לולאת pip
    for attempt in range(1, max_retries+1):
        log(f"\n🔄 ניסיון pip {attempt}/{max_retries}", Colors.BOLD)
        
        success, output = attempt_pip_install()
        if success:
            break
        
        # זהה שגיאות
        errors = detect_errors(output)
        if not errors:
            log("\n  ⚠ לא זיהיתי שגיאה מוכרת, מנסה תיקונים כלליים...", Colors.YELLOW)
            # נסה לשחרר גרסאות - כבר ב-requirements החדש
            # נסה עם --only-binary
            log("  🔧 מנסה עם --only-binary=:all:...", Colors.YELLOW)
            venv_python = ROOT / "venv" / "Scripts" / "python.exe" if os.name == 'nt' else ROOT / "venv" / "bin" / "python"
            python_exe = str(venv_python) if venv_python.exists() else sys.executable
            code, out, err = run_cmd([python_exe, "-m", "pip", "install", "--only-binary=:all:", "-r", str(REQUIREMENTS)], cwd=ROOT)
            if code == 0:
                log("  ✓ הצליח עם only-binary!", Colors.GREEN)
                success = True
                break
            continue
        
        log(f"\n  🔍 זיהיתי {len(errors)} שגיאות:", Colors.YELLOW)
        for err in errors:
            log(f"    - {err['id']}: {err['description']}", Colors.YELLOW)
        
        # תקן requirements.txt
        content = read_requirements()
        original = content
        
        for err in errors:
            fix_name = err["fix"]
            if fix_name in FIX_FUNCTIONS:
                if fix_name == "fix_npm_dir":
                    # זה לא שגיאת pip, דלג
                    continue
                if fix_name == "npm_legacy":
                    continue
                log(f"  🛠 מתקן {err['id']}...", Colors.CYAN)
                content = FIX_FUNCTIONS[fix_name](content)
        
        if content != original:
            write_requirements(content)
            log(f"  ✓ תיקנתי {len(errors)} דברים, מנסה שוב...", Colors.GREEN)
            time.sleep(1)
        else:
            log("  ⚠ אין מה לתקן יותר בקובץ", Colors.YELLOW)
            # אם זה npm error, אל תיכנס ללולאה אינסופית
            if any(e["id"].startswith("npm") for e in errors):
                break
    
    else:
        log("\n✗ נכשל אחרי כל הניסיונות pip", Colors.RED, bold=True)
        pip_success = False
    pip_success = success if 'success' in locals() else False
    
    # לולאת npm
    log(f"\n{'='*60}", Colors.CYAN)
    log("מנסה להתקין Frontend (npm)...", Colors.CYAN)
    
    # תיקון תיקייה - ודא שאנחנו ב-frontend
    if not (FRONTEND / "package.json").exists():
        log(f"  ✗ לא מוצא package.json ב-{FRONTEND}", Colors.RED)
        log(f"  מחפש...", Colors.YELLOW)
        for p in ROOT.rglob("package.json"):
            if "node_modules" not in str(p):
                log(f"    Found: {p}", Colors.BLUE)
        npm_success = False
    else:
        npm_success, _ = attempt_npm_install()
    
    # סיכום
    log(f"\n{'='*60}", Colors.CYAN, bold=True)
    log("  סיכום התקנה", Colors.CYAN, bold=True)
    log(f"{'='*60}", Colors.CYAN)
    
    if pip_success:
        log("  ✓ Python dependencies - הותקן בהצלחה!", Colors.GREEN, bold=True)
    else:
        log("  ✗ Python dependencies - נכשל, אבל יש fallback", Colors.RED)
        log("    אפשר להריץ עם: python backend/gui_fallback.py", Colors.YELLOW)
    
    if npm_success:
        log("  ✓ Frontend (npm) - הותקן בהצלחה!", Colors.GREEN, bold=True)
    else:
        log("  ⚠ Frontend (npm) - נכשל (Node 24 חדש מדי?)", Colors.YELLOW)
        log("    פתרונות:", Colors.YELLOW)
        log("    1. התקן Node 20 LTS מ- nodejs.org", Colors.YELLOW)
        log("    2. או הרץ: cd frontend && npm install --legacy-peer-deps", Colors.YELLOW)
        log("    3. או הרץ רק Backend: python launcher.py --backend-only", Colors.YELLOW)
    
    if pip_success:
        log(f"\n  🎉 הכל מוכן! הרץ: python launcher.py או scripts\\run.bat", Colors.GREEN, bold=True)
        return True
    else:
        log(f"\n  ⚠ חלקי - אפשר попробовать fallback", Colors.YELLOW)
        return False

if __name__ == "__main__":
    if "--check" in sys.argv:
        content = read_requirements()
        print(content)
        sys.exit(0)
    
    success = auto_fix_loop()
    sys.exit(0 if success else 1)
