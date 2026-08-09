"""
Wrapper for scripts/auto_installer.py - allows running from backend folder
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

try:
    from auto_installer import auto_fix_loop
    if __name__ == "__main__":
        auto_fix_loop()
except ImportError:
    print("Running fallback installer...")
    import subprocess
    subprocess.run([sys.executable, str(Path(__file__).parent.parent / "scripts" / "auto_installer.py")] + sys.argv[1:])
