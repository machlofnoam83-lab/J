"""
Adiel Junior - Self-Healing Error Recovery System
מערכת שמתקנת שגיאות בזמן ריצה - לב המוח העמיד

- מזהה שגיאות import / runtime
- מנסה להתקין / לתקן אוטומטית
- נופלת ל-fallback כדי לא לקרוס
- לומדת מטעויות (שומרת log)
"""
import os
import sys
import traceback
import importlib
import subprocess
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

ROOT = Path(__file__).parent.parent.parent
ERROR_LOG = ROOT / "backend" / "data" / "error_recovery_log.json"

class ErrorRecovery:
    def __init__(self):
        self.errors = self._load_log()
        self.fixes_attempted = {}
        
    def _load_log(self) -> Dict:
        if ERROR_LOG.exists():
            try:
                return json.loads(ERROR_LOG.read_text(encoding='utf-8'))
            except:
                return {"errors": [], "fixes": {}}
        return {"errors": [], "fixes": {}}
    
    def _save_log(self):
        try:
            ERROR_LOG.parent.mkdir(parents=True, exist_ok=True)
            ERROR_LOG.write_text(json.dumps(self.errors, ensure_ascii=False, indent=2), encoding='utf-8')
        except Exception as e:
            print(f"[Recovery] Failed to save log: {e}")
    
    def log_error(self, error_type: str, error_msg: str, fixed=False, fix_method=None):
        entry = {
            "timestamp": datetime.now().isoformat(),
            "type": error_type,
            "message": str(error_msg)[:500],
            "fixed": fixed,
            "fix_method": fix_method
        }
        self.errors["errors"].append(entry)
        # שמור רק 100 אחרונים
        if len(self.errors["errors"]) > 100:
            self.errors["errors"] = self.errors["errors"][-100:]
        self._save_log()
        print(f"[Recovery] Logged: {error_type} -> fixed={fixed}")

    def try_import_with_fix(self, module_name: str, pip_name: str = None, alternative: str = None):
        """
        מנסה לייבא מודול, אם נכשל מנסה לתקן אוטומטית
        """
        pip_name = pip_name or module_name
        
        try:
            mod = importlib.import_module(module_name)
            return mod, True
        except ImportError as e:
            print(f"[Recovery] Import failed for {module_name}: {e}")
            self.log_error(f"import_{module_name}", str(e), fixed=False)
            
            # נסה alternative אם יש
            if alternative:
                try:
                    mod = importlib.import_module(alternative)
                    print(f"[Recovery] Using alternative {alternative} for {module_name}")
                    self.log_error(f"import_{module_name}", f"Using {alternative}", fixed=True, fix_method=f"alternative_{alternative}")
                    return mod, True
                except:
                    pass
            
            # נסה להתקין אוטומטית
            print(f"[Recovery] Attempting pip install {pip_name}...")
            try:
                result = subprocess.run(
                    [sys.executable, "-m", "pip", "install", pip_name, "--break-system-packages"],
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                if result.returncode == 0:
                    print(f"[Recovery] Installed {pip_name}, retrying import...")
                    try:
                        mod = importlib.import_module(module_name)
                        self.log_error(f"import_{module_name}", f"Fixed by pip install {pip_name}", fixed=True, fix_method=f"pip_{pip_name}")
                        return mod, True
                    except:
                        pass
                else:
                    print(f"[Recovery] pip install failed: {result.stderr[-200:]}")
            except Exception as install_e:
                print(f"[Recovery] Install attempt failed: {install_e}")
            
            # נכשל - החזר None אבל אל תקריס
            self.log_error(f"import_{module_name}", str(e), fixed=False, fix_method="failed")
            return None, False

    def safe_execute(self, func, fallback=None, error_context=""):
        """
        מריץ פונקציה עם הגנה, אם נכשלת מחזיר fallback
        """
        try:
            return func(), True
        except Exception as e:
            tb = traceback.format_exc()
            print(f"[Recovery] Error in {error_context}: {e}")
            print(tb[-500:])
            self.log_error(f"runtime_{error_context}", f"{e}\n{tb[-1000:]}", fixed=False)
            
            # נסה fallback
            if fallback is not None:
                try:
                    if callable(fallback):
                        result = fallback()
                    else:
                        result = fallback
                    print(f"[Recovery] Used fallback for {error_context}")
                    self.log_error(f"runtime_{error_context}", "Recovered with fallback", fixed=True, fix_method="fallback")
                    return result, False
                except Exception as fb_e:
                    print(f"[Recovery] Fallback also failed: {fb_e}")
            
            return None, False

    def get_health_report(self) -> Dict[str, Any]:
        """דוח בריאות המערכת"""
        total_errors = len(self.errors.get("errors", []))
        fixed = sum(1 for e in self.errors.get("errors", []) if e.get("fixed"))
        recent = self.errors.get("errors", [])[-5:]
        
        return {
            "total_errors": total_errors,
            "fixed_count": fixed,
            "fix_rate": fixed / total_errors if total_errors else 1.0,
            "recent_errors": recent,
            "status": "healthy" if total_errors < 10 or fixed/total_errors > 0.7 else "needs_attention"
        }

# Global instance
_global_recovery = None

def get_recovery() -> ErrorRecovery:
    global _global_recovery
    if _global_recovery is None:
        _global_recovery = ErrorRecovery()
    return _global_recovery

# Decorator for auto-recovery
def auto_recover(fallback=None, context=""):
    def decorator(func):
        def wrapper(*args, **kwargs):
            recovery = get_recovery()
            def try_func():
                return func(*args, **kwargs)
            result, success = recovery.safe_execute(try_func, fallback=fallback, error_context=context or func.__name__)
            return result
        return wrapper
    return decorator
