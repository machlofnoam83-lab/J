"""
console_fix.py - תיקון קידוד קונסולה ב-Windows
=================================================
ב-Windows הקונסולה רצה בדרך כלל ב-cp1255/cp862, וכל print עם אימוג'י או
סימנים (✓ ⚡ 🚀 🧠 🎯 📖...) קורס עם UnicodeEncodeError.

הקריסה הזו הייתה הרסנית: היא התפוצצה באמצע __init__ של מודולים (STT, TTS,
Brain, Frames...) וגרמה להם להירשם כ"לא זמינים" למרות שהם עובדים.

הפתרון: reconfigure של stdout/stderr ל-UTF-8 עם errors='replace' — חייב
להיקרא *לפני* כל import של מודולים אחרים שמדפיסים.

בנוסף אנחנו מספקים safe_print לשימוש בנקודות רגישות במיוחד.
"""
import io
import os
import sys


def force_utf8_console() -> None:
    """מכריח UTF-8 על stdout/stderr. בטוח לקרוא מספר פעמים."""
    # לתהליכים בנים (pip, whisper...): שירשו UTF-8
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    os.environ.setdefault("PYTHONUTF8", "1")

    for name in ("stdout", "stderr"):
        stream = getattr(sys, name, None)
        if stream is None:
            continue
        try:
            # Python 3.7+ - TextIOWrapper.reconfigure (ידידותי ל-IDE/קונסולה)
            stream.reconfigure(encoding="utf-8", errors="replace")
            continue
        except (AttributeError, io.UnsupportedOperation, ValueError):
            pass
        except Exception:
            pass
        try:
            buffer = getattr(stream, "buffer", None)
            if buffer is not None:
                wrapped = io.TextIOWrapper(buffer, encoding="utf-8",
                                           errors="replace", line_buffering=True)
                setattr(sys, name, wrapped)
        except Exception:
            # המלטאות אחרונה - מעטפת שמחליפה תווים לא נתמכים
            try:
                class _SafeStream:
                    def __init__(self, raw):
                        self._raw = raw

                    def write(self, s):
                        try:
                            self._raw.write(s)
                        except UnicodeEncodeError:
                            enc = getattr(self._raw, "encoding", None) or "utf-8"
                            self._raw.write(s.encode(enc, errors="replace").decode(enc, errors="replace"))

                    def flush(self):
                        try:
                            self._raw.flush()
                        except Exception:
                            pass

                    def __getattr__(self, attr):
                        return getattr(self._raw, attr)

                if not isinstance(stream, _SafeStream):
                    setattr(sys, name, _SafeStream(stream))
            except Exception:
                pass


def safe_print(*args, **kwargs) -> None:
    """print שלא יכול לקרוס מקידוד - לשימוש בנקודות קריטיות."""
    try:
        print(*args, **kwargs)
    except UnicodeEncodeError:
        try:
            text = " ".join(str(a) for a in args)
            enc = getattr(sys.stdout, "encoding", None) or "utf-8"
            print(text.encode(enc, errors="replace").decode(enc, errors="replace"), **kwargs)
        except Exception:
            pass
    except Exception:
        pass
