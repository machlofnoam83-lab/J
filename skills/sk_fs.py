"""Filesystem skills — read, write, search, tree, hash, watch.

Every mutating call is a WRITE/CRITICAL action and therefore goes through the
Permission Firewall before it ever touches the disk.
"""

from __future__ import annotations

import fnmatch
import hashlib
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from skills.registry import REGISTRY, SkillResult  # noqa: E402

MAX_READ_BYTES = 2 * 1024 * 1024
MAX_RESULTS = 400


def _resolve(path: str) -> Path:
    p = Path(str(path).strip().strip('"').strip("'")).expanduser()
    if not p.is_absolute():
        p = (Path.home() / p).resolve() if str(p).startswith("~") else p.resolve()
    return p


@REGISTRY.register(
    "fs.read", risk="SAFE", agent="argus",
    description_he="קורא את תוכן הקובץ ומחזיר אותו כטקסט",
    required=("path",), triggers_he=("קרא את הקובץ", "תקרא", "read file"),
)
def fs_read(path: str = "", tail: int = 0, encoding: str = "utf-8") -> SkillResult:
    p = _resolve(path)
    if not p.exists():
        return SkillResult(ok=False, error=f"no such file: {p}")
    if p.is_dir():
        return SkillResult(ok=False, error=f"{p} is a directory, use fs.tree")
    size = p.stat().st_size
    if size > MAX_READ_BYTES and not tail:
        return SkillResult(ok=False, error=f"file is {_mb(size)} — pass tail=N to read the last N lines")
    try:
        text = p.read_text(encoding=encoding, errors="replace")
    except Exception as exc:
        return SkillResult(ok=False, error=f"could not decode {p}: {exc}")
    lines = text.splitlines()
    if tail and tail > 0:
        lines = lines[-int(tail):]
        text = "\n".join(lines)
    return SkillResult(ok=True, value=text,
                       data={"path": str(p), "bytes": size, "lines": len(text.splitlines()),
                             "summary_he": f"קראתי את {p.name}: {len(lines)} שורות, {_mb(size)}."})


@REGISTRY.register(
    "fs.write", risk="WRITE", agent="hermes",
    description_he="כותב טקסט לקובץ (יוצר תיקיות אם צריך)",
    required=("path", "content"), triggers_he=("כתוב לקובץ", "שמור קובץ"),
)
def fs_write(path: str = "", content: str = "", append: bool = False,
             encoding: str = "utf-8") -> SkillResult:
    p = _resolve(path)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a" if append else "w", encoding=encoding, newline="\n") as fh:
            fh.write(str(content))
    except Exception as exc:
        return SkillResult(ok=False, error=f"write failed: {exc}")
    verb = "הוספתי ל" if append else "כתבתי"
    return SkillResult(ok=True,
                       value=f"{verb}־{p.name}: {len(str(content))} תווים בנתיב {p}.",
                       data={"path": str(p), "chars": len(str(content)), "bytes": p.stat().st_size})


@REGISTRY.register(
    "fs.delete", risk="CRITICAL", agent="hermes",
    description_he="מוחק קובץ או תיקייה — פעולה בלתי הפיכה, דורשת אישור",
    required=("path",), triggers_he=("מחק את הקובץ", "delete"),
)
def fs_delete(path: str = "", recursive: bool = False, backup: bool = True) -> SkillResult:
    p = _resolve(path)
    if not p.exists():
        return SkillResult(ok=False, error=f"no such path: {p}")
    if p.is_dir() and any(p.iterdir()) and not recursive:
        return SkillResult(ok=False, error=f"{p} is a non-empty directory — pass recursive=true")
    moved_to = None
    if backup:
        # Safety first: we move to a trash folder instead of destroying bytes.
        trash = ROOT / "data" / "trash"
        trash.mkdir(parents=True, exist_ok=True)
        moved_to = trash / f"{int(time.time())}_{p.name}"
        try:
            shutil.move(str(p), str(moved_to))
        except Exception as exc:
            return SkillResult(ok=False, error=f"could not move to trash: {exc}")
    else:
        try:
            if p.is_dir():
                shutil.rmtree(p)
            else:
                p.unlink()
        except Exception as exc:
            return SkillResult(ok=False, error=f"delete failed: {exc}")
    where = f" (הועבר לאשפה ב־{moved_to} — אפשר לשחזר)" if moved_to else " (נמחק לצמיתות)"
    return SkillResult(ok=True, value=f"הסרתי את {p.name}{where}.",
                       data={"path": str(p), "recoverable": bool(moved_to),
                             "trash": str(moved_to) if moved_to else None})


@REGISTRY.register(
    "fs.search", risk="SAFE", agent="argus",
    description_he="מחפש קבצים לפי תבנית שם מתחת לנתיב נתון",
    required=("pattern",), triggers_he=("חפש קבצים", "מצא קובץ", "find"),
)
def fs_search(pattern: str = "*", root: str = ".", limit: int = 60,
              extension: str = "", max_depth: int = 8) -> SkillResult:
    base = _resolve(root)
    if not base.exists():
        return SkillResult(ok=False, error=f"no such directory: {base}")
    pat = str(pattern).strip() or "*"
    if "." in pat and "*" not in pat and "?" not in pat:
        pat = f"*{pat}*"
    found: List[Dict[str, Any]] = []
    base_depth = len(base.parts)
    for dirpath, dirnames, filenames in os.walk(base):
        depth = len(Path(dirpath).parts) - base_depth
        if depth > max_depth:
            dirnames[:] = []
            continue
        dirnames[:] = [d for d in dirnames if d not in {".git", "node_modules", "__pycache__", ".venv"}]
        for name in filenames:
            if not fnmatch.fnmatch(name.lower(), pat.lower()):
                continue
            if extension and not name.lower().endswith(extension.lower().lstrip(".")):
                continue
            full = Path(dirpath) / name
            try:
                st = full.stat()
                found.append({"path": str(full), "name": name, "bytes": st.st_size,
                              "modified": time.strftime("%Y-%m-%d %H:%M", time.localtime(st.st_mtime))})
            except OSError:
                continue
            if len(found) >= min(int(limit), MAX_RESULTS):
                break
        if len(found) >= min(int(limit), MAX_RESULTS):
            break
    if not found:
        return SkillResult(ok=True, value=f"לא מצאתי קבצים התואמים ל־{pat} מתחת ל־{base}.",
                           data={"count": 0, "files": []})
    lines = [f"{f['name']} ({_mb(f['bytes'])}, {f['modified']})" for f in found[:8]]
    more = f" ועוד {len(found) - 8}" if len(found) > 8 else ""
    return SkillResult(ok=True, value=f"מצאתי {len(found)} קבצים: {', '.join(lines)}{more}.",
                       data={"count": len(found), "files": found})


@REGISTRY.register(
    "fs.tree", risk="SAFE", agent="argus",
    description_he="מציג את עץ התיקיות והקבצים",
)
def fs_tree(path: str = ".", depth: int = 2, limit: int = 120) -> SkillResult:
    base = _resolve(path)
    if not base.exists():
        return SkillResult(ok=False, error=f"no such path: {base}")
    lines: List[str] = [base.name + "/"]
    count = 0

    def walk(d: Path, prefix: str, level: int) -> None:
        nonlocal count
        if level > depth or count >= limit:
            return
        try:
            entries = sorted(d.iterdir(), key=lambda e: (not e.is_dir(), e.name.lower()))
        except OSError:
            return
        entries = [e for e in entries if e.name not in {".git", "node_modules", "__pycache__", ".venv"}]
        for i, e in enumerate(entries):
            if count >= limit:
                return
            last = i == len(entries) - 1
            branch = "└── " if last else "├── "
            suffix = "/" if e.is_dir() else f" ({_mb(e.stat().st_size)})"
            lines.append(prefix + branch + e.name + suffix)
            count += 1
            if e.is_dir():
                walk(e, prefix + ("    " if last else "│   "), level + 1)

    walk(base, "", 1)
    return SkillResult(ok=True, value="\n".join(lines),
                       data={"count": count, "root": str(base)})


@REGISTRY.register(
    "fs.stat", risk="SAFE", agent="argus",
    description_he="פרטי קובץ: גודל, תאריכים, הרשאות, hash",
    required=("path",),
)
def fs_stat(path: str = "", hash_it: bool = True) -> SkillResult:
    p = _resolve(path)
    if not p.exists():
        return SkillResult(ok=False, error=f"no such path: {p}")
    st = p.stat()
    data: Dict[str, Any] = {
        "path": str(p), "is_dir": p.is_dir(), "bytes": st.st_size, "size": _mb(st.st_size),
        "modified": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(st.st_mtime)),
        "created": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(st.st_ctime)),
        "mode": oct(st.st_mode)[-3:],
    }
    if hash_it and p.is_file() and st.st_size < 64 * 1024 * 1024:
        h = hashlib.sha256()
        with p.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        data["sha256"] = h.hexdigest()[:32]
    return SkillResult(ok=True,
                       value=f"{p.name}: {_mb(st.st_size)}, נערך לאחרונה {data['modified']}.",
                       data=data)


@REGISTRY.register(
    "fs.grep", risk="SAFE", agent="argus",
    description_he="מחפש טקסט בתוך קבצים ומחזיר את השורות התואמות",
    required=("query",),
)
def fs_grep(query: str = "", root: str = ".", extension: str = "", limit: int = 40,
            ignore_case: bool = True) -> SkillResult:
    base = _resolve(root)
    if not base.is_dir():
        return SkillResult(ok=False, error=f"{base} is not a directory")
    q = str(query)
    needle = q.lower() if ignore_case else q
    hits: List[Dict[str, Any]] = []
    for dirpath, dirnames, filenames in os.walk(base):
        dirnames[:] = [d for d in dirnames if d not in {".git", "node_modules", "__pycache__", ".venv"}]
        for name in filenames:
            if extension and not name.lower().endswith(extension.lower().lstrip(".")):
                continue
            full = Path(dirpath) / name
            if full.stat().st_size > 4 * 1024 * 1024:
                continue
            try:
                text = full.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            for lineno, line in enumerate(text.splitlines(), 1):
                hay = line.lower() if ignore_case else line
                if needle in hay:
                    hits.append({"file": str(full), "line": lineno, "text": line.strip()[:200]})
                    if len(hits) >= limit:
                        break
            if len(hits) >= limit:
                break
        if len(hits) >= limit:
            break
    if not hits:
        return SkillResult(ok=True, value=f"לא מצאתי את {q!r} בקבצים מתחת ל־{base}.",
                           data={"count": 0, "hits": []})
    return SkillResult(ok=True,
                       value=f"מצאתי {len(hits)} התאמות ל־{q!r}. הראשונה: {Path(hits[0]['file']).name} שורה {hits[0]['line']}.",
                       data={"count": len(hits), "hits": hits})


@REGISTRY.register(
    "fs.disk_usage", risk="SAFE", agent="argus",
    description_he="ניצול דיסק לפי תיקייה",
)
def fs_disk_usage(path: str = ".") -> SkillResult:
    base = _resolve(path)
    total, used, free = shutil.disk_usage(str(base if base.exists() else Path.home()))
    return SkillResult(ok=True,
                       value=f"הדיסק: {_mb(total)} סה״כ, {_mb(used)} בשימוש, {_mb(free)} פנוי "
                             f"({round(used / total * 100, 1)} אחוז תפוס).",
                       data={"total": total, "used": used, "free": free,
                             "percent": round(used / total * 100, 2)})


def _mb(n: float) -> str:
    return _human(n)


def _human(num: float) -> str:
    for step in ("B", "KB", "MB", "GB", "TB"):
        if abs(num) < 1024:
            return f"{num:.0f}{step}" if step == "B" else f"{num:.1f}{step}"
        num /= 1024
    return f"{num:.1f}PB"
