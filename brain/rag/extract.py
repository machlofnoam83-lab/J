"""Local-file ingestion: discovery, encoding repair, binary rejection, policy.

Why this file exists and why it is not three lines of ``read_text``
-------------------------------------------------------------------
The user runs JARVIS on **Windows, in Hebrew**. That single fact breaks naive
ingestion in three measurable ways, and each one is handled here explicitly:

1. **Encoding.** A Hebrew ``.txt``/``.log`` saved by Notepad or by an old
   installer is usually *cp1255*, not UTF-8. Reading it as UTF-8 raises, and
   reading it as ``latin-1`` "succeeds" while turning every letter into mojibake
   that the retriever can never match. So we try a chain and *score* the result
   by how much real Hebrew it contains, instead of taking the first decode that
   does not throw.

2. **Binary files.** Indexing a ``.png`` or a ``.sqlite3`` produces garbage
   chunks that then out-score real prose on n-gram noise. We reject by content
   (NUL bytes / non-text byte ratio), not by extension alone, because extensions
   lie in both directions.

3. **Blast radius.** A RAG indexer that happily walks the whole disk will read
   ``~/.ssh/id_rsa`` and ``.env`` files and copy their contents into a SQLite
   index that the user may later zip and share. The deny-list in
   :class:`RagPolicy` is therefore a hard gate checked *before* a single byte is
   read, and it is unit-tested in ``tests/test_rag.py``.
"""

from __future__ import annotations

import fnmatch
import hashlib
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from brain.tokenizer import normalize  # noqa: E402

# ------------------------------------------------------------------ policy ---
#: Directories never descended into, whatever the user asks for.
HARD_DENY_DIRS = frozenset({
    ".git", ".hg", ".svn", ".venv", "venv", "env", "node_modules", "__pycache__",
    ".mypy_cache", ".pytest_cache", ".ruff_cache", ".tox", ".nox", ".gradle",
    ".terraform", ".idea", ".vscode", "site-packages", "AppData", "$RECYCLE.BIN",
    "System Volume Information", "Windows", "$WinREAgent", "Recovery",
    ".cache", ".npm", ".cargo", ".rustup", "dist-packages",
})

#: File names / patterns never read. These are the ones that actually hurt:
#: private keys, credential stores, browser databases, wallet files.
HARD_DENY_PATTERNS = (
    "id_rsa", "id_dsa", "id_ecdsa", "id_ed25519", "authorized_keys", "known_hosts",
    "*.pem", "*.key", "*.p12", "*.pfx", "*.jks", "*.keystore",
    ".env", ".env.*", "*.env", "*.env.*", "*.netrc", "_netrc", ".netrc",
    "credentials", "credentials.json",
    ".git-credentials", ".gitconfig", ".npmrc", ".pypirc", ".aws/credentials",
    "*.sqlite-wal", "*.sqlite-shm", "cookies.sqlite", "cookies.txt",
    "logins.json", "key3.db", "key4.db", "cert8.db", "cert9.db",
    "wallet.dat", "*.kdbx", "*.gpg", "*.asc",
    "sam", "security", "system", "ntds.dit",
)

#: What we can actually read. ``kind`` drives the chunking strategy.
TEXT_EXTENSIONS: Dict[str, str] = {
    # prose
    ".txt": "prose", ".md": "prose", ".markdown": "prose", ".rst": "prose",
    ".tex": "prose", ".adoc": "prose", ".rtf": "prose", ".org": "prose",
    ".html": "prose", ".htm": "prose", ".xml": "prose", ".xhtml": "prose",
    # code
    ".py": "code", ".pyi": "code", ".js": "code", ".mjs": "code", ".cjs": "code",
    ".ts": "code", ".tsx": "code", ".jsx": "code", ".java": "code", ".c": "code",
    ".h": "code", ".hpp": "code", ".cpp": "code", ".cc": "code", ".cs": "code",
    ".go": "code", ".rs": "code", ".rb": "code", ".php": "code", ".swift": "code",
    ".kt": "code", ".scala": "code", ".lua": "code", ".pl": "code", ".r": "code",
    ".sh": "code", ".bash": "code", ".zsh": "code", ".ps1": "code", ".bat": "code",
    ".cmd": "code", ".sql": "code", ".vue": "code", ".svelte": "code",
    # structured / data
    ".json": "data", ".jsonl": "data", ".yaml": "data", ".yml": "data",
    ".toml": "data", ".ini": "data", ".cfg": "data", ".conf": "data",
    ".csv": "data", ".tsv": "data", ".env.example": "data", ".gitignore": "data",
    ".srt": "data", ".vtt": "data", ".log": "prose", ".properties": "data",
}

#: Files whose *name* (not extension) marks them as text.
TEXT_BASENAMES = frozenset({
    "README", "LICENSE", "NOTICE", "Makefile", "Dockerfile", "CMakeLists.txt",
    "requirements.txt", "TODO", "CHANGELOG", ".gitignore", ".gitattributes",
    ".editorconfig", "package.json",
})

_HEBREW_RE = re.compile(r"[\u0590-\u05ff]")
# Control bytes that essentially never appear in a text file. Everything at or
# above 0x80 is *deliberately* allowed: UTF-8 Hebrew is encoded as 0xD7/0xD8
# followed by a continuation byte in 0x90–0xBF, so a rule that treated high bytes
# as "non-text" rejected every single Hebrew UTF-8 file in the repository.
_CONTROL_BYTES = frozenset(
    set(range(0x00, 0x09)) | {0x0B, 0x0C} | set(range(0x0E, 0x20)) | {0x7F}
)

#: Directories the app writes to itself. Indexing them lets JARVIS retrieve its
#: own earlier answers and present them as sources — a self-reinforcing loop that
#: looks like grounding and is not. Matched as *absolute* paths under the repo
#: root, so a ``Documents/data`` folder of the user's own is never touched.
SELF_FEEDBACK_PATHS: Tuple[Path, ...] = (ROOT / "data", ROOT / "logs")


class RagPolicy:
    """The gate between the indexer and the filesystem.

    Two layers: a *hard* deny-list that cannot be overridden (secrets, VCS
    internals, OS directories) and a *soft* user filter (include/exclude globs,
    size caps) supplied per indexing run.
    """

    def __init__(self, protected: Sequence[str] = ()) -> None:
        self.protected: Tuple[str, ...] = tuple(
            str(Path(p).expanduser()).replace("\\", "/").lower().rstrip("/") for p in protected
        )

    # ------------------------------------------------------------ hard gate --
    def deny_reason(self, path: Path) -> str:
        """'' when the path may be read, otherwise a human-readable reason."""
        try:
            resolved = str(path.resolve()).replace("\\", "/")
        except OSError:
            resolved = str(path).replace("\\", "/")
        low = resolved.lower()
        for part in path.parts:
            if part in HARD_DENY_DIRS:
                return f"inside protected directory '{part}'"
        name = path.name.lower()
        for pat in HARD_DENY_PATTERNS:
            if fnmatch.fnmatch(name, pat) or fnmatch.fnmatch(low, "*/" + pat):
                return f"matches secret pattern '{pat}'"
        for prot in self.protected:
            if low == prot or low.startswith(prot + "/"):
                return f"inside configured protected path '{prot}'"
        return ""

    def is_denied(self, path: Path) -> bool:
        return bool(self.deny_reason(path))

    # ----------------------------------------------------------- soft gate ---
    @staticmethod
    def is_text(path: Path) -> bool:
        if path.name in TEXT_BASENAMES or path.name.lower() in TEXT_BASENAMES:
            return True
        return path.suffix.lower() in TEXT_EXTENSIONS

    @staticmethod
    def kind(path: Path) -> str:
        return TEXT_EXTENSIONS.get(path.suffix.lower(), "prose")

    @staticmethod
    def matches_any(path: Path, patterns: Sequence[str]) -> bool:
        if not patterns:
            return False
        name = path.name
        posix = path.as_posix()
        for pat in patterns:
            if not pat:
                continue
            if fnmatch.fnmatch(name, pat) or fnmatch.fnmatch(posix, pat) \
                    or fnmatch.fnmatch(posix, "*/" + pat):
                return True
        return False


# ---------------------------------------------------------------- decoding ---
def _hebrew_score(text: str) -> float:
    """Fraction of letters that are Hebrew — the signal used to pick a codec."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for c in letters if _HEBREW_RE.match(c)) / len(letters)


def _mojibake_score(text: str) -> float:
    """How much the text looks like cp1255-read-as-latin1 garbage.

    Mojibake from a wrong 8-bit codec is dominated by Latin-1 supplement
    characters (U+0080–U+00FF) that no real Hebrew or English text contains.
    """
    if not text:
        return 0.0
    odd = sum(1 for c in text if 0x80 <= ord(c) <= 0xFF)
    return odd / max(1, len(text))


def decode_text(raw: bytes) -> Tuple[str, str]:
    """Decode arbitrary bytes into text, preferring real Hebrew over mojibake.

    Returns ``(text, encoding_used)``. Never raises: the last resort is
    ``utf-8`` with ``errors='replace'``, because a partially-garbled page is
    still more useful to the retriever than dropping the file silently.
    """
    if not raw:
        return "", "empty"
    if raw.startswith(b"\xef\xbb\xbf"):
        raw = raw[3:]
        try:
            return raw.decode("utf-8"), "utf-8-sig"
        except UnicodeDecodeError:
            pass
    elif raw.startswith(b"\xff\xfe") or raw.startswith(b"\xfe\xff"):
        for enc in ("utf-16", "utf-16-le", "utf-16-be"):
            try:
                return raw.decode(enc), enc
            except UnicodeDecodeError:
                continue

    # 1. strict UTF-8 is unambiguous when it works — no heuristic needed.
    try:
        return raw.decode("utf-8"), "utf-8"
    except UnicodeDecodeError:
        pass

    # 2. 8-bit candidates, ranked by how Hebrew the result looks. cp1255 is the
    #    Windows Hebrew codepage and by far the most likely non-UTF-8 encoding
    #    on this user's machine.
    best_text, best_enc, best_key = "", "latin-1", (-1.0, 1.0)
    for enc in ("cp1255", "cp1252", "iso-8859-8", "latin-1"):
        try:
            cand = raw.decode(enc)
        except (UnicodeDecodeError, LookupError):
            continue
        key = (_hebrew_score(cand), -_mojibake_score(cand))
        if key > best_key:
            best_text, best_enc, best_key = cand, enc, key
    if best_text:
        return best_text, best_enc

    # 3. last resort — keep the file, mark the damage.
    return raw.decode("utf-8", errors="replace"), "utf-8-replace"


def looks_binary(sample: bytes, text: Optional[str] = None) -> bool:
    """Content-based binary detection (extensions lie; bytes do not).

    Three independent signals, all cheap:

    * a NUL byte in the first 8 KB — no text format contains one;
    * a high density of *control* bytes (not high bytes: UTF-8 Hebrew lives
      above 0x80 and must not be penalised for it);
    * a successful decode that still yields mostly U+FFFD, i.e. bytes that were
      not text in any encoding we know.
    """
    if not sample:
        return False
    head = sample[:8192]
    if b"\x00" in head:
        return True
    controls = sum(1 for b in head if b in _CONTROL_BYTES)
    if controls / max(1, len(head)) > 0.05:
        return True
    if text is not None:
        if text and text.count("\ufffd") / max(1, len(text)) > 0.10:
            return True
    return False


# --------------------------------------------------------------- extraction --
@dataclass
class ExtractResult:
    """Outcome of reading one file. ``skipped`` is always honest about *why*."""

    path: Path
    text: str = ""
    encoding: str = ""
    kind: str = "prose"
    bytes: int = 0
    sha1: str = ""
    mtime: float = 0.0
    lines: int = 0
    skipped: str = ""
    truncated: bool = False
    warnings: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.skipped and bool(self.text)

    def to_dict(self) -> Dict[str, object]:
        return {
            "path": self.path.as_posix(), "encoding": self.encoding, "kind": self.kind,
            "bytes": self.bytes, "sha1": self.sha1, "lines": self.lines,
            "skipped": self.skipped, "truncated": self.truncated, "ok": self.ok,
        }


def extract_file(path: Path, *, max_bytes: int = 4 * 1024 * 1024,
                 policy: Optional[RagPolicy] = None) -> ExtractResult:
    """Read one file into normalised text, or explain exactly why we did not."""
    path = Path(path)
    policy = policy or RagPolicy()
    res = ExtractResult(path=path, kind=policy.kind(path))

    deny = policy.deny_reason(path)
    if deny:
        res.skipped = f"denied: {deny}"
        return res
    if not path.exists():
        res.skipped = "missing"
        return res
    if not path.is_file():
        res.skipped = "not a regular file"
        return res
    try:
        st = path.stat()
    except OSError as exc:
        res.skipped = f"stat failed: {exc}"
        return res
    res.bytes = st.st_size
    res.mtime = st.st_mtime
    if res.bytes == 0:
        res.skipped = "empty file"
        return res
    if res.bytes > max_bytes:
        res.skipped = f"too large ({res.bytes} bytes > {max_bytes})"
        return res

    try:
        with path.open("rb") as fh:
            head = fh.read(65536)
            if looks_binary(head):
                res.skipped = "binary content"
                return res
            fh.seek(0)
            raw = fh.read(max_bytes + 1)
    except OSError as exc:
        res.skipped = f"read failed: {exc}"
        return res

    if len(raw) > max_bytes:
        raw = raw[:max_bytes]
        res.truncated = True
        res.warnings.append(f"truncated to {max_bytes} bytes")

    text, enc = decode_text(raw)
    res.encoding = enc
    if looks_binary(raw[:65536], text):
        res.skipped = "binary content"
        return res

    text = normalize(text)                    # folds final forms, strips niqqud
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Collapse pathological runs of blank lines (log files, generated dumps).
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    if not text.strip():
        res.skipped = "no readable text"
        return res

    res.text = text
    res.lines = text.count("\n") + 1
    res.sha1 = hashlib.sha1(raw).hexdigest()
    return res


def _is_self_feedback(path: Path) -> bool:
    """True for the app's own ``data/`` and ``logs/`` under the repository root."""
    try:
        p = path.resolve()
    except OSError:
        return False
    for base in SELF_FEEDBACK_PATHS:
        try:
            p.relative_to(base.resolve())
            return True
        except (ValueError, OSError):
            continue
    return False


def walk_files(roots: Sequence[Path | str], *,
               include: Sequence[str] = (),
               exclude: Sequence[str] = (),
               policy: Optional[RagPolicy] = None,
               max_bytes: int = 4 * 1024 * 1024,
               max_files: int = 20_000,
               allow_self_feedback: bool = False,
               follow_symlinks: bool = False) -> Iterator[Path]:
    """Yield readable text files under ``roots``, honouring deny-list and globs.

    Directory pruning happens *before* descent, so ``node_modules`` costs one
    ``scandir`` rather than 40,000 ``stat`` calls — the difference between an
    indexer that finishes and one that does not.
    """
    policy = policy or RagPolicy()
    block_self = not allow_self_feedback
    seen: set = set()
    yielded = 0
    stack: List[Path] = [Path(r).expanduser() for r in roots]
    while stack:
        current = stack.pop()
        try:
            current = current.resolve()
        except OSError:
            continue
        key = str(current)
        if key in seen:
            continue
        seen.add(key)
        if not current.exists():
            continue
        if current.is_file():
            if policy.is_text(current) and not policy.is_denied(current) \
                    and not (block_self and _is_self_feedback(current)) \
                    and not RagPolicy.matches_any(current, exclude) \
                    and (not include or RagPolicy.matches_any(current, include)):
                yield current
                yielded += 1
                if yielded >= max_files:
                    return
            continue
        if block_self and _is_self_feedback(current):
            # Prune the app's own output before descending: one scandir saved is
            # 10,000 files we will never have to stat.
            continue
        try:
            entries = list(current.iterdir())
        except (OSError, PermissionError):
            continue
        for entry in entries:
            try:
                is_link = entry.is_symlink()
                if is_link and not follow_symlinks:
                    continue
                if entry.is_dir():
                    if entry.name in HARD_DENY_DIRS or policy.is_denied(entry):
                        continue
                    if block_self and _is_self_feedback(entry):
                        continue
                    if RagPolicy.matches_any(entry, exclude):
                        continue
                    stack.append(entry)
                elif entry.is_file():
                    if not policy.is_text(entry) or policy.is_denied(entry):
                        continue
                    if block_self and _is_self_feedback(entry):
                        continue
                    if RagPolicy.matches_any(entry, exclude):
                        continue
                    if include and not RagPolicy.matches_any(entry, include):
                        continue
                    try:
                        if entry.stat().st_size > max_bytes:
                            continue
                    except OSError:
                        continue
                    yield entry
                    yielded += 1
                    if yielded >= max_files:
                        return
            except OSError:
                continue


def file_sha1(path: Path, limit: int = 4 * 1024 * 1024) -> str:
    """Cheap content hash used to decide whether a document must be re-indexed."""
    h = hashlib.sha1()
    try:
        with path.open("rb") as fh:
            h.update(fh.read(limit))
    except OSError:
        return ""
    return h.hexdigest()


def human_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f}{unit}" if unit != "B" else f"{n}B"
        n /= 1024  # type: ignore[assignment]
    return f"{n}B"


if __name__ == "__main__":  # pragma: no cover - manual probe
    pol = RagPolicy()
    roots = [ROOT]
    n = 0
    for p in walk_files(roots, policy=pol):
        r = extract_file(p, policy=pol)
        n += 1
        if n <= 12:
            print(f"{'OK ' if r.ok else 'SKIP'} {r.path.name:<28} {r.encoding:<12} "
                  f"{r.lines:>6} lines  {r.skipped}")
    print(f"\n{n} candidate text files under {ROOT}")
