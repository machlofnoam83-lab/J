#!/usr/bin/env python3
"""Build JARVIS-full.zip — the whole system, nothing fetched, nothing external.

The HUD has always advertised a download button. It pointed at a file nobody
built, so clicking it produced a 404 wearing a feature's clothes. This is the
thing behind it: one zip carrying the code, the trained models, the voicebank,
the STT template bank and the HUD, so a person can walk away with the system
they are looking at.

What is left out, and why:
  .git/, .venv/, node_modules/, __pycache__  — the checkout and its interpreter,
    not the product; both are rebuilt by setup_windows.bat.
  data/  — this machine's runtime state: transcripts, memory database, sandbox
    scratch, audit trail. Shipping it would hand over a conversation history and
    a memory that belong to whoever ran it.
  dist/  — the output directory itself.

Two callers, one code path: `build_bundle()` returns the counts and checksum so
the HUD can state the real size before a byte moves, and the CLI writes a file
for offline use (`python tools/make_bundle.py`).

Usage:
    python tools/make_bundle.py [output.zip]
"""

from __future__ import annotations

import hashlib
import os
import sys
import zipfile
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]
NAME = "JARVIS-full.zip"

# Directories that never belong in a bundle, whatever they contain.
_SKIP_DIRS = {
    ".git", ".venv", "venv", "node_modules", "__pycache__",
    ".mypy_cache", ".pytest_cache", ".ruff_cache", ".cache", ".arena",
    "dist", "data", ".next", ".turbo", "build", "out", "target",
    "voicebank_raw",
}
# Suffixes that are build residue rather than product.
_SKIP_SUFFIX = {".pyc", ".pyo", ".log", ".tmp", ".swp", ".orig", ".rej"}


def iter_shippable(root: Path):
    """Every file worth shipping, in stable order."""
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in _SKIP_DIRS
                             and not d.endswith(".egg-info"))
        for fn in sorted(filenames):
            p = Path(dirpath) / fn
            if p.suffix.lower() in _SKIP_SUFFIX:
                continue
            if p.is_symlink() or not p.is_file():
                continue
            yield p


def build_bundle(dest: Path | str | None = None, root: Path | None = None) -> Dict[str, Any]:
    """Zip the project. Returns stats; writes the file when `dest` is given."""
    root = Path(root or ROOT)
    files = list(iter_shippable(root))
    raw = sum(p.stat().st_size for p in files)

    import io
    sink: Any = io.BytesIO()
    with zipfile.ZipFile(sink, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for p in files:
            rel = p.relative_to(root).as_posix()
            # Everything lands under one folder, so unpacking never scatters
            # sixty files across a desktop.
            zf.write(p, f"JARVIS/{rel}")
        entries = zf.namelist()

    if isinstance(sink, io.BytesIO):
        blob = sink.getvalue()
    else:  # pragma: no cover — defensive
        blob = sink.read()

    out = Path(dest) if dest else None
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(blob)

    return {
        "name": NAME,
        "path": str(out) if out else None,
        "bytes": len(blob),
        "mb": round(len(blob) / (1024 * 1024), 1),
        "files": len(entries),
        "uncompressed": raw,
        "ratio": round(len(blob) / raw, 3) if raw else 0.0,
        "sha256": hashlib.sha256(blob).hexdigest(),
    }


def bundle_stats(root: Path | None = None) -> Dict[str, Any]:
    """What the bundle would contain — no compression, no zip, no waiting.

    The HUD calls this to say the truth about size up front. Building the real
    archive to measure it would cost seconds on every page load.
    """
    root = Path(root or ROOT)
    files = list(iter_shippable(root))
    total = sum(p.stat().st_size for p in files)
    by_top: Dict[str, int] = {}
    for p in files:
        rel = p.relative_to(root).as_posix()
        top = rel.split("/")[0] if "/" in rel else "(שורש)"
        by_top[top] = by_top.get(top, 0) + p.stat().st_size
    biggest = sorted(by_top.items(), key=lambda kv: -kv[1])[:8]
    return {
        "name": NAME,
        "files": len(files),
        "bytes": total,
        "mb": round(total / (1024 * 1024), 1),
        "by_area": [{"area": k, "mb": round(v / (1024 * 1024), 2)} for k, v in biggest],
        "excluded": sorted(_SKIP_DIRS),
    }


def main(argv: list[str]) -> int:
    dest = Path(argv[1]) if len(argv) > 1 else ROOT / "dist" / NAME
    st = bundle_stats()
    print(f"[*] {st['files']} files, {st['mb']} MB uncompressed")
    print(f"[*] excluded: {', '.join(sorted(_SKIP_DIRS))}")
    rep = build_bundle(dest)
    print(f"[✓] {dest}  ({rep['mb']} MB, {rep['files']} entries)")
    print(f"[✓] sha256 {rep['sha256'][:16]}…")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
