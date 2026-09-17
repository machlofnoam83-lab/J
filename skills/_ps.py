"""Safe PowerShell invocation: user data never enters the script text.

Every place JARVIS shells out to PowerShell on Windows used to build the script
with an f-string, dropping a user-supplied value straight into a quoted literal:

    f"$bmp.Save('{out}'); ..."
    f"...CreateTextNode('{title}')); ...CreateTextNode('{msg}')..."

A value containing a single quote closes the literal and the rest is parsed as
code. With `path="x.png'); Remove-Item -Recurse -Force C:\\Users; ('"` the
destructive statement becomes part of the script, and because these skills are
graded WRITE and SAFE the firewall passes them without a confirmation prompt —
SAFE never prompts at all. Escaping the quotes would have closed that instance,
but it leaves the design one forgotten `.replace()` away from the next one, and
escaping for PowerShell has edges (embedded newlines, `list2cmdline` quoting when
the argument reaches `CreateProcess`) that are easy to get subtly wrong.

So this module takes the other approach: the value travels in the child
process's environment and the script reads it back as `$env:NAME`. Environment
variables are passed as structured data to `subprocess`, never spliced into text
that a parser will re-read, so there is no quoting to get wrong and nothing for
an attacker to terminate. The script body is then a constant the caller writes
once.

Use `run_ps` for anything that needs a user-supplied value. If a script genuinely
has no variable input, calling `subprocess.run` directly is still fine.
"""

from __future__ import annotations

import os
import subprocess
from typing import Any, Dict, Mapping, Optional, Tuple

__all__ = ["run_ps", "ps_available", "reject_control_chars", "PsError"]


class PsError(RuntimeError):
    """Raised for a value that cannot be carried safely to PowerShell."""


def ps_available() -> bool:
    """True if a PowerShell interpreter is on PATH."""
    from shutil import which
    return bool(which("powershell") or which("pwsh"))


def _interpreter() -> str:
    """Full path to a PowerShell interpreter, or "" when there is none.

    Returns "" rather than defaulting to "pwsh": naming an interpreter that is
    not installed turns a missing dependency into a confusing FileNotFoundError
    from subprocess, and callers on non-Windows platforms have no PowerShell at
    all — which is a normal condition, not an error to be discovered late.
    """
    from shutil import which
    return which("powershell") or which("pwsh") or ""


def reject_control_chars(value: str, label: str = "value") -> str:
    """Refuse NUL and line breaks in data headed for the child process.

    A NUL truncates paths at the OS layer, and a newline inside an environment
    value is the kind of thing that turns into two commands on some shells.
    Neither belongs in a filename, a notification title or a body, so refusing is
    not a restriction anyone would notice — it just fails loudly instead of
    producing a mangled or split value downstream.
    """
    s = str(value)
    for bad, name in (("\x00", "NUL"), ("\n", "newline"), ("\r", "carriage return")):
        if bad in s:
            raise PsError(f"{label} contains a {name}, which cannot be passed safely")
    return s


def run_ps(script: str, env: Optional[Mapping[str, str]] = None,
           timeout: float = 20.0) -> Tuple[int, str, str]:
    """Run a constant PowerShell script, with `env` exposed as `$env:*` inside it.

    `script` must be a literal written by the caller. Anything that varies per
    request goes in `env`, not in `script` — that separation is the whole point
    of this module, and interpolating into `script` reintroduces the injection.

    Returns (returncode, stdout, stderr). Never raises on a non-zero exit; the
    caller decides what a failure means. Raises PsError if PowerShell is absent
    or if an env value carries a control character — a clear failure beats a raw
    FileNotFoundError from deep in subprocess.
    """
    if env:
        for k, v in env.items():
            reject_control_chars(v, f"env[{k}]")

    exe = _interpreter()
    if not exe:
        raise PsError("no PowerShell interpreter on PATH (looked for powershell, pwsh)")

    child: Dict[str, str] = dict(os.environ)
    if env:
        child.update({str(k): str(v) for k, v in env.items()})

    proc = subprocess.run(
        [exe, "-NoProfile", "-NonInteractive", "-Command", script],
        capture_output=True, text=True, errors="replace", timeout=timeout, env=child)
    return proc.returncode, proc.stdout or "", proc.stderr or ""


def ps_env_path(script: str, path_var: str, path: Any,
                timeout: float = 20.0) -> Tuple[int, str, str]:
    """Convenience wrapper: one filesystem path, exposed as `$env:<path_var>`.

    Paths are the most common thing to hand PowerShell — `Bitmap.Save`, `Get-Item`,
    redirection targets — and they are also the most likely to contain a quote,
    since people name files things like `O'Brien's notes.png`.
    """
    return run_ps(script, {path_var: str(path)}, timeout=timeout)
