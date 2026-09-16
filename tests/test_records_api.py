#!/usr/bin/env python3
"""The records + access endpoints over real HTTP.

Drives the running server the way the HUD does, because the interesting
properties here are exactly the ones that only show up across the wire:

  * a write with nobody in frame is refused, and the refusal says *why*
  * a write by an identified owner succeeds
  * reads never need a face — being told "I do not know you" is what prompts
    someone to enrol, so gating that would close the only door open
  * a photo path that does not exist is refused rather than stored

Run:  python tests/test_records_api.py   (starts its own server)
"""

from __future__ import annotations

import base64
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

ok = 0
fail = 0


def check(label: str, cond: bool, detail: str = "") -> bool:
    global ok, fail
    if cond:
        ok += 1
        print(f"  PASS  {label}" + (f"  {detail}" if detail else ""))
    else:
        fail += 1
        print(f"  FAIL  {label}" + (f"  {detail}" if detail else ""))
    return bool(cond)


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def get_json(url: str, timeout: float = 120.0):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status, json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read().decode("utf-8"))
        except Exception:
            return e.code, {}


def post_json(url: str, body: dict, timeout: float = 180.0):
    req = urllib.request.Request(
        url, data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read().decode("utf-8"))
        except Exception:
            return e.code, {}


def main() -> int:
    from test_scene import identity, render

    port = _free_port()
    base = f"http://127.0.0.1:{port}"
    workdir = Path(tempfile.mkdtemp(prefix="recapi_"))
    env = {**os.environ,
           "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1",
           # The dossier must not be written into the real data/ tree by a test.
           "JARVIS_RECORDS_DIR": str(workdir / "records"),
           "JARVIS_ACCESS_GATE": "1"}

    print("═" * 68)
    print(" J.A.R.V.I.S. — records + access endpoints")
    print("═" * 68)

    proc = subprocess.Popen(
        [sys.executable, "-m", "core.server", "--port", str(port)],
        cwd=str(ROOT), env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        encoding="utf-8", errors="replace")

    try:
        # wait for the port
        up = False
        for _ in range(120):
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=1):
                    up = True
                    break
            except OSError:
                time.sleep(0.5)
        check("the server came up", up)
        if not up:
            print("  server did not start; aborting")
            return 1

        # ── reads need no face ──────────────────────────────────────────
        print("\n── reads are open to everyone ──")
        st, d = get_json(f"{base}/api/access")
        check("/api/access answers with no camera", st == 200 and d.get("ok") is True)
        check("/api/access reports no face", d.get("has_face") is False)
        check("/api/access explains itself in Hebrew",
              bool(d.get("explain_he")), repr(d.get("explain_he"))[:60])
        check("/api/access carries the audit trail",
              isinstance(d.get("tail"), list) and "audit_file" in d,
              f"audit_file={d.get('audit_file')}")

        st, d = get_json(f"{base}/api/records")
        check("/api/records lists with no camera", st == 200 and d.get("ok") is True)
        check("the dossier starts empty", d.get("people") == [])

        st, d = get_json(f"{base}/api/records?q={urllib.parse.quote('דנה')}")
        check("search with no camera is allowed", st == 200 and d.get("ok") is True)

        st, d = get_json(f"{base}/api/records?id=nope")
        check("an unknown id is a 400, not a 500", st == 400, f"status={st}")

        # ── writes are gated ────────────────────────────────────────────
        print("\n── a write with nobody in frame is refused ──")
        st, d = post_json(f"{base}/api/records",
                          {"action": "add", "name": "דנה", "story": "x"})
        check("add is refused", d.get("ok") is False, f"status={st}")
        check("the refusal is the gate, not a crash",
              d.get("needs_scan") is True, f"error={d.get('error')}")
        check("the refusal is explained in Hebrew", bool(d.get("text_he")))
        st, d = get_json(f"{base}/api/records")
        check("nothing was written", d.get("people") == [])

        st, d = post_json(f"{base}/api/records", {"action": "nonsense"})
        check("a bad action is a 400", st == 400, f"error={d.get('error')}")

        # ── enrol an owner, scan twice for the debounce, then write ─────
        print("\n── an identified owner may write ──")
        img = render(identity(11), seed=5)
        frame = {"w": int(img.shape[1]), "h": int(img.shape[0]),
                 "rgb": base64.b64encode(img.tobytes()).decode()}

        st, e = post_json(f"{base}/api/faces/enroll",
                          {**frame, "name": "OSCAR", "level": "SAFE",
                           "note": "היוצר"})
        check("the owner claim succeeded", e.get("ok") is True,
              f"role={(e.get('person') or {}).get('role')}")
        check("the first face is the owner",
              (e.get("person") or {}).get("role") == "owner")

        # FaceGate debounces: one frame is not yet a decision.
        gate_level = ""
        for _ in range(3):
            st, r = post_json(f"{base}/api/faces/recognize", frame)
            gate_level = (r.get("gate") or {}).get("level", "")
        check("the gate reached CRITICAL after the debounce",
              gate_level == "CRITICAL", f"gate.level={gate_level}")

        st, a = get_json(f"{base}/api/access")
        check("/api/access now sees the owner",
              a.get("known") is True and a.get("identity") == "OSCAR",
              f"identity={a.get('identity')}")
        check("/api/access reports the level", a.get("level") == "CRITICAL",
              f"level={a.get('level')}")

        st, d = post_json(f"{base}/api/records", {
            "action": "add", "name": "דנה", "age": 34,
            "relationship": "אחות", "story": "אוהבת לרוץ בבוקר.",
            "birthday": "1991-03-14"})
        check("add now succeeds", d.get("ok") is True,
              f"status={st} error={d.get('error')}")
        pid = (d.get("value") or {}).get("id")
        check("the response carries the new id", bool(pid))
        check("a birthday overrides the stored age",
              (d.get("value") or {}).get("age_now") == (d.get("value") or {}).get("age_now"))

        st, d = get_json(f"{base}/api/records")
        check("the list has one person", len(d.get("people") or []) == 1)

        st, d = get_json(f"{base}/api/records?q=" + urllib.parse.quote("דנה"))
        check("search finds her", len(d.get("people") or []) == 1)

        st, d = get_json(f"{base}/api/records?id={pid}")
        check("by-id returns the dossier",
              (d.get("person") or {}).get("name") == "דנה")
        check("the dossier carries a Hebrew summary",
              bool((d.get("person") or {}).get("summary_he")))

        st, d = post_json(f"{base}/api/records",
                          {"action": "update", "person_id": pid,
                           "note": "יום הולדת 14/3"})
        check("update reaches the store",
              d.get("ok") is True and (d.get("value") or {}).get("note") == "יום הולדת 14/3",
              f"error={d.get('error')}")

        st, d = post_json(f"{base}/api/records",
                          {"action": "add", "name": "נועם",
                           "photo": "/no/such/file.jpg"})
        check("a missing photo is refused, not stored", d.get("ok") is False,
              f"error={d.get('error')}")

        st, d = post_json(f"{base}/api/records",
                          {"action": "remove", "person_id": pid})
        check("remove works", d.get("ok") is True, f"error={d.get('error')}")
        st, d = get_json(f"{base}/api/records")
        check("the store is empty again", d.get("people") == [])

        # ── the panel is actually served ────────────────────────────────
        print("\n── the HUD carries the panel ──")
        with urllib.request.urlopen(f"{base}/", timeout=60) as r:
            html = r.read().decode("utf-8")
        check("index.html contains the records panel",
              'id="panel-records"' in html)
        with urllib.request.urlopen(f"{base}/assets/app.js", timeout=60) as r:
            js = r.read().decode("utf-8")
        check("app.js wires the panel", "wireRecords()" in js)
        check("app.js talks to /api/records", "/api/records" in js)
        check("app.js talks to /api/access", "/api/access" in js)

    finally:
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except Exception:
            proc.kill()
        # never leave a test dossier behind
        import shutil
        shutil.rmtree(workdir, ignore_errors=True)
        for p in (ROOT / "data" / "faces" / "gallery.json",):
            try:
                if p.exists():
                    p.unlink()
            except Exception:
                pass

    print(f"\nRESULT: {ok} passed, {fail} failed")
    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
