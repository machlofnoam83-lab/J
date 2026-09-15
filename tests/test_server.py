#!/usr/bin/env python3
"""JARVIS server tests — the bridge between the brain and the HUD.

Starts the real server on a free port as a subprocess and drives it over
HTTP + WebSocket exactly the way the Electron HUD does.

Covers: static HUD serving · CORS for the file:// renderer · every JSON API ·
WebSocket chat turn · audio frames reaching the UI · the human-in-the-loop
CRITICAL permission prompt answered over the same socket · kill switch ·
bus event streaming.

Run:  python tests/test_server.py
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

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


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def get(url: str, timeout: float = 180.0):
    req = urllib.request.Request(url, headers={"Origin": "null"})   # Electron file:// origin
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = r.read()
        return r.status, dict(r.headers), body


def get_json(url: str, timeout: float = 180.0):
    status, headers, body = get(url, timeout)
    return status, headers, json.loads(body.decode("utf-8"))


def post(url: str, data: bytes = b"", ctype: str = "application/octet-stream", timeout: float = 240.0):
    req = urllib.request.Request(url, data=data, method="POST",
                                 headers={"Content-Type": ctype, "Origin": "null"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, dict(r.headers), r.read()
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), e.read()


async def ws_session(base_ws: str, script, timeout: float = 240.0):
    """Connect, then run `script(ws, log)`; returns the collected message log."""
    import websockets
    log: list[dict] = []
    async with websockets.connect(base_ws, max_size=32 * 1024 * 1024, open_timeout=60) as ws:
        await script(ws, log)
    return log


def wait_for(log: list[dict], pred, timeout: float = 60.0):
    """Poll a growing log until pred(msg) is true."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        for m in log:
            if pred(m):
                return m
        time.sleep(0.05)
    return None


def main() -> int:
    port = free_port()
    base = f"http://127.0.0.1:{port}"
    base_ws = f"ws://127.0.0.1:{port}/ws"

    proc = subprocess.Popen(
        [sys.executable, "-m", "core.server", "--host", "127.0.0.1", "--port", str(port)],
        cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace",
        env={**__import__("os").environ, "PYTHONUNBUFFERED": "1"},
    )

    try:
        # ------------------------------------------------------------ startup --
        print("\n== startup ==")
        up = False
        t0 = time.time()
        while time.time() - t0 < 240:
            if proc.poll() is not None:
                out = proc.stdout.read() if proc.stdout else ""
                print(out[-2500:])
                break
            try:
                st, _, _ = get(f"{base}/api/status", timeout=200)
                if st == 200:
                    up = True
                    break
            except Exception:
                time.sleep(0.4)
        check("server comes up and answers /api/status", up, f"({time.time()-t0:.1f}s)")
        if not up:
            return 1

        # --------------------------------------------------------------- HTTP --
        print("\n== HTTP API ==")
        st, hdr, body = get(f"{base}/")
        html = body.decode("utf-8", "replace")
        check("HUD page served", st == 200 and "J.A.R.V.I.S." in html, f"({len(body)} bytes)")
        check("HUD page declares Hebrew RTL", 'lang="he"' in html and 'dir="rtl"' in html)
        check("HUD page has a strict CSP", "Content-Security-Policy" in html)
        # the interactive surfaces added since the first HUD: command palette,
        # posture panel, screen analysis and the sentinel block. A missing one
        # means the page silently degrades to an older interface.
        for needle, label in [('id="palette"', "command palette overlay"),
                              ('id="mode-chips"', "posture panel"),
                              ('id="screen-read"', "screen analysis block"),
                              ('id="sentinel-box"', "sentinel block"),
                              ('id="boot-skip"', "boot overlay skip"),
                              ('id="jump-latest"', "jump to latest")]:
            check(f"HUD page carries the {label}", needle in html)
        # two panels both titled "חומת הרשאות" — one carrying a bar, one the
        # controls — repeated level, dry-run and the allow/block counts on the
        # same screen. The merge is pinned here so the duplication cannot creep
        # back in as "just a small addition".
        check("exactly one firewall panel is served",
              html.count('id="panel-firewall"') == 1
              and 'id="panel-permissions"' not in html
              and 'id="panel-security"' not in html)
        check("the firewall panel owns queue, controls, audit and sentinel",
              all(n in html for n in ('id="perm-queue"', 'id="sel-level"',
                                      'id="audit-stream"', 'id="sentinel-box"')))
        check("the redundant block-percentage bar is gone", 'id="bar-fw"' not in html)

        st, hdr, body = get(f"{base}/assets/style.css")
        check("stylesheet served", st == 200 and len(body) > 5000, f"({len(body)} bytes)")
        for name in ("app.js", "hud.js", "audio.js"):
            st, _, body = get(f"{base}/assets/{name}")
            check(f"{name} served", st == 200 and len(body) > 1000, f"({len(body)} bytes)")

        check("CORS allows the file:// renderer", hdr.get("Access-Control-Allow-Origin") == "*",
              str(hdr.get("Access-Control-Allow-Origin")))

        st, _, d = get_json(f"{base}/api/status")
        check("/api/status returns the full panel",
              all(k in d for k in ("brain", "voice", "security", "skills", "memory", "telemetry")),
              f"skills={d.get('skills')} voice={d.get('voice', {}).get('engine')}")

        st, _, d = get_json(f"{base}/api/skills")
        check("/api/skills lists the catalogue", d.get("count", 0) > 20 and d.get("skills"),
              f"({d.get('count')} skills)")
        first = (d.get("skills") or [{}])[0]
        check("skill schema is machine-readable", all(k in first for k in ("name", "risk", "args")), str(first)[:100])

        st, _, d = get_json(f"{base}/api/memory")
        check("/api/memory responds", st == 200 and "stats" in d, str(d.get("stats")))
        st, _, d = get_json(f"{base}/api/audit")
        check("/api/audit responds", st == 200 and "stats" in d and "tail" in d,
              f"level={d['stats'].get('level')} entries={d['stats'].get('total')}")

        st, _, d = get_json(f"{base}/api/sentinel")
        check("/api/sentinel reports the watchdog", st == 200 and "stats" in d and "tail" in d,
              f"alerts={d['stats'].get('alerts')} acting={d['stats'].get('acting')}")
        st, _, d = get_json(f"{base}/api/bundle")
        check("/api/bundle measures the download without building it",
              st == 200 and d.get("ok") and d.get("files", 0) > 300
              and d.get("bytes", 0) > 10 * 1024 * 1024,
              f"{d.get('files')} files {d.get('mb')}MB")
        check("the bundle excludes this machine's private state",
              all(x in (d.get("excluded") or []) for x in ("data", ".git", ".venv")),
              str(d.get("excluded"))[:60])

        st, hdr, body = get(f"{base}/api/download")
        check("/api/download serves a real zip",
              st == 200 and body[:2] == b"PK" and len(body) > 5 * 1024 * 1024,
              f"{len(body) // 1024}KB magic={body[:2]!r}")
        names = zipfile.ZipFile(io.BytesIO(body)).namelist() if body[:2] == b"PK" else []
        check("the zip carries the whole system under one folder",
              any(n == "JARVIS/core/server.py" for n in names)
              and any(n == "JARVIS/ui/index.html" for n in names)
              and any(n.startswith("JARVIS/models/") for n in names)
              and any(n.startswith("JARVIS/brain/voicebank/") for n in names),
              f"{len(names)} entries")
        check("the zip leaves the machine's transcripts and memory behind",
              not any(n.startswith("JARVIS/data/") for n in names))
        check("the old dead link is gone from the HUD",
              "/ui/JARVIS-full.zip" not in open(
                  os.path.join(HERE, "..", "ui", "assets", "app.js"),
                  encoding="utf-8").read())

        # The skip button was dead for the whole wait for the brain, because it
        # was wired inside the boot ceremony rather than before it. Pin the order.
        js = open(os.path.join(HERE, "..", "ui", "assets", "app.js"),
                  encoding="utf-8").read()
        wired = js.find("\n  wireBootSkip();")
        waited = js.find("wait for the brain")
        check("the boot overlay's skip button is wired before the brain wait",
              0 <= wired < waited, f"wired@{wired} wait@{waited}")

        st, _, d = get_json(f"{base}/api/modes")
        check("/api/modes lists the twelve postures", st == 200 and len(d.get("modes", [])) == 12,
              f"current={d.get('current')}")

        # ── /api/screen/read: the HTTP face of the in-house PNG codec ──────
        # A real PNG written by our own encoder, read back through the route.
        import numpy as _np
        from vision import png as _png
        _shot = _np.full((60, 90, 3), 240, dtype=_np.uint8)
        for _y in range(12, 50, 8):
            _shot[_y:_y + 3, 14:76] = (30, 30, 30)
        _p = ROOT / "data" / "screenshot_server_test.png"
        _p.parent.mkdir(parents=True, exist_ok=True)
        _png.write_file(_p, _shot)
        try:
            st, _, d = get_json(f"{base}/api/screen/read?path={_p}")
            check("/api/screen/read decodes a real PNG", st == 200 and d.get("ok") is True,
                  str(d.get("value") or d.get("error"))[:70])
            _dd = d.get("data") or {}
            check("...and reports the decoded dimensions, not the filename",
                  _dd.get("width") == 90 and _dd.get("height") == 60,
                  f"{_dd.get('width')}x{_dd.get('height')}")
            check("...and states plainly that no text was extracted",
                  _dd.get("text_extracted") is False and "OCR" in str(_dd.get("note", "")),
                  str(_dd.get("note"))[:56])
            check("...and measures the bright screen as bright",
                  (_dd.get("brightness_mean") or 0) > 0.5, str(_dd.get("brightness_mean")))

            st, _, d = get_json(f"{base}/api/screen/read?path={ROOT / 'requirements.txt'}")
            check("/api/screen/read refuses a non-image cleanly",
                  st == 200 and d.get("ok") is False and "PNG" in str(d.get("error", "")),
                  str(d.get("error"))[:60])

            st, _, d = get_json(f"{base}/api/screen/read?path={ROOT / 'data' / 'absent.png'}")
            check("/api/screen/read reports a missing file cleanly",
                  st == 200 and d.get("ok") is False and bool(d.get("error")),
                  str(d.get("error"))[:60])

            # Capture goes through the firewall with the agent's real level, so
            # it either gets denied by policy or fails because a headless sandbox
            # has no display to photograph. Either is acceptable; what matters is
            # that it comes back as a reported result with a reason attached,
            # never a 500 or a stack trace.
            st, _, d = get_json(f"{base}/api/screen/read?capture=1")
            _clean = (st == 200 and isinstance(d, dict) and "ok" in d
                      and d.get("stage") == "capture"
                      and (d.get("ok") is True or bool(d.get("error"))))
            check("/api/screen/read?capture=1 is clean when headless", _clean,
                  f"HTTP {st} ok={d.get('ok')} stage={d.get('stage')} "
                  f"risk={d.get('risk')} {str(d.get('error') or d.get('value'))[:40]}")
            check("...and a denial carries the firewall's own reason",
                  d.get("ok") is True or "reason" in d or "error" in d,
                  str(d.get("reason") or d.get("error"))[:56])
        finally:
            _p.unlink(missing_ok=True)
        st, _, d = get_json(f"{base}/api/permissions")
        check("/api/permissions responds", st == 200 and isinstance(d.get("pending"), list),
              f"timeout={d.get('timeout')}")
        st, _, d = get_json(f"{base}/api/events?n=20")
        check("/api/events responds", st == 200 and isinstance(d.get("events"), list),
              f"({len(d.get('events', []))} events)")
        st, _, d = get_json(f"{base}/api/history")
        check("/api/history responds", st == 200 and isinstance(d.get("turns"), list))

        st, _, d = get_json(f"{base}/api/boot")
        check("/api/boot runs the honest POST", st == 200 and len(d.get("report", [])) == 15,
              f"({sum(1 for r in d.get('report', []) if r['ok'])}/15 ok)")

        st, _, d = get_json(f"{base}/api/stt")
        check("/api/stt reports the recogniser", st == 200 and "commands" in d,
              f"(available={d.get('available')} commands={d.get('commands')} templates={d.get('templates')})")

        st, hdr, body = post(f"{base}/api/listen", b"")
        d = json.loads(body.decode("utf-8", "replace"))
        check("empty /api/listen is rejected cleanly", st == 200 and d.get("ok") is False, d.get("error", "")[:70])

        # the full voice loop: synthesise a command, post it as a WAV, get an answer
        if True:
            import io as _io
            import wave as _wave
            import numpy as _np
            sys.path.insert(0, str(ROOT))
            from voice.stt import COMMANDS as _CMDS
            from voice.tts import get_voice as _gv
            _r = _gv().synthesize(_CMDS["time"]["say"][0], rate=1.0)
            _pcm = (_np.clip(_np.asarray(_r.samples, dtype="float32"), -1, 1) * 32767).astype("<i2").tobytes()
            _buf = _io.BytesIO()
            with _wave.open(_buf, "wb") as _w:
                _w.setnchannels(1); _w.setsampwidth(2); _w.setframerate(_r.sample_rate)
                _w.writeframes(_pcm)
            st, _, body = post(f"{base}/api/listen?speak=0", _buf.getvalue())
            d = json.loads(body.decode("utf-8", "replace"))
            check("POST /api/listen recognises a spoken command",
                  st == 200 and d.get("ok") is True and d.get("label") == "time",
                  str({k: d.get(k) for k in ("ok", "label", "text", "confidence")})[:130])
            turn = d.get("turn") or {}
            check("the recognised command reached the brain",
                  bool(turn) and "שעה" in str((turn.get("answer") or {}).get("text", "")),
                  str((turn.get("answer") or {}).get("text", ""))[:90])

        # ------------------------------------------------------- websocket ----
        print("\n== WebSocket ==")

        async def hello_script(ws, log):
            raw = await asyncio.wait_for(ws.recv(), timeout=120)
            log.append(json.loads(raw))

        log = asyncio.run(ws_session(base_ws, hello_script))
        hello = log[0] if log else {}
        check("socket greets with hello + status", hello.get("type") == "hello" and "status" in hello,
              str(list(hello))[:80])

        async def ping_script(ws, log):
            log.append(json.loads(await asyncio.wait_for(ws.recv(), timeout=120)))  # hello
            await ws.send(json.dumps({"type": "ping"}))
            t_end = time.time() + 30
            while time.time() < t_end:
                msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=30))
                log.append(msg)
                if msg.get("type") == "pong":
                    break

        log = asyncio.run(ws_session(base_ws, ping_script))
        check("ping/pong keepalive", any(m.get("type") == "pong" for m in log), str([m.get("type") for m in log])[:90])

        async def chat_script(ws, log):
            # drain the hello
            log.append(json.loads(await asyncio.wait_for(ws.recv(), timeout=120)))
            await ws.send(json.dumps({"type": "user_text", "text": "כמה זה 17 כפול 23", "speak": True}))
            t_end = time.time() + 180
            while time.time() < t_end:
                try:
                    msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=180))
                except asyncio.TimeoutError:
                    break
                log.append(msg)
                if msg.get("type") == "answer":
                    break

        log = asyncio.run(ws_session(base_ws, chat_script))
        kinds = [m.get("type") for m in log]
        answer = next((m for m in log if m.get("type") == "answer"), None)
        check("chat turn streams thinking → answer", "answer" in kinds, str(kinds)[:120])
        check("bus events are mirrored to the HUD", "event" in kinds, str(kinds)[:150])
        audio = next((m for m in log if m.get("type") == "audio"), None)
        check("audio frame delivered to the HUD", audio is not None and bool(audio.get("wav_b64")))
        if audio:
            wav = base64.b64decode(audio["wav_b64"])
            check("audio frame is RIFF/WAVE PCM", wav[:4] == b"RIFF" and wav[8:12] == b"WAVE",
                  f"({len(wav)} bytes @ {audio.get('sample_rate')}Hz)")
            secs = len(wav) / (2 * (audio.get("sample_rate") or 24000))
            check("audio frame has real duration", 0.2 < secs < 60, f"({secs:.2f}s)")
        if answer:
            a = answer.get("answer", {})
            check("answer is exact and grounded", "391" in a.get("text", "") and a.get("grounded") is True,
                  a.get("text", "")[:70])
            check("answer carries a HUD-ready trace", isinstance(a.get("trace"), dict) and a["trace"],
                  str(list(a.get("trace", {})))[:90])
            check("answer reports latency", (answer.get("ms") or 0) > 0, f"({answer.get('ms')}ms)")

        # the reader loop must stay responsive *while* a worker thread is parked
        perm_file = Path.home() / "jarvis_server_perm_test.txt"
        perm_file.write_text("delete me", encoding="utf-8")

        async def perm_script(ws, log):
            log.append(json.loads(await asyncio.wait_for(ws.recv(), timeout=120)))
            await ws.send(json.dumps({"type": "invoke", "req": 7, "skill": "fs.delete",
                                      "args": {"path": str(perm_file)}}))
            ask = None
            t_end = time.time() + 90
            while time.time() < t_end:
                msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=90))
                log.append(msg)
                ev = msg.get("event") or {}
                if ev.get("topic") == "security.permission.request":
                    ask = ev.get("data") or {}
                    break
            check("CRITICAL invoke raises a permission prompt", bool(ask), str(ask)[:120])
            if not ask:
                return
            check("prompt names the action and its risk",
                  ask.get("action") == "fs.delete" and ask.get("level") == "CRITICAL", str(ask)[:110])
            check("prompt shows the arguments to the human",
                  str(perm_file) in json.dumps(ask.get("args")), str(ask.get("args"))[:100])
            await ws.send(json.dumps({"type": "ping"}))
            pong = None
            t_end = time.time() + 20
            while time.time() < t_end:
                msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=20))
                log.append(msg)
                if msg.get("type") == "pong":
                    pong = msg
                    break
            check("socket stays responsive while a worker waits for the human", pong is not None)
            await ws.send(json.dumps({"type": "permission", "id": ask["id"], "allow": True}))
            t_end = time.time() + 60
            while time.time() < t_end:
                msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=60))
                log.append(msg)
                if msg.get("type") == "skill_result" and msg.get("req") == 7:
                    break
            res = next((m for m in log if m.get("type") == "skill_result"), None)
            check("approved action returns its result", res is not None, str(res)[:140])
            check("approved CRITICAL action actually ran", bool(res and res.get("ok")),
                  str(res.get("error") if res else "")[:90])
            check("the file is really gone", not perm_file.exists(), str(perm_file))
            check("approval is audited on the bus",
                  any((m.get("event") or {}).get("topic") == "security.permission.answer" for m in log))

        log = asyncio.run(ws_session(base_ws, perm_script))

        # a client that lies about permissions must still be stopped
        async def bypass_script(ws, log):
            log.append(json.loads(await asyncio.wait_for(ws.recv(), timeout=120)))
            await ws.send(json.dumps({"type": "invoke", "skill": "fs.delete", "granted": True,
                                      "args": {"path": str(Path.home() / "jarvis_never_exists.txt")}}))
            t_end = time.time() + 90
            while time.time() < t_end:
                msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=90))
                log.append(msg)
                ev = msg.get("event") or {}
                if ev.get("topic") == "security.permission.request":
                    await ws.send(json.dumps({"type": "permission", "id": ev["data"]["id"], "allow": False}))
                if msg.get("type") == "skill_result":
                    break

        log = asyncio.run(ws_session(base_ws, bypass_script))
        check("`granted:true` from the client cannot bypass the firewall",
              any((m.get("event") or {}).get("topic") == "security.permission.request" for m in log),
              str([(m.get("event") or {}).get("topic") for m in log if m.get("type") == "event"][-4:]))

        deny_file = Path.home() / "jarvis_server_deny_test.txt"
        deny_file.write_text("keep me", encoding="utf-8")

        async def deny_script(ws, log):
            log.append(json.loads(await asyncio.wait_for(ws.recv(), timeout=120)))
            await ws.send(json.dumps({"type": "invoke", "skill": "fs.delete",
                                      "args": {"path": str(deny_file)}}))
            ask = None
            t_end = time.time() + 90
            while time.time() < t_end:
                msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=90))
                log.append(msg)
                ev = msg.get("event") or {}
                if ev.get("topic") == "security.permission.request":
                    ask = ev.get("data") or {}
                    break
            if ask:
                await ws.send(json.dumps({"type": "permission", "id": ask["id"], "allow": False}))
            t_end = time.time() + 60
            while time.time() < t_end:
                msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=60))
                log.append(msg)
                if msg.get("type") == "skill_result":
                    break

        log = asyncio.run(ws_session(base_ws, deny_script))
        res = next((m for m in log if m.get("type") == "skill_result"), None)
        check("denied CRITICAL action is refused", res is not None and res.get("ok") is False,
              str(res.get("error") if res else "")[:100])
        check("denial reason is reported to the human",
              bool(res) and "denied" in str(res.get("error", "")).lower(), str(res.get("error") if res else "")[:100])
        check("the denied file survived", deny_file.exists(), str(deny_file))
        check("denial is audited", any((m.get("event") or {}).get("topic") == "security.permission.deny"
                                       for m in log), str([ (m.get("event") or {}).get("topic") for m in log if m.get("type") == "event"][-6:]))

        # ---------------------------------------------------------- controls --
        print("\n== controls ==")

        async def kill_script(ws, log):
            log.append(json.loads(await asyncio.wait_for(ws.recv(), timeout=120)))
            await ws.send(json.dumps({"type": "kill", "reason": "test"}))
            t_end = time.time() + 30
            while time.time() < t_end:
                msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=30))
                log.append(msg)
                if msg.get("type") == "control":
                    break

        log = asyncio.run(ws_session(base_ws, kill_script))
        ctl = next((m for m in log if m.get("type") == "control"), None)
        check("kill switch acknowledged", bool(ctl and ctl.get("killed") is True), str(ctl)[:90])
        st, _, d = get_json(f"{base}/api/status")
        check("kill visible over HTTP", d["security"]["killed"] is True)

        async def revive_script(ws, log):
            log.append(json.loads(await asyncio.wait_for(ws.recv(), timeout=120)))
            await ws.send(json.dumps({"type": "revive"}))
            t_end = time.time() + 30
            while time.time() < t_end:
                msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=30))
                log.append(msg)
                if msg.get("type") == "control":
                    break
            await ws.send(json.dumps({"type": "set_level", "level": "safe"}))
            t_end = time.time() + 30
            while time.time() < t_end:
                msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=30))
                log.append(msg)
                if msg.get("type") == "control" and msg.get("level"):
                    break

        log = asyncio.run(ws_session(base_ws, revive_script))
        check("revive acknowledged", any(m.get("type") == "control" and m.get("killed") is False for m in log),
              str([m for m in log if m.get("type") == "control"])[:110])
        check("permission level change acknowledged",
              any(m.get("type") == "control" and str(m.get("level")) == "SAFE" for m in log),
              str([m.get("level") for m in log if m.get("type") == "control"]))
        st, _, d = get_json(f"{base}/api/status")
        check("level persisted in the firewall", d["security"]["level"] == "SAFE", str(d["security"])[:90])

        async def theme_script(ws, log):
            log.append(json.loads(await asyncio.wait_for(ws.recv(), timeout=120)))
            await ws.send(json.dumps({"type": "set_theme", "theme": "mark2"}))
            t_end = time.time() + 30
            while time.time() < t_end:
                msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=30))
                log.append(msg)
                if msg.get("type") == "control":
                    break
            await ws.send(json.dumps({"type": "nonsense"}))
            t_end = time.time() + 20
            while time.time() < t_end:
                msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=20))
                log.append(msg)
                if msg.get("type") == "error":
                    break

        log = asyncio.run(ws_session(base_ws, theme_script))
        check("theme change acknowledged", any(m.get("theme") == "mark2" for m in log), str(log[-3:])[:130])
        check("unknown command returns a clean error",
              any(m.get("type") == "error" and "unknown type" in str(m.get("message")) for m in log))

        # ------------------------------------------- REST twin of the socket ----
        print("\n== REST transport (/api/command) ==")

        def cmd(payload: dict):
            st_, hdr_, body_ = post(f"{base}/api/command",
                                    json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                                    ctype="application/json")
            return st_, json.loads(body_.decode("utf-8", "replace"))

        st_, d = cmd({"type": "user_text", "text": "כמה זה 12 בחזקת 3", "speak": True})
        a = d.get("answer") or {}
        # the math engine formats thousands with a separator — compare digits only
        digits = str(a.get("text", "")).replace(",", "").replace("\u00a0", "")
        check("REST chat returns the same answer shape", st_ == 200 and d.get("type") == "answer"
              and "1728" in digits and a.get("grounded") is True, str(a.get("text"))[:70])
        frames = d.get("audio") or []
        check("REST chat carries the speech inline", len(frames) == 1 and frames[0].get("wav_b64"),
              f"({len(frames)} frame(s))")
        if frames:
            wav = base64.b64decode(frames[0]["wav_b64"])
            check("inline audio is RIFF/WAVE", wav[:4] == b"RIFF" and wav[8:12] == b"WAVE",
                  f"({len(wav)} bytes)")
            secs = len(wav) / (2 * (frames[0].get("sample_rate") or 24000))
            check("inline audio has real duration", 0.2 < secs < 60, f"({secs:.2f}s)")

        st_, d = cmd({"type": "status"})
        check("REST status command", st_ == 200 and d.get("type") == "hello" and "status" in d,
              f"(skills {(d.get('status') or {}).get('skills')})")
        st_, d = cmd({"type": "invoke", "skill": "time.now", "args": {}})
        check("REST runs a SAFE skill", st_ == 200 and d.get("ok") is True and d.get("value"),
              str(d.get("value"))[:60])
        st_, d = cmd({"type": "invoke", "skill": "nope.nope", "args": {}})
        check("REST rejects an unknown skill", d.get("ok") is False and "unknown skill" in str(d.get("error")),
              str(d.get("error"))[:60])
        st_, d = cmd({"type": "set_theme", "theme": "mark1"})
        check("REST control command", d.get("type") == "control" and d.get("theme") == "mark1", str(d)[:80])
        st_, d = cmd({"type": "nonsense"})
        check("REST reports unknown commands cleanly",
              d.get("type") == "error" and "unknown type" in str(d.get("message")), str(d)[:80])
        st_, hdr_, body_ = post(f"{base}/api/command", b"{not json", ctype="application/json")
        check("REST rejects a malformed body", st_ == 200 and b"invalid JSON" in body_, str(body_)[:70])

        # Unknown API routes must fail loudly and as JSON — never as index.html,
        # which the REST fallback in the UI could not tell apart from a success.
        def any_method(method: str, url: str, data: bytes | None = None):
            req = urllib.request.Request(url, data=data, method=method,
                                         headers={"Origin": "null"})
            try:
                with urllib.request.urlopen(req, timeout=60) as r:
                    return r.status, r.read()
            except urllib.error.HTTPError as e:
                return e.code, e.read()

        for method, path in (("GET", "/api/health"), ("POST", "/api/nope"),
                             ("GET", "/api/command"), ("DELETE", "/api/status/x"),
                             ("PUT", "/api/status")):
            st_, body_ = any_method(method, f"{base}{path}")
            check(f"unmatched {method} {path} answers JSON 404",
                  st_ == 404 and b'"type": "error"' in body_ and b"no such API route" in body_,
                  f"[{st_}] {body_[:64]!r}")
        st_, body_ = any_method("GET", f"{base}/")
        check("the UI catch-all still serves the HUD", st_ == 200 and b"<html" in body_.lower(),
              f"[{st_}]")
        st_, body_ = any_method("GET", f"{base}/no-such-page")
        check("an unknown UI path still falls back to the HUD", st_ == 200 and b"<html" in body_.lower(),
              f"[{st_}]")

        # the firewall applies to REST exactly as it does to the socket:
        # the level is still SAFE from the control test above, so WRITE must fail
        target = Path.home() / "jarvis_rest_write.txt"
        st_, d = cmd({"type": "invoke", "skill": "fs.write",
                      "args": {"path": str(target), "content": "x"}})
        check("REST cannot exceed the firewall level", d.get("ok") is False
              and "blocked" in str(d.get("error")), str(d.get("error"))[:90])
        check("the blocked write never touched the disk", not target.exists())

        st_, d = cmd({"type": "set_level", "level": "write"})
        check("level raised over REST", str(d.get("level")) == "WRITE", str(d)[:80])
        st_, d = cmd({"type": "invoke", "skill": "fs.write",
                      "args": {"path": str(target), "content": "x"}})
        check("REST WRITE skill runs once policy allows it", d.get("ok") is True,
              str(d.get("error"))[:90])
        check("the file really was written", target.exists() and target.read_text() == "x")
        try:
            target.unlink()
        except Exception:
            pass

        # ------------------------------------------------------------- load ----
        print("\n== resilience ==")
        st, _, d = get_json(f"{base}/api/history?n=5")
        check("turns survived the session", len(d.get("turns", [])) >= 1, f"({len(d.get('turns', []))} turns)")
        st, _, d = get_json(f"{base}/api/events?n=400")
        check("bus history retained for the HUD timeline", len(d.get("events", [])) > 20,
              f"({len(d.get('events', []))} events)")
        st, _, body = get(f"{base}/no/such/page")
        check("unknown path falls back to the HUD shell", st == 200 and b"J.A.R.V.I.S." in body)
        st, hdr, body = post(f"{base}/api/listen", b"NOT_A_WAV_AT_ALL" * 40)
        d = json.loads(body.decode("utf-8", "replace"))
        check("garbage audio is rejected, not crashed", st == 200 and d.get("ok") is False,
              str(d.get("error"))[:90])
        check("process still healthy after abuse", proc.poll() is None)

        for f in (Path.home() / "jarvis_server_perm_test.txt",
                  Path.home() / "jarvis_server_deny_test.txt"):
            try:
                if f.exists():
                    f.unlink()
            except Exception:
                pass

    finally:
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except Exception:
            proc.kill()

    print(f"\nRESULT: {ok} passed, {fail} failed")
    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
