"""JARVIS local server — the bridge between the brain and the HUD.

One process serves:
  * the UI (static files from ``ui/``)
  * a WebSocket carrying the full event stream, commands and audio
  * a small JSON API for status/history/audit

Security posture
  * binds to 127.0.0.1 by default — the app is local-only
  * ``--preview`` binds 0.0.0.0 for the sandbox live-preview harness only
  * the browser never talks to anything but this process

Run:  python -m core.server            (then open the printed URL)
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
UI_DIR = ROOT / "ui"

from aiohttp import WSMsgType, web  # noqa: E402

from core.bus import BUS, T  # noqa: E402
from core.voice_session import (active_sessions, get_session, start_session,  # noqa: E402
                                stop_session)
from core.config import CONFIG  # noqa: E402

EXECUTOR = ThreadPoolExecutor(max_workers=4, thread_name_prefix="jarvis-worker")
AGENT = None            # built lazily in a worker thread (loading the core is heavy)


# ------------------------------------------------------------------ plumbing --
def _json(data: Any) -> web.Response:
    return web.json_response(data, dumps=lambda o: json.dumps(o, ensure_ascii=False, default=str))


class Client:
    """One connected HUD.

    Sends are serialised through the event loop and a per-client lock, because
    frames originate from several threads (bus callbacks, worker threads, tasks).
    """

    __slots__ = ("ws", "loop", "lock", "tasks", "sid")

    def __init__(self, ws: web.WebSocketResponse, loop: asyncio.AbstractEventLoop) -> None:
        self.ws = ws
        self.loop = loop
        self.lock = asyncio.Lock()
        self.tasks: set = set()
        self.sid = uuid.uuid4().hex[:12]      # identifies this HUD's voice session

    async def send(self, payload: Dict[str, Any]) -> None:
        text = json.dumps(payload, ensure_ascii=False, default=str)
        async with self.lock:
            if not self.ws.closed:
                await self.ws.send_str(text)

    def push(self, payload: Dict[str, Any]) -> bool:
        """Thread-safe fire-and-forget broadcast to this client."""
        if self.ws.closed:
            return False
        try:
            text = json.dumps(payload, ensure_ascii=False, default=str)
        except Exception:
            return False
        try:
            fut = asyncio.run_coroutine_threadsafe(self._send_text(text), self.loop)
            fut.add_done_callback(lambda f: f.exception())
            return True
        except Exception:
            return False

    async def _send_text(self, text: str) -> None:
        async with self.lock:
            if not self.ws.closed:
                await self.ws.send_str(text)

    def spawn(self, coro) -> None:
        task = self.loop.create_task(coro)
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)


CLIENTS: List[Client] = []
_AGENT_LOCK = threading.Lock()
_CAPTURE = threading.local()          # per-call audio capture for the REST path
_REST_VOICE: Dict[str, List[Dict[str, Any]]] = {}   # pending events per REST sid
_REST_VOICE_LOCK = threading.Lock()


def _rest_emit(sid: str):
    """A session created over REST has no socket to push through, so its events
    queue up here and the next /api/audio call carries them back."""
    def emit(payload: Dict[str, Any]) -> None:
        with _REST_VOICE_LOCK:
            _REST_VOICE.setdefault(sid, []).append(payload)
    return emit


def _rest_drain(sid: str) -> List[Dict[str, Any]]:
    with _REST_VOICE_LOCK:
        return _REST_VOICE.pop(sid, [])


def _voice_opts(data: Dict[str, Any]) -> Dict[str, Any]:
    opts: Dict[str, Any] = {}
    if "ack" in data:
        opts["ack"] = str(data.get("ack") or "")
    for key in ("idle_timeout", "min_speech_ms", "end_silence_ms",
                "max_utterance_s", "wake_threshold"):
        if data.get(key) is not None:
            try:
                opts[key] = float(data[key])
            except (TypeError, ValueError):
                pass
    return opts


def _broadcast(payload: Dict[str, Any]) -> None:
    for c in list(CLIENTS):
        if not c.push(payload):
            try:
                CLIENTS.remove(c)
            except ValueError:
                pass


def _on_bus_event(ev) -> None:
    _broadcast({"type": "event", "event": ev.to_dict()})


def get_agent():
    """Build the orchestrator once (heavy: loads the neural core + voicebank)."""
    global AGENT
    if AGENT is None:
        with _AGENT_LOCK:
            if AGENT is None:
                from agents.jarvis import JarvisAgent
                BUS.on("*", _on_bus_event)

                def audio_sink(wav_bytes: bytes, sr: int) -> bool:
                    """The HUD is the loudspeaker. No frame is ever dropped to disk.

                    Two deliveries are possible at once: broadcast to every open
                    socket, and/or hand the frame back to the HTTP call that asked
                    for it (the REST transport has no socket to push through).
                    """
                    b64 = base64.b64encode(wav_bytes).decode("ascii")
                    buf = getattr(_CAPTURE, "buf", None)
                    if buf is not None:
                        buf.append({"wav_b64": b64, "sample_rate": sr})
                    if CLIENTS:
                        _broadcast({"type": "audio", "sample_rate": sr, "ts": time.time(),
                                    "wav_b64": b64})
                    return bool(CLIENTS) or buf is not None

                AGENT = JarvisAgent(audio_sink=audio_sink)
    return AGENT


# ------------------------------------------------------------------ HTTP ----
async def api_status(request: web.Request) -> web.Response:
    agent = await asyncio.get_event_loop().run_in_executor(EXECUTOR, get_agent)
    return _json(agent.status())


async def api_boot(request: web.Request) -> web.Response:
    agent = await asyncio.get_event_loop().run_in_executor(EXECUTOR, get_agent)
    report = await asyncio.get_event_loop().run_in_executor(EXECUTOR, agent.boot)
    return _json({"ok": True, "report": report})


async def api_events(request: web.Request) -> web.Response:
    agent = await asyncio.get_event_loop().run_in_executor(EXECUTOR, get_agent)
    return _json({"events": agent.events(int(request.query.get("n", 150)))})


async def api_skills(request: web.Request) -> web.Response:
    from skills.registry import REGISTRY
    import skills as skills_pkg
    skills_pkg.load_all()
    return _json({"count": len(REGISTRY.names()), "skills": REGISTRY.schema()})


async def api_audit(request: web.Request) -> web.Response:
    agent = await asyncio.get_event_loop().run_in_executor(EXECUTOR, get_agent)
    return _json({"stats": agent.firewall.stats(), "tail": agent.firewall.tail(int(request.query.get("n", 40)))})


async def api_permissions(request: web.Request) -> web.Response:
    agent = await asyncio.get_event_loop().run_in_executor(EXECUTOR, get_agent)
    return _json({"pending": agent.pending_permissions(), "stats": agent.firewall.stats(),
                  "timeout": agent.confirm_timeout})


async def api_memory(request: web.Request) -> web.Response:
    agent = await asyncio.get_event_loop().run_in_executor(EXECUTOR, get_agent)
    return _json({"stats": agent.memory.stats(), "facts": agent.memory.all_facts()[:60],
                  "recent": agent.memory.recent_episodes(14)})


async def api_stt(request: web.Request) -> web.Response:
    agent = await asyncio.get_event_loop().run_in_executor(EXECUTOR, get_agent)
    return _json(agent._stt_stats())


async def api_history(request: web.Request) -> web.Response:
    agent = await asyncio.get_event_loop().run_in_executor(EXECUTOR, get_agent)
    return _json({"turns": agent.history(int(request.query.get("n", 20)))})


async def api_listen(request: web.Request) -> web.Response:
    """POST a WAV captured in the HUD → local speech recognition → JARVIS answers."""
    loop = asyncio.get_event_loop()
    agent = await loop.run_in_executor(EXECUTOR, get_agent)
    payload = await request.read()
    if not payload:
        return _json({"ok": False, "error": "empty audio body", "text": ""})

    speak = request.query.get("speak", "1") not in ("0", "false", "no")
    try:
        conf = request.query.get("min_confidence")
        res = await loop.run_in_executor(
            EXECUTOR, agent.listen, payload, speak, float(conf) if conf else None)
    except Exception as exc:
        res = {"ok": False, "error": f"{type(exc).__name__}: {exc}", "text": ""}
    return _json(res)


async def api_voice(request: web.Request) -> web.Response:
    """POST /api/voice — the REST twin of the socket's voice_start/voice_stop.

    Body: {"action": "start"|"stop"|"state", "sid": "...", ...tuning}
    A REST client has no socket to push events through, so anything the session
    emits comes back on the next /api/audio response (and on "state").
    """
    loop = asyncio.get_event_loop()
    agent = await loop.run_in_executor(EXECUTOR, get_agent)
    try:
        data = json.loads((await request.read()).decode("utf-8", "replace") or "{}")
    except Exception:
        return _json({"type": "error", "message": "invalid JSON body"})
    action = str(data.get("action", "state")).lower()
    sid = str(data.get("sid") or request.query.get("sid") or "rest-default")[:40]

    if action == "start":
        session = await loop.run_in_executor(
            EXECUTOR, lambda: start_session(sid, agent, _rest_emit(sid), **_voice_opts(data)))
        return _json({"type": "control", "action": "voice_start", "sid": sid,
                      **session.snapshot(), "events": _rest_drain(sid)})
    if action == "stop":
        stopped = await loop.run_in_executor(EXECUTOR, stop_session, sid)
        return _json({"type": "control", "action": "voice_stop", "ok": stopped,
                      "events": _rest_drain(sid)})
    session = get_session(sid)
    return _json({"type": "voice.state", "sid": sid,
                  **(session.snapshot() if session else {"state": "off"}),
                  "sessions": active_sessions(), "events": _rest_drain(sid)})


async def api_audio(request: web.Request) -> web.Response:
    """POST /api/audio — stream one microphone chunk (raw Int16 LE mono PCM).

    The response carries everything the session emitted while chewing on it, plus
    any speech the answer produced, inline as base64 WAV frames — the same deal
    the REST command transport offers.
    """
    loop = asyncio.get_event_loop()
    sid = str(request.query.get("sid") or "rest-default")[:40]
    session = get_session(sid)
    if session is None:
        return web.json_response(
            {"type": "error", "state": "off",
             "message": "no voice session for this sid — POST /api/voice first"}, status=404)
    payload = await request.read()
    if not payload:
        return _json({"type": "voice.events", "events": _rest_drain(sid), "audio": [],
                      "message": "empty audio body"})

    # _CAPTURE is thread-local and the session runs in a worker thread, so the
    # buffer has to be opened *there* — setting it in this coroutine would leave
    # the audio sink writing to nothing and the spoken reply would be dropped.
    def work() -> List[Dict[str, Any]]:
        frames: List[Dict[str, Any]] = []
        _CAPTURE.buf = frames
        try:
            session.feed(payload)
        finally:
            _CAPTURE.buf = None
        return frames

    try:
        frames = await loop.run_in_executor(EXECUTOR, work)
    except Exception as exc:
        return _json({"type": "error", "message": f"{type(exc).__name__}: {exc}",
                      "events": _rest_drain(sid), "audio": []})
    return _json({"type": "voice.events", "state": session.state,
                  "events": _rest_drain(sid), "audio": frames})


# ------------------------------------------------------------- commands ----
def dispatch_command(agent, data: Dict[str, Any], capture_audio: bool = False) -> Dict[str, Any]:
    """Execute one HUD command. WebSocket and REST share this single code path,
    so both behave identically — including the permission firewall, which a REST
    caller can no more talk its way past than a socket can.

    ``capture_audio`` collects the speech frames produced by this call and hands
    them back inline; the REST transport has no socket to push them through.
    """
    owns = False
    if capture_audio and getattr(_CAPTURE, "buf", None) is None:
        _CAPTURE.buf = []
        owns = True
    try:
        out = _dispatch_inner(agent, data)
        if capture_audio and isinstance(out, dict):
            out["audio"] = list(getattr(_CAPTURE, "buf", None) or [])
        return out
    finally:
        if owns:
            _CAPTURE.buf = None


def _dispatch_inner(agent, data: Dict[str, Any]) -> Dict[str, Any]:
    from skills.registry import REGISTRY

    kind = str(data.get("type", ""))

    if kind == "user_text":
        text = str(data.get("text", "")).strip()
        if not text:
            return {"type": "error", "message": "empty text"}
        return {"type": "answer", **agent.handle_text(text, bool(data.get("speak", True)))}

    if kind == "speak":
        return {"type": "speak_result", **agent.speak_text(str(data.get("text", "")))}

    if kind == "clear_chat":
        return {"type": "control", "action": "clear_chat", **agent.clear_transcript()}

    if kind == "boot":
        return {"type": "boot", "report": agent.boot()}

    if kind == "status":
        return {"type": "hello", "status": agent.status()}

    if kind == "kill":
        return {"type": "control", "action": kind, **agent.kill(str(data.get("reason", "user")))}
    if kind == "revive":
        return {"type": "control", "action": kind, **agent.revive()}
    if kind == "set_level":
        return {"type": "control", "action": kind, **agent.set_level(str(data.get("level", "write")))}
    if kind == "set_dry_run":
        return {"type": "control", "action": kind, **agent.set_dry_run(bool(data.get("on", False)))}
    if kind == "set_theme":
        return {"type": "control", "action": kind, **agent.set_theme(str(data.get("theme", "mark1")))}
    if kind == "permission":
        return {"type": "control", "action": kind,
                **agent.grant(int(data.get("id", 0)), bool(data.get("allow", False)))}

    if kind == "invoke":
        name = str(data.get("skill", ""))
        args = data.get("args") or {}
        skill = REGISTRY.get(name)
        if skill is None:
            return {"type": "skill_result", "ok": False, "skill": name,
                    "error": f"unknown skill '{name}'"}
        decision = agent.firewall.check(name, skill.risk, args, "hud")
        if not decision.allowed:
            return {"type": "skill_result", "ok": False, "skill": name,
                    "error": decision.reason, "risk": skill.risk, **decision.to_dict()}
        return {"type": "skill_result", **REGISTRY.invoke(name, args, True).to_dict()}

    return {"type": "error", "message": f"unknown type {kind!r}"}


async def api_command(request: web.Request) -> web.Response:
    """REST twin of the WebSocket — used when a proxy will not upgrade sockets."""
    loop = asyncio.get_event_loop()
    agent = await loop.run_in_executor(EXECUTOR, get_agent)
    try:
        data = await request.json()
    except Exception:
        return _json({"type": "error", "message": "invalid JSON body"})
    if not isinstance(data, dict):
        return _json({"type": "error", "message": "body must be a JSON object"})

    def work() -> Dict[str, Any]:
        return dispatch_command(agent, data, capture_audio=True)

    try:
        out = await loop.run_in_executor(EXECUTOR, work)
    except Exception as exc:
        out = {"type": "error", "message": f"{type(exc).__name__}: {exc}"}
    return _json(out)


# ---------------------------------------------------------------- websocket --
# Long-running commands run as tasks so the reader loop never blocks. This is
# what makes the human-in-the-loop permission prompt possible: a CRITICAL skill
# parks a worker thread waiting for the answer that arrives on the same socket.
async def ws_handler(request: web.Request) -> web.WebSocketResponse:
    ws = web.WebSocketResponse(max_msg_size=32 * 1024 * 1024, heartbeat=25)
    await ws.prepare(request)
    loop = asyncio.get_event_loop()
    client = Client(ws, loop)
    CLIENTS.append(client)

    async def call(fn, *a, **kw):
        return await loop.run_in_executor(EXECUTOR, lambda: fn(*a, **kw))

    try:
        agent = await call(get_agent)
        await client.send({"type": "hello", "status": agent.status()})

        # ---- background command runners -------------------------------
        async def run_command(data: Dict[str, Any]) -> None:
            """Run one command off the reader loop, then answer on this socket."""
            rid = data.get("req")
            kind = str(data.get("type", ""))
            if kind == "user_text":
                await client.send({"type": "thinking", "text": str(data.get("text", ""))[:200],
                                   "req": rid})
            try:
                out = await call(dispatch_command, agent, data, False)
            except Exception as exc:
                out = {"type": "error", "message": f"{type(exc).__name__}: {exc}"}
            out["req"] = rid
            await client.send(out)

        # ---- reader loop (never blocks on a command) --------------------
        async for msg in ws:
            if msg.type == WSMsgType.ERROR:
                break

            # ---- streaming microphone audio (hands-free conversation) ----
            if msg.type == WSMsgType.BINARY:
                session = get_session(client.sid)
                if session is None:
                    await client.send({"type": "error",
                                       "message": "no voice session — send voice_start first"})
                    continue
                await call(session.feed, msg.data)
                continue

            if msg.type != WSMsgType.TEXT:
                continue
            try:
                data = json.loads(msg.data)
            except Exception:
                await client.send({"type": "error", "message": "invalid JSON"})
                continue
            kind = data.get("type", "")

            if kind == "ping":
                await client.send({"type": "pong", "ts": time.time()})
            elif kind in ("user_text", "invoke", "speak", "boot", "kill", "revive",
                          "set_level", "set_dry_run", "set_theme", "permission",
                          "clear_chat"):
                client.spawn(run_command(data))
            elif kind == "status":
                await client.send({"type": "hello", "status": await call(agent.status)})
            elif kind == "voice_start":
                session = await call(start_session, client.sid, agent, client.push,
                                     **_voice_opts(data))
                await client.send({"type": "control", "action": "voice_start",
                                   "sid": client.sid, **session.snapshot()})
            elif kind == "voice_stop":
                stopped = await call(stop_session, client.sid)
                await client.send({"type": "control", "action": "voice_stop", "ok": stopped})
            elif kind == "voice_state":
                session = get_session(client.sid)
                await client.send({"type": "voice.state",
                                   **(session.snapshot() if session else {"state": "off"})})
            else:
                await client.send({"type": "error", "message": f"unknown type {kind!r}"})
    except Exception as exc:
        try:
            await client.send({"type": "error", "message": f"{type(exc).__name__}: {exc}"})
        except Exception:
            pass
    finally:
        for t in list(client.tasks):
            t.cancel()
        stop_session(client.sid)          # never leave a microphone open behind
        if client in CLIENTS:
            CLIENTS.remove(client)
    return ws


async def api_not_found(request: web.Request) -> web.Response:
    """JSON 404 for any unmatched /api/* route, for any method."""
    return web.json_response(
        {"type": "error",
         "message": f"no such API route: {request.method} {request.path}"},
        status=404)


# ------------------------------------------------------------------- static --
async def index(request: web.Request) -> web.StreamResponse:
    # The catch-all below would otherwise answer an unknown /api/* route with
    # index.html — which the REST fallback in the UI cannot tell apart from a
    # real response. Unknown API paths must say so, in JSON, with a 404.
    if request.path.startswith("/api/") or request.path == "/api":
        return web.json_response(
            {"type": "error", "message": f"no such API route: {request.method} {request.path}"},
            status=404)
    path = UI_DIR / (request.match_info.get("tail", "") or "index.html")
    if not str(path.resolve()).startswith(str(UI_DIR.resolve())):
        raise web.HTTPForbidden()
    if not path.exists() or path.is_dir():
        path = UI_DIR / "index.html"
    if not path.exists():
        return web.Response(text="UI not built yet", status=503)
    return web.FileResponse(path)


# ------------------------------------------------------------------ telemetry --
async def telemetry_task(app: web.WebApplication) -> None:
    from skills.registry import REGISTRY
    import skills as skills_pkg
    skills_pkg.load_all()
    loop = asyncio.get_event_loop()
    while True:
        try:
            res = await loop.run_in_executor(EXECUTOR, REGISTRY.invoke, "sys.telemetry", {}, True)
            if res.ok:
                _broadcast({"type": "telemetry", "data": res.data, "ts": time.time()})
        except Exception:
            pass
        await asyncio.sleep(2.0)


async def _start_bg(app: web.WebApplication) -> None:
    app["telemetry"] = asyncio.create_task(telemetry_task(app))


async def _stop_bg(app: web.WebApplication) -> None:
    app["telemetry"].cancel()
    EXECUTOR.shutdown(wait=False)


CORS_HEADERS = {
    # The Electron renderer runs from file:// (Origin: null); the browser preview
    # runs from the sandbox host. Loopback-only binding is what keeps this safe.
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type",
    "Access-Control-Max-Age": "600",
}


@web.middleware
async def cors_middleware(request: web.Request, handler) -> web.StreamResponse:
    if request.method == "OPTIONS":
        return web.Response(status=204, headers=CORS_HEADERS)
    try:
        resp = await handler(request)
    except web.HTTPException as exc:
        exc.headers.update(CORS_HEADERS)
        raise
    resp.headers.update(CORS_HEADERS)
    return resp


def build_app() -> web.Application:
    app = web.Application(client_max_size=32 * 1024 * 1024, middlewares=[cors_middleware])
    app.router.add_get("/ws", ws_handler)
    app.router.add_get("/api/status", api_status)
    app.router.add_get("/api/boot", api_boot)
    app.router.add_get("/api/events", api_events)
    app.router.add_get("/api/skills", api_skills)
    app.router.add_get("/api/audit", api_audit)
    app.router.add_get("/api/memory", api_memory)
    app.router.add_get("/api/history", api_history)
    app.router.add_get("/api/permissions", api_permissions)
    app.router.add_get("/api/stt", api_stt)
    app.router.add_post("/api/listen", api_listen)
    app.router.add_post("/api/voice", api_voice)
    app.router.add_post("/api/audio", api_audio)
    app.router.add_post("/api/command", api_command)
    # Unknown /api/* must answer as JSON for every method. Registered before the
    # UI catch-all below (aiohttp resolves resources in registration order), so
    # a typo or a stale client gets a real 404 instead of index.html or a bare 405.
    app.router.add_route("*", "/api/{tail:.*}", api_not_found)
    if UI_DIR.exists():
        app.router.add_static("/ui/", path=str(UI_DIR), show_index=False)
        app.router.add_static("/assets/", path=str(UI_DIR / "assets"), show_index=False) \
            if (UI_DIR / "assets").exists() else None
    app.router.add_get("/", index)
    app.router.add_get("/{tail:.*}", index)
    app.on_startup.append(_start_bg)
    app.on_cleanup.append(_stop_bg)
    return app


def main() -> int:
    ap = argparse.ArgumentParser(description="JARVIS local server")
    ap.add_argument("--host", default="")
    ap.add_argument("--port", type=int, default=0)
    ap.add_argument("--preview", action="store_true",
                    help="bind 0.0.0.0 (sandbox live preview only; never for production)")
    ap.add_argument("--no-speak", action="store_true")
    args = ap.parse_args()

    host = args.host or (CONFIG.server.sandbox_preview_host if args.preview else CONFIG.server.host)
    port = args.port or CONFIG.server.port
    if args.no_speak:
        os.environ["JARVIS_SPEAK"] = "0"

    print(f"[server] JARVIS brain server on http://{host}:{port}")
    print(f"[server] binding {'PUBLIC (preview mode)' if host == '0.0.0.0' else 'localhost only'}")
    web.run_app(build_app(), host=host, port=port, print=None, access_log=None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
