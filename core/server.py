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
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
UI_DIR = ROOT / "ui"

from aiohttp import WSMsgType, web  # noqa: E402

from core.bus import BUS, T  # noqa: E402
from core.voice_session import (active_sessions, get_session, start_session,  # noqa: E402
                                stop_session)
from core.config import CONFIG, DATA as CONFIG_DATA  # noqa: E402

EXECUTOR = ThreadPoolExecutor(max_workers=4, thread_name_prefix="jarvis-worker")
AGENT = None            # built lazily in a worker thread (loading the core is heavy)


# ------------------------------------------------------------------ plumbing --
def _json(data: Any, status: int = 200) -> web.Response:
    """JSON with Hebrew left readable (ensure_ascii=False) and a settable status."""
    return web.json_response(data, status=status,
                             dumps=lambda o: json.dumps(o, ensure_ascii=False, default=str))


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
    if data.get("arm_on_start") is not None:
        opts["arm_on_start"] = str(data["arm_on_start"]).lower() not in ("0", "false", "no")
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


async def api_modes(request: web.Request) -> web.Response:
    """The twelve operational postures and which one is held."""
    from brain.modes import STATE
    return _json({"current": STATE.get().id, "modes": STATE.schema(),
                  "stats": STATE.stats()})


async def api_mode_set(request: web.Request) -> web.Response:
    """Switch posture. SAFE: changing how JARVIS behaves grants no new power —
    every action still passes the same firewall at its own risk level."""
    from brain.modes import STATE
    try:
        data = await request.json()
    except Exception:
        data = {}
    mid = str((data or {}).get("id") or request.query.get("id") or "").strip()
    m = STATE.set(mid)
    if m is None:
        return _json({"ok": False, "error": f"unknown mode {mid!r}",
                      "modes": [x["id"] for x in STATE.schema()]}, status=400)
    BUS.emit("mode.change", {"mode": m.id, "he": m.he}, source="hud")
    return _json({"ok": True, "mode": m.id, "he": m.he, "en": m.en,
                  "desc": m.desc_he, "persona": m.persona_he})


async def api_boot_progress(request: web.Request) -> web.Response:
    """Checks completed so far, without waiting for the whole POST.

    The boot POST only returns once every subsystem has been measured, and two of
    those measurements are minutes-long by design. The HUD used to wait for that
    single response behind a black overlay, which on a slow machine reads as a
    dead app. This endpoint lets it stream the results as they land instead.
    """
    agent = await asyncio.get_event_loop().run_in_executor(EXECUTOR, get_agent)
    return _json({"ok": True, **agent.boot_progress()})


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


async def api_sentinel(request: web.Request) -> web.Response:
    """The anomaly watchdog: burst/repeat/denial shapes across the session."""
    agent = await asyncio.get_event_loop().run_in_executor(EXECUTOR, get_agent)
    s = getattr(agent, "sentinel", None)
    if s is None:
        return _json({"stats": {}, "tail": []})
    return _json({"stats": s.stats(), "tail": s.tail(int(request.query.get("n", 20)))})


# ── the download bundle ─────────────────────────────────────────────────────
# The HUD's download button used to point at /ui/JARVIS-full.zip, a file nobody
# ever built: a 404 wearing a feature's clothes. These two routes are the real
# thing. Stats first, so the interface can state the true size before a byte
# moves, then the archive itself — built once and reused until a source file is
# newer than the zip.
_BUNDLE: Dict[str, Any] = {"path": None, "stamp": 0.0, "bytes": 0, "files": 0, "mb": 0.0}


def _bundle_stamp() -> float:
    """Newest modification time among everything the bundle would ship."""
    try:
        from tools.make_bundle import ROOT, iter_shippable
        return max((p.stat().st_mtime for p in iter_shippable(ROOT)), default=0.0)
    except Exception:
        return 0.0


def _bundle_build() -> Dict[str, Any]:
    """Build (or reuse) dist/JARVIS-full.zip. Runs in the executor, never here."""
    try:
        from tools.make_bundle import ROOT, bundle_stats, build_bundle
        dest = ROOT / "dist" / "JARVIS-full.zip"
        stamp = _bundle_stamp()
        cached = _BUNDLE.get("path")
        if cached and Path(cached).exists() and stamp <= _BUNDLE.get("stamp", 0.0):
            return {"ok": True, "cached": True, "path": cached,
                    "bytes": _BUNDLE["bytes"], "mb": _BUNDLE["mb"],
                    "files": _BUNDLE["files"], **bundle_stats()}
        rep = build_bundle(dest)
        _BUNDLE.update({"path": str(dest), "stamp": max(stamp, Path(dest).stat().st_mtime),
                        "bytes": rep["bytes"], "files": rep["files"], "mb": rep["mb"]})
        return {"ok": True, "cached": False, "path": str(dest), "sha256": rep["sha256"],
                **bundle_stats(), "zip_bytes": rep["bytes"], "zip_mb": rep["mb"]}
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


async def api_bundle(request: web.Request) -> web.Response:
    """What the download contains, measured without compressing anything."""
    try:
        from tools.make_bundle import bundle_stats
        return _json({"ok": True, **bundle_stats()})
    except Exception as exc:
        return _json({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, status=500)


async def api_download(request: web.Request) -> web.Response:
    """JARVIS-full.zip: code, trained models, voicebank, STT bank and the HUD."""
    loop = asyncio.get_event_loop()
    rep = await loop.run_in_executor(EXECUTOR, _bundle_build)
    if not rep.get("ok") or not rep.get("path"):
        return _json({"ok": False, "error": rep.get("error", "bundle unavailable")},
                     status=500)
    return web.FileResponse(Path(rep["path"]), headers={
        "Content-Disposition": 'attachment; filename="JARVIS-full.zip"',
        "X-JARVIS-Files": str(rep.get("files", "")),
        "X-JARVIS-MB": str(rep.get("mb", "")),
        "Cache-Control": "no-store",
    })


async def api_screen_read(request: web.Request) -> web.Response:
    """Describe a screenshot, decoding it with the in-house PNG codec.

    Two modes, because they have different risk profiles and the UI offers both:

      ?path=<file>   read that file (SAFE — reading something already on disk)
      ?capture=1     take a fresh screenshot, then read it (WRITE — the capture
                     step drives the OS, so it goes through the firewall like
                     any other screen.capture call)

    With neither, the most recent saved screenshot is read.

    This is the visible end of the capture→understand loop that `vision.png`
    exists to close. The response always carries `text_extracted: false`: pixel
    statistics are derivable offline, reading characters is not, and the panel
    would rather show an honest gap than imply it understood the screen.
    """
    from skills.registry import REGISTRY
    import skills as skills_pkg
    skills_pkg.load_all()

    agent = await asyncio.get_event_loop().run_in_executor(EXECUTOR, get_agent)
    path = request.query.get("path", "").strip()
    capture = request.query.get("capture", "").strip().lower() in ("1", "true", "yes")

    def _work() -> Dict[str, Any]:
        shot_path = path
        if capture:
            # Same sequence the WS `invoke` path uses: ask the firewall with the
            # agent's real level, and only then run it. Hardcoding
            # permission_granted=False here made the button permanently dead —
            # it could never succeed at any level. Going through check() keeps
            # the kill switch, the SAFE ceiling, dry-run and the CRITICAL
            # confirmation hook all in force, and the reason comes back to the
            # panel verbatim so a denial explains itself.
            skill = REGISTRY.get("screen.capture")
            if skill is None:
                return {"ok": False, "stage": "capture",
                        "error": "screen.capture is not registered"}
            decision = agent.firewall.check("screen.capture", skill.risk, {}, "hud")
            if not decision.allowed:
                return {"ok": False, "stage": "capture", "error": decision.reason,
                        "risk": skill.risk, **decision.to_dict()}
            cap = REGISTRY.invoke("screen.capture", {}, permission_granted=True)
            if not cap.ok:
                return {"ok": False, "stage": "capture",
                        "error": cap.error or "the capture did not succeed",
                        "value": cap.value or ""}
            shot_path = str((cap.data or {}).get("path") or "")
            if not shot_path:
                return {"ok": False, "stage": "capture",
                        "error": "the capture reported success but returned no path"}
        res = REGISTRY.invoke("screen.read", {"path": shot_path} if shot_path else {},
                              permission_granted=True)
        out: Dict[str, Any] = {"ok": bool(res.ok), "stage": "read", "value": res.value or ""}
        if res.ok:
            out["data"] = res.data
        else:
            out["error"] = res.error
        return out

    return _json(await asyncio.get_event_loop().run_in_executor(EXECUTOR, _work))


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


# ── local-file retrieval (RAG) ──────────────────────────────────────────────
# Four routes, and the split matters:
#
#   GET  /api/rag/status   cheap, polled by the HUD panel
#   POST /api/rag/search   retrieval only — "show me where"
#   POST /api/rag/ask      retrieval + cited extractive answer
#   POST /api/rag/index    the only one that touches the disk
#
# ``/api/rag/index`` goes through the same Permission Firewall as every other
# mutating call, so it is refused outright when the level is SAFE or when the
# kill switch is engaged — the indexer never gets a private channel around the
# safety layer just because it happens to be new.
async def api_rag_status(request: web.Request) -> web.Response:
    from brain.rag.engine import get_engine

    def work() -> Dict[str, Any]:
        eng = get_engine()
        return {"ok": True, "status": eng.status(),
                "docs": [{"path": d["path"], "chunks": d["chunks"], "bytes": d["bytes"],
                          "kind": d["kind"], "title": d["title"]}
                         for d in eng.docs(int(request.query.get("n", 100)))]}

    try:
        return _json(await asyncio.get_event_loop().run_in_executor(EXECUTOR, work))
    except Exception as exc:
        return _json({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, status=500)


async def api_rag_search(request: web.Request) -> web.Response:
    from brain.rag.engine import get_engine
    try:
        data = await request.json()
    except Exception:
        return _json({"ok": False, "error": "invalid JSON body"}, status=400)
    query = str(data.get("query") or "").strip()
    if not query:
        return _json({"ok": False, "error": "query is required"}, status=400)

    def work() -> Dict[str, Any]:
        eng = get_engine()
        hits = eng.search(query, k=int(data.get("k") or 8),
                          path_filter=str(data.get("path_filter") or ""))
        return {"ok": True, "query": query, "found": len(hits),
                "hits": [h.to_dict() for h in hits],
                "expansions": eng.retriever.last_expansions}

    try:
        return _json(await asyncio.get_event_loop().run_in_executor(EXECUTOR, work))
    except Exception as exc:
        return _json({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, status=500)


async def api_rag_ask(request: web.Request) -> web.Response:
    from brain.rag.engine import get_engine
    try:
        data = await request.json()
    except Exception:
        return _json({"ok": False, "error": "invalid JSON body"}, status=400)
    query = str(data.get("query") or "").strip()
    if not query:
        return _json({"ok": False, "error": "query is required"}, status=400)

    def work() -> Dict[str, Any]:
        eng = get_engine()
        ans = eng.ask(query, k=int(data.get("k") or 8),
                      path_filter=str(data.get("path_filter") or ""))
        verified, problems = eng.verify(ans)
        payload = ans.to_dict()
        # The grounding invariant is re-checked against the files on disk for
        # every answer we serve, and the result is published. A client that
        # sees verified=false knows not to trust the citations.
        payload["ok"] = True
        payload["verified"] = verified
        payload["verify_problems"] = problems
        payload["expansions"] = eng.retriever.last_expansions
        return payload

    try:
        return _json(await asyncio.get_event_loop().run_in_executor(EXECUTOR, work))
    except Exception as exc:
        return _json({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, status=500)


async def api_rag_index(request: web.Request) -> web.Response:
    from brain.rag.engine import get_engine
    try:
        data = await request.json()
    except Exception:
        data = {}
    roots = data.get("roots") or []
    if isinstance(roots, str):
        roots = [p.strip() for p in roots.replace("\n", ",").split(",") if p.strip()]

    agent = await asyncio.get_event_loop().run_in_executor(EXECUTOR, get_agent)

    def work() -> Dict[str, Any]:
        decision = agent.firewall.check("rag.index", "WRITE", {"roots": roots})
        if not decision.allowed:
            return {"ok": False, "error": f"blocked by permission firewall: {decision.reason}",
                    "decision": decision.to_dict()}
        if agent.firewall.dry_run:
            return {"ok": True, "dry_run": True,
                    "would_index": [str(r) for r in roots],
                    "decision": decision.to_dict()}
        eng = get_engine()
        report = eng.index(roots or None, force=bool(data.get("force")))
        report["ok"] = not report.get("error")
        report["decision"] = decision.to_dict()
        return report

    try:
        out = await asyncio.get_event_loop().run_in_executor(EXECUTOR, work)
        return _json(out, status=200 if out.get("ok") else 403)
    except Exception as exc:
        return _json({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, status=500)


async def api_rag_roots(request: web.Request) -> web.Response:
    """Set which folders are indexed. Never triggers a scan by itself."""
    from brain.rag.engine import get_engine
    try:
        data = await request.json()
    except Exception:
        return _json({"ok": False, "error": "invalid JSON body"}, status=400)
    roots = data.get("roots") or []
    if isinstance(roots, str):
        roots = [p.strip() for p in roots.replace("\n", ",").split(",") if p.strip()]

    def work() -> Dict[str, Any]:
        eng = get_engine()
        clean = eng.set_roots([str(r) for r in roots])
        return {"ok": True, "roots": clean}

    try:
        return _json(await asyncio.get_event_loop().run_in_executor(EXECUTOR, work))
    except Exception as exc:
        return _json({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, status=500)


ENROLL_DIR = CONFIG_DATA / "enroll"
#: what the HUD asks the user to say when training their own voice
ENROLL_SCRIPT = ["wake", "time", "status", "help", "stop", "joke"]


async def api_enroll(request: web.Request) -> web.Response:
    """/api/enroll — teach JARVIS the voice that is actually in the room.

    GET returns the training script; POST a WAV of one phrase and its MFCC
    trajectory is appended to the template bank. This is not a nicety: the bank
    ships synthesised from our own TTS, and a voice with a different timbre
    scores 0.000 on it. Recording the listener is the only thing that makes the
    wake word and the commands answer to *them*.
    """
    from voice.stt import COMMANDS, get_engine, read_wav_bytes

    loop = asyncio.get_event_loop()
    engine = await loop.run_in_executor(EXECUTOR, get_engine)
    if not engine.available:
        return web.json_response({"ok": False, "error": "template bank unavailable"}, status=503)

    if request.method == "GET":
        counts: Dict[str, int] = {}
        for e in engine.enrolled:
            counts[str(e.get("label"))] = counts.get(str(e.get("label")), 0) + 1
        return _json({
            "ok": True,
            "script": [{"label": lab, "text": COMMANDS[lab]["text"],
                        "say": COMMANDS[lab].get("say", []), "enrolled": counts.get(lab, 0)}
                       for lab in ENROLL_SCRIPT if lab in COMMANDS],
            "vocabulary": [{"label": v["label"], "text": v["text"]} for v in engine.vocabulary()],
            "enrolled_total": len(engine.enrolled),
            "templates": engine.stats().get("templates", 0),
            "threshold": getattr(engine, "min_confidence", 0.5),
        })

    label = str(request.query.get("label") or "wake")[:32]
    if label not in COMMANDS:
        return web.json_response({"ok": False, "error": f"unknown label {label!r}",
                                  "known": sorted(COMMANDS)}, status=404)
    payload = await request.read()
    if len(payload) < 1024:
        return _json({"ok": False, "label": label, "error": "recording is empty or too short"})

    def work() -> Dict[str, Any]:
        ENROLL_DIR.mkdir(parents=True, exist_ok=True)
        path = ENROLL_DIR / f"{label}-{int(time.time() * 1000)}.wav"
        path.write_bytes(payload)
        try:
            res = engine.enroll_wav(label, path, source="hud")
        except Exception as exc:
            return {"ok": False, "label": label, "error": f"{type(exc).__name__}: {exc}"}
        # Score the very clip they just handed us — instant proof either way.
        try:
            x, sr = read_wav_bytes(payload)
            if label == "wake":
                score = float(engine.wake_score(x, sr))
            else:
                v = engine.recognize(x, sr)
                score = float(getattr(v, "confidence", 0.0) or 0.0)
                res["matched"] = getattr(v, "label", None)
        except Exception as exc:
            score, res["score_error"] = 0.0, str(exc)
        distinct = len({str(e.get("label")) for e in engine.enrolled})
        res.update({"text": COMMANDS[label]["text"], "score": round(score, 3),
                    "heard": score >= 0.55, "seconds": round(len(payload) / 32000.0, 2),
                    "enrolled_labels": distinct})
        if distinct < 3:
            # Measured: with a single enrolled phrase, that phrase's template
            # becomes an attractor for the rest of the speaker's foreign speech
            # (timbre dominates content in MFCC+DTW). Training 4 phrases restored
            # clean discrimination — every trained command at 0.92-1.00 and an
            # untrained one still rejected. So say so instead of letting one
            # sample look like a finished training.
            res["advise"] = ("כדאי לאמן לפחות 3-4 משפטים: דגימה בודדת יכולה למשוך "
                             "גם משפטים אחרים אל התווית שלה")
        return res

    out = await loop.run_in_executor(EXECUTOR, work)
    BUS.emit("stt.enrolled", {"label": label, "ok": bool(out.get("ok")),
                              "score": out.get("score"), "bank": out.get("bank")},
             source="server")
    return _json(out)


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
    if action == "unmute":
        session = get_session(sid)
        if session is None:
            return web.json_response({"type": "error", "state": "off",
                                      "message": "no voice session for this sid"}, status=404)
        await loop.run_in_executor(EXECUTOR, session.unmute)
        return _json({"type": "control", "action": "voice_unmute", "ok": True,
                      "state": session.state, "events": _rest_drain(sid)})
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
        level = str(data.get("level", "write"))
        out = agent.set_level(level)
        # A level chosen by hand is also the level to fall back to when nobody is
        # in front of the camera. Without this the presence gate keeps restoring
        # whatever the firewall happened to be at the moment vision started up.
        gate = FACE_STATE.get("gate")
        if gate is not None:
            gate.default_level = level.upper()
            if gate.identity is None:
                gate.poll()
        return {"type": "control", "action": kind, **out}
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


# ══════════════════════════════ sight ══════════════════════════════
# Who is in front of the machine decides what JARVIS may do to it. Detection,
# recognition and the presence gate are all built in-house (vision/faces.py): no
# OpenCV, no downloaded weights, no cloud. Frames arrive from the HUD as raw RGB
# pixels read off a canvas, so there is no image codec in the path either.
FACE_STATE: Dict[str, Any] = {"store": None, "gate": None}


def get_faces() -> Tuple[Any, Any]:
    """Lazily build the face store and its presence gate (heavy first call)."""
    if FACE_STATE["store"] is None:
        from security.permissions import FIREWALL
        from vision.faces import FaceGate, FaceStore

        cfg = getattr(CONFIG, "vision", None)
        get = (lambda k, d: getattr(cfg, k, d)) if cfg is not None else (lambda k, d: d)
        store = FaceStore()
        gate = FaceGate(
            store, firewall=FIREWALL,
            default_level=str(get("default_level", FIREWALL.level)),
            ceiling=str(get("ceiling", "CRITICAL")),
            ttl=float(get("presence_ttl", 12.0)),
            debounce=int(get("debounce", 2)),
            unknown_level=str(get("unknown_level", "SAFE")),
        )
        gate.armed = bool(get("arm_firewall", True))
        FACE_STATE["store"], FACE_STATE["gate"] = store, gate
    return FACE_STATE["store"], FACE_STATE["gate"]


def _face_frame(payload: Dict[str, Any]):
    """Turn a HUD frame payload into an RGB array (raw pixels, or grey)."""
    from vision.faces import frame_from_gray, frame_from_rgb
    if payload.get("rgb") or payload.get("data"):
        return frame_from_rgb(payload)
    return frame_from_gray(payload)


async def api_faces(request: web.Request) -> web.Response:
    """GET /api/faces — the gallery, the gate and what it currently grants."""
    from vision.faces import CONFIDENCE_THRESHOLD
    store, gate = await asyncio.get_event_loop().run_in_executor(EXECUTOR, get_faces)
    return _json({"ok": True, "people": store.list(), "gate": gate.state(),
                  "threshold": CONFIDENCE_THRESHOLD,
                  "calibrated": {"within": round(store._within, 3),
                                 "between": round(store._between, 3)}})


async def api_faces_recognize(request: web.Request) -> web.Response:
    """POST /api/faces/recognize — who is this, and what may they do?"""
    store, gate = await asyncio.get_event_loop().run_in_executor(EXECUTOR, get_faces)
    try:
        payload = await request.json()
    except Exception:
        return _json({"ok": False, "error": "body must be JSON"}, status=400)
    loop = asyncio.get_event_loop()

    def work() -> Dict[str, Any]:
        from vision.faces import detect
        from vision.scene import describe
        try:
            frame = _face_frame(payload)
        except Exception as exc:
            return {"ok": False, "error": f"bad frame: {exc}"}
        found = detect(frame, max_faces=4)
        match = store.recognize(frame)
        # Scene understanding is measured on every frame, not only when a face is
        # enrolled — "it's too dark to tell you who you are" is an answer, and a
        # more useful one than silence.
        scene = describe(frame, identity=match)
        state = gate.observe(match)
        # Liveness can only ever subtract. A frame that looks like a display never
        # unlocks anything, whatever the eigenface distance said.
        if scene.liveness is not None and scene.liveness.verdict == "suspect":
            state = dict(state)
            state["withheld"] = "liveness"
            state["withheld_reason"] = scene.summary_he
            if state.get("level") not in ("SAFE",):
                state["level"] = "SAFE"
        # Feed the brain. This is the wire that was missing: without it the
        # reasoning engine had no idea anybody was in the room, because the only
        # connection was FaceGate pushing a level into the firewall behind the
        # brain's back. Now every frame updates what the brain believes.
        try:
            from brain.presence import get_presence
            get_presence().observe(match, scene, gate)
        except Exception:
            pass
        return {"ok": True, "match": match.to_dict(), "gate": state,
                "scene": scene.to_dict(),
                "boxes": [f.to_dict() for f in found],
                "frame": {"w": int(frame.shape[1]), "h": int(frame.shape[0])}}

    out = await loop.run_in_executor(EXECUTOR, work)
    return _json(out, status=200 if out.get("ok") else 400)


async def api_faces_enroll(request: web.Request) -> web.Response:
    """POST /api/faces/enroll — teach JARVIS a face and what it is allowed to do."""
    store, gate = await asyncio.get_event_loop().run_in_executor(EXECUTOR, get_faces)
    try:
        payload = await request.json()
    except Exception:
        return _json({"ok": False, "error": "body must be JSON"}, status=400)
    name = str(payload.get("name") or "").strip()
    level = str(payload.get("level") or "SAFE").upper()
    if not name:
        return _json({"ok": False, "error": "a name is required"}, status=400)
    loop = asyncio.get_event_loop()

    def work() -> Dict[str, Any]:
        from vision.faces import LEVELS, encode_frame
        if level not in LEVELS:
            return {"ok": False, "error": f"level must be one of {list(LEVELS)}"}
        try:
            frame = _face_frame(payload)
        except Exception as exc:
            return {"ok": False, "error": f"bad frame: {exc}"}
        face = encode_frame(frame)
        if face is None:
            return {"ok": False, "error": "לא נמצאו פנים בתמונה — התקרב למצלמה והאר אותה"}
        person = store.enroll(name, face.vector, level=level,
                              person_id=payload.get("id"), note=str(payload.get("note") or ""))
        BUS.emit("vision.enroll", {"name": person.name, "level": person.level,
                                   "samples": len(person.vectors), "eyes": bool(face.eyes)},
                 source="faces")
        return {"ok": True, "person": person.to_dict(), "gate": gate.state(),
                "aligned_on_eyes": bool(face.eyes),
                "warning": None if face.eyes else
                "העיניים לא אותרו — היישור לפי תיבת הפנים בלבד, פחות מדויק"}

    out = await loop.run_in_executor(EXECUTOR, work)
    return _json(out, status=200 if out.get("ok") else 400)


async def api_faces_admin(request: web.Request) -> web.Response:
    """POST /api/faces/admin — remove, re-level, arm/disarm, poll the lease."""
    store, gate = await asyncio.get_event_loop().run_in_executor(EXECUTOR, get_faces)
    try:
        payload = await request.json()
    except Exception:
        return _json({"ok": False, "error": "body must be JSON"}, status=400)
    action = str(payload.get("action") or "").lower()
    loop = asyncio.get_event_loop()

    def work() -> Dict[str, Any]:
        if action == "remove":
            ok = store.remove(str(payload.get("id") or ""))
            return {"ok": ok, "error": None if ok else "no such person", "gate": gate.poll()}
        if action == "set_level":
            pid = str(payload.get("id") or "")
            lvl = str(payload.get("level") or "").upper()
            person = store.get(pid)
            if person is None:
                return {"ok": False, "error": "no such person", "gate": gate.poll()}
            ok = store.set_level(pid, lvl)
            if ok:
                BUS.emit("vision.level.grant", {"id": pid, "level": lvl}, source="faces")
                return {"ok": True, "error": None, "gate": gate.poll()}
            # Say which of the two it was. "no such person or bad level" on a
            # protected owner sends the user hunting for a typo that is not there.
            from vision.faces import LEVELS, OWNER_LEVEL
            if lvl not in LEVELS:
                err = f"level must be one of {list(LEVELS)}"
            elif person.is_owner:
                err = (f"{person.name} הוא הבעלים — הרשאת הבעלים ({OWNER_LEVEL}) מוגנת. "
                       "העבר בעלות קודם אם זה באמת מה שרצית.")
            else:
                err = "not changed"
            return {"ok": False, "error": err, "gate": gate.poll()}
        if action == "set_owner":
            ok = store.set_owner(str(payload.get("id") or ""))
            if ok:
                o = store.owner()
                BUS.emit("vision.owner", {"id": o.id, "name": o.name}, source="faces")
            return {"ok": ok, "error": None if ok else "no such person",
                    "gate": gate.poll()}
        if action == "arm":
            gate.armed = True
            return {"ok": True, "gate": gate.poll()}
        if action == "disarm":
            gate.disarm()
            return {"ok": True, "gate": gate.state()}
        if action == "poll":
            return {"ok": True, "gate": gate.poll()}
        if action == "configure":
            if payload.get("ttl"):
                gate.ttl = max(1.0, float(payload["ttl"]))
            if payload.get("ceiling"):
                gate.ceiling = str(payload["ceiling"]).upper()
            if payload.get("default_level"):
                gate.default_level = str(payload["default_level"]).upper()
            return {"ok": True, "gate": gate.poll()}
        return {"ok": False, "error": "action must be remove|set_level|arm|disarm|poll|configure"}

    out = await loop.run_in_executor(EXECUTOR, work)
    return _json(out, status=200 if out.get("ok") else 400)


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

    async def call(fn, *a, **kw):
        return await loop.run_in_executor(EXECUTOR, lambda: fn(*a, **kw))

    try:
        agent = await call(get_agent)
        # Greet before joining the broadcast list. The startup boot runs in a
        # background thread and, with warm caches, can finish within seconds —
        # close enough to a new connection that its system.ready event used to
        # overtake the hello on the wire. The HUD treats a leading event as
        # harmless, but "the first frame is the greeting" is a contract worth
        # keeping deterministic; anything emitted in the gap is covered by the
        # sync the HUD runs on open.
        await client.send({"type": "hello", "status": agent.status()})
        CLIENTS.append(client)

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
            elif kind == "voice_unmute":
                session = get_session(client.sid)
                if session is None:
                    await client.send({"type": "error",
                                       "message": "no voice session — send voice_start first"})
                else:
                    await call(session.unmute)
                    await client.send({"type": "control", "action": "voice_unmute", "ok": True,
                                       "state": session.state})

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
    app.router.add_get("/api/boot/progress", api_boot_progress)
    app.router.add_get("/api/modes", api_modes)
    app.router.add_post("/api/mode", api_mode_set)
    app.router.add_get("/api/events", api_events)
    app.router.add_get("/api/skills", api_skills)
    app.router.add_get("/api/audit", api_audit)
    app.router.add_get("/api/sentinel", api_sentinel)
    app.router.add_get("/api/bundle", api_bundle)
    app.router.add_get("/api/download", api_download)
    app.router.add_get("/api/screen/read", api_screen_read)
    app.router.add_get("/api/memory", api_memory)
    app.router.add_get("/api/rag/status", api_rag_status)
    app.router.add_post("/api/rag/search", api_rag_search)
    app.router.add_post("/api/rag/ask", api_rag_ask)
    app.router.add_post("/api/rag/index", api_rag_index)
    app.router.add_post("/api/rag/roots", api_rag_roots)
    app.router.add_get("/api/history", api_history)
    app.router.add_get("/api/permissions", api_permissions)
    app.router.add_get("/api/stt", api_stt)
    app.router.add_post("/api/listen", api_listen)
    app.router.add_post("/api/voice", api_voice)
    app.router.add_post("/api/audio", api_audio)
    app.router.add_get("/api/enroll", api_enroll)
    app.router.add_post("/api/enroll", api_enroll)
    app.router.add_get("/api/faces", api_faces)
    app.router.add_post("/api/faces/recognize", api_faces_recognize)
    app.router.add_post("/api/faces/enroll", api_faces_enroll)
    app.router.add_post("/api/faces/admin", api_faces_admin)
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
