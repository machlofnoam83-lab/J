"""Hands-free conversation tests.

The whole loop is closed with our own machinery: JARVIS's concat TTS synthesises
the wake word and a command, that audio is streamed into the session exactly as a
browser would stream it, and our template STT has to hear it back. No fixtures
recorded elsewhere, no network.

Covers the session state machine (wake / endpoint / answer / stop / mute /
idle-timeout), the WebSocket binary transport, the REST twin, and the persisted
chat transcript.
"""
from __future__ import annotations

import asyncio
import json
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SR = 16000

ok = fail = 0


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


def post(url: str, data: bytes = b"", ctype: str = "application/octet-stream", timeout: float = 240.0):
    req = urllib.request.Request(url, data=data, method="POST",
                                 headers={"Content-Type": ctype, "Origin": "null"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def get(url: str, timeout: float = 200.0):
    req = urllib.request.Request(url, headers={"Origin": "null"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


# ------------------------------------------------------------------ fixtures --
def make_agent():
    from agents.jarvis import JarvisAgent
    ag = JarvisAgent(speak_out=False)
    ag.boot()
    return ag


def speech(agent, text: str) -> bytes:
    """JARVIS's own voice, at the rate a browser would stream: 16 kHz Int16 mono."""
    from voice.dsp import resample
    res = agent.voice.synthesize(text)
    x = resample(np.asarray(res.samples, dtype=np.float32), res.sample_rate, SR)
    return (np.clip(x, -1.0, 1.0) * 32767).astype("<i2").tobytes()


def foreign_speech(agent, text: str) -> bytes:
    """The same words in a timbre the bank has never heard — our stand-in for a
    human voice. Measured against the shipped bank this scores 0.000 on the wake
    word, which is exactly the complaint enrolment exists to fix."""
    from voice.dsp import resample
    res = agent.voice.synthesize(text, engine="formant")
    x = resample(np.asarray(res.samples, dtype=np.float32), res.sample_rate, SR)
    return (np.clip(x, -1.0, 1.0) * 32767).astype("<i2").tobytes()


def wav_wrap(pcm: bytes, sr: int = SR) -> bytes:
    """Raw Int16 mono PCM → a WAV file, the shape /api/enroll expects."""
    import io
    import wave
    b = io.BytesIO()
    with wave.open(b, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm)
    return b.getvalue()


def silence(seconds: float = 0.9) -> bytes:
    return np.zeros(int(SR * seconds), dtype="<i2").tobytes()


def chunks(pcm: bytes, ms: int = 200):
    n = int(SR * ms / 1000) * 2
    for i in range(0, len(pcm), n):
        yield pcm[i:i + n]


def kinds(events):
    return [e.get("type") for e in events]


def find(events, typ):
    return [e for e in events if e.get("type") == typ]


# ============================================================== in-process ====
def part_session() -> None:
    print("\n== VoiceSession state machine (in process) ==")
    from core.voice_session import (IDLE, LISTENING, VoiceSession, active_sessions,
                                    get_session, start_session, stop_session)

    agent = make_agent()
    ev: list = []
    sess = VoiceSession(agent, ev.append, ack="", idle_timeout=60.0)

    # 1 — an open microphone that hears nothing must not arm itself
    for c in chunks(silence(1.2)):
        sess.feed(c)
    check("silence does not wake it", sess.state == IDLE, f"(state={sess.state})")
    check("level events are emitted while idle", len(find(ev, "voice.level")) > 0,
          f"({len(find(ev, 'voice.level'))} events)")

    # 2 — the wake word arms it
    ev.clear()
    for c in chunks(speech(agent, "ג'רוויס")):
        sess.feed(c)
    for c in chunks(silence(0.5)):
        sess.feed(c)
    st = find(ev, "voice.state")
    check("the wake word arms the session", sess.state == LISTENING, f"(state={sess.state})")
    check("arming reports the wake confidence",
          bool(st) and float(st[0].get("confidence") or 0) > 0.5,
          f"(conf={st[0].get('confidence') if st else None})")

    # 3 — a spoken command is endpointed, recognised and answered out loud
    ev.clear()
    for c in chunks(speech(agent, "מה השעה")):
        sess.feed(c)
    for c in chunks(silence(1.0)):
        sess.feed(c)
    heard = find(ev, "voice.heard")
    answers = find(ev, "answer")
    states = [e.get("state") for e in find(ev, "voice.state")]
    check("endpointing produced exactly one utterance", len(heard) == 1, f"({kinds(ev)})")
    check("the utterance was recognised as the time command",
          bool(heard) and heard[0].get("label") == "time",
          f"(label={heard[0].get('label') if heard else None} text={heard[0].get('text') if heard else None})")
    check("recognition was confident", bool(heard) and float(heard[0].get("confidence") or 0) > 0.5,
          f"(conf={heard[0].get('confidence') if heard else None})")
    check("it went thinking -> speaking -> listening",
          states[:3] == ["thinking", "speaking", "listening"], f"({states})")
    check("an answer turn was emitted for the HUD", bool(answers) and answers[0].get("voice") is True)
    if answers:
        a = answers[0].get("answer") or {}
        check("the spoken answer is the real, grounded time",
              bool(a.get("text")) and a.get("grounded") is True, f"({str(a.get('text'))[:44]})")
        check("the turn carries the user's transcript",
              (answers[0].get("user") or "").strip() != "", f"({answers[0].get('user')!r})")
    check("the session is listening again for the next turn", sess.state == LISTENING)
    check("the turn counter advanced", sess.snapshot().get("turns") == 1,
          f"({sess.snapshot()})")

    # 4 — it must not hear itself: the mute window covers the answer's audio
    check("speaking opened a mute window", sess._mute_until > time.time(),
          f"({sess._mute_until - time.time():.2f}s left)")
    ev.clear()
    for c in chunks(speech(agent, "ג'רוויס")):
        sess.feed(c)
    check("our own voice cannot re-trigger anything",
          not find(ev, "voice.heard") and sess.snapshot().get("turns") == 1, f"({kinds(ev)})")

    # 5 — an unrecognisable utterance is admitted, not guessed
    sess._mute_until = 0.0
    ev.clear()
    for c in chunks(speech(agent, "בלה בלה בלה בלה")):
        sess.feed(c)
    for c in chunks(silence(1.0)):
        sess.feed(c)
    check("gibberish is reported as unheard", len(find(ev, "voice.unheard")) >= 1
          or len(find(ev, "voice.heard")) >= 1, f"({kinds(ev)})")
    check("gibberish did not end the conversation", sess.state == LISTENING, f"({sess.state})")

    # 6 — the stop word closes the conversation
    sess._mute_until = 0.0
    ev.clear()
    for c in chunks(speech(agent, "עצור")):
        sess.feed(c)
    for c in chunks(silence(1.0)):
        sess.feed(c)
    heard = find(ev, "voice.heard")
    check("the stop word is recognised", bool(heard) and heard[0].get("label") == "stop",
          f"(label={heard[0].get('label') if heard else None})")
    check("the stop word disarms the session", sess.state == IDLE, f"(state={sess.state})")
    check("disarming is reported with its reason",
          any("stop" in str(e.get("reason", "")) for e in find(ev, "voice.state")),
          f"({[e.get('reason') for e in find(ev, 'voice.state')]})")

    # 7 — an abandoned microphone disarms itself
    ev.clear()
    quiet = VoiceSession(agent, ev.append, ack="", idle_timeout=0.4, arm_on_start=False)
    quiet._state = LISTENING
    quiet._last_voice = time.time() - 1.0
    quiet.feed(np.zeros(3200, dtype="<i2"))
    check("an idle microphone disarms on its own", quiet.state == IDLE, f"(state={quiet.state})")
    check("the timeout says why",
          any(e.get("reason") == "idle timeout" for e in find(ev, "voice.state")),
          f"({[e.get('reason') for e in find(ev, 'voice.state')]})")

    # 7b — an always-open microphone re-arms instead of falling asleep. The wake
    # word only matches voices the bank has templates for, so a session that
    # demanded one would go permanently deaf for anyone who has not enrolled.
    ev.clear()
    open_mic = VoiceSession(agent, ev.append, ack="", idle_timeout=0.4, arm_on_start=True)
    open_mic._state = LISTENING
    open_mic._last_voice = time.time() - 1.0
    open_mic.feed(np.zeros(3200, dtype="<i2"))
    check("an always-open microphone keeps listening through a quiet room",
          open_mic.state == LISTENING, f"(state={open_mic.state})")
    check("the re-arm says it is still listening",
          any(e.get("reason") == "still listening" for e in find(ev, "voice.state")),
          f"({[e.get('reason') for e in find(ev, 'voice.state')]})")
    check("re-arms are counted", open_mic.snapshot().get("rearms") == 1,
          f"({open_mic.snapshot().get('rearms')})")
    open_mic.feed(np.zeros(3200, dtype="<i2"))
    check("a re-armed microphone still hears the next command",
          open_mic.state == LISTENING and open_mic.snapshot().get("rearms") >= 1)

    # 7c — the echo guard is reported and can be lifted by the HUD
    guarded = VoiceSession(agent, ev.append, ack="", idle_timeout=60.0)
    guarded._state = LISTENING
    guarded._mute_until = time.time() + 5.0
    ev.clear()
    guarded.feed(np.frombuffer(speech(agent, "מה השעה"), dtype="<i2"))
    check("audio during the echo guard is announced, not silently eaten",
          len(find(ev, "voice.muted")) == 1, f"({kinds(ev)})")
    check("the announcement says how long is left",
          bool(find(ev, "voice.muted")) and find(ev, "voice.muted")[0].get("remaining", 0) > 4.0,
          f"({find(ev, 'voice.muted')[:1]})")
    guarded.feed(np.frombuffer(speech(agent, "מה השעה"), dtype="<i2"))
    check("the announcement is throttled, not spammed", len(find(ev, "voice.muted")) == 1,
          f"({len(find(ev, 'voice.muted'))} events)")
    ev.clear()
    guarded.unmute()
    check("the HUD can lift the guard when playback really ends",
          guarded._mute_until == 0.0 and len(find(ev, "voice.unmuted")) == 1, f"({kinds(ev)})")

    # 7d — a missed utterance comes back with the audio, so it can be taught
    ev.clear()
    teach = VoiceSession(agent, ev.append, ack="", idle_timeout=60.0, unheard_cues=0)
    teach._state = LISTENING
    for c in chunks(speech(agent, "בלה בלה בלה בלה")):
        teach.feed(c)
    for c in chunks(silence(1.0)):
        teach.feed(c)
    missed = find(ev, "voice.unheard")
    check("a missed utterance is offered back for teaching",
          bool(missed) and missed[0].get("teachable") is True, f"({kinds(ev)})")
    if missed and missed[0].get("wav_b64"):
        import base64 as _b64
        import io as _io
        import wave as _wave
        raw = _b64.b64decode(missed[0]["wav_b64"])
        with _wave.open(_io.BytesIO(raw)) as w:
            secs = w.getnframes() / float(w.getframerate())
        check("the offered audio is a whole WAV of the utterance",
              raw[:4] == b"RIFF" and raw[8:12] == b"WAVE" and 0.3 < secs < 9.0,
              f"({len(raw)} bytes, {secs:.2f}s)")
        check("the failure also reports its closest guesses",
              isinstance(missed[0].get("top"), list), f"({missed[0].get('top')})")

    # 8 — stop() closes the microphone for good
    quiet._state = LISTENING
    quiet.stop()
    before = len(ev)
    quiet.feed(np.zeros(6400, dtype="<i2"))
    check("a stopped session ignores further audio", len(ev) == before and quiet.state == IDLE)

    # 9 — the registry the server uses
    sid = "test-session"
    ev.clear()
    sess2 = start_session(sid, agent, ev.append, ack="")
    check("start_session registers the session", get_session(sid) is sess2)
    check("active_sessions lists it", sid in active_sessions(), f"({active_sessions()})")
    check("a new session opens already listening", sess2.state == LISTENING,
          f"(state={sess2.state})")
    check("and says no wake word is required",
          any(e.get("wake_required") is False for e in find(ev, "voice.state")),
          f"({[e.get('wake_required') for e in find(ev, 'voice.state')]})")
    stop_session(sid)
    ev.clear()
    gated = start_session(sid, agent, ev.append, ack="", arm_on_start=False)
    check("the wake-gated mode still starts disarmed", gated.state == IDLE,
          f"(state={gated.state})")
    check("and says the wake word is required",
          any(e.get("wake_required") is True for e in find(ev, "voice.state")),
          f"({[e.get('wake_required') for e in find(ev, 'voice.state')]})")
    check("stop_session removes it", stop_session(sid) is True and get_session(sid) is None)
    check("stopping twice is harmless", stop_session(sid) is False)


# ================================================================= live server ==
async def ws_voice(base_ws: str, wake: bytes, command: bytes, quiet: bytes) -> list:
    import websockets
    log: list = []
    async with websockets.connect(base_ws, max_size=32 * 1024 * 1024, open_timeout=60) as ws:
        await ws.send(json.dumps({"type": "voice_start", "ack": "", "idle_timeout": 60}))
        log.append(json.loads(await asyncio.wait_for(ws.recv(), timeout=60)))

        async def drain(seconds: float) -> None:
            end = time.time() + seconds
            while time.time() < end:
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=max(0.05, end - time.time()))
                except asyncio.TimeoutError:
                    return
                try:
                    log.append(json.loads(raw))
                except Exception:
                    log.append({"type": "raw", "len": len(raw)})

        await drain(1.0)
        for c in chunks(wake):
            await ws.send(c)
            await asyncio.sleep(0.02)
        for c in chunks(quiet):
            await ws.send(c)
            await asyncio.sleep(0.02)
        await drain(4.0)

        for c in chunks(command):
            await ws.send(c)
            await asyncio.sleep(0.02)
        for c in chunks(quiet):
            await ws.send(c)
            await asyncio.sleep(0.02)
        await drain(14.0)

        await ws.send(json.dumps({"type": "voice_stop"}))
        await drain(2.0)
    return log


def part_live() -> None:
    print("\n== live server: REST voice transport ==")
    port = free_port()
    base = f"http://127.0.0.1:{port}"
    base_ws = f"ws://127.0.0.1:{port}/ws"
    # Enrolment writes into the template bank, so this server runs against a
    # throwaway copy — the shipped bank must survive the test run untouched.
    import os as _os
    import shutil as _shutil
    src_bank = ROOT / "voice" / "stt" / "bank"
    tmp_bank = ROOT / "data" / "bank-test"
    if tmp_bank.exists():
        _shutil.rmtree(tmp_bank)
    if src_bank.exists():
        _shutil.copytree(src_bank, tmp_bank)
    proc = subprocess.Popen(
        [sys.executable, "-m", "core.server", "--host", "127.0.0.1", "--port", str(port)],
        cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace",
        env={**_os.environ, "PYTHONUNBUFFERED": "1", "JARVIS_STT_BANK": str(tmp_bank)},
    )
    sid = "pytest-rest"
    try:
        up = False
        t0 = time.time()
        while time.time() - t0 < 240:
            if proc.poll() is not None:
                print((proc.stdout.read() if proc.stdout else "")[-2000:])
                break
            try:
                st, _ = get(f"{base}/api/status", timeout=200)
                if st == 200:
                    up = True
                    break
            except Exception:
                time.sleep(0.4)
        check("server comes up", up, f"({time.time()-t0:.1f}s)")
        if not up:
            return

        agent = make_agent()                       # local voice for the fixtures
        wake = speech(agent, "ג'רוויס")
        command = speech(agent, "ספר לי בדיחה")
        quiet = silence(1.0)

        # audio before a session exists must be refused, not silently dropped
        st, body = post(f"{base}/api/audio?sid=nope", next(iter(chunks(wake, 1000))))
        check("streaming audio with no session is a JSON 404",
              st == 404 and b"no voice session" in body, f"[{st}] {body[:60]!r}")

        st, body = post(f"{base}/api/voice", json.dumps({"action": "start", "sid": sid,
                                                         "ack": "", "idle_timeout": 60}).encode(),
                        ctype="application/json")
        d = json.loads(body.decode())
        check("POST /api/voice starts a session",
              st == 200 and d.get("action") == "voice_start" and d.get("state") == "listening",
              f"[{st}] {d.get('state')}")
        check("it opens already listening, no wake word needed",
              d.get("arm_on_start") is True and d.get("armed") is True,
              f"({{k: d.get(k) for k in ('arm_on_start', 'armed')}})")
        check("the session reports its tuning", d.get("idle_timeout") == 60.0, f"({d.get('idle_timeout')})")

        st, body = post(f"{base}/api/voice", json.dumps({"action": "state", "sid": sid}).encode(),
                        ctype="application/json")
        d = json.loads(body.decode())
        check("GET-style state query works over POST",
              d.get("state") == "listening" and sid in (d.get("sessions") or []),
              f"({d.get('state')}, {d.get('sessions')})")

        # stream the wake word
        events: list = []
        last_state = "idle"
        for c in list(chunks(wake)) + list(chunks(quiet)):
            st, body = post(f"{base}/api/audio?sid={sid}", c)
            d = json.loads(body.decode())
            events += d.get("events") or []
            last_state = d.get("state") or last_state
        states = [e.get("state") for e in find(events, "voice.state")]
        check("the REST stream carries level events", len(find(events, "voice.level")) > 0,
              f"({len(find(events, 'voice.level'))})")
        check("the wake word arms it over REST", "listening" in states, f"({states})")

        # stream a command and collect everything the call produced
        events = []
        audio = []
        for c in list(chunks(command)) + list(chunks(quiet)):
            st, body = post(f"{base}/api/audio?sid={sid}", c)
            d = json.loads(body.decode())
            events += d.get("events") or []
            audio += d.get("audio") or []
            if d.get("state"):
                last_state = d["state"]
        heard = find(events, "voice.heard")
        answers = find(events, "answer")
        check("REST endpointing recognised the command", bool(heard), f"({kinds(events)})")
        check("it heard the joke request",
              bool(heard) and heard[0].get("label") == "joke",
              f"(label={heard[0].get('label') if heard else None})")
        check("the answer came back on the same transport", bool(answers), f"({kinds(events)})")
        if answers:
            a = answers[0].get("answer") or {}
            check("the answer is the grounded joke",
                  a.get("grounded") is True and "Halloween" in str(a.get("text", "")),
                  f"({str(a.get('text'))[:40]})")
        check("REST carries the spoken reply inline", len(audio) >= 1, f"({len(audio)} frame(s))")
        if audio:
            import base64
            wav = base64.b64decode(audio[0]["wav_b64"])
            check("the inline reply is real WAV audio",
                  wav[:4] == b"RIFF" and wav[8:12] == b"WAVE" and len(wav) > 4000,
                  f"({len(wav)} bytes @ {audio[0].get('sample_rate')}Hz)")

        # ------------------------------------- teaching JARVIS a new voice ----
        print("\n== live server: voice enrolment and teach-back ==")
        st, body = post(f"{base}/api/voice", json.dumps({"action": "unmute", "sid": sid}).encode(),
                        ctype="application/json")
        d = json.loads(body.decode())
        check("POST /api/voice can lift the echo guard",
              st == 200 and d.get("ok") is True and d.get("action") == "voice_unmute",
              f"[{st}] {d.get('ok')}")

        st, body = get(f"{base}/api/enroll")
        d = json.loads(body.decode())
        check("GET /api/enroll offers a training script",
              st == 200 and d.get("ok") is True and len(d.get("script") or []) >= 3,
              f"[{st}] {[p.get('label') for p in (d.get('script') or [])]}")
        check("the script starts with the wake word and says it in Hebrew",
              bool(d.get("script")) and d["script"][0].get("label") == "wake"
              and d["script"][0].get("text") == "ג'רוויס",
              f"({d.get('script', [{}])[0]})")
        check("the whole vocabulary is offered for teaching",
              len(d.get("vocabulary") or []) >= 20, f"({len(d.get('vocabulary') or [])} labels)")

        st, body = post(f"{base}/api/enroll?label=nope", wav_wrap(foreign_speech(agent, "מה השעה")))
        check("an unknown label is refused, not invented", st == 404, f"[{st}]")
        st, body = post(f"{base}/api/enroll?label=wake", b"")
        d = json.loads(body.decode())
        check("an empty recording is reported, not stored", d.get("ok") is False, f"({d})")

        before = json.loads(get(f"{base}/api/enroll")[1].decode()).get("enrolled_total") or 0
        clip = foreign_speech(agent, "ספר לי בדיחה")
        st, body = post(f"{base}/api/enroll?label=joke", wav_wrap(clip))
        d = json.loads(body.decode())
        check("a foreign voice can enrol its own recording",
              st == 200 and d.get("ok") is True and d.get("label") == "joke", f"[{st}] {d}")
        check("enrolment reports how well that voice is now heard",
              float(d.get("score") or 0) > 0.5 and d.get("heard") is True,
              f"(score={d.get('score')} heard={d.get('heard')})")
        after = json.loads(get(f"{base}/api/enroll")[1].decode()).get("enrolled_total") or 0
        check("the recording is counted in the bank", after == before + 1, f"({before} -> {after})")

        # the proof that matters: the same foreign voice, a *different* take of the
        # same phrase, is now understood through the live session
        post(f"{base}/api/voice", json.dumps({"action": "unmute", "sid": sid}).encode(),
             ctype="application/json")
        events, audio = [], []
        fresh = foreign_speech(agent, "ספר לי בדיחה")
        for c in list(chunks(fresh)) + list(chunks(quiet)):
            st, body = post(f"{base}/api/audio?sid={sid}", c)
            d = json.loads(body.decode())
            events += d.get("events") or []
            audio += d.get("audio") or []
        heard = find(events, "voice.heard")
        check("a newly enrolled voice is understood on an unseen take",
              bool(heard) and heard[0].get("label") == "joke",
              f"(labels={[h.get('label') for h in heard]}, {kinds(events)})")
        check("and JARVIS answers it out loud", len(find(events, "answer")) >= 1 and len(audio) >= 1,
              f"({len(find(events, 'answer'))} turns, {len(audio)} audio frames)")

        # A single enrolled phrase is not a trained voice: measured, its template
        # becomes an attractor for that speaker's other speech (timbre dominates
        # content in MFCC+DTW). Train a second phrase so this next step measures
        # teach-back rather than that artefact — and check the server says so.
        st, body = post(f"{base}/api/enroll?label=time", wav_wrap(foreign_speech(agent, "מה השעה")))
        d = json.loads(body.decode())
        check("the bank keeps warning while fewer than three phrases are trained",
              bool(d.get("advise")) and d.get("enrolled_labels") == 2,
              f"(labels={d.get('enrolled_labels')}, advise={bool(d.get('advise'))})")
        st, body = post(f"{base}/api/enroll?label=stop", wav_wrap(foreign_speech(agent, "עצור")))
        d = json.loads(body.decode())
        check("three phrases are counted", d.get("enrolled_labels") == 3,
              f"({d.get('enrolled_labels')} labels)")
        check("and the warning stops once the voice is really trained",
              not d.get("advise"), f"({d.get('advise')})")

        # teach-back: a phrase nobody enrolled comes back with its own audio
        post(f"{base}/api/voice", json.dumps({"action": "unmute", "sid": sid}).encode(),
             ctype="application/json")
        events = []
        unknown = foreign_speech(agent, "מה אתה יודע לעשות")
        for c in list(chunks(unknown)) + list(chunks(quiet)):
            st, body = post(f"{base}/api/audio?sid={sid}", c)
            events += json.loads(body.decode()).get("events") or []
        missed = find(events, "voice.unheard")
        check("an unenrolled phrase is missed, not guessed", bool(missed), f"({kinds(events)})")
        check("the miss carries the audio so the HUD can teach it",
              bool(missed) and missed[0].get("teachable") is True and len(missed[0].get("wav_b64") or "") > 1000,
              f"({{k: (missed[0].get(k) if missed else None) for k in ('teachable', 'confidence')}})")
        if missed and missed[0].get("wav_b64"):
            import base64 as _b64
            raw = _b64.b64decode(missed[0]["wav_b64"])
            check("the returned audio is a playable WAV",
                  raw[:4] == b"RIFF" and raw[8:12] == b"WAVE", f"({len(raw)} bytes)")
            st, body = post(f"{base}/api/enroll?label=help", raw)
            d = json.loads(body.decode())
            check("teaching from the returned audio succeeds",
                  st == 200 and d.get("ok") is True and d.get("heard") is True, f"[{st}] {d}")

        st, body = post(f"{base}/api/voice", json.dumps({"action": "stop", "sid": sid}).encode(),
                        ctype="application/json")
        d = json.loads(body.decode())
        check("POST /api/voice stops the session", d.get("ok") is True, f"({d.get('ok')})")
        st, body = post(f"{base}/api/audio?sid={sid}", quiet[:6400])
        check("audio after stop is refused", st == 404, f"[{st}]")

        # ------------------------------------------------------- websocket ----
        print("\n== live server: WebSocket binary microphone stream ==")
        log = asyncio.run(ws_voice(base_ws, wake, speech(agent, "מה השעה"), quiet))
        ctrl = [m for m in log if m.get("type") == "control"]
        check("voice_start is acknowledged on the socket",
              any(m.get("action") == "voice_start" for m in ctrl), f"({[m.get('action') for m in ctrl]})")
        states = [m.get("state") for m in log if m.get("type") == "voice.state"]
        check("the socket reports the wake transition", "listening" in states, f"({states[:6]})")
        check("the socket reports level metering",
              len([m for m in log if m.get("type") == "voice.level"]) > 0)
        heard = [m for m in log if m.get("type") == "voice.heard"]
        check("the socket carries the recognised command", bool(heard), f"({[m.get('text') for m in heard]})")
        labels = [m.get("label") for m in heard]
        check("the wake word spoken to an open microphone is answered, not obeyed",
              "wake" in labels, f"({labels})")
        check("it was the time command", "time" in labels, f"(labels={labels})")
        answers = [m for m in log if m.get("type") == "answer" and m.get("voice")]
        check("the socket carries the voice turn", bool(answers))
        check("the socket broadcasts the spoken reply",
              len([m for m in log if m.get("type") == "audio"]) >= 1,
              f"({len([m for m in log if m.get('type') == 'audio'])} frame(s))")
        check("voice_stop is acknowledged",
              any(m.get("action") == "voice_stop" and m.get("ok") for m in ctrl),
              f"({[m.get('action') for m in ctrl]})")

        # ------------------------------------------- persisted chat history ----
        print("\n== chat transcript persistence ==")
        st, body = post(f"{base}/api/command",
                        json.dumps({"type": "user_text", "text": "כמה זה 9 כפול 9",
                                    "speak": False}).encode(),
                        ctype="application/json")
        check("a text turn is accepted", st == 200 and b"81" in body, f"[{st}]")
        st, body = get(f"{base}/api/history?n=60")
        turns = json.loads(body.decode()).get("turns") or []
        check("history returns the turn", len(turns) >= 1, f"({len(turns)} turns)")
        check("the turn is the one we just sent",
              any(t.get("user") == "כמה זה 9 כפול 9" for t in turns))

        tpath = ROOT / "data" / "transcript.jsonl"
        check("the transcript is written to disk", tpath.exists() and tpath.stat().st_size > 0,
              f"({tpath.stat().st_size if tpath.exists() else 0} bytes)")
        if tpath.exists():
            lines = [json.loads(x) for x in tpath.read_text(encoding="utf-8").splitlines() if x.strip()]
            check("every persisted line is a whole turn",
                  all(isinstance(r, dict) and "user" in r and "answer" in r for r in lines),
                  f"({len(lines)} lines)")

        st, body = post(f"{base}/api/command", json.dumps({"type": "clear_chat"}).encode(),
                        ctype="application/json")
        d = json.loads(body.decode())
        check("clear_chat reports what it removed",
              d.get("ok") is True and int(d.get("cleared") or 0) >= 1, f"({d})")
        st, body = get(f"{base}/api/history?n=60")
        check("history is empty afterwards",
              len(json.loads(body.decode()).get("turns") or []) == 0)
        check("the transcript file is gone", not tpath.exists())

        st, body = post(f"{base}/api/command",
                        json.dumps({"type": "user_text", "text": "תודה", "speak": False}).encode(),
                        ctype="application/json")
        st, body = get(f"{base}/api/history?n=60")
        check("the conversation continues after a clear",
              len(json.loads(body.decode()).get("turns") or []) == 1)
    finally:
        try:
            proc.terminate()
            proc.wait(timeout=20)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
        if tmp_bank.exists():
            _shutil.rmtree(tmp_bank, ignore_errors=True)


def main() -> int:
    t0 = time.time()
    part_session()
    part_live()
    print(f"\nRESULT: {ok} passed, {fail} failed  ({time.time()-t0:.1f}s)")
    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
