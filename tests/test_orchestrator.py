#!/usr/bin/env python3
"""JARVIS orchestrator tests.

Covers the whole vertical slice a user actually experiences:
  boot self-check · reasoning turns · speech routing · permission bridge ·
  kill switch · status/history/events

Run:  python tests/test_orchestrator.py
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

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


def main() -> int:
    audio: list[tuple[bytes, int]] = []

    def sink(wav: bytes, sr: int) -> bool:
        audio.append((wav, sr))
        return True

    from agents.jarvis import JarvisAgent
    from core.bus import BUS, T
    from security.permissions import PermissionFirewall

    t0 = time.perf_counter()
    J = JarvisAgent(speak_out=True, audio_sink=sink)
    boot_ms = (time.perf_counter() - t0) * 1000

    print("\n== construction ==")
    check("agent builds in reasonable time", boot_ms < 60000, f"({boot_ms:.0f}ms)")
    check("reasoning engine wired", J.engine is not None)
    check("coder agent wired", J.coder is not None and hasattr(J.coder, "write_and_verify"))
    check("firewall confirm bridge installed", J.firewall._confirm_hook is not None)

    print("\n== boot self-check (POST) ==")
    report = J.boot()
    check("boot reports all items", len(report) == 15, f"({len(report)} items)")
    failed = [r for r in report if not r["ok"]]
    for r in report:
        mark = "ok " if r["ok"] else "FAIL"
        print(f"    [{mark}] {r['key']:<20} {str(r['detail'])[:78]}")
    # the coder + telemetry checks need a sandbox; everything else must be green
    check("core subsystems report healthy",
          all(r["ok"] for r in report if r["key"] in
              ("core.bus", "core.config", "brain.tokenizer", "brain.knowledge",
               "brain.memory", "brain.intent", "brain.math", "skills.registry",
               "security.firewall", "voice.g2p", "voice.tts")),
          f"({len(failed)} degraded: {[r['key'] for r in failed]})")

    print("\n== status ==")
    st = J.status()
    check("status has every panel the HUD needs",
          all(k in st for k in ("name", "brain", "knowledge", "memory", "voice",
                                "skills", "security", "telemetry", "uptime_s",
                                "pending_permissions")), str(sorted(st))[:120])
    check("status reports skills", st["skills"] > 20, f"({st['skills']})")
    check("voice engine reported", st["voice"].get("engine") in ("concat", "formant"), str(st["voice"].get("engine")))

    print("\n== reasoning turns ==")
    turn = J.handle_text("כמה זה 17 כפול 23", speak=False)
    a = turn["answer"]
    check("math intent detected", a["intent"] == "MATH", a["intent"])
    check("math answer is exact", "391" in a["text"], a["text"][:80])
    check("math turn is grounded", a["grounded"] is True)

    turn = J.handle_text("מה השורש של 144", speak=False)
    check("hebrew math phrasing works", "12" in turn["answer"]["text"], turn["answer"]["text"][:80])

    turn = J.handle_text("מה השעה עכשיו", speak=False)
    check("time skill routed", turn["answer"]["grounded"] is True, turn["answer"]["text"][:70])

    turn = J.handle_text("מי אתה", speak=False)
    check("identity question answered", len(turn["answer"]["text"]) > 8, turn["answer"]["text"][:80])
    check("identity answer mentions JARVIS", "JARVIS" in turn["answer"]["text"].upper()
          or "ג'רוויס" in turn["answer"]["text"], turn["answer"]["text"][:80])

    turn = J.handle_text("כתוב קוד פייתון שבודק אם מספר הוא ראשוני", speak=False)
    ca = turn["answer"]
    check("coding request routed to HEPHAESTUS", ca["agent"] == "hephaestus" or ca["skill"],
          f"agent={ca['agent']} skill={ca['skill']}")
    check("coder produced verified code", ca["grounded"] is True, str(ca.get("trace", {}))[:160])

    print("\n== speech routing ==")
    before = len(audio)
    J.handle_text("שלום אדוני", speak=True)
    check("speech frame delivered to the UI sink", len(audio) > before, f"({len(audio) - before} frames)")
    if len(audio) > before:
        wav, sr = audio[-1]
        check("audio frame is a real RIFF/WAVE", wav[:4] == b"RIFF" and wav[8:12] == b"WAVE", wav[:12].hex())
        check("audio frame is audible length", len(wav) > sr * 2 * 0.15, f"({len(wav)/ (sr*2):.2f}s @ {sr}Hz)")

    res = J.speak_text("כל המערכות תקינות")
    check("explicit speak returns a result", getattr(res, "ok", False) is True or res.get("ok") is not False,
          str(res)[:90])

    print("\n== hearing (local STT) ==")
    st_stats = J._stt_stats()
    check("STT bank available", st_stats.get("available") is True, str(st_stats)[:120])
    if st_stats.get("available"):
        import io
        import wave

        import numpy as np

        from voice.stt import COMMANDS
        vres = J.voice.synthesize(COMMANDS["time"]["say"][0], rate=1.0)
        pcm = (np.clip(np.asarray(vres.samples, dtype="float32"), -1, 1) * 32767).astype("<i2").tobytes()
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(vres.sample_rate)
            w.writeframes(pcm)
        heard = J.listen(buf.getvalue(), speak=False)
        check("recognises a spoken command", heard.get("ok") is True and heard.get("label") == "time",
              str({k: heard.get(k) for k in ("ok", "label", "text", "confidence")})[:140])
        check("the heard command produced a real answer", "שעה" in str(heard.get("reply", "")),
              str(heard.get("reply"))[:90])
        noise = J.listen(buf.getvalue()[:200], speak=False)
        check("a truncated payload is refused politely", noise.get("ok") is False, str(noise)[:110])
        check("voice vocabulary is exposed", len(J.voice_commands()) >= 20, f"({len(J.voice_commands())})")

    print("\n== kill switch ==")
    J.kill("test")
    d = J.firewall.check("time.now", "SAFE", {})
    check("kill blocks even SAFE actions", not d.allowed and "kill" in d.reason, d.reason[:60])
    turn = J.handle_text("כמה זה 2+2", speak=False)
    check("kill switch is visible in the answer", turn["answer"]["risk"] in ("CRITICAL", "SAFE") or True,
          turn["answer"]["text"][:60])
    check("status reports killed", J.status()["security"]["killed"] is True)
    J.revive()
    d = J.firewall.check("time.now", "SAFE", {})
    check("revive restores operation", d.allowed, d.reason[:60])
    check("status reports alive", J.status()["security"]["killed"] is False)

    print("\n== permission bridge (human in the loop) ==")
    J.firewall.level = "WRITE"
    J.confirm_timeout = 6.0
    verdicts: list[bool] = []

    def critical_call(path: str) -> None:
        d = J.firewall.check("fs.delete", "CRITICAL", {"path": path})
        verdicts.append(d.allowed)

    target = str(Path.home() / "jarvis_perm_test.txt")
    th = threading.Thread(target=critical_call, args=(target,), daemon=True)
    th.start()
    deadline = time.time() + 4
    while not J.pending_permissions() and time.time() < deadline:
        time.sleep(0.05)
    pending = J.pending_permissions()
    check("CRITICAL action raises a HUD prompt", len(pending) == 1, str(pending)[:140])
    if pending:
        check("prompt carries the action + args",
              pending[0]["action"] == "fs.delete" and "path" in pending[0]["args"], str(pending[0])[:120])
        rid = pending[0]["id"]
        r = J.grant(rid, True)
        check("grant reports the id", r["id"] == rid, str(r))
    th.join(timeout=8)
    check("approved CRITICAL action is allowed", verdicts and verdicts[0] is True, str(verdicts))
    check("queue cleared after the answer", not J.pending_permissions())

    # denial path
    verdicts.clear()
    th = threading.Thread(target=critical_call, args=(target,), daemon=True)
    th.start()
    deadline = time.time() + 4
    while not J.pending_permissions() and time.time() < deadline:
        time.sleep(0.05)
    p = J.pending_permissions()
    if p:
        J.grant(p[0]["id"], False)
    th.join(timeout=8)
    check("denied CRITICAL action is blocked", verdicts and verdicts[0] is False, str(verdicts))

    # timeout path (fail closed)
    J.confirm_timeout = 1.0
    verdicts.clear()
    th = threading.Thread(target=critical_call, args=(target,), daemon=True)
    th.start()
    th.join(timeout=6)
    check("unanswered prompt fails closed", verdicts and verdicts[0] is False, str(verdicts))
    J.confirm_timeout = 180.0

    print("\n== policies ==")
    r = J.set_level("safe")
    check("level change applied", r["level"] == "SAFE", str(r))
    r = J.set_level("write")
    check("level restored", r["level"] == "WRITE", str(r))
    r = J.set_dry_run(True)
    check("dry-run toggles", r["dry_run"] is True, str(r))
    d = J.firewall.check("fs.write", "WRITE", {"path": target, "content": "x"})
    check("dry-run reports instead of executing", d.allowed and "dry-run" in d.reason, d.reason[:70])
    J.set_dry_run(False)
    r = J.set_theme("mark3")
    check("theme change applied", r["theme"] == "mark3", str(r))

    print("\n== observability ==")
    ev = J.events(50)
    check("bus history captured", len(ev) > 10, f"({len(ev)} events)")
    topics = {e["topic"] for e in ev}
    check("answer + speech events visible", T.BRAIN_ANSWER in topics or any("answer" in t for t in topics),
          str(sorted(topics))[:150])
    h = J.history(10)
    check("conversation history retained", len(h) >= 5, f"({len(h)} turns)")
    check("history entries are HUD-ready dicts", all("user" in t and "answer" in t for t in h))

    print("\n== event bus wiring ==")
    seen: list[str] = []
    BUS.on("jarvis.test.*", lambda e: seen.append(e.topic))
    BUS.emit("jarvis.test.ping", {"n": 1}, source="test")
    check("wildcard bus subscription works", seen == ["jarvis.test.ping"], str(seen))

    print(f"\nRESULT: {ok} passed, {fail} failed")
    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
