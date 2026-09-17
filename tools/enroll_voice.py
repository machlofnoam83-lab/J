#!/usr/bin/env python3
"""Enrol YOUR voice into JARVIS's speech recogniser.

The bank ships with templates JARVIS synthesised from its own voice, which
recognises *that* voice well. Template matching is speaker-sensitive, so the
single biggest accuracy upgrade available is teaching it how **you** say each
command — no downloads, no cloud, thirty seconds of speaking.

    python tools/enroll_voice.py --list
    python tools/enroll_voice.py --record            # mic, one command at a time
    python tools/enroll_voice.py --label time --wav me_time.wav
    python tools/enroll_voice.py --dir my_recordings # files named <label>.wav
    python tools/enroll_voice.py --verify            # test the bank after enrolment

Every recording is stored as an MFCC trajectory in voice/stt/bank — the audio
itself never leaves the folder you point at.
"""

from __future__ import annotations

import argparse
import sys
import time
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from voice.stt import COMMANDS, SttEngine, get_engine  # noqa: E402


def print_vocabulary() -> None:
    print("\n  label          what to say                     JARVIS hears")
    print("  " + "─" * 68)
    for label, spec in COMMANDS.items():
        say = " / ".join(spec.get("say", [spec["text"]]))
        wake = "  ← wake word" if spec.get("wake") else ""
        print(f"  {label:<14} {say:<33} {spec['text']}{wake}")
    print()


def record_one(label: str, seconds: float = 3.0, sr: int = 16000) -> np.ndarray | None:
    try:
        import sounddevice as sd  # type: ignore
    except Exception:
        print("  [!] sounddevice is not installed — pip install sounddevice")
        print("      or record with any app and pass --wav <file>.")
        return None
    spec = COMMANDS[label]
    print(f"  say: {' / '.join(spec.get('say', [spec['text']]))}")
    for i in (3, 2, 1):
        print(f"    {i}…", end="\r", flush=True)
        time.sleep(0.6)
    print("    ● recording          ")
    audio = sd.rec(int(seconds * sr), samplerate=sr, channels=1, dtype="float32")
    sd.wait()
    x = np.asarray(audio, dtype="float32").reshape(-1)
    peak = float(np.abs(x).max()) if len(x) else 0.0
    print(f"    captured {len(x)/sr:.2f}s, peak {peak:.3f}")
    if peak < 0.01:
        print("    [!] that looks like silence — check the input device")
        return None
    return x


def save_wav(x: np.ndarray, sr: int, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype("<i2").tobytes())


def verify(eng: SttEngine, rounds: int = 1) -> int:
    from voice.tts import get_voice
    voice = get_voice()
    print("\n  verification (synthesised speech):")
    hits = total = 0
    for label, spec in COMMANDS.items():
        for phrase in spec.get("say", [])[:rounds + 1]:
            res = voice.synthesize(phrase, rate=1.0)
            r = eng.recognize(res.samples, res.sample_rate)
            total += 1
            good = r.ok and r.label == label
            hits += good
            print(f"    {'✓' if good else '✗'} {label:<13} {phrase!r:<26} -> "
                  f"{r.label or '—':<13} conf={r.confidence:.2f}")
    print(f"\n  recognised {hits}/{total}")
    return 0 if hits == total else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="Enrol your voice into JARVIS's recogniser")
    ap.add_argument("--list", action="store_true", help="show the command vocabulary")
    ap.add_argument("--record", action="store_true", help="record each command from the microphone")
    ap.add_argument("--label", default="", help="command label to enrol")
    ap.add_argument("--wav", default="", help="WAV file containing that command")
    ap.add_argument("--dir", default="", help="folder of <label>.wav recordings")
    ap.add_argument("--only", default="", help="comma separated labels for --record")
    ap.add_argument("--seconds", type=float, default=3.0)
    ap.add_argument("--verify", action="store_true", help="verify the bank")
    ap.add_argument("--keep-wav", action="store_true", help="keep recorded WAVs in data/enrolled/")
    args = ap.parse_args()

    if args.list:
        print_vocabulary()
        return 0

    eng = get_engine()
    if not eng.available:
        print("[*] no template bank yet — building it (JARVIS enrols itself)…")
        eng.ensure_bank()
    print(f"[*] bank: {eng.stats()['templates']} templates / {eng.stats()['commands']} commands")

    done = 0
    if args.record:
        labels = [s.strip() for s in args.only.split(",") if s.strip()] or list(COMMANDS)
        for label in labels:
            if label not in COMMANDS:
                print(f"  [!] unknown label {label!r}")
                continue
            x = record_one(label, args.seconds)
            if x is None:
                continue
            if args.keep_wav:
                save_wav(x, 16000, ROOT / "data" / "enrolled" / f"{label}.wav")
            tmp = ROOT / "data" / "enrolled" / f"_tmp_{label}.wav"
            save_wav(x, 16000, tmp)
            res = eng.enroll_wav(label, tmp, source="user-mic")
            if not args.keep_wav:
                tmp.unlink(missing_ok=True)
            print(f"    {'✓' if res.get('ok') else '✗'} enrolled {label}: {res}")
            done += bool(res.get("ok"))

    if args.label and args.wav:
        if args.label not in COMMANDS:
            print(f"[!] unknown label {args.label!r} — see --list")
            return 2
        res = eng.enroll_wav(args.label, args.wav, source="user-file")
        print(f"{'✓' if res.get('ok') else '✗'} {res}")
        done += bool(res.get("ok"))

    if args.dir:
        folder = Path(args.dir)
        if not folder.is_dir():
            print(f"[!] {folder} is not a folder")
            return 2
        for wav in sorted(folder.glob("*.wav")):
            label = wav.stem.strip().lower()
            if label not in COMMANDS:
                print(f"  [skip] {wav.name} — no such label (see --list)")
                continue
            res = eng.enroll_wav(label, wav, source="user-file")
            print(f"  {'✓' if res.get('ok') else '✗'} {label}: {res.get('frames', 0)} frames")
            done += bool(res.get("ok"))

    if done:
        eng2 = SttEngine()
        st = eng2.stats()
        print(f"\n[✓] enrolled {done} recording(s). bank now {st['templates']} templates "
              f"({st['enrolled']} from you).")

    if args.verify:
        return verify(eng)

    if not (args.record or args.label or args.dir or args.verify):
        print_vocabulary()
        print("nothing to do — pass --record, --label/--wav, --dir or --verify")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
