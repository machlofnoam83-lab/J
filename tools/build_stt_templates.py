#!/usr/bin/env python3
"""Build (or rebuild) JARVIS's speech-recognition template bank.

JARVIS enrols itself: every command phrase is synthesised with our own TTS,
converted to an MFCC trajectory and stored. Then the bank is self-calibrated by
measuring same-phrase vs different-phrase DTW distances, which is what turns a
raw distance into an honest confidence score.

    python tools/build_stt_templates.py                 # default bank
    python tools/build_stt_templates.py --rates 0.9 1.0 1.1
    python tools/build_stt_templates.py --engines concat formant
    python tools/build_stt_templates.py --calibrate-only
    python tools/build_stt_templates.py --report
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from voice.stt import BANK_DIR, COMMANDS, SttEngine, build_bank, calibrate  # noqa: E402


def report() -> None:
    eng = SttEngine()
    if not eng.available:
        print(f"[!] no bank at {BANK_DIR} — build it first")
        return
    st = eng.stats()
    manifest = json.loads((BANK_DIR / "manifest.json").read_text(encoding="utf-8"))
    print(f"\n  bank        {BANK_DIR}")
    print(f"  built       {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(manifest['built_at']))} "
          f"({manifest['build_seconds']}s)")
    print(f"  templates   {st['templates']} across {st['commands']} commands")
    print(f"  engines     {', '.join(manifest.get('engines', []))} @ rates {manifest.get('rates')}")
    print(f"  sample rate {st['sample_rate']} Hz · {manifest.get('n_mfcc')} MFCC + Δ")
    cal = st["calibration"]
    print(f"  calibration self {cal.get('self_ref', 0):.2f} / cross {cal.get('cross_ref', 0):.2f} "
          f"(from {cal.get('samples', {}).get('self', 0)} + {cal.get('samples', {}).get('cross', 0)} pairs)")
    print(f"  enrolments  {st['enrolled']} from the user")
    size = (BANK_DIR / "templates.npz").stat().st_size
    print(f"  size        {size/1024:.0f} KB")
    print("\n  vocabulary:")
    for c in manifest["commands"]:
        wake = "  ← wake" if c.get("wake") else ""
        print(f"    {c['label']:<14} {c['text']:<22} {c['templates']} templates{wake}")
    print()


def main() -> int:
    ap = argparse.ArgumentParser(description="Build the JARVIS STT template bank")
    ap.add_argument("--rates", type=float, nargs="*", default=[0.94, 1.06])
    ap.add_argument("--engines", nargs="*", default=["concat"], choices=["concat", "formant"])
    ap.add_argument("--calibrate-only", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if args.report:
        report()
        return 0

    if args.calibrate_only:
        t0 = time.perf_counter()
        cal = calibrate(BANK_DIR)
        print(f"[✓] recalibrated in {time.perf_counter()-t0:.1f}s: "
              f"self {cal['self_ref']:.2f} / cross {cal['cross_ref']:.2f}")
        return 0

    print(f"[*] synthesising {len(COMMANDS)} commands with our own TTS "
          f"({', '.join(args.engines)} @ {args.rates})…")
    t0 = time.perf_counter()
    manifest = build_bank(bank_dir=BANK_DIR, rates=tuple(args.rates),
                          engines=tuple(args.engines), verbose=args.verbose)
    cal = manifest["calibration"]
    print(f"[✓] {manifest['templates']} templates / {len(manifest['commands'])} commands "
          f"in {time.perf_counter()-t0:.1f}s")
    print(f"[✓] calibration: same-phrase {cal['self_ref']:.2f} · different-phrase "
          f"{cal['cross_ref']:.2f} · separation {cal['cross_ref']/max(cal['self_ref'],1e-6):.1f}x")
    print(f"[✓] bank at {BANK_DIR} ({(BANK_DIR/'templates.npz').stat().st_size/1024:.0f} KB)")

    eng = SttEngine()
    if eng.available:
        from voice.tts import get_voice
        voice = get_voice()
        hits = total = 0
        t0 = time.perf_counter()
        for label, spec in COMMANDS.items():
            for phrase in spec.get("say", []):
                res = voice.synthesize(phrase, rate=1.0)
                r = eng.recognize(res.samples, res.sample_rate)
                total += 1
                hits += bool(r.ok and r.label == label)
        dt = time.perf_counter() - t0
        print(f"[✓] self-check: {hits}/{total} phrasings recognised, "
              f"{dt/max(total,1)*1000:.0f}ms each")
        if hits < total:
            print("    (run with --verbose to inspect the bank, or enrol your voice: "
                  "tools/enroll_voice.py --record)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
