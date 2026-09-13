#!/usr/bin/env python3
"""Quick conversational probe — how does the trained core actually talk?

Run:  python tools/probe_brain.py [question ...]
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEFAULT = [
    "האם אתה יכול ללמוד דברים חדשים",
    "מה אתה יודע לעשות",
    "למה השמיים כחולים",
    "ספר לי בדיחה",
    "מה דעתך על בינה מלאכותית",
    "תודה ג'רוויס",
    "כמה זה 48 חלקי 6",
    "מה מצב המחשב",
    "כתוב פונקציה שמחשבת את מספר פיבונאצ'י",
]


def main() -> int:
    from agents.jarvis import JarvisAgent

    J = JarvisAgent(speak_out=False)
    st = J.status()
    tr = (st["brain"].get("train") or {})
    print(f"[probe] core: {st['brain'].get('params', 0)/1e6:.2f}M params · "
          f"dev ppl {tr.get('ppl')} · {st['skills']} skills · voice {st['voice'].get('engine')}")

    for q in (sys.argv[1:] or DEFAULT):
        t0 = time.perf_counter()
        a = J.handle_text(q, speak=False)["answer"]
        ms = (time.perf_counter() - t0) * 1000
        print(f"\n❯ {q}")
        print(f"  [{a['intent']} grounded={a['grounded']} conf={a['confidence']:.2f} "
              f"skill={a['skill'] or '—'} agent={a['agent'] or '—'} {ms:.0f}ms]")
        text = a["text"].strip().replace("\n", "\n  ")
        print(f"  {text[:600]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
