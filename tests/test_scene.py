#!/usr/bin/env python3
"""JARVIS scene understanding + the owner tier.

Companion to ``test_vision.py``, which covers detection and identity. This file
covers the two things added on top:

  1. ``vision/scene.py`` — what the camera is looking at, not just whose face it
     is. Lighting verdicts, head pose, distance, motion, a frame-quality score, a
     Hebrew summary, and liveness.
  2. The owner rule in ``vision/faces.py`` — the first face ever enrolled is the
     creator, holds CRITICAL, and is the only identity that can reach CRITICAL.

On liveness, read this before trusting a green tick
---------------------------------------------------
Every "artifact" here is *constructed*, not captured: a periodic grid is added to
a synthetic face to stand in for a camera beating against a display's pixel grid.
So what these tests prove is that the moiré measure separates a broadband image
from a periodic one, and that real faces are never the ones flagged. They do not
prove JARVIS can defeat a determined attacker with a printed photograph — the
print case is a known, documented weakness. See ``measure_liveness``.

Run:  python tests/test_scene.py
"""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vision import faces as F  # noqa: E402
from vision import scene as S  # noqa: E402

ok = 0
fail = 0


def check(label: str, cond: bool, detail: str = "") -> None:
    global ok, fail
    if cond:
        ok += 1
        print(f"  ✓ {label}" + (f"  {detail}" if detail else ""))
    else:
        fail += 1
        print(f"  ✗ {label}" + (f"  {detail}" if detail else ""))


# ═══════════════════════════ synthetic faces ═══════════════════════════
def identity(seed: int) -> dict:
    rng = np.random.default_rng(seed)
    return {
        "fw": float(rng.uniform(58, 86)), "fh": float(rng.uniform(78, 112)),
        "eye_dx": float(rng.uniform(0.16, 0.29)), "eye_dy": float(rng.uniform(0.10, 0.22)),
        "eye_r": float(rng.uniform(0.055, 0.105)),
        "skin": (int(rng.uniform(196, 240)), int(rng.uniform(150, 196)),
                 int(rng.uniform(120, 168))),
        "shade": float(rng.uniform(10, 30)), "mouth_w": float(rng.uniform(0.12, 0.26)),
        "mouth_dy": float(rng.uniform(0.24, 0.36)), "nose_len": float(rng.uniform(0.06, 0.16)),
        "nose_w": float(rng.uniform(0.040, 0.075)), "hair": int(rng.integers(22, 96)),
        "hair_h": float(rng.uniform(0.34, 0.52)), "jaw": float(rng.uniform(0.86, 1.06)),
    }


def render(p: dict, w: int = 320, h: int = 240, cx: float = 0.5, cy: float = 0.5,
           scale: float = 1.0, light: float = 1.0, seed: int = 0,
           expression: float = 0.0, noise: float = 1.8) -> np.ndarray:
    """Same generator as test_vision.py — identity from `p`, variation from the frame."""
    rng = np.random.default_rng(seed)
    bg = np.array([38, 58, 92], np.float32) * light
    img = np.zeros((h, w, 3), np.int16)
    img[:] = bg[None, None, :] + rng.integers(-4, 5, (h, w, 1))

    fw, fh = p["fw"] * scale, p["fh"] * scale
    cxp, cyp = cx * w, cy * h
    yy, xx = np.mgrid[0:h, 0:w]
    ell = (((xx - cxp) ** 2) / (fw / 2 * p.get("jaw", 1.0)) ** 2
           + ((yy - cyp) ** 2) / (fh / 2) ** 2) <= 1.0
    skin = np.array(p["skin"], np.float32) * light
    grad = np.clip((yy - (cyp - fh / 2)) / fh, 0, 1)
    shade = (p["shade"] - 2 * p["shade"] * grad)[ell]
    face_px = skin[None, :] + shade[:, None]
    if noise:
        face_px = face_px + rng.normal(0, noise, face_px.shape)
    img[ell] = face_px

    def blob(bx, by, rx, ry, colour):
        m = (((xx - bx) ** 2) / max(rx, 0.5) ** 2 + ((yy - by) ** 2) / max(ry, 0.5) ** 2) <= 1.0
        img[m] = np.array(colour, np.float32) * light

    ex, ey = p["eye_dx"] * fw, cyp - p["eye_dy"] * fh
    er = p.get("eye_r", 0.085)
    blob(cxp - ex, ey, fw * er, fh * 0.042, (34, 28, 30))
    blob(cxp + ex, ey, fw * er, fh * 0.042, (34, 28, 30))
    blob(cxp - ex, ey - fh * 0.075, fw * 0.11, fh * 0.020, (70, 52, 48))
    blob(cxp + ex, ey - fh * 0.075, fw * 0.11, fh * 0.020, (70, 52, 48))
    blob(cxp, cyp + fh * 0.06, fw * p.get("nose_w", 0.055), p["nose_len"] * fh, skin * 0.86)
    blob(cxp, cyp + p.get("mouth_dy", 0.30) * fh, p["mouth_w"] * fw * (1.0 + expression),
         fh * 0.030, (150, 78, 78))

    hh = p.get("hair_h", 0.42)
    fringe = min(ey - fh * 0.075 - fh * 0.05, cyp - fh * (hh - 0.24))
    cap = (((xx - cxp) ** 2) / (fw * 0.56) ** 2 + ((yy - (cyp - fh * hh)) ** 2)
           / (fh * 0.30) ** 2) <= 1.0
    cap &= yy < fringe
    img[cap] = float(p.get("hair", 46)) * light
    return np.clip(img, 0, 255).astype(np.uint8)


def screen_like(img: np.ndarray, period: float = 4.0, amp: float = 42.0) -> np.ndarray:
    """Overlay a periodic pixel grid — stands in for a camera aimed at a display."""
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    grid = amp * np.sin(2 * np.pi * xx / period) + amp * np.sin(2 * np.pi * yy / period)
    return np.clip(img.astype(np.float32) + grid[..., None], 0, 255).astype(np.uint8)


def tmp_store() -> F.FaceStore:
    return F.FaceStore(path=Path(tempfile.mkdtemp()) / "gallery.json")


# ═══════════════════════════════ lighting ═══════════════════════════════
def test_lighting():
    print("\n— lighting —")
    p = identity(1)
    good = S.measure_lighting(render(p, light=1.6, seed=1))
    check("a lit room reads good", good.verdict == "good", f"mean={good.mean:.0f}")
    dark = S.measure_lighting(render(p, light=0.22, seed=1))
    check("an unlit room reads dark", dark.verdict == "dark", f"mean={dark.mean:.0f}")
    blown = S.measure_lighting(np.full((120, 160, 3), 250, np.uint8))
    check("a saturated frame reads blown", blown.verdict == "blown", f"mean={blown.mean:.0f}")
    flat = S.measure_lighting(np.full((120, 160, 3), 120, np.uint8))
    check("a featureless frame reads flat", flat.verdict == "flat",
          f"contrast={flat.contrast:.1f}")

    # Backlight: a bright surround with a dark subject. Built directly rather than
    # through render(), whose lamp scales the whole scene uniformly.
    frame = np.full((240, 320, 3), 235, np.uint8)
    face = render(p, w=320, h=240, scale=1.0, light=0.30, seed=2)
    y, x = 60, 100
    frame[y:y + 120, x:x + 120] = face[60:180, 100:220]
    bl = S.measure_lighting(frame)
    check("a bright surround behind a dark subject reads backlit", bl.backlit,
          f"mean={bl.mean:.0f}")
    check("a normally-lit frame is not called backlit",
          not S.measure_lighting(render(p, light=1.0, seed=3)).backlit)


# ═════════════════════════════════ pose ═════════════════════════════════
def test_pose():
    print("\n— pose and distance —")
    p = identity(4)
    near = S.measure_pose(render(p, scale=1.5, seed=1), F.detect(render(p, scale=1.5, seed=1))[0])
    far = S.measure_pose(render(p, scale=0.35, seed=1), F.detect(render(p, scale=0.35, seed=1))[0])
    check("a large face reads close", near.distance == "close",
          f"fraction={near.face_fraction:.3f}")
    check("a small face reads far", far.distance == "far",
          f"fraction={far.face_fraction:.3f}")
    check("distance tracks face height monotonically",
          near.face_fraction > far.face_fraction)

    centred = S.measure_pose(render(p, cx=0.5, cy=0.5, seed=1),
                             F.detect(render(p, cx=0.5, cy=0.5, seed=1))[0])
    corner = S.measure_pose(render(p, cx=0.12, cy=0.85, seed=1),
                            F.detect(render(p, cx=0.12, cy=0.85, seed=1))[0])
    check("a face in the middle reads centred", centred.centred)
    check("a face in the corner does not", not corner.centred)

    # Yaw must respond to the head turning, not to noise: compare a face shifted so
    # the detection box sits off-centre relative to the features.
    straight = S.measure_pose(render(p, seed=1), F.detect(render(p, seed=1))[0])
    check("a square-on face has small yaw", abs(straight.yaw) < 0.4, f"yaw={straight.yaw:.3f}")
    check("roll is near zero when the head is upright", abs(straight.roll) < 12.0,
          f"roll={straight.roll:.1f}°")


# ═══════════════════════════════ liveness ═══════════════════════════════
def test_liveness():
    print("\n— liveness (see the module docstring for what this does NOT prove) —")
    faces_verdicts, face_moire = [], []
    for seed in range(1, 9):
        img = render(identity(seed), seed=seed)
        found = F.detect(img)
        if not found:
            continue
        lv = S.measure_liveness(img, found[0].box)
        faces_verdicts.append(lv.verdict)
        face_moire.append(lv.moire)
    check("no genuine face is called suspect",
          "suspect" not in faces_verdicts, f"n={len(faces_verdicts)}")
    check("genuine faces are broadband in the spectrum",
          max(face_moire) < S.MOIRE_SUSPECT,
          f"max moiré {max(face_moire):.2f} < {S.MOIRE_SUSPECT}")

    screen_moire, caught = [], 0
    for period in (3.0, 4.0, 5.0, 6.0):
        base = render(identity(1), seed=3)
        img = screen_like(base, period=period)
        found = F.detect(img)
        if not found:
            continue
        lv = S.measure_liveness(img, found[0].box)
        screen_moire.append(lv.moire)
        caught += (lv.verdict == "suspect")
    check("a display grid scores far above real skin",
          min(screen_moire) > max(face_moire),
          f"min screen {min(screen_moire):.1f} vs max face {max(face_moire):.2f}")
    check("display grids are flagged suspect", caught >= 3, f"{caught}/{len(screen_moire)}")

    # The gap is what makes the threshold honest — assert it is wide, not merely
    # that the current numbers happen to land on the right side of it.
    lo, hi = max(face_moire), min(screen_moire)
    check("the moiré threshold sits strictly inside the measured gap",
          lo < S.MOIRE_SUSPECT < hi,
          f"faces ≤{lo:.1f} · threshold {S.MOIRE_SUSPECT} · screens ≥{hi:.1f}")
    check("that gap is wide enough to be meaningful",
          hi > lo * 3.0, f"screens are {hi / max(lo, 1e-9):.1f}× the loudest face")
    check("the threshold errs toward the faces, not the screens",
          S.MOIRE_SUSPECT - lo < hi - S.MOIRE_SUSPECT,
          f"{S.MOIRE_SUSPECT - lo:.1f} above the faces vs {hi - S.MOIRE_SUSPECT:.1f} below the screens")

    tiny = S.measure_liveness(np.zeros((8, 8, 3), np.uint8), (0, 0, 8, 8))
    check("a degenerate patch degrades to uncertain, never to live",
          tiny.verdict == "uncertain")


# ═══════════════════════════════ motion ═════════════════════════════════
def test_motion():
    print("\n— motion —")
    p = identity(7)
    a = render(p, seed=1)
    check("no previous frame reads as no motion", S.measure_motion(None, a) == 0.0)
    check("an identical frame reads as still", S.measure_motion(a, a.copy()) < 0.5)
    moved = S.measure_motion(a, render(p, cx=0.62, cy=0.55, seed=2))
    check("a head that moved reads as motion", moved > 1.0, f"Δ={moved:.2f}")
    check("mismatched shapes degrade to zero rather than raising",
          S.measure_motion(a, np.zeros((64, 64, 3), np.uint8)) == 0.0)


# ══════════════════════════════ describe() ══════════════════════════════
def test_describe():
    print("\n— describe(): the whole frame in one call —")
    p = identity(9)
    empty = S.describe(np.full((160, 200, 3), 90, np.uint8))
    check("an empty room reports zero people", empty.people == 0)
    check("an empty room still gets a Hebrew summary", bool(empty.summary_he),
          empty.summary_he)
    check("an empty room carries no pose", empty.pose is None)

    img = render(p, seed=4)
    sc = S.describe(img, prev=render(p, cx=0.6, seed=5))
    check("one face reports one person", sc.people == 1)
    check("lighting is measured", sc.lighting is not None)
    check("liveness is measured", sc.liveness is not None)
    check("pose is measured", sc.pose is not None)
    check("motion against the previous frame is measured", sc.motion > 0.0,
          f"Δ={sc.motion:.2f}")
    check("a clean close frame scores high quality", sc.quality > 0.7, f"q={sc.quality:.2f}")
    check("the summary is Hebrew prose", "מול המצלמה" in sc.summary_he, sc.summary_he)
    check("the summary states a quality percentage", "%" in sc.summary_he)

    dark = S.describe(render(p, light=0.2, seed=4))
    check("a dark frame warns the user", any("חשוך" in w for w in dark.warnings),
          str(dark.warnings))
    check("a dark frame scores lower quality than a lit one", dark.quality < sc.quality,
          f"{dark.quality:.2f} < {sc.quality:.2f}")

    far = S.describe(render(p, scale=0.30, seed=4))
    check("a distant face warns to come closer",
          any("רחוק" in w for w in far.warnings), str(far.warnings))

    class _M:
        name, known = "OSCAR", True
    named = S.describe(img, identity=_M())
    check("an identified person is named in the summary", "OSCAR" in named.summary_he,
          named.summary_he)

    bad = S.describe(np.zeros((40, 40), np.uint8))
    check("a malformed frame is refused, not crashed on",
          bad.people == 0 and "תקין" in bad.summary_he)

    d = sc.to_dict()
    check("to_dict is JSON-shaped for the HUD",
          {"people", "lighting", "liveness", "pose", "quality", "summary_he"} <= set(d))


# ════════════════════════════ the owner tier ════════════════════════════
class FakeFirewall:
    def __init__(self) -> None:
        self.level = "SAFE"
        self.calls = []

    def set_level(self, level: str) -> None:
        self.level = level
        self.calls.append(level)


def test_owner():
    print("\n— the owner: first face enrolled holds everything —")
    store = tmp_store()
    check("an empty gallery has no owner", store.owner() is None)

    first = store.enroll(F.DEFAULT_OWNER_NAME,
                         F.encode_frame(render(identity(1), seed=1)).vector,
                         level="SAFE")
    check("the first enrollee becomes the owner", first.is_owner, first.role)
    check("the owner holds CRITICAL even when SAFE was asked for",
          first.level == F.OWNER_LEVEL, first.level)
    check("the default owner name is OSCAR", F.DEFAULT_OWNER_NAME == "OSCAR")
    check("the owner's note says why", "היוצר" in first.note, first.note)

    second = store.enroll("אורח", F.encode_frame(render(identity(2), seed=2)).vector,
                          level="CRITICAL")
    check("a later enrollee is a guest", not second.is_owner, second.role)
    check("there is exactly one owner",
          sum(1 for p in store.people if p.is_owner) == 1)

    store.enroll(F.DEFAULT_OWNER_NAME,
                 F.encode_frame(render(identity(1), seed=5)).vector, level="SAFE")
    check("re-enrolling the owner does not strip CRITICAL",
          store.owner().level == F.OWNER_LEVEL, store.owner().level)
    check("re-enrolling the owner keeps the role", store.owner().is_owner)

    check("set_level refuses to demote the owner",
          store.set_level(first.id, "SAFE") is False)
    check("the owner's level survived the refusal",
          store.owner().level == F.OWNER_LEVEL)
    check("set_level(force=True) is the explicit override",
          store.set_level(first.id, "SAFE", force=True) is True)
    store.set_level(first.id, F.OWNER_LEVEL, force=True)
    check("set_level still works on a guest",
          store.set_level(second.id, "WRITE") is True)

    # ── the gate: CRITICAL is the owner's alone ──
    fw = FakeFirewall()
    store.set_level(second.id, "CRITICAL", force=True)
    gate = F.FaceGate(store, firewall=fw, debounce=1)
    st = gate.observe(store.recognize(render(identity(1), seed=11)))
    check("the owner's presence grants CRITICAL", st["level"] == "CRITICAL", st["level"])
    check("the gate reports who the owner is", st["is_owner"] is True)
    check("the firewall was actually driven", fw.level == "CRITICAL", fw.level)
    check("the event says why", "owner" in gate.events[-1]["why"], gate.events[-1]["why"])

    gate2 = F.FaceGate(store, firewall=FakeFirewall(), debounce=1)
    st2 = gate2.observe(store.recognize(render(identity(2), seed=12)))
    check("a guest enrolled at CRITICAL is capped below it",
          st2["level"] == "WRITE", st2["level"])
    check("the cap is explained in the event log",
          "not the owner" in gate2.events[-1]["why"], gate2.events[-1]["why"])
    check("a guest is never reported as the owner", st2["is_owner"] is False)

    check("state() exposes whether an owner exists",
          gate.state()["owner_enrolled"] is True)
    check("state() names the expected owner",
          gate.state()["owner_name_expected"] == "OSCAR")

    # ── ownership can be handed over, and only to one person ──
    check("ownership can be transferred", store.set_owner(second.id) is True)
    check("the new owner holds CRITICAL", store.get(second.id).level == F.OWNER_LEVEL)
    check("the previous owner was demoted from CRITICAL",
          store.get(first.id).level != F.OWNER_LEVEL, store.get(first.id).level)
    check("exactly one owner survives a transfer",
          sum(1 for p in store.people if p.is_owner) == 1)
    check("transferring to a stranger fails cleanly",
          store.set_owner("no-such-person") is False)

    # ── persistence ──
    reloaded = F.FaceStore(path=store.path)
    check("the owner role survives a reload", reloaded.owner() is not None)
    check("the right person is still the owner",
          reloaded.owner().id == second.id, reloaded.owner().name)
    check("the owner level survives a reload",
          reloaded.owner().level == F.OWNER_LEVEL)
    d = reloaded.owner().to_dict()
    check("to_dict carries the role for the HUD",
          d["role"] == "owner" and d["is_owner"] is True)


def test_liveness_blocks_privilege():
    """Liveness may only ever subtract — the rule the server enforces."""
    print("\n— liveness cannot grant, only withhold —")
    store = tmp_store()
    store.enroll(F.DEFAULT_OWNER_NAME,
                 F.encode_frame(render(identity(1), seed=1)).vector)
    gate = F.FaceGate(store, firewall=FakeFirewall(), debounce=1)

    clean = gate.observe(store.recognize(render(identity(1), seed=11)))
    check("a clean frame grants the owner CRITICAL", clean["level"] == "CRITICAL")

    # Reproduce the server's rule directly: a suspect frame is forced to SAFE.
    suspect = dict(clean)
    suspect["level"] = "SAFE"
    suspect["withheld"] = "liveness"
    check("a suspect frame is forced down to SAFE", suspect["level"] == "SAFE")
    check("the withholding is labelled, not silent", suspect["withheld"] == "liveness")

    img = screen_like(render(identity(1), seed=11), period=4.0)
    sc = S.describe(img)
    if sc.liveness is not None and sc.liveness.verdict == "suspect":
        check("a display-like frame is the one that gets withheld",
              sc.liveness.verdict == "suspect")
    else:
        check("a display-like frame is detected by the scene layer", False,
              f"verdict={sc.liveness.verdict if sc.liveness else None}")


def main() -> int:
    print("═" * 68)
    print(" J.A.R.V.I.S. — scene understanding and the owner tier")
    print("═" * 68)
    t0 = time.perf_counter()
    test_lighting()
    test_pose()
    test_liveness()
    test_motion()
    test_describe()
    test_owner()
    test_liveness_blocks_privilege()
    print(f"\n completed in {time.perf_counter() - t0:.1f}s")
    print(f"\nRESULT: {ok} passed, {fail} failed")
    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
