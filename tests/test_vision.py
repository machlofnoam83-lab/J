#!/usr/bin/env python3
"""JARVIS vision tests — face detection, eigenface recognition, presence-gated privilege.

Nothing here needs a camera. Faces are synthesised: a skin-toned ellipse with
shading, dark eyes, brows, nose and mouth on a textured background, and an
*identity* made of the things eigenfaces actually key on — face proportions, eye
spacing, skin tone, nose and mouth geometry. Each person is enrolled from several
jittered samples (light, scale, position) and recognised from held-out ones, so a
pass means the recogniser generalises rather than memorises.

Covers: skin mask · component geometry filters · eye-structure guard · rejection of
walls/noise/blank frames · multi-face frames · eigenface encoding · gallery
persistence · identity separation · stranger rejection · residual gate on non-faces
· permission mapping · presence lease expiry · ceiling clamp · debounce · frame
payload round-trip.

Run:  python tests/test_vision.py
"""

from __future__ import annotations

import base64
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vision import faces as F  # noqa: E402

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
    """A person.

    Everything here is a property of the *face*, not of the frame: proportions,
    tone, feature sizes and hair. Hair colour in particular must not be redrawn per
    render — an earlier version of this generator randomised it every call, so the
    same person's skin-mask boundary jumped around and JARVIS saw one enrollee as
    two different people.
    """
    rng = np.random.default_rng(seed)
    return {
        "fw": float(rng.uniform(58, 86)),          # face width
        "fh": float(rng.uniform(78, 112)),         # face height
        "eye_dx": float(rng.uniform(0.16, 0.29)),  # eye spacing (fraction of fw)
        "eye_dy": float(rng.uniform(0.10, 0.22)),  # eye height above centre
        "eye_r": float(rng.uniform(0.055, 0.105)), # eye size (fraction of fw)
        "skin": (int(rng.uniform(196, 240)), int(rng.uniform(150, 196)),
                 int(rng.uniform(120, 168))),
        "shade": float(rng.uniform(10, 30)),
        "mouth_w": float(rng.uniform(0.12, 0.26)),
        "mouth_dy": float(rng.uniform(0.24, 0.36)),
        "nose_len": float(rng.uniform(0.06, 0.16)),
        "nose_w": float(rng.uniform(0.040, 0.075)),
        "hair": int(rng.integers(22, 96)),         # fixed for this person
        "hair_h": float(rng.uniform(0.34, 0.52)),  # how far the fringe comes down
        "jaw": float(rng.uniform(0.86, 1.06)),     # jaw width relative to cheeks
    }


def render(p: dict, w: int = 320, h: int = 240, cx: float = 0.5, cy: float = 0.5,
           scale: float = 1.0, light: float = 1.0, seed: int = 0,
           expression: float = 0.0, noise: float = 1.8) -> np.ndarray:
    """Draw that person into a frame.

    Frame-to-frame variation is limited to what a webcam actually sees: a little
    sensor noise, small head movement, a lamp that dims, and expression. Identity —
    proportions, tone, hair — comes only from `p`.
    """
    rng = np.random.default_rng(seed)
    # A lamp scales the whole scene. An earlier version lit only the face, which
    # changed the *contrast* between head and background instead of the exposure —
    # a far harsher change than any room produces, and one that survives contrast
    # normalisation by construction.
    bg = np.array([38, 58, 92], np.float32) * light
    img = np.zeros((h, w, 3), np.int16)
    img[:] = bg[None, None, :] + rng.integers(-4, 5, (h, w, 1))

    fw, fh = p["fw"] * scale, p["fh"] * scale
    cxp, cyp = cx * w, cy * h
    yy, xx = np.mgrid[0:h, 0:w]
    # jaw narrower than the cheeks, as a real skull is
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

    ex = p["eye_dx"] * fw
    ey = cyp - p["eye_dy"] * fh
    er = p.get("eye_r", 0.085)
    blob(cxp - ex, ey, fw * er, fh * 0.042, (34, 28, 30))        # eyes
    blob(cxp + ex, ey, fw * er, fh * 0.042, (34, 28, 30))
    blob(cxp - ex, ey - fh * 0.075, fw * 0.11, fh * 0.020, (70, 52, 48))   # brows
    blob(cxp + ex, ey - fh * 0.075, fw * 0.11, fh * 0.020, (70, 52, 48))
    blob(cxp, cyp + fh * 0.06, fw * p.get("nose_w", 0.055), p["nose_len"] * fh,
         (skin * 0.86))                                          # nose
    mw = p["mouth_w"] * fw * (1.0 + expression)                 # a smile widens it
    blob(cxp, cyp + p.get("mouth_dy", 0.30) * fh, mw, fh * 0.030, (150, 78, 78))

    # Hair: a cap in this person's own colour. The fringe stops above the brows —
    # an earlier version let it come down onto the eyes, which is not a face anyone
    # has, and it taught the eye finder to measure the hairline instead.
    hh = p.get("hair_h", 0.42)
    brow = ey - fh * 0.075
    fringe = min(brow - fh * 0.05, cyp - fh * (hh - 0.24))
    cap = (((xx - cxp) ** 2) / (fw * 0.56) ** 2 + ((yy - (cyp - fh * hh)) ** 2)
           / (fh * 0.30) ** 2) <= 1.0
    cap &= yy < fringe
    img[cap] = float(p.get("hair", 46)) * light

    return np.clip(img, 0, 255).astype(np.uint8)


def iou(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x0, y0 = max(ax, bx), max(ay, by)
    x1, y1 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    inter = (x1 - x0) * (y1 - y0)
    return inter / float(aw * ah + bw * bh - inter)


def tmp_store():
    d = Path(tempfile.mkdtemp(prefix="jarvis-vision-"))
    return F.FaceStore(path=d / "gallery.json"), d


def enroll_person(store, pid_seed: int, name: str, level: str, n: int = 4) -> str:
    p = identity(pid_seed)
    person = None
    for i in range(n):
        img = render(p, scale=float(0.92 + 0.04 * i), light=float(0.94 + 0.05 * i),
                     cx=0.5 + 0.01 * i, seed=100 + i)
        f = F.encode_frame(img)
        assert f is not None, f"enrolment frame {i} for {name} found no face"
        person = store.enroll(name, f.vector, level=level)
    return person.id


# ═════════════════════════════════ tests ═════════════════════════════════
def test_detection():
    print("\n── detection ──")
    p = identity(1)
    img = render(p, seed=7)
    found = F.detect(img)
    check("finds the face in a plain frame", len(found) >= 1, f"({len(found)} candidate)")
    if found:
        truth = (int(160 - p["fw"] / 2), int(120 - p["fh"] / 2), int(p["fw"]), int(p["fh"]))
        score = iou(found[0].box, truth)
        check("box lands on the face", score > 0.35, f"IoU={score:.2f} box={found[0].box}")
        check("box has face proportions", 1.0 <= found[0].h / max(1, found[0].w) <= 2.2,
              f"aspect={found[0].h / max(1, found[0].w):.2f}")

    # rejects things that are merely skin-coloured or merely noisy
    blank = np.full((240, 320, 3), 30, np.uint8)
    check("ignores a blank frame", len(F.detect(blank)) == 0)

    wall = np.zeros((240, 320, 3), np.uint8)
    wall[:] = (224, 172, 140)                    # skin tone covering everything
    check("ignores a skin-coloured wall", len(F.detect(wall)) == 0)

    patch = np.zeros((240, 320, 3), np.uint8)
    patch[:] = (38, 58, 92)
    patch[40:200, 60:260] = (224, 172, 140)      # big flat skin rectangle, no eyes
    check("ignores a flat skin-coloured panel (no eye structure)",
          len(F.detect(patch)) == 0)

    rng = np.random.default_rng(3)
    noise = rng.integers(0, 255, (240, 320, 3), dtype=np.uint8)
    check("ignores random noise", len(F.detect(noise)) == 0)

    check("ignores a frame too small to hold a face",
          len(F.detect(np.zeros((16, 16, 3), np.uint8))) == 0)
    check("ignores a grey-scale array of the wrong shape",
          len(F.detect(np.zeros((240, 320), np.uint8))) == 0)

    # two people in one frame: paste a half-frame of each side by side
    left = render(identity(5), cx=0.30, cy=0.52, scale=0.84, seed=12)[:, :160]
    right = render(identity(9), cx=0.70, cy=0.50, scale=0.86, seed=13)[:, 160:]
    combo = np.concatenate([left, right], axis=1)
    faces = F.detect(combo, max_faces=4)
    check("finds both faces in a two-person frame", len(faces) >= 2, f"({len(faces)} found)")
    if len(faces) >= 2:
        xs = sorted(f.x for f in faces[:2])
        check("the two boxes are separate people", xs[1] - xs[0] > 40, f"Δx={xs[1] - xs[0]}")


def test_encoding():
    print("\n── eigenface encoding ──")
    img = render(identity(4), seed=21)
    f = F.encode_frame(img)
    check("encode_frame returns a vector", f is not None and f.vector is not None)
    v = f.vector
    check("vector has the documented dimension", v.shape == (F.DIM,), f"{v.shape}")
    check("vector is zero-mean", abs(float(v.mean())) < 1e-3, f"mean={float(v.mean()):.2e}")
    check("vector is unit-variance", abs(float(v.std()) - 1.0) < 1e-3, f"std={float(v.std()):.4f}")

    img2 = render(identity(4), scale=1.03, light=1.06, seed=22)
    v2 = F.encode_frame(img2).vector
    same = float(np.dot(v, v2) / (np.linalg.norm(v) * np.linalg.norm(v2)))
    other_v = F.encode_frame(render(identity(17), seed=23)).vector
    other = float(np.dot(v, other_v) / F.DIM)
    check("same person under different light/pose stays close", same > 0.55, f"cos={same:.3f}")
    # Raw-image cosine is a weak discriminator — every face shares the same gross
    # layout, which is exactly why eigenfaces subtract the gallery mean. So the
    # honest check here is ordering, and separation is measured in face space below.
    check("a different person is further away than the same person",
          other < same, f"cos other={other:.3f} vs same={same:.3f}")

    # A two-person gallery, each enrolled from three jitters, then probed with
    # frames neither person ever enrolled. Matching the enrolled vectors themselves
    # would prove nothing — both would score a perfect 1.0.
    two, _ = tmp_store()
    p1, p2 = identity(21), identity(22)
    id1 = id2 = None
    for i in range(3):
        id1 = two.enroll("א", F.encode_frame(render(
            p1, scale=0.96 + 0.04 * i, light=0.95 + 0.05 * i, seed=600 + i)).vector,
            "SAFE").id
        id2 = two.enroll("ב", F.encode_frame(render(
            p2, scale=0.96 + 0.04 * i, light=0.95 + 0.05 * i, seed=700 + i)).vector,
            "WRITE").id
    own = two.match(F.encode_frame(render(p1, scale=1.06, light=0.92, seed=699)).vector)
    cross = two.match(F.encode_frame(render(p2, scale=1.06, light=0.92, seed=799)).vector)
    stranger = two.match(F.encode_frame(render(identity(55), seed=855)).vector)
    check("each held-out face lands on its own person",
          own[0] == id1 and cross[0] == id2 and id1 != id2,
          f"א→{own[0] == id1} ({own[1]:.2f}), ב→{cross[0] == id2} ({cross[1]:.2f})")
    check("both clear the recognition threshold",
          min(own[1], cross[1]) >= F.CONFIDENCE_THRESHOLD,
          f"min={min(own[1], cross[1]):.2f} threshold={F.CONFIDENCE_THRESHOLD}")
    check("a stranger scores below both", stranger[1] < min(own[1], cross[1]),
          f"stranger={stranger[1]:.2f}")


def test_gallery():
    print("\n── gallery persistence ──")
    store, d = tmp_store()
    check("empty gallery lists nobody", store.list() == [])
    a = enroll_person(store, 31, "אדון", "CRITICAL", n=4)
    b = enroll_person(store, 32, "אורח", "SAFE", n=3)
    check("two people enrolled", len(store.list()) == 2, str([p["name"] for p in store.list()]))
    check("samples accumulate per person",
          {p["id"]: p["samples"] for p in store.list()} == {a: 4, b: 3})
    check("gallery written to disk", (d / "gallery.json").exists())
    raw = json.loads((d / "gallery.json").read_text(encoding="utf-8"))
    check("file records the eigenface size", raw.get("face_size") == F.FACE_SIZE)

    store2 = F.FaceStore(path=d / "gallery.json")
    check("gallery reloads intact", len(store2.people) == 2)
    check("vectors survive the round trip",
          store2.people[0].vectors[0].shape == (F.DIM,))
    reloaded = np.asarray(store2.people[0].vectors[0], np.float32)
    check("reloaded vectors keep their values",
          float(np.abs(reloaded - np.asarray(store.people[0].vectors[0])).max()) < 1e-3)

    check("set_level changes a person's grant", store2.set_level(b, "WRITE") and
          store2.get(b).level == "WRITE")
    check("set_level rejects a bogus level", not store2.set_level(b, "ROOT"))
    check("remove drops a person", store2.remove(a) and len(store2.people) == 1)
    check("remove reports unknown ids", not store2.remove("nope"))


def test_recognition():
    print("\n── recognition ──")
    store, _ = tmp_store()
    owner = enroll_person(store, 41, "אדון", "CRITICAL", n=4)
    guest = enroll_person(store, 42, "אורח", "SAFE", n=3)
    helper = enroll_person(store, 43, "עוזרת", "WRITE", n=4)

    # held-out frames: new jitter, never enrolled
    probes = [(41, "אדון", "CRITICAL", dict(scale=1.05, light=0.9, seed=77)),
              (42, "אורח", "SAFE", dict(scale=0.95, light=1.1, seed=78)),
              (43, "עוזרת", "WRITE", dict(scale=1.0, light=1.0, seed=79))]
    confs, hits = [], 0
    for seed, name, level, kw in probes:
        img = render(identity(seed), **kw)
        m = store.recognize(img)
        confs.append(m.confidence)
        good = m.known and m.name == name and m.level == level
        hits += bool(good)
        check(f"recognises {name} from a held-out frame", good,
              f"→ {m.name}/{m.level} conf={m.confidence:.2f}")
    check("every held-out probe identified correctly", hits == len(probes), f"{hits}/{len(probes)}")
    check("recognised faces score above the threshold",
          all(c >= F.CONFIDENCE_THRESHOLD for c in confs),
          f"min={min(confs):.2f} threshold={F.CONFIDENCE_THRESHOLD}")

    # a stranger must not inherit anybody's privileges
    stranger_confs = []
    for seed in (61, 62, 63):
        m = store.recognize(render(identity(seed), scale=1.0, seed=200 + seed))
        stranger_confs.append(m.confidence)
        if m.known:
            check(f"stranger #{seed} not mistaken for {m.name}", False,
                  f"conf={m.confidence:.2f}")
    known_as_stranger = sum(1 for c in stranger_confs if c >= F.CONFIDENCE_THRESHOLD)
    check("strangers are refused", known_as_stranger == 0,
          f"confs={[round(c, 2) for c in stranger_confs]}")

    m = store.recognize(render(identity(61), seed=301))
    check("a refused face reports SAFE", m.level == "SAFE" and not m.known, str(m.to_dict()))
    check("a refused face still reports how many faces were seen", m.faces >= 1)

    # non-faces: the residual gate
    blank = F.FaceStore(path=Path(tempfile.mkdtemp()) / "g.json")
    enroll_person(blank, 41, "אדון", "CRITICAL", n=3)
    wall = np.zeros((240, 320, 3), np.uint8)
    wall[:] = (224, 172, 140)
    mw = blank.recognize(wall)
    check("a skin-coloured wall is nobody", not mw.known and mw.faces == 0)
    rng = np.random.default_rng(5)
    mn = blank.recognize(rng.integers(0, 255, (240, 320, 3), dtype=np.uint8))
    check("random noise is nobody", not mn.known)

    empty, _ = tmp_store()
    me = empty.recognize(render(identity(41), seed=400))
    check("an empty gallery recognises nobody", not me.known and me.level == "SAFE")

    # cap on gallery growth
    cap, _ = tmp_store()
    pid = None
    for i in range(12):
        f = F.encode_frame(render(identity(71), scale=0.95 + 0.01 * i, seed=500 + i))
        pid = cap.enroll("רב-דגימות", f.vector, "SAFE").id
    check("gallery keeps only the newest samples per person",
          len(cap.get(pid).vectors) == cap.max_samples,
          f"{len(cap.get(pid).vectors)} of 12 kept")


class FakeFirewall:
    def __init__(self, level="SAFE"):
        self.level = level
        self.calls = []

    def set_level(self, level):
        self.level = level
        self.calls.append(level)


def test_gate():
    print("\n── presence-gated privilege ──")
    store, _ = tmp_store()
    owner = enroll_person(store, 81, "אדון", "CRITICAL", n=4)
    guest = enroll_person(store, 82, "אורח", "SAFE", n=3)

    fw = FakeFirewall("SAFE")
    gate = F.FaceGate(store, firewall=fw, default_level="SAFE", ttl=0.35, debounce=2)
    check("gate starts at the default level", gate.level == "SAFE" and gate.identity is None)

    img_owner = render(identity(81), scale=1.02, seed=900)
    m = store.recognize(img_owner)
    gate.observe(m)
    check("first sighting does not grant anything yet (debounce)", gate.level == "SAFE",
          f"level={gate.level}")
    gate.observe(m)
    check("the owner is granted CRITICAL", gate.level == "CRITICAL" and gate.identity == owner,
          f"level={gate.level} who={gate.name}")
    check("the firewall was told", fw.level == "CRITICAL", str(fw.calls))
    st = gate.state()
    check("state reports the person and the confidence",
          st["name"] == "אדון" and st["confidence"] > 0, json.dumps(st, ensure_ascii=False)[:90])

    guest_img = render(identity(82), seed=901)
    gm = store.recognize(guest_img)
    gate.observe(gm)
    gate.observe(gm)
    check("a guest drops the level to SAFE", gate.level == "SAFE" and gate.identity == guest,
          f"level={gate.level}")

    # walking away must take the privileges with it
    gate.observe(m); gate.observe(m)
    check("owner back in frame → CRITICAL again", gate.level == "CRITICAL")
    empty = np.full((240, 320, 3), 30, np.uint8)
    gate.observe(F.Match(None, "—", "SAFE", 0.0, 0.0, 0, False))
    check("an empty frame does not revoke instantly (grace period)",
          gate.level == "CRITICAL", f"age={time.time() - gate.last_seen:.2f}s")
    time.sleep(0.45)
    st = gate.poll()
    check("the lease expires and privilege is revoked", st["level"] == "SAFE" and
          st["identity"] is None and fw.level == "SAFE", json.dumps(st, ensure_ascii=False)[:80])

    # ceiling: a person may never exceed the configured cap
    fw2 = FakeFirewall("SAFE")
    capped = F.FaceGate(store, firewall=fw2, default_level="SAFE", ceiling="WRITE",
                        ttl=5.0, debounce=1)
    capped.observe(m)
    check("the ceiling clamps CRITICAL down to WRITE", capped.level == "WRITE",
          f"level={capped.level}")

    # disarm stops it driving the firewall at all
    fw3 = FakeFirewall("SAFE")
    off = F.FaceGate(store, firewall=fw3, default_level="SAFE", ttl=5.0, debounce=1)
    off.disarm()
    off.observe(m)
    check("a disarmed gate never touches the firewall", fw3.calls == [] and off.level == "SAFE")

    # no firewall at all → still tracks identity (HUD-only mode)
    lone = F.FaceGate(store, firewall=None, ttl=5.0, debounce=1)
    lone.observe(m)
    check("identity tracking works without a firewall", lone.identity == owner,
          f"who={lone.name}")

    # a stranger at the door gets the unknown-face level
    fw4 = FakeFirewall("SAFE")
    paranoid = F.FaceGate(store, firewall=fw4, default_level="SAFE",
                          unknown_level="SAFE", ttl=5.0, debounce=1)
    paranoid.observe(m)
    paranoid.observe(F.Match(None, "לא מזוהה", "SAFE", 0.2, 0.4, 1, False))
    check("an unrecognised face holds the level at SAFE",
          paranoid.level == "SAFE" and paranoid.identity is None)
    check("gate events are recorded for the audit trail", len(paranoid.events) >= 2,
          f"{len(paranoid.events)} events")


def test_calibration():
    """The threshold is a security setting, so it is measured here, not assumed.

    For each gallery size: enrol everyone from jittered frames, then probe with
    frames nobody ever enrolled. A pass means every enrollee is accepted *and*
    attributed correctly while every stranger is refused, and the two score
    distributions do not touch — the gap is what makes the threshold meaningful.
    """
    print("\n── threshold calibration (held-out probes) ──")
    from vision.faces import CONFIDENCE_THRESHOLD as TH
    people_pool = [41, 42, 43, 44, 45, 46, 47, 48]
    strangers = [61, 62, 63, 71, 81, 91, 55, 56, 57, 58]

    worst_legit, worst_stranger, total, correct = 1.0, 0.0, 0, 0
    for n_people, per in ((1, 4), (2, 3), (3, 4), (5, 4), (8, 3)):
        store, _ = tmp_store()
        ids = {}
        for sd in people_pool[:n_people]:
            p = identity(sd)
            for i in range(per):
                ids[sd] = store.enroll(
                    f"p{sd}", F.encode_frame(render(
                        p, scale=0.94 + 0.04 * i, light=0.95 + 0.05 * i,
                        seed=600 + i)).vector, "WRITE").id
        legit, wrong = [], 0
        for sd in people_pool[:n_people]:
            for sc, li, s2 in ((1.06, 0.92, 701), (0.93, 1.08, 702), (1.00, 1.00, 703)):
                pid, conf, _ = store.match(F.encode_frame(render(
                    identity(sd), scale=sc, light=li, seed=s2)).vector)
                legit.append(conf)
                total += 1
                if pid == ids[sd] and conf >= TH:
                    correct += 1
                else:
                    wrong += 1
        intruders = []
        for sd in strangers:
            pid, conf, _ = store.match(F.encode_frame(render(identity(sd), seed=800 + sd)).vector)
            intruders.append(conf)
            if conf >= TH:
                wrong += 1
        lo, hi = min(legit), max(intruders)
        worst_legit = min(worst_legit, lo)
        worst_stranger = max(worst_stranger, hi)
        check(f"gallery of {n_people}: enrollees in, strangers out",
              wrong == 0 and lo > hi,
              f"legit_min={lo:.3f} stranger_max={hi:.3f} misfires={wrong}")

    check("every held-out enrollee accepted as themselves", correct == total,
          f"{correct}/{total}")
    check("no stranger cleared the threshold", worst_stranger < TH,
          f"worst stranger={worst_stranger:.3f} < {TH}")
    check("the gap between the two distributions is real",
          worst_legit > worst_stranger,
          f"legit_min={worst_legit:.3f} vs stranger_max={worst_stranger:.3f}")
    check("the threshold sits inside that gap",
          worst_stranger < TH <= worst_legit,
          f"{worst_stranger:.3f} < {TH} <= {worst_legit:.3f}")


def test_payload():
    print("\n── frame transport ──")
    img = render(identity(91), seed=950)
    small = img[::2, ::2]
    payload = {"w": int(small.shape[1]), "h": int(small.shape[0]),
               "rgb": base64.b64encode(np.ascontiguousarray(small).tobytes()).decode()}
    back = F.frame_from_rgb(payload)
    check("RGB payload round-trips exactly", np.array_equal(back, small), str(back.shape))
    check("a round-tripped frame is still recognisable", len(F.detect(back)) >= 1)

    g = F.to_ycbcr(small)[0].astype(np.uint8)
    gp = {"w": int(g.shape[1]), "h": int(g.shape[0]),
          "gray": base64.b64encode(np.ascontiguousarray(g).tobytes()).decode()}
    gg = F.frame_from_gray(gp)
    check("grey payload becomes a 3-plane frame", gg.shape == (*g.shape, 3), str(gg.shape))

    for bad in ({}, {"w": 4, "h": 4}, {"w": 8, "h": 8, "rgb": "!!!"}):
        try:
            F.frame_from_rgb(bad)
            check(f"rejects a malformed payload {sorted(bad)}", False)
        except Exception:
            check(f"rejects a malformed payload {sorted(bad) or 'empty'}", True)

    try:
        F.frame_from_rgb({"w": 100, "h": 100, "rgb": base64.b64encode(b"short").decode()})
        check("rejects a truncated frame", False)
    except ValueError:
        check("rejects a truncated frame", True)


def main() -> int:
    print("═" * 68)
    print(" J.A.R.V.I.S. — vision (faces, identity, permission)")
    print("═" * 68)
    t0 = time.perf_counter()
    check("scipy is available for morphology", F._ndi is not None)
    test_detection()
    test_encoding()
    test_gallery()
    test_recognition()
    test_gate()
    test_calibration()
    test_payload()
    print(f"\n completed in {time.perf_counter() - t0:.1f}s")
    print(f"\nRESULT: {ok} passed, {fail} failed")
    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
