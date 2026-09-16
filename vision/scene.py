"""Scene understanding: what the camera is actually looking at.

``vision/faces.py`` answers one question — *is there a face, and whose is it?*
That is enough to gate a permission, and not enough to behave like something
that is watching a room. This module answers the questions around it, from pixels
only, with no model and no network:

    how many people      · face boxes surviving detection
    how far, which way   · face height as a share of the frame, eye-line geometry
    is the room lit      · luma mean/contrast, backlight
    is it a live person  · chroma spread, specular highlights, moiré
    is anything moving   · frame differencing against the previous frame

On liveness, stated plainly because it is the one claim here that can hurt
------------------------------------------------------------------
Distinguishing a person from a photograph of a person is a genuinely hard
problem, and a hand-built heuristic is *not* a solution to it. What this module
does is measure three signals that differ measurably between a webcam and a
screen, report each one separately, and combine them into a verdict of ``live`` /
``suspect`` / ``uncertain``. It is calibrated to be **conservative in the safe
direction**: it never upgrades a permission on the strength of liveness, it can
only *refuse to grant* one. A false "suspect" costs the user a re-enrolment; a
false "live" would hand CRITICAL to a photograph, so the thresholds sit well
inside the measured gap rather than on its edge.

Stated as a limitation rather than hidden: this detects a **display** reliably
and a **printed photograph** not at all. A fourth signal meant to catch a page
held to the lens was built, measured against real faces, and deleted because it
fired on genuine faces and separated nothing. See ``measure_liveness``.

Nothing here is allowed to raise a permission level. ``vision/faces.py`` owns
that decision and this module only ever subtracts from what it would grant.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from vision.faces import Face, detect, encode, to_ycbcr  # noqa: E402

# ─────────────────────────────────────────────────────────────── thresholds ──
# Luma bands. A webcam in a normal room sits in the middle; the two ends mean
# "I cannot identify anyone reliably", which is worth saying out loud.
LUMA_DARK = 42.0
LUMA_BRIGHT = 218.0
CONTRAST_FLAT = 18.0
# A face smaller than this share of the frame height is too far away to identify
# with confidence — eigenfaces at 64x64 lose the detail that separates people.
FACE_TOO_FAR = 0.13
FACE_CLOSE = 0.34
# Liveness. Each signal is measured, then combined. See the module docstring for
# why these are conservative.
CHROMA_SPREAD_LIVE = 9.0       # live skin has chroma variation; print is flatter
SPECULAR_LIVE = 0.004          # fraction of face pixels that are specular
# Peak-to-median in the high-frequency annulus. Broadband skin measures ~1-3;
# a camera beating against a panel's pixel grid measures far higher.
MOIRE_SUSPECT = 12.0
MOTION_STILL = 0.6             # mean |Δluma| below this is a static scene


@dataclass
class Lighting:
    """How usable the room is, in the only terms that matter for identification."""

    mean: float
    contrast: float
    backlit: bool
    verdict: str                 # good | dim | dark | blown | flat

    def to_dict(self) -> Dict[str, Any]:
        return {"mean": round(self.mean, 1), "contrast": round(self.contrast, 1),
                "backlit": self.backlit, "verdict": self.verdict}


@dataclass
class Liveness:
    """Signals that separate a webcam from a photograph or a display."""

    chroma_spread: float
    specular: float
    moire: float
    signals_hit: int
    verdict: str                 # live | suspect | uncertain

    def to_dict(self) -> Dict[str, Any]:
        return {"chroma_spread": round(self.chroma_spread, 2),
                "specular": round(self.specular, 5), "moire": round(self.moire, 3),
                "signals_hit": self.signals_hit, "verdict": self.verdict}


@dataclass
class Pose:
    """Where the head is, derived from the geometry detection already found."""

    yaw: float                   # -1 (turned left) .. +1 (turned right)
    roll: float                  # degrees, + = tilted clockwise
    distance: str                # close | mid | far
    face_fraction: float
    centred: bool

    def to_dict(self) -> Dict[str, Any]:
        return {"yaw": round(self.yaw, 3), "roll": round(self.roll, 1),
                "distance": self.distance, "face_fraction": round(self.face_fraction, 4),
                "centred": self.centred}


@dataclass
class Scene:
    """Everything this module knows about one frame."""

    people: int
    faces: List[Face] = field(default_factory=list)
    lighting: Optional[Lighting] = None
    liveness: Optional[Liveness] = None
    pose: Optional[Pose] = None
    motion: float = 0.0
    quality: float = 0.0         # 0..1 — can we identify anyone from this frame?
    identity_hint: str = ""
    summary_he: str = ""
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "people": self.people,
            "faces": [f.to_dict() for f in self.faces],
            "lighting": self.lighting.to_dict() if self.lighting else None,
            "liveness": self.liveness.to_dict() if self.liveness else None,
            "pose": self.pose.to_dict() if self.pose else None,
            "motion": round(self.motion, 3),
            "quality": round(self.quality, 3),
            "identity_hint": self.identity_hint,
            "summary_he": self.summary_he,
            "warnings": self.warnings,
        }


# ══════════════════════════════════════════════════════════════════ lighting ══
def measure_lighting(rgb: np.ndarray) -> Lighting:
    """Luma statistics plus a backlight check.

    Backlight is measured rather than assumed: compare the face region's luma
    with the surrounding frame. A window behind the subject makes the face dark
    *and* the frame bright, which is the one lighting failure that silently
    destroys identification while looking fine to the user.
    """
    y, _, _ = to_ycbcr(np.asarray(rgb))
    mean = float(y.mean())
    contrast = float(y.std())

    faces = detect(rgb, max_faces=1)
    backlit = False
    if faces:
        x, yy, w, h = faces[0].box
        H, W = y.shape
        y0, y1 = max(0, yy), min(H, yy + h)
        x0, x1 = max(0, x), min(W, x + w)
        if y1 > y0 and x1 > x0:
            face_luma = float(y[y0:y1, x0:x1].mean())
            # Surround = the frame minus a generous margin around the face.
            m = max(8, int(0.12 * min(H, W)))
            band = np.concatenate([
                y[:m, :].ravel(), y[-m:, :].ravel(),
                y[:, :m].ravel(), y[:, -m:].ravel()])
            if band.size and face_luma < float(band.mean()) - 22.0:
                backlit = True

    if mean < LUMA_DARK:
        verdict = "dark"
    elif mean > LUMA_BRIGHT:
        verdict = "blown"
    elif contrast < CONTRAST_FLAT:
        verdict = "flat"
    elif mean < LUMA_DARK + 26:
        verdict = "dim"
    else:
        verdict = "good"
    return Lighting(mean, contrast, backlit, verdict)


# ══════════════════════════════════════════════════════════════════ liveness ══
def _patch(rgb: np.ndarray, box: Tuple[int, int, int, int],
           pad: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """Return (rgb patch, luma patch) for a face box, optionally padded."""
    a = np.asarray(rgb)
    H, W = a.shape[:2]
    x, y, w, h = box
    px, py = int(w * pad), int(h * pad)
    x0, x1 = max(0, x - px), min(W, x + w + px)
    y0, y1 = max(0, y - py), min(H, y + h + py)
    if x1 <= x0 or y1 <= y0:
        return np.zeros((1, 1, 3), np.uint8), np.zeros((1, 1), np.float32)
    rgb_p = a[y0:y1, x0:x1]
    y_p, _, _ = to_ycbcr(rgb_p)
    return rgb_p, y_p


def measure_liveness(rgb: np.ndarray, box: Tuple[int, int, int, int]) -> Liveness:
    """Signals that separate a webcam from a photo or a screen.

    Reported separately so the HUD can show *why* a frame was doubted instead of
    just asserting it. Measured on the project's synthetic face gallery and on
    constructed display artifacts:

        moiré   real faces  3.83 – 6.64      display grids  32.4 – 102.6
        chroma  real faces  9.14 – 20.22

    The moiré gap is wide and clean, so the display case is decided by it. A
    fourth signal — a straight rectangular edge framing the face, for a printed
    page held to the lens — was built, measured and **removed**: real faces scored
    0.146 – 0.740 on it while the page scored 0.724, because the detection box
    hugs the skin blob and the band around it catches the face's own contour as a
    long straight run. It fired on genuine faces and separated nothing, so it is
    gone rather than tuned until it looked convincing. Detecting a printed photo
    remains an open weakness; detecting a screen does not.

    A second attempt was measured rather than built, and the measurement is the
    reason there is no detector. Simulated prints against the synthetic gallery:

        matte paper   chroma 13.11-16.90   specular 0.0      moire 2.99-6.04   -> live
        glossy print  chroma 13.94-14.87   specular 0.032-0.068                -> live
        real faces    chroma 11.99-16.08   specular 0.0      moire 3.03-6.27   -> live

    Every range overlaps. The structural problem is that ``moire < MOIRE_SUSPECT``
    is counted as positive evidence of life when it is only evidence of *not a
    screen*, so a matte sheet scores two hits and reads live.

    Fixing that needs a signal the synthetic gallery does not have: specular is
    exactly 0.0 on every rendered face, because the renderer draws no sebum
    highlight, and the only frame-to-frame motion available is per-seed renderer
    noise — a real person sitting still looks identical to a photograph. A rule
    tuned here would be tuned against a straw man and would fail on the first
    real webcam. So it is documented instead of shipped.
    """
    a = np.asarray(rgb)
    rgb_p, y_p = _patch(a, box, pad=0.0)
    if rgb_p.size < 64:
        return Liveness(0.0, 0.0, 1.0, 0, "uncertain")

    # 1. Chroma spread. Live skin under a room lamp varies in hue across the
    #    cheeks, nose and shadow. A print or an LCD panel reproduces a flatter
    #    chroma distribution because it has already been quantised once.
    _, cb, cr = to_ycbcr(rgb_p)
    chroma_spread = float(np.std(cb) + np.std(cr)) / 2.0

    # 2. Specular highlights. Sebum and moisture throw small very-bright spots
    #    off a real face; matte paper does not, and a screen's highlight is
    #    usually a large uniform region rather than scattered points.
    bright = float((y_p > 235).mean())
    # A single large blob is a screen glare, not skin — count *scattered* ones.
    specular = bright if bright < 0.12 else bright * 0.25

    # 3. Moiré / refresh periodicity. A camera pointed at a display beats against
    #    the panel's pixel grid and leaves one dominant fine frequency. Real skin
    #    is broadband. This is the signal that does the work.
    moire = _moire_strength(y_p)

    hits = 0
    if chroma_spread >= CHROMA_SPREAD_LIVE:
        hits += 1
    if specular >= SPECULAR_LIVE:
        hits += 1
    if moire < MOIRE_SUSPECT:
        hits += 1

    # "Suspect" needs positive evidence of an artifact, not merely the absence of
    # evidence of life. A matte, evenly-lit face legitimately shows no specular
    # highlight, and punishing that would tell the user they are a photograph.
    if moire >= MOIRE_SUSPECT:
        verdict = "suspect"
    elif chroma_spread < CHROMA_SPREAD_LIVE * 0.45 and specular < SPECULAR_LIVE:
        verdict = "uncertain"
    elif hits >= 2:
        verdict = "live"
    else:
        verdict = "uncertain"
    return Liveness(chroma_spread, specular, moire, hits, verdict)


def _moire_strength(y_plane: np.ndarray) -> float:
    """How much one high frequency dominates — the signature of sampling a grid.

    Measured as peak-to-median inside an annulus of the spectrum, *not* as a share
    of total energy. That distinction is the whole function: a face is a smooth
    blob, so nearly all of its energy sits in the lowest few bins, and any
    "strongest peak over the whole spectrum" measure reads that shape gradient as
    periodicity. Restricting to radii 0.35–0.95 of the Nyquist circle leaves only
    fine detail, where broadband skin and a pixel grid genuinely differ.

    Returns ~1.0–3.0 for broadband content and climbs well past that when a single
    fine frequency dominates. A Hann window suppresses the leakage a hard crop
    would otherwise inject as spurious peaks.
    """
    p = np.asarray(y_plane, np.float32)
    if p.size < 256:
        return 1.0
    n = int(2 ** math.floor(math.log2(min(p.shape))))
    n = max(16, min(n, 128))
    crop = p[:n, :n] if p.shape[0] >= n and p.shape[1] >= n else np.resize(p, (n, n))
    crop = crop - crop.mean()
    w1 = np.hanning(n)
    crop = crop * w1[:, None] * w1[None, :]
    try:
        spec = np.abs(np.fft.fftshift(np.fft.fft2(crop)))
    except Exception:                                  # pragma: no cover
        return 1.0
    c = n // 2
    gy, gx = np.mgrid[0:n, 0:n]
    r = np.sqrt((gy - c) ** 2 + (gx - c) ** 2) / float(c)
    band = (r >= 0.35) & (r <= 0.95)
    vals = spec[band]
    if vals.size < 32:
        return 1.0
    med = float(np.median(vals))
    if med <= 1e-9:
        return 1.0
    return float(vals.max() / med)


# ══════════════════════════════════════════════════════════════════════ pose ══
def measure_pose(rgb: np.ndarray, face: Face, eyes: Optional[Tuple[Tuple[float, float],
                                                                   Tuple[float, float]]] = None
                 ) -> Pose:
    """Head position from the geometry detection already produced.

    Yaw comes from how asymmetrically the eyes sit inside the box: turn your head
    and the far eye crowds toward the centre. Roll is the angle of the line
    between them. Both are estimates, and they are reported as such — the point is
    to tell the user "turn toward the camera", not to reconstruct a head.
    """
    a = np.asarray(rgb)
    H, W = a.shape[:2]
    x, y, w, h = face.box
    frac = h / float(H)

    yaw, roll = 0.0, 0.0
    if eyes is not None:
        (lx, ly), (rx, ry) = eyes
        span = max(1e-6, rx - lx)
        mid = (lx + rx) / 2.0
        # Where the eye pair sits inside the box, relative to its centre.
        offset = (mid - (x + w / 2.0)) / (w / 2.0)
        yaw = float(max(-1.0, min(1.0, offset)))
        roll = float(math.degrees(math.atan2(ry - ly, span)))
    else:
        # No eyes found: fall back on the skin blob's centroid inside the box.
        y_plane, _, _ = to_ycbcr(a)
        y0, y1 = max(0, y), min(H, y + h)
        x0, x1 = max(0, x), min(W, x + w)
        if y1 > y0 and x1 > x0:
            sub = y_plane[y0:y1, x0:x1]
            cols = sub.mean(axis=0)
            if cols.size > 2:
                idx = np.arange(cols.size, dtype=np.float32)
                wsum = float(cols.sum()) or 1.0
                centroid = float((idx * cols).sum() / wsum)
                yaw = float(max(-1.0, min(1.0, (centroid - (x1 - x0) / 2.0) / ((x1 - x0) / 2.0))))

    if frac >= FACE_CLOSE:
        distance = "close"
    elif frac >= FACE_TOO_FAR:
        distance = "mid"
    else:
        distance = "far"

    cx = (x + w / 2.0) / float(W)
    cy = (y + h / 2.0) / float(H)
    centred = 0.28 <= cx <= 0.72 and 0.22 <= cy <= 0.78
    return Pose(yaw, roll, distance, frac, centred)


# ════════════════════════════════════════════════════════════════════ motion ══
def measure_motion(prev: Optional[np.ndarray], cur: np.ndarray) -> float:
    """Mean absolute luma difference between two frames, 0..~255 scale.

    Deliberately crude. The question worth answering is "is the room static or is
    something happening", and a frame difference answers it in microseconds
    without tracking a single point.
    """
    if prev is None:
        return 0.0
    a, b = np.asarray(prev), np.asarray(cur)
    if a.shape != b.shape:
        return 0.0
    ya, _, _ = to_ycbcr(a)
    yb, _, _ = to_ycbcr(b)
    return float(np.abs(ya.astype(np.float32) - yb.astype(np.float32)).mean())


# ═══════════════════════════════════════════════════════════════════ describe ══
def describe(rgb: np.ndarray, *, identity: Optional[Any] = None,
             prev: Optional[np.ndarray] = None, max_faces: int = 4) -> Scene:
    """Build a full :class:`Scene` for one frame.

    ``identity`` is an optional ``vision.faces.Match``; when supplied the summary
    names the person and the quality reflects whether the frame is good enough to
    have trusted that identification.
    """
    a = np.asarray(rgb)
    if a.ndim != 3 or a.shape[2] < 3:
        return Scene(0, summary_he="הפריים לא תקין.", warnings=["invalid frame"])

    faces = detect(a, max_faces=max_faces)
    lighting = measure_lighting(a)
    warnings: List[str] = []

    liveness: Optional[Liveness] = None
    pose: Optional[Pose] = None
    primary: Optional[Face] = None

    if faces:
        # The primary face is the largest — the person closest to the camera, who
        # is the one a permission decision should be about.
        primary = max(faces, key=lambda f: f.w * f.h)
        liveness = measure_liveness(a, primary.box)
        eyes = None
        if primary.eyes:
            try:
                from vision.faces import _find_eyes, to_ycbcr as _y
                y_plane, _, _ = _y(a)
                found = _find_eyes(y_plane, primary.box)
                eyes = found
            except Exception:
                eyes = None
        pose = measure_pose(a, primary, eyes)

    motion = measure_motion(prev, a)

    # ── quality: can this frame support an identification at all? ──
    quality = 0.0
    if primary is not None and pose is not None:
        quality = 0.55
        if pose.distance == "close":
            quality += 0.18
        elif pose.distance == "mid":
            quality += 0.10
        if lighting.verdict == "good":
            quality += 0.14
        elif lighting.verdict == "dim":
            quality += 0.06
        if pose.centred:
            quality += 0.06
        if abs(pose.yaw) < 0.35:
            quality += 0.07
        if liveness is not None and liveness.verdict == "live":
            quality += 0.10
        quality = float(max(0.0, min(1.0, quality)))

    if lighting.verdict == "dark":
        warnings.append("חשוך מדי לזיהוי")
    elif lighting.verdict == "blown":
        warnings.append("מואר מדי — הפנים שרופות")
    if lighting.backlit:
        warnings.append("תאורה אחורית — כבה את האור שמאחוריך")
    if pose is not None and pose.distance == "far":
        warnings.append("רחוק מדי מהמצלמה")
    if pose is not None and abs(pose.yaw) > 0.55:
        warnings.append("הפנה את הפנים למצלמה")
    if liveness is not None and liveness.verdict == "suspect":
        warnings.append("נראה כמו תמונה או מסך, לא פנים חיות")

    identity_hint = ""
    if identity is not None:
        name = getattr(identity, "name", "") or ""
        if getattr(identity, "known", False):
            identity_hint = name
        elif faces:
            identity_hint = "לא מזוהה"

    return Scene(
        people=len(faces), faces=faces, lighting=lighting, liveness=liveness,
        pose=pose, motion=motion, quality=quality, identity_hint=identity_hint,
        summary_he=_summary(len(faces), lighting, pose, liveness, motion,
                            identity_hint, quality),
        warnings=warnings,
    )


def _summary(people: int, lighting: Lighting, pose: Optional[Pose],
             liveness: Optional[Liveness], motion: float, identity: str,
             quality: float) -> str:
    """One honest Hebrew sentence about the frame — this is what JARVIS says."""
    if people == 0:
        if lighting.verdict == "dark":
            return "אני לא רואה אף אחד — חשוך מדי בחדר."
        return "אין אף אחד מול המצלמה."

    who = identity if identity and identity != "לא מזוהה" else None
    if people == 1:
        head = f"{who} מול המצלמה" if who else "אדם אחד מול המצלמה, לא מזוהה"
    else:
        head = f"{people} אנשים מול המצלמה"
        if who:
            head += f", הקרוב הוא {who}"

    bits: List[str] = []
    if pose is not None:
        dist = {"close": "קרוב", "mid": "במרחק סביר", "far": "רחוק"}[pose.distance]
        bits.append(dist)
        if abs(pose.yaw) > 0.55:
            bits.append("מופנה הצידה")
        elif abs(pose.roll) > 18:
            bits.append("ראש מוטה")
    light_he = {"good": "תאורה טובה", "dim": "תאורה חלשה", "dark": "חשוך",
                "blown": "מואר מדי", "flat": "תאורה שטוחה"}[lighting.verdict]
    bits.append(light_he)
    if lighting.backlit:
        bits.append("תאורה אחורית")
    if liveness is not None and liveness.verdict == "suspect":
        bits.append("ייתכן שזו תמונה ולא פנים חיות")
    if motion < MOTION_STILL:
        bits.append("הסצנה סטטית")

    tail = f" (איכות זיהוי {int(quality * 100)}%)" if quality else ""
    return f"{head} — {', '.join(bits)}.{tail}"
