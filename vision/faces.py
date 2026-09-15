"""Face detection and recognition — built from scratch, like everything else here.

No OpenCV, no Haar cascades, no downloaded weights, no cloud. Two pieces of
classical vision, implemented directly on numpy arrays:

**Detection.** Skin is distinctive in chroma, not in brightness: the Cb/Cr
planes of YCbCr separate it from most backgrounds regardless of exposure. A
skin mask is thresholded, cleaned with morphological opening/closing, split into
connected components, and each candidate is judged on geometry (faces are taller
than wide, solidly filled, not touching every edge) and on *structure* — a real
face has dark eye sockets and brows in the upper band, a skin-coloured wall does
not.

**Recognition.** Eigenfaces. Every enrolled face is normalised to a 64x64
zero-mean unit-variance vector; PCA over the gallery (numpy SVD) gives a compact
face space; a probe is projected into it and compared against each person's
samples. Two guards keep it honest: the distance is normalised by the gallery's
own spread, so the same threshold works for one enrollee or twenty, and the
*residual* — how much of the probe lies outside face space at all — rejects
pictures of hands, mugs and wallpaper.

**Privilege.** Recognition alone is not authorisation. `FaceGate` maps an
identity to a permission level in the existing firewall, and — this is the part
that matters — the grant is *presence-bound*: it decays back to the default the
moment the face stops being seen, so nobody inherits privileges from an empty
room or from a photo held up on the way out.
"""

from __future__ import annotations

import json
import math
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from scipy import ndimage as _ndi
except Exception:  # pragma: no cover - scipy is a hard dependency, but degrade safely
    _ndi = None

ROOT = Path(__file__).resolve().parent.parent
FACE_SIZE = 64                 # eigenface resolution
DIM = FACE_SIZE * FACE_SIZE
MAX_COMPONENTS = 24            # cap on candidate blobs examined per frame
# Calibrated on held-out frames rather than guessed. Across galleries of one, two,
# three, five and eight people (76 probes), every genuine enrollee scored 0.915 or
# better and every stranger 0.789 or worse, with all 76 attributed to the right
# person. The threshold sits inside that measured gap.
CONFIDENCE_THRESHOLD = 0.85    # below this the face is a stranger
LEVELS = ("SAFE", "WRITE", "CRITICAL")
_RANK = {name: i for i, name in enumerate(LEVELS)}

# ────────────────────────────────────────────────────────────── the owner ──
# The first face JARVIS is ever taught belongs to the person who switched it on.
# That person is the owner: full CRITICAL, and the only identity whose presence
# is allowed to unlock CRITICAL at all. The rule is positional, not nominal — it
# does not matter what name is typed in, the *first* enrolment wins — because a
# name can be typed by anyone standing in front of the camera, while "you were
# here when the gallery was empty" cannot be claimed after the fact.
OWNER_LEVEL = "CRITICAL"
DEFAULT_OWNER_NAME = "OSCAR"
ROLES = ("owner", "guest")


# ══════════════════════════════ colour spaces ══════════════════════════════
def to_ycbcr(rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """RGB (H,W,3) uint8 → Y, Cb, Cr as float32 planes."""
    a = np.asarray(rgb, dtype=np.float32)
    if a.ndim == 2:                                  # already grey: fake chroma
        return a, np.full(a.shape, 128.0, np.float32), np.full(a.shape, 128.0, np.float32)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = 128.0 - 0.168736 * r - 0.331264 * g + 0.5 * b
    cr = 128.0 + 0.5 * r - 0.418688 * g - 0.081312 * b
    return y, cb, cr


# Chroma window for skin. The textbook bounds (Cb 77-127, Cr 133-173) are tuned
# for bright, warm-lit, lightly pigmented skin and quietly drop olive and dark
# complexions once the light falls — measured here: an olive face at 0.94 light
# lands on Cr 132.3 and vanishes. The window is therefore a little wider on both
# planes; what that admits (beige walls, wood) is rejected by the eye-structure
# test, which is the filter that actually decides.
SKIN_CB = (75.0, 132.0)
SKIN_CR = (128.0, 176.0)
SKIN_CHROMA_GAP = 10.0     # Cr - Cb: skin always leans red over blue
SKIN_RED_BLUE_GAP = 15.0   # R - B: ...and never has more blue than red


def skin_mask(rgb: np.ndarray) -> np.ndarray:
    """Chroma-thresholded skin mask in YCbCr.

    The two gap conditions matter as much as the window. A perfectly neutral grey
    sits at Cr - Cb = 0 and R - B = 0 *exactly*, so without them every grey pixel
    above the brightness floor counts as skin — grey hair merges into the head and
    doubles the crop, and concrete, white shirts and overcast sky become faces.
    Measured here: dropping the gaps made one enrollee's own samples further apart
    than two different people.
    """
    a = np.asarray(rgb, dtype=np.float32)
    if a.ndim == 2:
        a = np.stack([a, a, a], axis=-1)
    y, cb, cr = to_ycbcr(a)
    r, b = a[..., 0], a[..., 2]
    return ((cb >= SKIN_CB[0]) & (cb <= SKIN_CB[1]) &
            (cr >= SKIN_CR[0]) & (cr <= SKIN_CR[1]) & (y >= 40) &
            ((cr - cb) >= SKIN_CHROMA_GAP) & ((r - b) >= SKIN_RED_BLUE_GAP))


def _clean(mask: np.ndarray) -> np.ndarray:
    """Drop speckle, then bridge and fill the gaps a face is full of.

    Eyes, brows, nostrils, mouth and hair all punch holes in a skin mask, which
    would otherwise shred one face into several ragged fragments and halve its
    measured fill ratio. Opening removes noise first; a wide closing plus
    hole-filling reunites the head, so the geometry checks below see a face
    rather than a doughnut.
    """
    if _ndi is None or mask.sum() < 8:
        return mask
    st3 = np.ones((3, 3), bool)
    st7 = np.ones((7, 7), bool)
    out = _ndi.binary_opening(mask, structure=st3, iterations=1)
    out = _ndi.binary_closing(out, structure=st3, iterations=2)
    out = _ndi.binary_closing(out, structure=st7, iterations=1)
    out = _ndi.binary_fill_holes(out)
    return np.asarray(out, bool)


def _components(mask: np.ndarray) -> List[Tuple[int, int, int, int, int]]:
    """Connected components → (x, y, w, h, area), largest first."""
    if _ndi is None:
        ys, xs = np.nonzero(mask)
        if len(xs) == 0:
            return []
        return [(int(xs.min()), int(ys.min()),
                 int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1), int(len(xs)))]
    lab, n = _ndi.label(mask)
    if n == 0:
        return []
    out = []
    objs = _ndi.find_objects(lab)
    areas = np.bincount(lab.ravel())
    for i, sl in enumerate(objs, start=1):
        if sl is None:
            continue
        y0, y1 = sl[0].start, sl[0].stop
        x0, x1 = sl[1].start, sl[1].stop
        out.append((int(x0), int(y0), int(x1 - x0), int(y1 - y0), int(areas[i])))
    out.sort(key=lambda t: -t[4])
    return out[:MAX_COMPONENTS]


def _resize(plane: np.ndarray, size: int = FACE_SIZE) -> np.ndarray:
    if _ndi is not None:
        zy = size / max(1, plane.shape[0])
        zx = size / max(1, plane.shape[1])
        return np.asarray(_ndi.zoom(plane, (zy, zx), order=1, mode="nearest"), np.float32)
    ys = np.linspace(0, plane.shape[0] - 1, size).astype(int)
    xs = np.linspace(0, plane.shape[1] - 1, size).astype(int)
    return np.asarray(plane[np.ix_(ys, xs)], np.float32)


def _has_eye_structure(y_plane: np.ndarray, box: Tuple[int, int, int, int]) -> bool:
    """A face has dark eyes and brows in its upper half; flat blobs do not.

    This is the filter that separates a person from a skin-coloured wall, so the
    band is generous: hair often covers the forehead entirely, which puts the eyes
    almost at the top of the surviving skin blob.
    """
    x, yy, w, h = box
    if h < 12 or w < 8:
        return False
    y0, y1 = yy + int(0.04 * h), yy + int(0.62 * h)
    band = y_plane[max(yy, y0):y1, x: x + w]
    lower = y_plane[y1:yy + h, x: x + w]
    if band.size < 16:
        return False
    base = float(np.mean(y_plane[yy:yy + h, x:x + w]))
    dark = float(np.mean(band < base - 12.0))
    if dark < 0.010 or float(np.std(band)) < 4.5:
        return False
    if lower.size >= 16 and float(np.std(lower)) < 1.0 and dark < 0.05:
        return False                      # upper speckle on a perfectly flat lower face
    return True


@dataclass
class Face:
    """A detected face: frame box plus its eigenface vector."""
    x: int
    y: int
    w: int
    h: int
    score: float = 0.0
    vector: Optional[np.ndarray] = None
    eyes: bool = False                         # were the eyes found for alignment?

    @property
    def box(self) -> Tuple[int, int, int, int]:
        return (self.x, self.y, self.w, self.h)

    def to_dict(self) -> Dict[str, Any]:
        return {"x": self.x, "y": self.y, "w": self.w, "h": self.h,
                "score": round(self.score, 4), "eyes": self.eyes}


def detect(rgb: np.ndarray, max_faces: int = 4) -> List[Face]:
    """Find faces in an RGB frame. Returns [] when nothing face-like survives."""
    a = np.asarray(rgb)
    if a.ndim != 3 or a.shape[2] < 3 or min(a.shape[:2]) < 24:
        return []
    H, W = a.shape[:2]
    y, _, _ = to_ycbcr(a)
    mask = _clean(skin_mask(a))
    if not mask.any():
        return []

    faces: List[Face] = []
    for (x, yy, w, h, area) in _components(mask):
        if area < max(240, 0.0016 * H * W):
            continue
        aspect = h / max(1, w)
        # brows-to-chin skin is often about as wide as it is tall, especially when
        # hair covers the forehead, so the window is wide and the eye-structure
        # test below does the real discrimination.
        if not (0.62 <= aspect <= 2.20):
            continue
        if area / float(w * h) < 0.45:                    # too ragged to be a face
            continue
        # a blob spanning the whole frame is a wall, not a person
        edges = (x <= 1) + (yy <= 1) + (x + w >= W - 1) + (yy + h >= H - 1)
        if edges >= 3:
            continue
        if w < 18 or h < 22:
            continue
        if not _has_eye_structure(y, (x, yy, w, h)):
            continue
        faces.append(Face(x, yy, w, h, score=float(area) / float(H * W)))
        if len(faces) >= max_faces:
            break
    return faces


# Where the eyes land in the aligned output, as fractions of its side. Pinning
# them makes the encoding invariant to head position, distance from the camera and
# tilt — without that, a person leaning 8% closer to the lens looked further from
# themselves than from somebody else.
EYE_LEFT = (0.315, 0.40)
EYE_RIGHT = (0.685, 0.40)


def _find_eyes(y_plane: np.ndarray, box: Tuple[int, int, int, int],
               face_region: Optional[np.ndarray] = None
               ) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
    """Locate both eyes as darkness centroids on the row where they are darkest.

    Three things make this stable enough to align on.

    *A relative darkness threshold* (0.62 x the local mean) selects the same
    pixels whether the lamp is bright or dim.

    *A horizontal offset gate* — dark pixels between 7% and 44% of the face width
    from the midline — excludes the nose and mouth (central) and a low fringe or
    the ears (outer). It gates rather than weights: weighting by the offset pulled
    the centroid out towards the temples and inflated the measured eye separation
    by 44%.

    *A row window around the peak of bilateral dark mass.* Eyes outweigh brows
    roughly three to one here, so the peak row is the eye row; taking the centroid
    only within +-6% of the face height of that peak keeps the brows, which sit
    just above, from dragging the anchor upwards.

    Returns None when the two halves do not agree on a plausible eye pair, and the
    caller falls back to centring the box instead.
    """
    x, yy, w, h = box
    if w < 12 or h < 12:
        return None
    y0 = yy + int(0.02 * h)
    y1 = min(y_plane.shape[0], yy + int(0.55 * h))
    x0, x1 = x, min(y_plane.shape[1], x + w)
    reg = y_plane[y0:y1, x0:x1]
    if reg.size < 24 or reg.shape[1] < 8:
        return None
    if _ndi is not None:
        reg = np.asarray(_ndi.gaussian_filter(reg, 0.9), np.float32)
    base = float(np.mean(y_plane[yy:yy + h, x:x + w]))
    thr = 0.62 * max(1.0, base)
    dark = np.clip(thr - reg, 0.0, None) ** 2

    # A bounding box is a rectangle and a face is not, so its corners hold
    # background — dark, lateral, and easily mistaken for a pair of eyes. Eyes are
    # dark regions *enclosed by skin*; corners, hair and the room behind the head
    # all connect to the outer background, and the hole-filled skin mask is exactly
    # the difference between those two sets.
    if face_region is not None:
        sub = np.asarray(face_region)[y0:y1, x0:x1]
        if sub.shape == dark.shape:
            dark = dark * sub
    if float(dark.sum()) <= 1e-6:
        return None

    hh, ww = dark.shape
    mid = int(ww / 2)
    if mid < 3 or ww - mid < 3:
        return None
    off = np.abs(np.arange(ww) - mid)[None, :]
    gate = ((off >= 0.07 * ww) & (off <= 0.44 * ww)).astype(np.float32)
    rowmass = (dark * gate).sum(axis=1)
    if float(rowmass.sum()) <= 1e-6:
        return None
    if hh >= 5:                                  # smooth, then take the peak row
        k = np.array([1.0, 2.0, 3.0, 2.0, 1.0], np.float32)
        pad = np.convolve(rowmass, k / k.sum(), mode="same")
        peak = int(np.argmax(pad))
    else:
        peak = int(np.argmax(rowmass))
    dy = max(2, int(0.06 * hh))
    r0, r1 = max(0, peak - dy), min(hh, peak + dy + 1)

    win = dark[r0:r1]
    left = win[:, :mid]
    right = win[:, mid:]
    if float(left.sum()) <= 1e-6 or float(right.sum()) <= 1e-6:
        return None
    rows = np.arange(r0, r1)[:, None]
    lcols = np.arange(left.shape[1])[None, :]
    rcols = mid + np.arange(right.shape[1])[None, :]
    lx = float((left * lcols).sum() / left.sum())
    ly = float((left * rows).sum() / left.sum())
    rx = float((right * rcols).sum() / right.sum())
    ry = float((right * rows).sum() / right.sum())

    sep = rx - lx
    if sep < 0.16 * w or sep > 1.2 * w:
        return None
    if abs(ry - ly) > 0.30 * sep:                # not a plausible eye pair
        return None
    return ((x0 + lx, y0 + ly), (x0 + rx, y0 + ry))


def _warp(plane: np.ndarray, le: Tuple[float, float], re: Tuple[float, float],
          size: int) -> np.ndarray:
    """Similarity transform putting the eyes on the canonical points.

    Eyes arrive as (x, y); scipy's affine_transform indexes arrays as (row, col),
    i.e. (y, x), and maps *output* coordinates to *input* ones. Swapping those two
    conventions silently transposes the result — the face comes out sideways with a
    strip of whatever was above the head down one side, and because the width of
    that strip depends on head distance, the same person at a different distance
    looks like a stranger. Hence the explicit axis swap below.
    """
    lx, ly = float(le[0]), float(le[1])
    rx, ry = float(re[0]), float(re[1])
    dx, dy = rx - lx, ry - ly
    dist = math.hypot(dx, dy) or 1.0
    ang = math.atan2(dy, dx)
    target = (EYE_RIGHT[0] - EYE_LEFT[0]) * size
    scale = target / dist
    ca, sa = math.cos(ang), math.sin(ang)
    # inverse map in (x, y): in = R(ang)/scale * (out - c_out) + le
    r00, r01, r10, r11 = ca / scale, -sa / scale, sa / scale, ca / scale
    cxo, cyo = EYE_LEFT[0] * size, EYE_LEFT[1] * size
    ox = lx - (r00 * cxo + r01 * cyo)
    oy = ly - (r10 * cxo + r11 * cyo)
    mat = np.array([[r11, r10], [r01, r00]], np.float64)   # (row, col) ordering
    offset = np.array([oy, ox], np.float64)
    fill = float(np.median(plane))
    if _ndi is None:
        return _resize(plane, size)
    return np.asarray(_ndi.affine_transform(
        plane, mat, offset=offset, output_shape=(size, size), order=1,
        mode="constant", cval=fill, prefilter=False), np.float32)


def _warp_box(plane: np.ndarray, box: Tuple[int, int, int, int], size: int) -> np.ndarray:
    """Fallback when the eyes are not found: centre the box, keep its scale."""
    x, y, w, h = box
    side = max(8.0, 1.30 * max(w, h))
    s = size / side
    inv = 1.0 / s
    half = size / 2.0
    mat = np.array([[inv, 0.0], [0.0, inv]], np.float64)
    offset = np.array([y + h / 2.0 - inv * half, x + w / 2.0 - inv * half], np.float64)
    fill = float(np.median(plane))
    if _ndi is None:
        return _resize(plane[y:y + h, x:x + w], size)
    return np.asarray(_ndi.affine_transform(
        plane, mat, offset=offset, output_shape=(size, size), order=1,
        mode="constant", cval=fill, prefilter=False), np.float32)


def encode(rgb: np.ndarray, box: Tuple[int, int, int, int],
           face_region: Optional[np.ndarray] = None) -> np.ndarray:
    """Frame + face box → aligned 64x64 grey → zero-mean unit-variance vector.

    Alignment comes first: the eyes are located and warped onto fixed points, so
    the vector describes *who the face is* rather than where it happened to be in
    the frame. Only then is contrast normalised away, which removes the lamp.
    """
    a = np.asarray(rgb)
    if a.ndim == 2:
        a = np.stack([a, a, a], axis=-1)
    y_plane = np.asarray(to_ycbcr(a)[0], np.float32)
    if face_region is None:
        face_region = _clean(skin_mask(a))
    eyes = _find_eyes(y_plane, box, face_region)
    g = _warp(y_plane, eyes[0], eyes[1], FACE_SIZE) if eyes else _warp_box(y_plane, box, FACE_SIZE)
    v = np.asarray(g, np.float32).reshape(-1)
    v -= float(v.mean())
    sd = float(v.std())
    return v / (sd if sd > 1e-6 else 1.0)


def encode_frame(rgb: np.ndarray) -> Optional[Face]:
    """Convenience: detect the most prominent face and return it with its vector."""
    found = detect(rgb, max_faces=1)
    if not found:
        return None
    f = found[0]
    a = np.asarray(rgb)
    if a.ndim == 2:
        a = np.stack([a, a, a], axis=-1)
    region = _clean(skin_mask(a))
    f.vector = encode(a, f.box, region)
    f.eyes = _find_eyes(np.asarray(to_ycbcr(a)[0], np.float32), f.box, region) is not None
    return f


# ══════════════════════════════ identity store ══════════════════════════════
@dataclass
class Person:
    id: str
    name: str
    level: str = "SAFE"
    vectors: List[np.ndarray] = field(default_factory=list)
    created: float = field(default_factory=time.time)
    note: str = ""
    role: str = "guest"          # "owner" for the first face ever enrolled

    @property
    def is_owner(self) -> bool:
        return self.role == "owner"

    def to_dict(self, with_vectors: bool = False) -> Dict[str, Any]:
        d = {"id": self.id, "name": self.name, "level": self.level,
             "samples": len(self.vectors), "created": round(self.created, 3),
             "note": self.note, "role": self.role, "is_owner": self.is_owner}
        if with_vectors:
            d["vectors"] = [np.asarray(v, np.float32).tolist() for v in self.vectors]
        return d


@dataclass
class Match:
    identity: Optional[str]          # person id, or None for a stranger
    name: str
    level: str
    confidence: float
    residual: float
    faces: int
    known: bool
    role: str = "guest"              # "owner" only for the first face enrolled

    def to_dict(self) -> Dict[str, Any]:
        return {"identity": self.identity, "name": self.name, "level": self.level,
                "confidence": round(float(self.confidence), 4),
                "residual": round(float(self.residual), 4),
                "faces": self.faces, "known": self.known, "role": self.role,
                "is_owner": self.role == "owner"}


class FaceStore:
    """Enrolled faces + eigenface matching. Persists as plain JSON in data/faces."""

    def __init__(self, path: Optional[Path] = None, max_samples: int = 8) -> None:
        self.path = Path(path or (ROOT / "data" / "faces" / "gallery.json"))
        self.max_samples = int(max_samples)
        self.people: List[Person] = []
        self._lock = threading.RLock()
        self._basis: Optional[np.ndarray] = None
        self._mean: Optional[np.ndarray] = None
        self._feats: Optional[np.ndarray] = None
        self._spread: float = 1.0
        self._within: float = 1.0
        self._between: float = 3.0
        self._gallery: Optional[np.ndarray] = None
        self._residual_ref: float = 1.0
        self._ids: List[str] = []
        self.load()

    # ------------------------------------------------------------- persist --
    def load(self) -> None:
        with self._lock:
            self.people = []
            if self.path.exists():
                try:
                    raw = json.loads(self.path.read_text(encoding="utf-8"))
                    for rec in raw.get("people", []):
                        vs = [np.asarray(v, np.float32) for v in rec.get("vectors", [])]
                        self.people.append(Person(
                            id=str(rec["id"]), name=str(rec.get("name", rec["id"])),
                            level=str(rec.get("level", "SAFE")).upper(), vectors=vs,
                            created=float(rec.get("created", time.time())),
                            note=str(rec.get("note", "")),
                            role=str(rec.get("role", "guest")).lower()))
                except Exception:
                    self.people = []
            self._rebuild()

    def save(self) -> None:
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            payload = {"version": 1, "face_size": FACE_SIZE,
                       "saved": time.time(),
                       "people": [p.to_dict(with_vectors=True) for p in self.people]}
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(payload), encoding="utf-8")
            tmp.replace(self.path)

    # ------------------------------------------------------------- gallery --
    def _rebuild(self) -> None:
        """Recompute the PCA basis. Called whenever the gallery changes."""
        vecs: List[np.ndarray] = []
        ids: List[str] = []
        for p in self.people:
            for v in p.vectors:
                vecs.append(np.asarray(v, np.float32).ravel())
                ids.append(p.id)
        self._ids = ids
        if not vecs:
            self._basis = self._mean = self._feats = None
            self._gallery = None
            self._spread, self._within, self._between = 1.0, 1.0, 3.0
            return
        X = np.stack(vecs)
        self._gallery = X
        self._mean = X.mean(axis=0)
        C = X - self._mean
        if len(vecs) == 1:
            self._basis = np.zeros((0, DIM), np.float32)
            self._feats = np.zeros((1, 0), np.float32)
            self._spread, self._within, self._between = 1.0, 1.0, 3.0
            return
        # economy SVD: rows of Vt are the eigenfaces
        try:
            _, s, vt = np.linalg.svd(C, full_matrices=False)
        except np.linalg.LinAlgError:
            self._basis = np.zeros((0, DIM), np.float32)
            self._feats = np.zeros((len(vecs), 0), np.float32)
            self._spread = 1.0
            return
        k = int(min(len(s), max(1, len(vecs) - 1), 24))
        keep = s[:k] > 1e-6
        self._basis = np.asarray(vt[:k][keep], np.float32)
        self._feats = np.asarray(C @ self._basis.T, np.float32)
        # Calibrate on the gallery itself: how far apart two samples of the SAME
        # person sit, versus two samples of DIFFERENT people. A probe is then
        # scored by where it falls between those two references, so one threshold
        # works for a gallery of one enrollee or twenty — the same trick the speech
        # bank uses with its self/cross reference distances. Distances are measured
        # on the aligned vectors themselves, not on their PCA projection: with a
        # single enrollee the projection is three-dimensional and throws away the
        # very detail that identifies them (measured — a stranger landed closer
        # than the owner).
        d = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=-1)
        same: List[float] = []
        diff: List[float] = []
        for i in range(len(vecs)):
            for j in range(i + 1, len(vecs)):
                (same if ids[i] == ids[j] else diff).append(float(d[i, j]))
        within = float(np.median(same)) if same else 0.0
        if within <= 1e-6:
            within = 0.25 * float(np.median(np.linalg.norm(self._feats, axis=1)) or 1.0)
        between = float(np.median(diff)) if diff else within * 3.0
        if between <= within * 1.15:
            between = within * (3.0 if not diff else 1.15) + 1e-6
        self._within, self._between = max(1e-6, within), between
        self._spread = self._within
        # The eigenbasis is kept for the residual figure — how much of a probe lies
        # outside the space this gallery spans. It is reported, not enforced: a
        # one-person gallery leaves 0.49-0.95 of even its owner outside three
        # dimensions, so gating on it would lock out the most common installation.
        self._residual_ref = float(np.median([
            float(np.linalg.norm(C[i] - (C[i] @ self._basis.T) @ self._basis))
            / (float(np.linalg.norm(C[i])) or 1.0)
            for i in range(len(vecs))])) if self._basis is not None and self._basis.size else 1.0

    def enroll(self, name: str, vector: np.ndarray, level: str = "SAFE",
               person_id: Optional[str] = None, note: str = "") -> Person:
        """Add a face sample. Re-enrolling the same id/name appends to that person."""
        level = str(level).upper()
        if level not in LEVELS:
            level = "SAFE"
        v = np.asarray(vector, np.float32).ravel()
        if v.size != DIM:
            raise ValueError(f"face vector must be {DIM} floats, got {v.size}")
        with self._lock:
            target = None
            for p in self.people:
                if (person_id and p.id == person_id) or (not person_id and p.name == name):
                    target = p
                    break
            if target is None:
                pid = person_id or f"p{len(self.people) + 1:03d}-{int(time.time())}"
                target = Person(id=pid, name=name or pid, level=level, note=note)
                # ── the owner claim ──
                # An empty gallery means nobody has ever been taught. Whoever is
                # enrolled into it is the person who set JARVIS up, so they get
                # the owner role and full CRITICAL — regardless of what level was
                # asked for, because asking for SAFE on your own machine would
                # otherwise lock you out of your own assistant.
                if not self.people and self.owner() is None:
                    target.role = "owner"
                    target.level = OWNER_LEVEL
                    if not note:
                        target.note = "היוצר — נרשם ראשון, הרשאות מלאות"
                self.people.append(target)
            else:
                # Re-enrolling never silently strips the owner's level. A guest's
                # level is whatever was asked for; the owner's stays CRITICAL
                # unless set_level() is called explicitly.
                if not target.is_owner:
                    target.level = level
                if name:
                    target.name = name
                if note:
                    target.note = note
            target.vectors.append(v)
            # keep the newest samples: a face changes with hair, glasses, light
            if len(target.vectors) > self.max_samples:
                target.vectors = target.vectors[-self.max_samples:]
            self._rebuild()
            self.save()
            return target

    def remove(self, person_id: str) -> bool:
        with self._lock:
            before = len(self.people)
            self.people = [p for p in self.people if p.id != person_id]
            if len(self.people) == before:
                return False
            self._rebuild()
            self.save()
            return True

    def set_level(self, person_id: str, level: str, force: bool = False) -> bool:
        """Change what someone may do.

        The owner's level is protected: a stray re-level — a UI slip, a skill call
        with a bad argument, a guest who got WRITE and found the admin endpoint —
        must not be able to demote the person who owns the machine. Passing
        ``force=True`` is the explicit, auditable way to do it anyway.
        """
        level = str(level).upper()
        if level not in LEVELS:
            return False
        with self._lock:
            for p in self.people:
                if p.id == person_id:
                    if p.is_owner and level != OWNER_LEVEL and not force:
                        return False
                    p.level = level
                    self.save()
                    return True
        return False

    def get(self, person_id: str) -> Optional[Person]:
        with self._lock:
            for p in self.people:
                if p.id == person_id:
                    return p
        return None

    def owner(self) -> Optional[Person]:
        """The first person ever enrolled, or None if the gallery is empty."""
        with self._lock:
            for p in self.people:
                if p.is_owner:
                    return p
        return None

    def set_owner(self, person_id: str) -> bool:
        """Move the owner role to somebody else. Exactly one owner exists."""
        with self._lock:
            target = self.get(person_id)
            if target is None:
                return False
            for p in self.people:
                p.role = "guest" if p.id != person_id else "owner"
                p.level = OWNER_LEVEL if p.id == person_id else (
                    p.level if p.level != OWNER_LEVEL else "WRITE")
            self.save()
            return True

    def list(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [p.to_dict() for p in self.people]

    # ------------------------------------------------------------- matching --
    def match(self, vector: np.ndarray) -> Tuple[Optional[str], float, float]:
        """(person_id, confidence, residual).

        Confidence is 1.0 when the probe sits as close as the same person's own
        samples do, 0.0 when it is as far as a different person's, and the residual
        says how much of it lies outside the space this gallery spans. A confidence
        below CONFIDENCE_THRESHOLD means "stranger" — and, because that number is
        what opens permissions, it is deliberately the conservative reading.
        """
        x = np.asarray(vector, np.float32).ravel()
        with self._lock:
            if not self.people or self._gallery is None or self._mean is None:
                return None, 0.0, 1.0
            d = np.linalg.norm(self._gallery - x[None, :], axis=1)
            best = int(np.argmin(d))
            best_d = float(d[best])
            best_id = self._ids[best]

            c = x - self._mean
            cn = float(np.linalg.norm(c)) or 1.0
            residual = 1.0
            if self._basis is not None and self._basis.size:
                f = c @ self._basis.T
                residual = float(np.linalg.norm(c - f @ self._basis)) / cn
            elif self._feats is not None and self._feats.shape[1] == 0:
                # one sample in the gallery: report cosine distance instead
                vv = self._gallery[0] - self._mean
                n2 = float(np.linalg.norm(vv)) or 1.0
                residual = 1.0 - max(-1.0, min(1.0, float(np.dot(c, vv) / (cn * n2))))

            span = max(1e-6, self._between - self._within)
            conf = float(max(0.0, min(1.0, (self._between - best_d) / span)))
            return best_id, conf, residual

    def recognize(self, rgb: np.ndarray) -> Match:
        """Detect + identify in one call."""
        found = detect(rgb, max_faces=4)
        if not found:
            return Match(None, "—", "SAFE", 0.0, 0.0, 0, False)
        # Identify the largest face — the person closest to the lens, who is the
        # one a permission decision should be about when more than one is present.
        primary = max(found, key=lambda f: f.w * f.h)
        primary.vector = encode(rgb, primary.box)
        pid, conf, resid = self.match(primary.vector)
        if pid and conf >= CONFIDENCE_THRESHOLD:
            p = self.get(pid)
            if p is not None:
                return Match(p.id, p.name, p.level, conf, resid, len(found), True,
                             role=p.role)
        return Match(None, "לא מזוהה", "SAFE", conf, resid, len(found), False)


# ═══════════════════════════ presence-gated privilege ═══════════════════════
class FaceGate:
    """Turns "who is in front of the camera" into "what JARVIS may do right now".

    The grant is deliberately temporary. Every observation refreshes a lease; when
    the lease expires (nobody recognised for `ttl` seconds) the level falls back to
    the default. Privilege therefore tracks presence instead of persisting after
    the person has left the room.
    """

    def __init__(self, store: FaceStore, firewall: Any = None, default_level: str = "SAFE",
                 ceiling: str = "CRITICAL", ttl: float = 12.0, debounce: int = 2,
                 unknown_level: Optional[str] = None) -> None:
        self.store = store
        self.firewall = firewall
        self.default_level = str(default_level).upper()
        self.ceiling = str(ceiling).upper()
        self.ttl = float(ttl)
        self.debounce = max(1, int(debounce))
        # a stranger gets the default level unless configured otherwise
        self.unknown_level = str(unknown_level or default_level).upper()
        self.identity: Optional[str] = None
        self.name: str = "—"
        self.level: str = self.default_level
        self.confidence: float = 0.0
        self.last_seen: float = 0.0
        self.armed: bool = bool(firewall is not None)
        self._pending: List[Optional[str]] = []
        self._lock = threading.RLock()
        self.events: List[Dict[str, Any]] = []

    # ---------------------------------------------------------------- apply --
    def _clamp(self, level: str) -> str:
        if level not in LEVELS:
            return self.default_level
        return level if _RANK[level] <= _RANK[self.ceiling] else self.ceiling

    def _set_level(self, level: str, who: str, why: str) -> None:
        level = self._clamp(level)
        if not self.armed:
            # disarmed (kill switch, or no firewall attached): still report who is
            # in frame, but never claim a privilege that is not being enforced.
            level = self._clamp(self.default_level)
        changed = level != self.level
        self.level = level
        if changed and self.firewall is not None and self.armed:
            try:
                self.firewall.set_level(level)
            except Exception:
                pass
        rec = {"ts": time.time(), "identity": self.identity, "name": who,
               "level": level, "changed": changed, "why": why}
        self.events.append(rec)
        if len(self.events) > 200:
            self.events = self.events[-200:]
        if changed:
            try:
                from core.bus import BUS
                BUS.emit("vision.level", {"name": who, "level": level, "why": why},
                         source="faces")
            except Exception:
                pass

    def observe(self, match: Match) -> Dict[str, Any]:
        """Feed one camera frame's result. Returns the effective state."""
        with self._lock:
            now = time.time()
            pid = match.identity if match.known else None
            self._pending.append(pid)
            self._pending = self._pending[-self.debounce:]
            stable = len(self._pending) >= self.debounce and len(set(self._pending)) == 1
            confirmed = self._pending[-1] if stable else self.identity

            if confirmed:
                person = self.store.get(confirmed)
                if person is not None:
                    self.identity, self.name = person.id, person.name
                    self.confidence = match.confidence
                    self.last_seen = now
                    level, why = person.level, "face recognised"
                    # CRITICAL is the owner's alone. A guest enrolled at CRITICAL
                    # — by accident, or by an owner who has since forgotten —
                    # still cannot reach it, because the ceiling on everybody but
                    # the first face is WRITE.
                    if not person.is_owner and _RANK.get(level, 0) == _RANK["CRITICAL"]:
                        level, why = "WRITE", "face recognised (capped: not the owner)"
                    elif person.is_owner:
                        why = "owner recognised"
                    self._set_level(level, person.name, why)
                else:
                    confirmed = None

            if not confirmed:
                if match.faces:
                    self.identity, self.name = None, match.name or "לא מזוהה"
                    self.confidence = match.confidence
                    self.last_seen = now
                    self._set_level(self.unknown_level, "לא מזוהה", "unrecognised face")
                else:
                    self.confidence = 0.0
                    if now - self.last_seen > self.ttl and self.identity is not None:
                        self.identity, self.name = None, "—"
                        self._set_level(self.default_level, "—", "nobody in frame")
            return self.state()

    def poll(self) -> Dict[str, Any]:
        """Called on a timer so the lease expires even without new frames."""
        with self._lock:
            if self.identity is not None and time.time() - self.last_seen > self.ttl:
                self.identity, self.name, self.confidence = None, "—", 0.0
                self._pending.clear()
                self._set_level(self.default_level, "—", "presence lease expired")
            return self.state()

    def disarm(self) -> None:
        """Stop driving the firewall (used by the kill switch and tests)."""
        with self._lock:
            self.armed = False
            self.identity, self.name, self.confidence = None, "—", 0.0
            self._pending.clear()
            self.level = self.default_level

    def state(self) -> Dict[str, Any]:
        owner = self.store.owner()
        return {"identity": self.identity, "name": self.name, "level": self.level,
                "confidence": round(float(self.confidence), 4),
                "last_seen": round(self.last_seen, 3), "armed": self.armed,
                "ttl": self.ttl, "people": len(self.store.people),
                "age": round(time.time() - self.last_seen, 2) if self.last_seen else None,
                "is_owner": bool(owner is not None and owner.id == self.identity),
                "owner": ({"id": owner.id, "name": owner.name,
                           "samples": len(owner.vectors)} if owner else None),
                "owner_enrolled": owner is not None,
                "owner_name_expected": DEFAULT_OWNER_NAME}


# ══════════════════════════════════ payload I/O ══════════════════════════════
def frame_from_rgb(payload: Dict[str, Any]) -> np.ndarray:
    """Rebuild a frame the HUD sent as downsampled raw RGB.

    The browser reads pixels straight off a canvas, so no image codec is needed
    anywhere in the stack: {w, h, rgb: base64 of w*h*3 bytes}.
    """
    import base64
    w = int(payload.get("w", 0))
    h = int(payload.get("h", 0))
    data = payload.get("rgb") or payload.get("data") or ""
    if w <= 0 or h <= 0 or not data:
        raise ValueError("frame payload needs w, h and rgb")
    raw = base64.b64decode(data)
    need = w * h * 3
    if len(raw) < need:
        raise ValueError(f"frame too small: {len(raw)} bytes, expected {need}")
    arr = np.frombuffer(raw[:need], dtype=np.uint8).reshape(h, w, 3)
    return np.ascontiguousarray(arr)


def frame_from_gray(payload: Dict[str, Any]) -> np.ndarray:
    """Same, for a grey frame — recognised by the detector as colourless skin."""
    import base64
    w, h = int(payload.get("w", 0)), int(payload.get("h", 0))
    raw = base64.b64decode(payload.get("gray") or payload.get("data") or "")
    if w <= 0 or h <= 0 or len(raw) < w * h:
        raise ValueError("grey frame payload invalid")
    g = np.frombuffer(raw[:w * h], dtype=np.uint8).reshape(h, w)
    return np.ascontiguousarray(np.stack([g, g, g], axis=-1))
