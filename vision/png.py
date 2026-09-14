"""A PNG codec built from the standard library and numpy — nothing else.

Why this exists. `screen.capture` writes a PNG and nothing in JARVIS could read it
back: Pillow and OpenCV are both absent, and neither may be assumed, because the
whole project runs offline with no downloaded packages beyond what
requirements.txt already pins. The result was an assistant that could photograph
the screen and never look at the photograph — every pixel that reached `vision/`
arrived as a raw base64 buffer straight from the browser canvas, and the one path
that produced an actual image file dead-ended on disk.

PNG is a reasonable thing to implement from scratch, because the format is a
container around deflate: chunk framing with CRC-32, a zlib stream, and a
per-scanline reversible filter. Python's stdlib already provides `zlib` and
`binascii.crc32`, so the only real work is the filtering and the colour-type
bookkeeping. No third-party dependency, no native build, no weights.

Supported on decode: bit depths 1/2/4/8/16, colour types 0 (greyscale), 2 (RGB),
3 (palette), 4 (greyscale+alpha), 6 (RGBA), all five scanline filters, `tRNS`
transparency, and Adam7 interlacing. `gAMA`/`sRGB`/`iCCP`/text chunks are skipped
deliberately — colour management is out of scope for reading a screen capture, and
pretending otherwise would mean silently applying a transform nobody asked for.

The encoder is included because a decoder with no way to produce known-good input
cannot be tested honestly. Round-tripping a synthesised image against a reference
implementation is not available offline, so instead the encoder and decoder are
verified against each other *and* against the real 256x256 icon that ships in
`electron/assets`, whose IHDR fields and pixel statistics can be checked
independently.
"""

from __future__ import annotations

import binascii
import struct
import zlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

__all__ = [
    "PngError",
    "PngImage",
    "decode",
    "decode_file",
    "encode",
    "write_file",
    "to_gray",
    "to_rgb",
]

SIGNATURE = b"\x89PNG\r\n\x1a\n"

#: channels per pixel for each colour type, before any bit-depth expansion
_CHANNELS = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}
_COLOR_NAMES = {0: "greyscale", 2: "rgb", 3: "palette", 4: "greyscale+alpha", 6: "rgba"}

_FILTER_NAMES = {0: "None", 1: "Sub", 2: "Up", 3: "Average", 4: "Paeth"}

# Adam7 interlace: (x_offset, y_offset, x_step, y_step) per pass
_ADAM7 = (
    (0, 0, 8, 8),
    (4, 0, 8, 8),
    (0, 4, 4, 8),
    (2, 0, 4, 4),
    (0, 2, 2, 4),
    (1, 0, 2, 2),
    (0, 1, 1, 2),
)


class PngError(Exception):
    """Raised for a malformed or unsupported PNG. Never for a missing file."""


class PngImage:
    """Decoded pixels plus the header facts that describe them.

    `data` is returned as the natural numpy dtype for the bit depth — uint8 for
    depths <= 8, uint16 for depth 16. Single-channel images (colour types 0 and 3)
    come back two-dimensional as (height, width); everything else is
    (height, width, channels). Both 8-bit and 16-bit greyscale follow that rule, so
    a caller can index `data[y, x]` without first checking the bit depth.
    """

    __slots__ = ("data", "width", "height", "bit_depth", "color_type",
                 "palette", "transparent", "interlaced", "chunks_seen")

    def __init__(self, data: np.ndarray, width: int, height: int, bit_depth: int,
                 color_type: int, palette: Optional[np.ndarray] = None,
                 transparent: Optional[Dict[str, int]] = None,
                 interlaced: bool = False,
                 chunks_seen: Optional[List[str]] = None) -> None:
        self.data = data
        self.width = int(width)
        self.height = int(height)
        self.bit_depth = int(bit_depth)
        self.color_type = int(color_type)
        self.palette = palette
        self.transparent = transparent or None
        self.interlaced = bool(interlaced)
        self.chunks_seen = list(chunks_seen or [])

    # ------------------------------------------------------------ convenience --
    @property
    def channels(self) -> int:
        return int(self.data.shape[2]) if self.data.ndim == 3 else 1

    @property
    def has_alpha(self) -> bool:
        return self.color_type in (4, 6) or self.transparent is not None

    @property
    def color_name(self) -> str:
        return _COLOR_NAMES.get(self.color_type, f"type{self.color_type}")

    def gray(self) -> np.ndarray:
        """Luminance as float32 in 0..1, (height, width)."""
        return to_gray(self.data, self.color_type, self.palette)

    def rgb(self) -> np.ndarray:
        """RGB as uint8, (height, width, 3). Palette entries are expanded.

        Alpha is dropped, as the name says — use `rgba()` when transparency
        matters.
        """
        return to_rgb(self.data, self.color_type, self.palette, self.bit_depth)

    def alpha(self) -> np.ndarray:
        """Alpha plane as uint8 in (height, width); fully opaque if none declared.

        This is where tRNS actually gets used. Parsing the chunk is not the same
        as applying it, and a decoder that stores transparency and then ignores
        it reports `has_alpha == True` while handing back pixels with no way to
        see the transparent parts — the bug this method exists to close.

        For colour types 4 and 6 the plane is already in the file and is
        returned directly. For 0, 2 and 3 tRNS is synthesised per the spec:
        greyscale and RGB name one transparent value, palettes carry one alpha
        byte per entry.
        """
        h, w = self.height, self.width
        opaque = np.full((h, w), 255, dtype=np.uint8)
        t = self.transparent
        if self.color_type in (4, 6):
            ch = 2 if self.color_type == 4 else 4
            if self.data.ndim == 3 and self.data.shape[2] == ch:
                return np.ascontiguousarray(self.data[:, :, -1].astype(np.uint8))
            return opaque
        if not t:
            return opaque

        if self.color_type == 3 and self.palette is not None:
            # Palette alpha: a byte per entry, entries past the tRNS chunk are
            # opaque. Indices map through it, so one lookup does the whole image.
            table = np.full(len(self.palette), 255, dtype=np.uint8)
            raw = t.get("bytes")  # type: ignore[union-attr]
            if raw is not None:
                n = min(len(table), len(raw))
                table[:n] = np.frombuffer(bytes(raw[:n]), dtype=np.uint8)
            idx = self.data.astype(np.int64).clip(0, len(self.palette) - 1)
            return table[idx].astype(np.uint8)

        if self.color_type == 0 and "gray" in t:  # type: ignore[operator]
            value = t["gray"]                      # type: ignore[index]
            return np.where(self.data == value, 0, 255).astype(np.uint8)

        if self.color_type == 2 and all(k in t for k in ("r", "g", "b")):  # type: ignore[operator]
            rgb = self.data.astype(np.int64)
            if rgb.ndim == 3 and rgb.shape[2] >= 3:
                match = ((rgb[:, :, 0] == t["r"]) & (rgb[:, :, 1] == t["g"])  # type: ignore[index]
                         & (rgb[:, :, 2] == t["b"]))                          # type: ignore[index]
                return np.where(match, 0, 255).astype(np.uint8)
        return opaque

    def rgba(self) -> np.ndarray:
        """RGB plus applied transparency as uint8, (height, width, 4).

        The form a compositor wants: colours resolved through the palette and
        alpha folded in, whether the file stored it per pixel or only named a
        transparent value in tRNS.
        """
        return np.dstack([self.rgb(), self.alpha()]).astype(np.uint8)

    def __repr__(self) -> str:
        return (f"<PngImage {self.width}x{self.height} {self.color_name} "
                f"depth={self.bit_depth} interlaced={self.interlaced}>")


# ===================================================================== decode ====
def decode(raw: bytes) -> PngImage:
    """Decode PNG bytes into a PngImage. Raises PngError on malformed input."""
    if not raw or len(raw) < 8:
        raise PngError("not a PNG: too short")
    if raw[:8] != SIGNATURE:
        raise PngError("not a PNG: bad signature")

    width = height = bit_depth = color_type = interlace = None
    palette: Optional[np.ndarray] = None
    trns: Optional[bytes] = None
    idat: List[bytes] = []
    seen: List[str] = []
    pos = 8
    have_iend = False

    while pos + 8 <= len(raw):
        (length,) = struct.unpack(">I", raw[pos:pos + 4])
        ctype = raw[pos + 4:pos + 8].decode("latin1")
        body_start = pos + 8
        body_end = body_start + length
        if body_end + 4 > len(raw):
            raise PngError(f"chunk {ctype!r} truncated")
        body = raw[body_start:body_end]
        (crc_stored,) = struct.unpack(">I", raw[body_end:body_end + 4])
        crc_actual = binascii.crc32(raw[pos + 4:body_end]) & 0xFFFFFFFF
        if crc_stored != crc_actual:
            raise PngError(f"chunk {ctype!r} failed CRC "
                           f"(stored {crc_stored:#010x}, computed {crc_actual:#010x})")
        seen.append(ctype)

        if ctype == "IHDR":
            if len(body) != 13:
                raise PngError("IHDR must be 13 bytes")
            (width, height, bit_depth, color_type,
             _comp, _filt, interlace) = struct.unpack(">IIBBBBB", body)
            if width == 0 or height == 0:
                raise PngError("IHDR has a zero dimension")
            if color_type not in _CHANNELS:
                raise PngError(f"unsupported colour type {color_type}")
            if bit_depth not in (1, 2, 4, 8, 16):
                raise PngError(f"unsupported bit depth {bit_depth}")
            if not _depth_ok(color_type, bit_depth):
                raise PngError(f"bit depth {bit_depth} invalid for colour type {color_type}")
        elif ctype == "PLTE":
            if len(body) % 3:
                raise PngError("PLTE length is not a multiple of 3")
            palette = np.frombuffer(body, dtype=np.uint8).reshape(-1, 3).copy()
        elif ctype == "tRNS":
            trns = body
        elif ctype == "IDAT":
            idat.append(body)
        elif ctype == "IEND":
            have_iend = True
            break
        # every other chunk (gAMA, sRGB, iCCP, pHYs, tEXt, ...) is skipped
        pos = body_end + 4

    if width is None:
        raise PngError("no IHDR chunk")
    if not idat:
        raise PngError("no IDAT chunk — nothing to decode")
    if not have_iend:
        # A truncated file is still decodable up to what arrived; record it rather
        # than discarding pixels the caller may want.
        seen.append("<missing IEND>")

    if color_type == 3 and palette is None:
        raise PngError("colour type 3 (palette) with no PLTE chunk")

    try:
        stream = zlib.decompress(b"".join(idat))
    except zlib.error as exc:
        raise PngError(f"inflate failed: {exc}") from exc

    transparent = _parse_trns(trns, color_type, palette)

    if interlace == 1:
        data = _unfilter_interlaced(stream, width, height, bit_depth, color_type)
    else:
        data = _unfilter(stream, width, height, bit_depth, color_type)

    if color_type == 3 and palette is not None:
        # An index past the palette is a corrupt file. Left unchecked it would
        # reach numpy's fancy indexing and raise IndexError deep inside rgb() —
        # or, where alpha() clips, quietly report the wrong colour. Catch it here
        # where the reason is still visible.
        idx = data.astype(np.int64)
        worst = int(idx.max()) if idx.size else 0
        if worst >= len(palette):
            raise PngError(f"palette index {worst} exceeds PLTE ({len(palette)} entries)")

    return PngImage(data=data, width=width, height=height, bit_depth=bit_depth,
                    color_type=color_type, palette=palette, transparent=transparent,
                    interlaced=(interlace == 1), chunks_seen=seen)


def decode_file(path) -> PngImage:
    """Decode a PNG from disk. Propagates OSError for a missing/unreadable file."""
    with open(path, "rb") as fh:
        return decode(fh.read())


def is_png(path) -> bool:
    """True if the file starts with the PNG signature.

    Reads eight bytes rather than trusting the extension, because a screenshot
    renamed or truncated on disk is a real possibility and the extension says
    nothing about the contents. Cheap enough to call before committing to a full
    decode.
    """
    try:
        with open(path, "rb") as fh:
            return fh.read(8) == SIGNATURE
    except OSError:
        return False


def _depth_ok(color_type: int, bit_depth: int) -> bool:
    """The spec allows only certain depth/colour-type pairs."""
    allowed = {
        0: (1, 2, 4, 8, 16),
        2: (8, 16),
        3: (1, 2, 4, 8),
        4: (8, 16),
        6: (8, 16),
    }
    return bit_depth in allowed.get(color_type, ())


def _parse_trns(trns: Optional[bytes], color_type: int,
                palette: Optional[np.ndarray]) -> Optional[Dict[str, int]]:
    if not trns:
        return None
    if color_type == 3:
        n = min(len(trns), len(palette) if palette is not None else 0)
        return {"palette_alpha": n, "bytes": trns[:n]}          # type: ignore[dict-item]
    if color_type == 0 and len(trns) >= 2:
        return {"gray": struct.unpack(">H", trns[:2])[0]}
    if color_type == 2 and len(trns) >= 6:
        r, g, b = struct.unpack(">HHH", trns[:6])
        return {"r": r, "g": g, "b": b}
    return None


def _bytes_per_pixel(bit_depth: int, color_type: int) -> int:
    """Bytes needed for one pixel, rounded up — used for the Paeth/Sub offset."""
    ch = _CHANNELS[color_type]
    bits = bit_depth * ch
    return max(1, bits // 8)


def _stride_bits(width: int, bit_depth: int, color_type: int) -> int:
    return width * bit_depth * _CHANNELS[color_type]


def _unfilter(stream: bytes, width: int, height: int,
              bit_depth: int, color_type: int) -> np.ndarray:
    bpp = _bytes_per_pixel(bit_depth, color_type)
    row_bytes = (_stride_bits(width, bit_depth, color_type) + 7) // 8
    need = (row_bytes + 1) * height
    if len(stream) < need:
        raise PngError(f"not enough decompressed data: have {len(stream)}, need {need}")

    raw = np.frombuffer(stream[:need], dtype=np.uint8).reshape(height, row_bytes + 1)
    filters = raw[:, 0].astype(np.uint8)
    scanlines = raw[:, 1:].astype(np.uint8).copy()

    out = _apply_filters(scanlines, filters, bpp)
    return _reshape_pixels(out, width, height, bit_depth, color_type)


def _apply_filters(scanlines: np.ndarray, filters: np.ndarray, bpp: int) -> np.ndarray:
    """Reverse the five PNG scanline filters, row by row.

    Each filter is inherently sequential across a row — Sub depends on the
    reconstructed byte `bpp` positions back — so this stays a Python loop. It is
    the honest cost of the format; vectorising it would require prefix sums that
    only exist for the Sub case.
    """
    height, row_bytes = scanlines.shape
    out = np.zeros((height, row_bytes), dtype=np.uint8)
    prev = np.zeros(row_bytes, dtype=np.int64)

    for y in range(height):
        f = int(filters[y])
        cur = scanlines[y].astype(np.int64)
        rec = cur.copy()

        if f == 0:                                   # None
            pass
        elif f == 1:                                 # Sub
            for x in range(bpp, row_bytes):
                rec[x] = (cur[x] + rec[x - bpp]) & 0xFF
        elif f == 2:                                 # Up
            rec = (cur + prev) & 0xFF
        elif f == 3:                                 # Average
            for x in range(row_bytes):
                a = int(rec[x - bpp]) if x >= bpp else 0
                rec[x] = (cur[x] + ((a + int(prev[x])) >> 1)) & 0xFF
        elif f == 4:                                 # Paeth
            for x in range(row_bytes):
                a = int(rec[x - bpp]) if x >= bpp else 0
                c = int(prev[x - bpp]) if x >= bpp else 0
                rec[x] = (cur[x] + _paeth(a, int(prev[x]), c)) & 0xFF
        else:
            raise PngError(f"unknown filter type {f} on row {y}")

        out[y] = rec.astype(np.uint8)
        prev = rec
    return out


def _paeth(a: int, b: int, c: int) -> int:
    p = a + b - c
    pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    return b if pb <= pc else c


def _reshape_pixels(out: np.ndarray, width: int, height: int,
                    bit_depth: int, color_type: int) -> np.ndarray:
    ch = _CHANNELS[color_type]

    if bit_depth == 16:
        if out.shape[1] != width * ch * 2:
            raise PngError("width/bit-depth mismatch in IDAT")
        # big-endian uint16, as the format specifies
        arr = out.reshape(height, width, ch, 2)
        wide = (arr[..., 0].astype(np.uint16) << 8 | arr[..., 1].astype(np.uint16))
        return wide[:, :, 0] if ch == 1 else wide

    if bit_depth == 8:
        if out.shape[1] != width * ch:
            raise PngError("width/bit-depth mismatch in IDAT")
        arr = out.reshape(height, width, ch)
        return arr[:, :, 0] if ch == 1 else arr

    # sub-byte depths: 1, 2 or 4 bits, packed most-significant-bit first
    bits = bit_depth
    per_byte = 8 // bits
    mask = (1 << bits) - 1
    if out.shape[1] != (width * bits + 7) // 8:
        raise PngError("width/bit-depth mismatch in IDAT")
    unpacked = np.unpackbits(out, axis=1).reshape(height, out.shape[1], 8)
    # unpackbits emits big-endian bit order within each byte, which matches PNG
    flat = unpacked.reshape(height, -1)
    # group every `bits` consecutive bit-planes into one sample
    grouped = np.zeros((height, out.shape[1] * per_byte), dtype=np.uint8)
    for b in range(bits):
        grouped |= flat[:, b::bits] << (bits - 1 - b)
    grouped = grouped[:, : width * ch].reshape(height, width, ch)
    grouped &= mask
    return grouped[:, :, 0] if ch == 1 else grouped


def _unfilter_interlaced(stream: bytes, width: int, height: int,
                         bit_depth: int, color_type: int) -> np.ndarray:
    """Adam7: seven reduced images, each independently filtered."""
    ch = _CHANNELS[color_type]
    bpp = _bytes_per_pixel(bit_depth, color_type)
    out = np.zeros((height, width, ch), dtype=np.uint16 if bit_depth == 16 else np.uint8)
    pos = 0

    for x_off, y_off, x_step, y_step in _ADAM7:
        pw = (width - x_off + x_step - 1) // x_step
        ph = (height - y_off + y_step - 1) // y_step
        if pw <= 0 or ph <= 0:
            continue
        row_bytes = (_stride_bits(pw, bit_depth, color_type) + 7) // 8
        need = (row_bytes + 1) * ph
        if pos + need > len(stream):
            raise PngError("interlaced IDAT truncated")
        block = np.frombuffer(stream[pos:pos + need], dtype=np.uint8).reshape(ph, row_bytes + 1)
        pos += need
        rec = _apply_filters(block[:, 1:].astype(np.uint8).copy(),
                             block[:, 0].astype(np.uint8), bpp)
        pixels = _reshape_pixels(rec, pw, ph, bit_depth, color_type)
        if pixels.ndim == 2:
            pixels = pixels[:, :, None]
        out[y_off::y_step, x_off::x_step] = pixels

    if ch == 1:
        return out[:, :, 0]
    return out


# ===================================================================== encode ====
def encode(data: np.ndarray, color_type: Optional[int] = None,
           palette: Optional[np.ndarray] = None, level: int = 6,
           interlace: bool = False, bit_depth: Optional[int] = None) -> bytes:
    """Encode a numpy array to PNG bytes.

    Accepts (h, w), (h, w, 1), (h, w, 3) or (h, w, 4). `interlace` is accepted for
    symmetry but only non-interlaced output is produced; writing Adam7 would mean
    seven filtered passes nobody downstream needs, and claiming otherwise in the
    signature would be a lie.

    Depth follows the dtype: uint16 writes 16-bit, everything else 8-bit. A
    `bit_depth` argument is only ever checked against that — asking for a depth
    this encoder cannot write is an error, not a silent reinterpretation of the
    caller's array.
    """
    if interlace:
        raise PngError("interlaced encoding is not implemented")

    arr = np.asarray(data)
    if arr.dtype.kind == "f":
        # Silently casting 0.0-1.0 floats to bytes would collapse the whole image
        # to black and hand back a file that decodes without complaint. Say so.
        raise PngError("cannot encode a float array — convert to uint8 or uint16 first "
                       "(0-255, not 0.0-1.0)")
    if arr.dtype.kind not in "ui":
        raise PngError(f"cannot encode dtype {arr.dtype}; expected an integer array")
    if arr.ndim == 2:
        arr = arr[:, :, None]
    if arr.ndim != 3:
        raise PngError(f"expected 2 or 3 dimensions, got {arr.ndim}")
    height, width, ch = arr.shape
    if width == 0 or height == 0:
        raise PngError("cannot encode an empty image")

    if color_type is None:
        # 2 channels is greyscale+alpha. It has to be inferable, because `decode`
        # returns exactly that shape for colour type 4 — without it a decoded
        # image could not be re-encoded, which is an asymmetry rather than a
        # restriction anybody asked for.
        color_type = {1: 0, 2: 4, 3: 2, 4: 6}.get(ch)
        if color_type is None:
            raise PngError(f"cannot infer colour type from {ch} channels")
    if ch != _CHANNELS.get(color_type):
        raise PngError(f"{ch} channels does not match colour type {color_type}")

    if arr.dtype == np.uint16:
        depth = 16
    else:
        arr = arr.astype(np.uint8)
        depth = 8
    if bit_depth is not None and int(bit_depth) != depth:
        raise PngError(f"bit_depth {int(bit_depth)} cannot be written from a "
                       f"{arr.dtype} array (this encoder writes {depth}-bit); "
                       f"sub-byte depths are decode-only")
    bit_depth = depth

    chunks: List[bytes] = []
    ihdr = struct.pack(">IIBBBBB", width, height, bit_depth, color_type, 0, 0, 0)
    chunks.append(_chunk(b"IHDR", ihdr))

    if color_type == 3:
        if palette is None:
            raise PngError("colour type 3 needs a palette")
        chunks.append(_chunk(b"PLTE", np.asarray(palette, dtype=np.uint8).tobytes()))

    row_bytes = width * ch * (2 if bit_depth == 16 else 1)
    flat = np.zeros((height, row_bytes + 1), dtype=np.uint8)
    if bit_depth == 16:
        # split each uint16 into big-endian byte pairs, as the format requires
        hi = (arr >> 8).astype(np.uint8)
        lo = (arr & 0xFF).astype(np.uint8)
        be = np.empty((height, width, ch, 2), dtype=np.uint8)
        be[..., 0] = hi
        be[..., 1] = lo
        src = be.reshape(height, row_bytes)
    else:
        src = arr.reshape(height, row_bytes)
    flat[:, 0] = 0                                  # filter 0 (None) on every row
    flat[:, 1:] = src

    chunks.append(_chunk(b"IDAT", zlib.compress(flat.tobytes(), level)))
    chunks.append(_chunk(b"IEND", b""))
    return SIGNATURE + b"".join(chunks)


def write_file(path, data: np.ndarray, **kw) -> str:
    """Encode and write to disk; returns the path.

    Parent directories are created, because the caller is usually writing a
    screenshot into a data directory that may not exist yet on a fresh install —
    failing there would be a confusing error for something that is trivially
    fixable at this end.
    """
    target = Path(path)
    if target.parent and str(target.parent) not in ("", "."):
        target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "wb") as fh:
        fh.write(encode(data, **kw))
    return str(target)


def _chunk(ctype: bytes, body: bytes) -> bytes:
    crc = binascii.crc32(ctype + body) & 0xFFFFFFFF
    return struct.pack(">I", len(body)) + ctype + body + struct.pack(">I", crc)


# =================================================================== convert ====
def to_gray(data: np.ndarray, color_type: int = 2,
            palette: Optional[np.ndarray] = None) -> np.ndarray:
    """Luminance in 0..1 as float32, shape (height, width).

    Uses the Rec. 601 luma coefficients rather than a plain channel mean, because
    a mean weights blue as heavily as green and the downstream face and text
    detectors were tuned against luminance that matches how the pixels were
    captured.
    """
    arr = np.asarray(data)
    if color_type == 3 and palette is not None:
        arr = np.asarray(palette, dtype=np.uint8)[arr.astype(np.int64)]
        color_type = 2

    if arr.dtype == np.uint16:
        arr = (arr >> 8).astype(np.uint8)
    if arr.ndim == 2:
        return arr.astype(np.float32) / 255.0

    if arr.shape[2] == 1:
        return arr[:, :, 0].astype(np.float32) / 255.0
    if arr.shape[2] == 2:                                  # grey + alpha
        return arr[:, :, 0].astype(np.float32) / 255.0
    rgb = arr[:, :, :3].astype(np.float32)
    return (0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]) / 255.0


def to_rgb(data: np.ndarray, color_type: int = 2,
           palette: Optional[np.ndarray] = None, bit_depth: int = 8) -> np.ndarray:
    """RGB uint8, shape (height, width, 3). Expands palettes and drops alpha."""
    arr = np.asarray(data)
    if color_type == 3 and palette is not None:
        arr = np.asarray(palette, dtype=np.uint8)[arr.astype(np.int64)]
    elif arr.dtype == np.uint16:
        arr = (arr >> 8).astype(np.uint8)
    else:
        arr = arr.astype(np.uint8)

    if arr.ndim == 2:
        return np.stack([arr, arr, arr], axis=-1)
    ch = arr.shape[2]
    if ch == 1:
        return np.concatenate([arr, arr, arr], axis=-1)
    if ch == 2:
        g = arr[:, :, 0:1]
        return np.concatenate([g, g, g], axis=-1)
    return np.ascontiguousarray(arr[:, :, :3])
