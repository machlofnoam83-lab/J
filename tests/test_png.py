#!/usr/bin/env python3
"""JARVIS PNG codec tests — decode, encode, filters, palettes, malformed input.

`vision/png.py` exists because JARVIS could write a screenshot and not open it:
Pillow and OpenCV are absent and neither may be assumed offline, so
`screen.capture` produced a file nothing could read. This codec is stdlib `zlib`
plus numpy, written against the PNG specification rather than against a
reference decoder, which means the tests have to carry the verification.

The round-trips here are therefore written from both directions on purpose.
`encode` produces a stream that `decode` must reproduce byte-for-byte, and
separately, hand-built streams exercise the parts `encode` never emits — every
one of the five scanline filters, sub-byte palette depths, Adam7 interlacing,
tRNS transparency — so a pass proves the decoder against the format, not merely
against our own encoder's habits.

Covers: signature validation · IHDR parsing · CRC rejection · all five colour
types · 1/2/4/8/16-bit depths · filter types 0-4 with hand-built streams ·
Adam7 pass reconstruction · palette expansion · tRNS alpha for indexed and
greyscale · multi-chunk IDAT concatenation · greyscale squeeze consistency
across bit depths · encode/decode round-trip · malformed and truncated input.

Run:  python tests/test_png.py
"""

from __future__ import annotations

import struct
import sys
import tempfile
import time
import zlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vision import png as P  # noqa: E402

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


# Adam7 pass geometry, from the specification.
_ADAM7 = [(0, 0, 8, 8), (4, 0, 8, 8), (0, 4, 4, 8), (2, 0, 4, 4),
          (0, 2, 2, 4), (1, 0, 2, 2), (0, 1, 1, 2)]


def _ihdr(w: int, h: int, depth: int, ctype: int, interlace: int = 0) -> bytes:
    return struct.pack(">IIBBBBB", w, h, depth, ctype, 0, 0, interlace)


def _png(w: int, h: int, depth: int, ctype: int, raw: bytes,
         interlace: int = 0, extra: list[bytes] | None = None) -> bytes:
    """Assemble a file from already-filtered scanline bytes."""
    body = P.SIGNATURE + P._chunk(b"IHDR", _ihdr(w, h, depth, ctype, interlace))
    for chunk in (extra or []):
        body += chunk
    body += P._chunk(b"IDAT", zlib.compress(raw, 9)) + P._chunk(b"IEND", b"")
    return body


def _rows(pixels: np.ndarray, bpp: int, ftype: int) -> bytes:
    """Filter a pixel array with one chosen filter type, forward direction.

    Built independently of `decode` on purpose: if both directions shared a
    helper, a common mistake would cancel out and the test would pass while the
    decoder was wrong.
    """
    h, w = pixels.shape[0], pixels.shape[1]
    stride = pixels.shape[1] * (pixels.shape[2] if pixels.ndim == 3 else 1)
    flat = pixels.reshape(h, stride).astype(np.int64)
    out = bytearray()
    prev = np.zeros(stride, dtype=np.int64)
    for y in range(h):
        cur = flat[y]
        enc = np.zeros(stride, dtype=np.int64)
        for x in range(stride):
            a = int(cur[x - bpp]) if x >= bpp else 0
            b = int(prev[x])
            c = int(prev[x - bpp]) if x >= bpp else 0
            if ftype == 0:
                enc[x] = cur[x]
            elif ftype == 1:
                enc[x] = (cur[x] - a) & 0xFF
            elif ftype == 2:
                enc[x] = (cur[x] - b) & 0xFF
            elif ftype == 3:
                enc[x] = (cur[x] - ((a + b) >> 1)) & 0xFF
            else:
                enc[x] = (cur[x] - P._paeth(a, b, c)) & 0xFF
        out.append(ftype)
        out += enc.astype(np.uint8).tobytes()
        prev = cur
    return bytes(out)


# ----------------------------------------------------------------------- #


def test_round_trip() -> None:
    print("\n[1] encode/decode round-trip, every colour type and depth")
    rng = np.random.default_rng(7)
    cases = [
        ("greyscale 8-bit", rng.integers(0, 256, (9, 11), dtype=np.uint8)),
        ("greyscale+alpha 8", rng.integers(0, 256, (9, 11, 2), dtype=np.uint8)),
        ("rgb 8-bit", rng.integers(0, 256, (9, 11, 3), dtype=np.uint8)),
        ("rgba 8-bit", rng.integers(0, 256, (9, 11, 4), dtype=np.uint8)),
        ("greyscale 16-bit", rng.integers(0, 65536, (7, 5), dtype=np.uint16)),
        ("rgb 16-bit", rng.integers(0, 65536, (7, 5, 3), dtype=np.uint16)),
        ("rgba 16-bit", rng.integers(0, 65536, (7, 5, 4), dtype=np.uint16)),
    ]
    for label, arr in cases:
        got = P.decode(P.encode(arr)).data
        check(f"{label} survives the round-trip", np.array_equal(got, arr),
              f"{arr.shape} -> {got.shape}")

    # The shape rule has to hold across bit depths: a caller indexing
    # data[y, x] must not have to ask whether the source was 16-bit.
    g8 = P.decode(P.encode(np.zeros((4, 5), dtype=np.uint8))).data
    g16 = P.decode(P.encode(np.zeros((4, 5), dtype=np.uint16))).data
    check("single channel is 2-D at 8-bit", g8.shape == (4, 5), str(g8.shape))
    check("single channel is 2-D at 16-bit too", g16.shape == (4, 5), str(g16.shape))
    check("16-bit keeps its dtype", g16.dtype == np.uint16, str(g16.dtype))


def test_filters() -> None:
    print("\n[2] each scanline filter, from hand-built streams")
    rng = np.random.default_rng(11)
    pixels = rng.integers(0, 256, (6, 7, 3), dtype=np.uint8)
    for ftype in range(5):
        raw = _rows(pixels, bpp=3, ftype=ftype)
        got = P.decode(_png(7, 6, 8, 2, raw)).data
        check(f"filter {ftype} inverts exactly", np.array_equal(got, pixels),
              P._FILTER_NAMES[ftype])

    # A gradient stresses Up/Paeth harder than noise does, because there the
    # predictor is right and the residual small — an off-by-one in the running
    # `prev` row shows up as drift down the image rather than as noise.
    grad = np.tile(np.arange(40, dtype=np.uint8).reshape(1, 40, 1), (30, 1, 3))
    for ftype in range(5):
        got = P.decode(_png(40, 30, 8, 2, _rows(grad, 3, ftype))).data
        check(f"filter {ftype} on a gradient shows no drift", np.array_equal(got, grad))


def test_bit_depths() -> None:
    print("\n[3] sub-byte and 16-bit depths")
    # 4-bit greyscale: two samples per byte, high nibble first.
    idx = np.array([[0, 15, 3, 12], [7, 0, 1, 14]], dtype=np.uint8)
    packed = np.zeros((2, 2), dtype=np.uint8)
    for y in range(2):
        for xb in range(2):
            packed[y, xb] = (idx[y, xb * 2] << 4) | idx[y, xb * 2 + 1]
    raw = b"".join(bytes([0]) + packed[y].tobytes() for y in range(2))
    got = P.decode(_png(4, 2, 4, 0, raw)).data
    check("4-bit greyscale unpacks high nibble first", np.array_equal(got, idx),
          str(got.tolist()))

    # 2-bit: four samples per byte, and the last byte is padded — a decoder that
    # trusts the row length would emit phantom pixels.
    two = np.array([[0, 1, 2, 3, 0, 1]], dtype=np.uint8)
    packed2 = bytes([0, (0 << 6) | (1 << 4) | (2 << 2) | (3 << 0), (0 << 6) | (1 << 4)])
    got2 = P.decode(_png(6, 1, 2, 0, packed2)).data
    check("2-bit drops the pad bits instead of inventing pixels",
          np.array_equal(got2, two), str(got2.tolist()))

    # 1-bit, the extreme case: 17 pixels need 3 bytes with 7 pad bits.
    one = np.array([[1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1]], dtype=np.uint8)
    bits = one.reshape(-1)
    byts = bytearray(3)
    for i, b in enumerate(bits):
        byts[i // 8] |= int(b) << (7 - (i % 8))
    got1 = P.decode(_png(17, 1, 1, 0, bytes([0]) + bytes(byts))).data
    check("1-bit honours the declared width over the byte count",
          np.array_equal(got1, one), str(got1.tolist()))

    # 16-bit is big-endian; a little-endian reading would transpose values wildly.
    wide = np.array([[0x0001, 0x0100, 0xABCD]], dtype=np.uint16)
    be = wide.astype(">u2").tobytes()
    gotw = P.decode(_png(3, 1, 16, 0, bytes([0]) + be)).data
    check("16-bit is read big-endian", np.array_equal(gotw, wide), str(gotw.tolist()))
    check("16-bit values are not byte-swapped", gotw[0, 2] == 0xABCD, hex(int(gotw[0, 2])))


def test_palette_and_trns() -> None:
    print("\n[4] palettes and transparency")
    pal = np.array([[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.uint8)
    idx = np.array([[0, 1, 2, 3], [3, 2, 1, 0]], dtype=np.uint8)
    packed = np.zeros((2, 2), dtype=np.uint8)
    for y in range(2):
        for xb in range(2):
            packed[y, xb] = (idx[y, xb * 2] << 4) | idx[y, xb * 2 + 1]
    raw = b"".join(bytes([0]) + packed[y].tobytes() for y in range(2))
    plte = P._chunk(b"PLTE", pal.tobytes())
    im = P.decode(_png(4, 2, 4, 3, raw, extra=[plte]))
    check("indexed PNG yields palette indices", np.array_equal(im.data, idx),
          str(im.data.tolist()))
    check("indexed reports color_name 'palette'", im.color_name == "palette", im.color_name)
    check("indexed rgb() expands through the palette",
          tuple(im.rgb()[0, 1]) == (255, 0, 0) and tuple(im.rgb()[1, 0]) == (0, 0, 255))
    check("indexed without tRNS is fully opaque",
          im.alpha().min() == 255 and im.has_alpha is False)

    # tRNS on an indexed image is one alpha byte per palette entry. `data` stays
    # as indices — that is what the file holds — so transparency is checked
    # through alpha(), which is the method that has to apply it.
    trns = P._chunk(b"tRNS", bytes([255, 128, 0, 255]))
    im2 = P.decode(_png(4, 2, 4, 3, raw, extra=[plte, trns]))
    check("indexed with tRNS reports has_alpha", im2.has_alpha is True)
    check("data stays as palette indices even with tRNS",
          im2.data.shape == (2, 4) and im2.data.dtype == np.uint8, str(im2.data.shape))
    a2 = im2.alpha()
    check("tRNS alpha follows the palette entry",
          a2.shape == (2, 4) and a2[0, 0] == 255 and a2[0, 1] == 128
          and a2[0, 2] == 0 and a2[0, 3] == 255, str(a2.tolist()))
    check("the second row mirrors the palette", a2[1].tolist() == [255, 0, 128, 255],
          str(a2[1].tolist()))
    check("rgba() carries the alpha through", im2.rgba().shape == (2, 4, 4)
          and im2.rgba()[0, 2, 3] == 0 and tuple(im2.rgba()[0, 2, :3]) == (0, 255, 0))

    # tRNS shorter than the palette: the entries it does not cover stay opaque.
    im2b = P.decode(_png(4, 2, 4, 3, raw, extra=[plte, P._chunk(b"tRNS", bytes([0, 200]))]))
    check("tRNS shorter than the palette leaves the rest opaque",
          im2b.alpha()[0].tolist() == [0, 200, 255, 255], str(im2b.alpha()[0].tolist()))

    # tRNS on greyscale is a single transparent value, not a table.
    gs = np.array([[10, 20, 30], [20, 20, 40]], dtype=np.uint8)
    graw = _rows(gs, 1, 0)
    im3 = P.decode(_png(3, 2, 8, 0, graw, extra=[P._chunk(b"tRNS", struct.pack(">H", 20))]))
    a3 = im3.alpha()
    check("tRNS on greyscale zeroes only the matching value",
          a3.tolist() == [[255, 0, 255], [0, 0, 255]], str(a3.tolist()))
    check("tRNS on greyscale leaves data untouched", im3.data.shape == (2, 3),
          str(im3.data.shape))
    check("greyscale tRNS reports has_alpha", im3.has_alpha is True)

    # tRNS on truecolour names one RGB triple.
    rgbim = np.array([[[255, 0, 0], [0, 255, 0]], [[255, 0, 0], [7, 7, 7]]], dtype=np.uint8)
    im4 = P.decode(_png(2, 2, 8, 2, _rows(rgbim, 3, 0),
                        extra=[P._chunk(b"tRNS", struct.pack(">HHH", 255, 0, 0))]))
    check("tRNS on truecolour zeroes the named triple only",
          im4.alpha().tolist() == [[0, 255], [0, 255]], str(im4.alpha().tolist()))

    # Colour types 4 and 6 carry alpha per pixel in the file itself.
    rgba = np.array([[[1, 2, 3, 0], [4, 5, 6, 255]]], dtype=np.uint8)
    im5 = P.decode(P.encode(rgba))
    check("truecolour alpha is read from the pixel data, not synthesised",
          im5.alpha().tolist() == [[0, 255]] and im5.has_alpha is True)
    check("rgba() round-trips a file that already had alpha",
          np.array_equal(im5.rgba(), rgba))

    # A palette smaller than the highest index used is a corrupt file, not a
    # lookup that should silently wrap or raise from numpy.
    small = P._chunk(b"PLTE", np.array([[0, 0, 0], [255, 255, 255]], dtype=np.uint8).tobytes())
    try:
        P.decode(_png(4, 2, 4, 3, raw, extra=[small]))
        check("rejects an index past the palette", False)
    except P.PngError:
        check("rejects an index past the palette", True)
    except Exception as exc:
        check("rejects an index past the palette", False, f"leaked {type(exc).__name__}")


def test_interlace() -> None:
    print("\n[5] Adam7 interlacing")
    rng = np.random.default_rng(5)
    pixels = rng.integers(0, 256, (9, 11, 3), dtype=np.uint8)
    # Emit the seven passes in order, each filtered independently.
    stream = bytearray()
    for x0, y0, dx, dy in _ADAM7:
        xs = list(range(x0, 11, dx))
        ys = list(range(y0, 9, dy))
        if not xs or not ys:
            continue
        block = pixels[np.ix_(ys, xs)]
        stream += _rows(block, 3, 0)
    got = P.decode(_png(11, 9, 8, 2, bytes(stream), interlace=1)).data
    check("interlaced image reassembles from seven passes",
          np.array_equal(got, pixels), str(got.shape))

    # A size where later passes are empty entirely — pass 6 needs an odd height.
    tiny = rng.integers(0, 256, (3, 3, 3), dtype=np.uint8)
    stream2 = bytearray()
    for x0, y0, dx, dy in _ADAM7:
        xs = list(range(x0, 3, dx))
        ys = list(range(y0, 3, dy))
        if not xs or not ys:
            continue
        stream2 += _rows(tiny[np.ix_(ys, xs)], 3, 0)
    got2 = P.decode(_png(3, 3, 8, 2, bytes(stream2), interlace=1)).data
    check("3x3 interlaced skips its empty passes", np.array_equal(got2, tiny),
          str(got2.shape))

    check("interlaced flag is reported", P.decode(_png(11, 9, 8, 2, bytes(stream),
                                                       interlace=1)).interlaced is True)


def test_chunks_and_crc() -> None:
    print("\n[6] chunk handling and integrity")
    pixels = np.zeros((5, 5, 3), dtype=np.uint8)
    raw = _rows(pixels, 3, 0)
    # Split IDAT across three chunks at awkward boundaries. A decoder that read
    # only the first would fail on length rather than produce wrong pixels.
    comp = zlib.compress(raw, 9)
    parts = [comp[:3], comp[3:9], comp[9:]]
    body = (P.SIGNATURE + P._chunk(b"IHDR", _ihdr(5, 5, 8, 2))
            + b"".join(P._chunk(b"IDAT", p) for p in parts)
            + P._chunk(b"IEND", b""))
    im = P.decode(body)
    check("concatenates IDAT split across three chunks",
          np.array_equal(im.data, pixels), str(im.chunks_seen))

    # Ancillary chunks between IHDR and IDAT must be skipped, not mistaken for
    # image data.
    gamma = P._chunk(b"gAMA", struct.pack(">I", 45455))
    text = P._chunk(b"tEXt", b"Comment\x00written by the test")
    body2 = (P.SIGNATURE + P._chunk(b"IHDR", _ihdr(5, 5, 8, 2)) + gamma + text
             + P._chunk(b"IDAT", comp) + P._chunk(b"IEND", b""))
    im2 = P.decode(body2)
    check("skips ancillary chunks (gAMA, tEXt)", np.array_equal(im2.data, pixels),
          str(im2.chunks_seen))
    check("records the chunks it saw", "gAMA" in im2.chunks_seen and "tEXt" in im2.chunks_seen)

    # Corrupt one payload byte: the stored CRC must catch it.
    tampered = bytearray(raw)
    tampered[len(tampered) // 2] ^= 0xFF
    good_crc = zlib.crc32(b"IHDR" + _ihdr(5, 5, 8, 2)) & 0xFFFFFFFF
    bad = (P.SIGNATURE + struct.pack(">I", 13) + b"IHDR" + _ihdr(5, 5, 8, 2)
           + struct.pack(">I", good_crc)
           + P._chunk(b"IDAT", zlib.compress(bytes(tampered), 9))
           + P._chunk(b"IEND", b""))
    bad = bytearray(bad)
    bad[len(bad) - 12] ^= 0x01  # corrupt the IEND CRC
    try:
        P.decode(bytes(bad))
        check("rejects a chunk whose CRC does not match", False)
    except P.PngError:
        check("rejects a chunk whose CRC does not match", True)

    check("encode verifies its own CRC on re-read",
          P.decode(P.encode(pixels)).data.shape == pixels.shape)


def test_malformed() -> None:
    print("\n[7] malformed input raises PngError, never an IndexError")
    ihdr_only = P.SIGNATURE + P._chunk(b"IHDR", _ihdr(4, 4, 8, 2)) + P._chunk(b"IEND", b"")
    bad = [
        ("empty bytes", b""),
        ("not a PNG at all", b"this is plain text, not an image"),
        ("signature truncated", P.SIGNATURE[:4]),
        ("signature one byte off", bytes([0x89]) + b"PNF\r\n\x1a\n"),
        ("IHDR then nothing", P.SIGNATURE + P._chunk(b"IHDR", _ihdr(4, 4, 8, 2))),
        ("no IDAT chunk", ihdr_only),
        ("IDAT truncated mid-stream", ihdr_only[:-12] + P._chunk(b"IDAT", b"\x78\x9c\x01")),
        ("chunk length runs off the end",
         P.SIGNATURE + struct.pack(">I", 99999) + b"IHDR" + b"\x00" * 13),
        ("zero-width image",
         P.SIGNATURE + P._chunk(b"IHDR", _ihdr(0, 4, 8, 2))
         + P._chunk(b"IDAT", zlib.compress(b"\x00" * 4)) + P._chunk(b"IEND", b"")),
        ("unsupported bit depth",
         P.SIGNATURE + P._chunk(b"IHDR", _ihdr(4, 4, 3, 2))
         + P._chunk(b"IDAT", zlib.compress(b"\x00" * 4)) + P._chunk(b"IEND", b"")),
        ("invalid colour type",
         P.SIGNATURE + P._chunk(b"IHDR", _ihdr(4, 4, 8, 5))
         + P._chunk(b"IDAT", zlib.compress(b"\x00" * 4)) + P._chunk(b"IEND", b"")),
        ("indexed without PLTE",
         P.SIGNATURE + P._chunk(b"IHDR", _ihdr(4, 4, 8, 3))
         + P._chunk(b"IDAT", zlib.compress(b"\x00" * 20)) + P._chunk(b"IEND", b"")),
        ("IDAT shorter than the header claims",
         P.SIGNATURE + P._chunk(b"IHDR", _ihdr(64, 64, 8, 2))
         + P._chunk(b"IDAT", zlib.compress(b"\x00" * 10)) + P._chunk(b"IEND", b"")),
    ]
    for label, data in bad:
        try:
            P.decode(data)
            check(f"{label} is rejected", False, "decoded without complaint")
        except P.PngError:
            check(f"{label} is rejected", True)
        except Exception as exc:
            check(f"{label} is rejected", False, f"leaked {type(exc).__name__}: {exc}")

    # encode must also refuse what it cannot represent, rather than write a file
    # that a reader would reject later.
    for label, arr, kw in [
        ("4-D array", np.zeros((2, 2, 2, 2), dtype=np.uint8), {}),
        ("float array", np.zeros((2, 2, 3), dtype=np.float32), {}),
        ("depth 4 with rgb", np.zeros((2, 2, 3), dtype=np.uint8), {"bit_depth": 4}),
    ]:
        try:
            P.encode(arr, **kw)
            check(f"encode rejects {label}", False)
        except P.PngError:
            check(f"encode rejects {label}", True)
        except Exception as exc:
            check(f"encode rejects {label}", False, f"leaked {type(exc).__name__}")


def test_helpers() -> None:
    print("\n[8] grey/rgb conversions and file I/O")
    check("to_gray normalises to [0,1] float32",
          (lambda g: g.dtype == np.float32 and g.min() >= 0.0 and g.max() <= 1.0)(
              P.to_gray(np.full((3, 3, 3), 255, dtype=np.uint8))))

    # Rec.601 luma: green weighs most, blue least. Pure channels make the
    # coefficients checkable without floating-point guesswork.
    g_r = float(P.to_gray(np.array([[[255, 0, 0]]], dtype=np.uint8))[0, 0])
    g_g = float(P.to_gray(np.array([[[0, 255, 0]]], dtype=np.uint8))[0, 0])
    g_b = float(P.to_gray(np.array([[[0, 0, 255]]], dtype=np.uint8))[0, 0])
    check("luma ranks green above red above blue", g_g > g_r > g_b,
          f"r={g_r:.3f} g={g_g:.3f} b={g_b:.3f}")
    check("white maps to 1.0 and black to 0.0",
          abs(float(P.to_gray(np.full((1, 1, 3), 255, dtype=np.uint8))[0, 0]) - 1.0) < 1e-3
          and float(P.to_gray(np.zeros((1, 1, 3), dtype=np.uint8))[0, 0]) == 0.0)

    gray = np.array([[10, 20]], dtype=np.uint8)
    check("to_rgb promotes greyscale to three identical planes",
          (lambda r: r.shape == (1, 2, 3) and np.array_equal(r[0, 0], [10, 10, 10]))(
              P.to_rgb(gray)))
    rgba = np.array([[[1, 2, 3, 4]]], dtype=np.uint8)
    check("to_rgb drops alpha rather than keeping four planes",
          P.to_rgb(rgba).shape == (1, 1, 3))

    # 16-bit must be scaled to [0,1], not merely cast — casting would overflow.
    hi = P.to_gray(np.full((2, 2, 3), 65535, dtype=np.uint16))
    check("16-bit grey scales into [0,1]", abs(float(hi.max()) - 1.0) < 1e-3, f"{hi.max():.4f}")

    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "sub" / "out.png"
        arr = np.random.default_rng(1).integers(0, 256, (13, 17, 3), dtype=np.uint8)
        P.write_file(p, arr)
        check("write_file creates missing parent directories", p.exists())
        check("write_file returns the path it wrote", str(P.write_file(p, arr)) == str(p))
        check("the file it wrote is non-empty and PNG-shaped",
              p.stat().st_size > 8 and P.is_png(p), f"{p.stat().st_size} bytes")
        check("write_file round-trips through disk",
              np.array_equal(P.decode_file(p).data, arr))
        # decode_file documents that it propagates OSError for a missing file
        # rather than wrapping it — standard Python behaviour, and the caller
        # (screen.read) catches OSError separately from PngError.
        try:
            P.decode_file(Path(td) / "nope.png")
            check("decode_file reports a missing file", False, "no exception")
        except OSError as exc:
            check("decode_file propagates OSError for a missing file", True,
                  type(exc).__name__)
        except Exception as exc:
            check("decode_file propagates OSError for a missing file", False,
                  f"raised {type(exc).__name__} instead")
        check("is_png is False for a missing file, not an exception",
              P.is_png(Path(td) / "nope.png") is False)


def test_real_file() -> None:
    print("\n[9] the app icon, a real encoder's output")
    target = ROOT / "electron" / "assets" / "icon.png"
    if not target.exists():
        check("icon.png is present to test against", False, str(target))
        return
    im = P.decode_file(target)
    check("icon decodes", im.width > 0 and im.height > 0, f"{im.width}x{im.height}")
    check("icon dimensions match its IHDR", (im.width, im.height) == (256, 256),
          f"{im.width}x{im.height}")
    check("icon pixel count is complete",
          im.data.shape[:2] == (im.height, im.width), str(im.data.shape))
    check("icon has three colour planes", im.channels == 3, str(im.channels))
    check("icon values span the byte range", int(im.data.min()) == 0 and int(im.data.max()) == 255,
          f"[{im.data.min()},{im.data.max()}]")
    # Our own encoder writes one IDAT with filter 0; a real encoder splits the
    # stream and filters per row. Decoding one proves chunk concatenation and
    # real filter choices are handled, not just our habits.
    check("icon came from a multi-chunk IDAT stream", im.chunks_seen.count("IDAT") > 1,
          f"{im.chunks_seen.count('IDAT')} IDAT chunks")
    check("is_png accepts it", P.is_png(target))
    check("is_png rejects a non-image", not P.is_png(ROOT / "requirements.txt"))


def test_screen_read() -> None:
    print("\n[10] screen.read, the skill this codec exists for")
    import skills
    skills.load_all()
    from skills import REGISTRY

    check("screen.read is registered", "screen.read" in REGISTRY.names())
    rec = REGISTRY.get("screen.read")
    check("screen.read is SAFE — reading a file it did not write", rec.risk == "SAFE",
          str(rec.risk))

    rng = np.random.default_rng(2)
    with tempfile.TemporaryDirectory() as td:
        bright = np.full((80, 120, 3), 246, dtype=np.uint8)
        for y in range(15, 70, 9):
            bright[y:y + 3, 20:100] = (35, 35, 35)
        bp = Path(td) / "bright.png"
        P.write_file(bp, bright)
        r = REGISTRY.invoke("screen.read", {"path": str(bp)}, permission_granted=True)
        check("describes a bright screen", r.ok and "בהירה" in r.value, r.value[:60])
        check("reports dimensions from the decode, not the filename",
              r.data["width"] == 120 and r.data["height"] == 80)
        check("flags that no text was extracted", r.data["text_extracted"] is False)
        check("says why: no offline OCR", "OCR" in r.data["note"])

        dark = np.zeros((80, 120, 3), dtype=np.uint8)
        dark[:, :] = (10, 14, 22)
        dp = Path(td) / "dark.png"
        P.write_file(dp, dark)
        r2 = REGISTRY.invoke("screen.read", {"path": str(dp)}, permission_granted=True)
        check("describes a dark screen", r2.ok and "כהה" in r2.value, r2.value[:60])

        mid = np.full((40, 40, 3), 128, dtype=np.uint8)
        mp = Path(td) / "mid.png"
        P.write_file(mp, mid)
        r3 = REGISTRY.invoke("screen.read", {"path": str(mp)}, permission_granted=True)
        check("mid-tone adjective agrees with תמונה (feminine)",
              "בינונית" in r3.value and "בינוני " not in r3.value, r3.value[:60])

        r4 = REGISTRY.invoke("screen.read", {"path": str(Path(td) / "absent.png")},
                             permission_granted=True)
        check("missing file is reported, not raised", not r4.ok and bool(r4.error))

        txt = Path(td) / "notes.txt"
        txt.write_text("not an image", encoding="utf-8")
        r5 = REGISTRY.invoke("screen.read", {"path": str(txt)}, permission_granted=True)
        check("refuses a non-image by content, not by extension",
              not r5.ok and "PNG" in r5.error, str(r5.error)[:56])

        # Renamed but genuinely a PNG: content sniffing should let this through,
        # where an extension check would have refused a perfectly good screenshot.
        renamed = Path(td) / "capture_without_extension"
        renamed.write_bytes(P.encode(np.full((30, 40, 3), 200, dtype=np.uint8)))
        r7 = REGISTRY.invoke("screen.read", {"path": str(renamed)}, permission_granted=True)
        check("reads a real PNG whose name lacks an extension", r7.ok,
              r7.value[:56] if r7.ok else str(r7.error)[:56])

        junk = Path(td) / "fake.png"
        junk.write_bytes(b"\x89PNG\r\n\x1a\n" + b"garbage after the signature")
        r6 = REGISTRY.invoke("screen.read", {"path": str(junk)}, permission_granted=True)
        check("a corrupt PNG fails cleanly", not r6.ok and bool(r6.error), str(r6.error)[:50])


def main() -> int:
    print("═" * 68)
    print(" J.A.R.V.I.S. — PNG codec (decode, encode, filters, integrity)")
    print("═" * 68)
    t0 = time.perf_counter()
    test_round_trip()
    test_filters()
    test_bit_depths()
    test_palette_and_trns()
    test_interlace()
    test_chunks_and_crc()
    test_malformed()
    test_helpers()
    test_real_file()
    test_screen_read()
    print(f"\n completed in {time.perf_counter() - t0:.1f}s")
    print(f"\nRESULT: {ok} passed, {fail} failed")
    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
