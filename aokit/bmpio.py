"""aokit.bmpio -- BMP read/write in pure numpy (no Pillow dependency).

Implemented (foundational). Mirrors the byte logic of ``src/bmp.c`` so that a
C/Python parity test (tests) is trivial. Supports uncompressed 8-bit grayscale
(palette-aware) and 24-bit BGR; bottom-up/top-down; 4-byte row padding.
See research/07 PART A.
"""
from __future__ import annotations

import numpy as np


def read_bmp_gray(path: str) -> np.ndarray:
    """Read a BMP into a float64 grayscale array (top-left origin, H x W)."""
    raw = np.fromfile(path, dtype=np.uint8)
    if raw.size < 54 or raw[0] != 0x42 or raw[1] != 0x4D:
        raise ValueError(f"{path}: not a BMP")
    off = int.from_bytes(raw[10:14].tobytes(), "little")
    w = int.from_bytes(raw[18:22].tobytes(), "little", signed=True)
    h = int.from_bytes(raw[22:26].tobytes(), "little", signed=True)
    bpp = int.from_bytes(raw[28:30].tobytes(), "little")
    comp = int.from_bytes(raw[30:34].tobytes(), "little")
    if comp != 0:
        raise ValueError(f"{path}: compressed BMP not supported")
    if bpp not in (8, 24):
        raise ValueError(f"{path}: only 8/24-bit supported (got {bpp})")

    bottom_up = h > 0
    H = abs(h)
    Bpp = bpp // 8
    stride = ((bpp * w + 31) // 32) * 4

    px = raw[off:off + stride * H].reshape(H, stride)
    px = px[:, : w * Bpp].reshape(H, w, Bpp)
    if Bpp == 1:
        img = px[..., 0].astype(np.float64)
        # 8-bit palette is assumed identity grayscale (matches our writer).
    else:
        # BGR -> luma (same weights as src/bmp.c)
        b = px[..., 0].astype(np.float64)
        g = px[..., 1].astype(np.float64)
        r = px[..., 2].astype(np.float64)
        img = (r * 299 + g * 587 + b * 114) / 1000.0

    if bottom_up:
        img = img[::-1]
    return np.ascontiguousarray(img)


def write_bmp_gray8(path: str, img) -> None:
    """Write a grayscale array as an 8-bit BMP with identity palette.

    ``img`` is H x W, values 0..255 (clipped/rounded). Bottom-up rows,
    4-byte padded, ``bfOffBits = 1078`` -- matches ``src/bmp.c``.
    """
    a = np.asarray(img)
    if a.ndim != 2:
        raise ValueError("img must be 2-D (H x W)")
    H, W = a.shape
    data = np.clip(np.rint(a), 0, 255).astype(np.uint8)

    stride = ((8 * W + 31) // 32) * 4
    pixoff = 54 + 256 * 4  # 1078
    imgsize = stride * H
    filesize = pixoff + imgsize

    fh = bytearray(14)
    fh[0:2] = b"BM"
    fh[2:6] = int(filesize).to_bytes(4, "little")
    fh[10:14] = int(pixoff).to_bytes(4, "little")

    ih = bytearray(40)
    ih[0:4] = (40).to_bytes(4, "little")
    ih[4:8] = int(W).to_bytes(4, "little")
    ih[8:12] = int(H).to_bytes(4, "little")          # positive => bottom-up
    ih[12:14] = (1).to_bytes(2, "little")            # planes
    ih[14:16] = (8).to_bytes(2, "little")            # bpp
    ih[16:20] = (0).to_bytes(4, "little")            # BI_RGB
    ih[20:24] = int(imgsize).to_bytes(4, "little")
    ih[24:28] = (2835).to_bytes(4, "little")
    ih[28:32] = (2835).to_bytes(4, "little")
    ih[32:36] = (256).to_bytes(4, "little")          # clrUsed
    ih[36:40] = (0).to_bytes(4, "little")

    # identity grayscale palette: B=G=R=i, reserved=0
    pal = np.zeros((256, 4), dtype=np.uint8)
    ramp = np.arange(256, dtype=np.uint8)
    pal[:, 0] = ramp
    pal[:, 1] = ramp
    pal[:, 2] = ramp

    # rows bottom-up, padded to stride
    rows = np.zeros((H, stride), dtype=np.uint8)
    rows[:, :W] = data[::-1]  # flip to bottom-up

    with open(path, "wb") as f:
        f.write(bytes(fh))
        f.write(bytes(ih))
        f.write(pal.tobytes())
        f.write(rows.tobytes())
