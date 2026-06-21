"""aokit.matio -- AOMX self-describing binary matrix format (read/write).

Implemented (not stubbed): this is the Python side of the C<->Python contract
and MUST byte-match ``src/matio.c``. See ARCHITECTURE.md S4.2.

Layout (little-endian, 32-byte header then row-major payload)::

    off size type    field
      0    4 char[4] magic   = b'AOMX'
      4    4 uint32  version = 1
      8    4 uint32  rows
     12    4 uint32  cols    (vector => cols = 1)
     16    4 uint32  dtype   0 = float32, 1 = float64
     20    4 uint32  layout  0 = row-major
     24    4 uint32  flags   bit0: 1 if semantically a vector
     28    4 uint32  checksum additive sum of payload bytes mod 2**32 (0 = skip)
     32  R*C*sz data   element(i, j) at i*cols + j
"""
from __future__ import annotations

import struct
import numpy as np

MAGIC = b"AOMX"
VERSION = 1
HEADER_BYTES = 32

DTYPE_F32 = 0
DTYPE_F64 = 1
LAYOUT_ROWMAJOR = 0
FLAG_VECTOR = 1

_DT_CODE = {"f32": DTYPE_F32, "f64": DTYPE_F64}
_NP_DTYPE = {DTYPE_F32: np.float32, DTYPE_F64: np.float64}


def _checksum(payload: bytes) -> int:
    """Additive sum of payload bytes mod 2**32 (matches src/matio.c)."""
    # np.uint64 sum then mask is exact and fast.
    return int(np.frombuffer(payload, dtype=np.uint8).sum(dtype=np.uint64) & 0xFFFFFFFF)


def write_aomx(path: str, array, dtype: str = "f32") -> None:
    """Write a 2-D (or 1-D) array to an AOMX file.

    Parameters
    ----------
    path : str
        Output file path.
    array : array_like
        1-D vector (written as cols=1) or 2-D matrix; stored row-major (C order).
    dtype : {"f32", "f64"}
        On-disk element type. Use "f32" for matrices fed to the real-time core.
    """
    if dtype not in _DT_CODE:
        raise ValueError(f"dtype must be 'f32' or 'f64', got {dtype!r}")
    a = np.asarray(array)
    if a.ndim == 1:
        rows, cols = a.shape[0], 1
    elif a.ndim == 2:
        rows, cols = a.shape
    else:
        raise ValueError(f"array must be 1-D or 2-D, got ndim={a.ndim}")

    np_dt = _NP_DTYPE[_DT_CODE[dtype]]
    payload = np.ascontiguousarray(a, dtype=np_dt).tobytes(order="C")

    flags = FLAG_VECTOR if cols == 1 else 0
    chk = _checksum(payload)
    header = MAGIC + struct.pack(
        "<7I", VERSION, rows, cols, _DT_CODE[dtype], LAYOUT_ROWMAJOR, flags, chk
    )
    assert len(header) == HEADER_BYTES, "AOMX header must be 32 bytes"
    with open(path, "wb") as f:
        f.write(header)
        f.write(payload)


def read_aomx(path: str, verify_checksum: bool = True) -> np.ndarray:
    """Read an AOMX file, returning a (rows, cols) numpy array.

    The returned dtype matches the on-disk dtype (float32 or float64). A 1-D
    semantics is *not* auto-squeezed; callers can ``.ravel()`` if cols == 1.
    """
    with open(path, "rb") as f:
        header = f.read(HEADER_BYTES)
        if len(header) != HEADER_BYTES or header[:4] != MAGIC:
            raise ValueError(f"{path}: not an AOMX file (bad magic)")
        (version, rows, cols, dtype, layout, _flags, chk) = struct.unpack(
            "<7I", header[4:]
        )
        if version != VERSION:
            raise ValueError(f"{path}: unsupported AOMX version {version}")
        if layout != LAYOUT_ROWMAJOR:
            raise ValueError(f"{path}: unsupported layout {layout}")
        if dtype not in _NP_DTYPE:
            raise ValueError(f"{path}: unsupported dtype {dtype}")
        np_dt = _NP_DTYPE[dtype]
        count = rows * cols
        payload = f.read(count * np.dtype(np_dt).itemsize)

    if verify_checksum and chk != 0 and _checksum(payload) != chk:
        raise ValueError(f"{path}: AOMX checksum mismatch")

    arr = np.frombuffer(payload, dtype=np_dt, count=count).reshape(rows, cols)
    return np.array(arr, copy=True)  # own the buffer (writeable)


def read_header(path: str) -> dict:
    """Read just the AOMX header, returning a dict of fields."""
    with open(path, "rb") as f:
        header = f.read(HEADER_BYTES)
    if len(header) != HEADER_BYTES or header[:4] != MAGIC:
        raise ValueError(f"{path}: not an AOMX file (bad magic)")
    (version, rows, cols, dtype, layout, flags, chk) = struct.unpack("<7I", header[4:])
    return dict(version=version, rows=rows, cols=cols, dtype=dtype,
                layout=layout, flags=flags, checksum=chk)
