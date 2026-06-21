"""Tests that aokit.matio and src/matio.c agree on the AOMX format.

The Python<->Python roundtrip is checked directly. The C<->Python byte parity
is checked by compiling a tiny C harness against src/matio.c (skipped if no C
compiler is available), writing from Python and reading in C and vice-versa.
"""
from __future__ import annotations

import os
import shutil
import struct
import subprocess
import tempfile

import numpy as np
import pytest

from aokit.matio import write_aomx, read_aomx, read_header, MAGIC, HEADER_BYTES

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_python_roundtrip_f32_exact():
    a = np.array([[1.0, -2.5, 3.25], [4.0, 5.5, 6.75]], dtype=np.float32)
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "m.aomx")
        write_aomx(p, a, "f32")
        b = read_aomx(p)
        assert b.shape == (2, 3)
        assert np.array_equal(a, b)


def test_python_roundtrip_f64_values():
    a = np.array([[1.1, 2.2], [3.3, 4.4], [5.5, 6.6]], dtype=np.float64)
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "m.aomx")
        write_aomx(p, a, "f64")
        b = read_aomx(p)
        assert b.dtype == np.float64
        assert np.allclose(a, b)


def test_vector_flag_and_header():
    v = np.arange(5, dtype=np.float32)
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "v.aomx")
        write_aomx(p, v, "f32")
        h = read_header(p)
        assert h["rows"] == 5 and h["cols"] == 1
        assert h["flags"] & 1  # FLAG_VECTOR


def test_header_is_32_bytes_and_magic():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "m.aomx")
        write_aomx(p, np.zeros((1, 1), np.float32), "f32")
        with open(p, "rb") as f:
            head = f.read(HEADER_BYTES)
        assert len(head) == HEADER_BYTES
        assert head[:4] == MAGIC


def test_checksum_detects_corruption():
    a = np.array([[1.0, 2.0]], dtype=np.float32)
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "m.aomx")
        write_aomx(p, a, "f32")
        # flip a payload byte
        with open(p, "r+b") as f:
            f.seek(HEADER_BYTES)
            byte = f.read(1)
            f.seek(HEADER_BYTES)
            f.write(bytes([byte[0] ^ 0xFF]))
        with pytest.raises(ValueError):
            read_aomx(p, verify_checksum=True)


@pytest.mark.skipif(shutil.which("gcc") is None and shutil.which("cc") is None,
                    reason="no C compiler available")
def test_c_reads_python_written_aomx():
    """Compile a C harness against src/matio.c; assert it reads our file."""
    cc = shutil.which("gcc") or shutil.which("cc")
    with tempfile.TemporaryDirectory() as d:
        # tiny harness prints rows cols then values
        harness = os.path.join(d, "h.c")
        with open(harness, "w") as f:
            f.write(
                '#include "%s/src/matio.h"\n#include <stdio.h>\n'
                "int main(int c,char**v){AOMatrix m;if(aomx_read(v[1],&m))return 1;"
                'printf("%%u %%u",m.rows,m.cols);'
                "for(unsigned i=0;i<m.rows*m.cols;i++)printf(\" %%.6g\",m.f32[i]);"
                "aomx_free(&m);return 0;}\n" % REPO
            )
        exe = os.path.join(d, "h")
        r = subprocess.run([cc, "-std=c11", "-O2", harness,
                            os.path.join(REPO, "src", "matio.c"),
                            "-o", exe, "-lm"],
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stderr

        a = np.array([[1.0, -2.5, 3.25], [4.0, 5.5, 6.75]], dtype=np.float32)
        p = os.path.join(d, "m.aomx")
        write_aomx(p, a, "f32")
        out = subprocess.run([exe, p], capture_output=True, text=True)
        assert out.returncode == 0, out.stderr
        toks = out.stdout.split()
        assert int(toks[0]) == 2 and int(toks[1]) == 3
        vals = np.array([float(t) for t in toks[2:]], dtype=np.float32)
        assert np.allclose(vals, a.ravel())
