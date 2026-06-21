/* matio.c -- AOMX binary matrix reader/writer (implemented).
 *
 * Implemented (not stubbed) because the byte layout is the C<->Python
 * contract; tests/test_matio_roundtrip.py checks parity with aokit/matio.py.
 * Explicit little-endian byte assembly keeps it portable.
 */
#include "matio.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint32_t rd_u32le(const uint8_t *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}
static void wr_u32le(uint8_t *p, uint32_t v) {
    p[0] = (uint8_t)(v & 0xFF);
    p[1] = (uint8_t)((v >> 8) & 0xFF);
    p[2] = (uint8_t)((v >> 16) & 0xFF);
    p[3] = (uint8_t)((v >> 24) & 0xFF);
}

static uint32_t payload_checksum(const uint8_t *data, size_t nbytes) {
    uint32_t acc = 0;
    for (size_t i = 0; i < nbytes; ++i) acc += data[i];
    return acc; /* mod 2^32 by overflow */
}

int aomx_read_header(const char *path, uint32_t *rows, uint32_t *cols,
                     uint32_t *dtype) {
    FILE *f = fopen(path, "rb");
    if (!f) return 1;
    uint8_t h[AOMX_HEADER_BYTES];
    if (fread(h, 1, AOMX_HEADER_BYTES, f) != AOMX_HEADER_BYTES) { fclose(f); return 2; }
    fclose(f);
    if (h[0] != AOMX_MAGIC0 || h[1] != AOMX_MAGIC1 ||
        h[2] != AOMX_MAGIC2 || h[3] != AOMX_MAGIC3) return 3;
    if (rd_u32le(h + 4) != AOMX_VERSION) return 4;
    if (rows)  *rows  = rd_u32le(h + 8);
    if (cols)  *cols  = rd_u32le(h + 12);
    if (dtype) *dtype = rd_u32le(h + 16);
    return 0;
}

int aomx_read(const char *path, AOMatrix *out) {
    if (!out) return 1;
    memset(out, 0, sizeof(*out));
    FILE *f = fopen(path, "rb");
    if (!f) return 1;
    uint8_t h[AOMX_HEADER_BYTES];
    if (fread(h, 1, AOMX_HEADER_BYTES, f) != AOMX_HEADER_BYTES) { fclose(f); return 2; }
    if (h[0] != AOMX_MAGIC0 || h[1] != AOMX_MAGIC1 ||
        h[2] != AOMX_MAGIC2 || h[3] != AOMX_MAGIC3) { fclose(f); return 3; }
    if (rd_u32le(h + 4) != AOMX_VERSION) { fclose(f); return 4; }
    uint32_t rows  = rd_u32le(h + 8);
    uint32_t cols  = rd_u32le(h + 12);
    uint32_t dtype = rd_u32le(h + 16);
    uint32_t layout = rd_u32le(h + 20);
    uint32_t chk   = rd_u32le(h + 28);
    if (layout != AOMX_LAYOUT_ROWMAJOR) { fclose(f); return 5; }

    size_t n = (size_t)rows * cols;
    size_t esz = (dtype == AOMX_DTYPE_F64) ? 8 : 4;
    size_t nbytes = n * esz;
    uint8_t *raw = (uint8_t *)malloc(nbytes ? nbytes : 1);
    if (!raw) { fclose(f); return 6; }
    if (fread(raw, 1, nbytes, f) != nbytes) { free(raw); fclose(f); return 7; }
    fclose(f);

    if (chk != 0 && payload_checksum(raw, nbytes) != chk) { free(raw); return 8; }

    float *fdata = (float *)malloc((n ? n : 1) * sizeof(float));
    if (!fdata) { free(raw); return 9; }
    if (dtype == AOMX_DTYPE_F64) {
        const double *d = (const double *)(const void *)raw;
        for (size_t i = 0; i < n; ++i) fdata[i] = (float)d[i];
    } else {
        memcpy(fdata, raw, n * sizeof(float));
    }
    free(raw);

    out->rows = rows;
    out->cols = cols;
    out->dtype = dtype;
    out->f32 = fdata;
    return 0;
}

void aomx_free(AOMatrix *m) {
    if (!m) return;
    free(m->f32);
    m->f32 = NULL;
    m->rows = m->cols = 0;
}

int aomx_write_f32(const char *path, uint32_t rows, uint32_t cols,
                   const float *data) {
    FILE *f = fopen(path, "wb");
    if (!f) return 1;
    size_t n = (size_t)rows * cols;
    size_t nbytes = n * sizeof(float);

    uint8_t h[AOMX_HEADER_BYTES];
    memset(h, 0, sizeof(h));
    h[0] = AOMX_MAGIC0; h[1] = AOMX_MAGIC1; h[2] = AOMX_MAGIC2; h[3] = AOMX_MAGIC3;
    wr_u32le(h + 4, AOMX_VERSION);
    wr_u32le(h + 8, rows);
    wr_u32le(h + 12, cols);
    wr_u32le(h + 16, AOMX_DTYPE_F32);
    wr_u32le(h + 20, AOMX_LAYOUT_ROWMAJOR);
    wr_u32le(h + 24, (cols == 1) ? AOMX_FLAG_VECTOR : 0u);
    wr_u32le(h + 28, payload_checksum((const uint8_t *)data, nbytes));

    if (fwrite(h, 1, AOMX_HEADER_BYTES, f) != AOMX_HEADER_BYTES) { fclose(f); return 2; }
    if (fwrite(data, 1, nbytes, f) != nbytes) { fclose(f); return 3; }
    fclose(f);
    return 0;
}

int aomx_selftest(void) {
    const char *tmp = "._aomx_selftest.aomx";
    float A[6] = {1.0f, -2.5f, 3.25f, 4.0f, 5.5f, 6.75f};
    if (aomx_write_f32(tmp, 2, 3, A) != 0) return 1;

    uint32_t r, c, dt;
    if (aomx_read_header(tmp, &r, &c, &dt) != 0) { remove(tmp); return 2; }
    if (r != 2 || c != 3 || dt != AOMX_DTYPE_F32) { remove(tmp); return 3; }

    AOMatrix m;
    if (aomx_read(tmp, &m) != 0) { remove(tmp); return 4; }
    int rc = 0;
    if (m.rows != 2 || m.cols != 3) rc = 5;
    for (int i = 0; !rc && i < 6; ++i)
        if (m.f32[i] != A[i]) rc = 6;
    aomx_free(&m);
    remove(tmp);
    return rc;
}
