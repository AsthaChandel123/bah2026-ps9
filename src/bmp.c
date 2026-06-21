/* bmp.c -- hand-rolled BMP reader/writer (implemented, no external libs).
 *
 * Supports uncompressed (BI_RGB) 8-bit grayscale (palette-aware) and 24-bit
 * BGR; bottom-up/top-down; 4-byte row padding. Byte layout per research/07 A.
 */
#include "bmp.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint32_t rd_u32(const uint8_t *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}
static uint16_t rd_u16(const uint8_t *p) {
    return (uint16_t)(p[0] | (p[1] << 8));
}
static void wr_u32(uint8_t *p, uint32_t v) {
    p[0] = (uint8_t)(v & 0xFF); p[1] = (uint8_t)((v >> 8) & 0xFF);
    p[2] = (uint8_t)((v >> 16) & 0xFF); p[3] = (uint8_t)((v >> 24) & 0xFF);
}
static void wr_u16(uint8_t *p, uint16_t v) {
    p[0] = (uint8_t)(v & 0xFF); p[1] = (uint8_t)((v >> 8) & 0xFF);
}

int bmp_dims(const char *path, int *out_w, int *out_h) {
    FILE *f = fopen(path, "rb");
    if (!f) return 1;
    uint8_t hdr[54];
    if (fread(hdr, 1, 54, f) != 54) { fclose(f); return 2; }
    fclose(f);
    if (hdr[0] != 'B' || hdr[1] != 'M') return 3;
    int32_t w = (int32_t)rd_u32(hdr + 18);
    int32_t h = (int32_t)rd_u32(hdr + 22);
    if (out_w) *out_w = (int)(w < 0 ? -w : w);
    if (out_h) *out_h = (int)(h < 0 ? -h : h);
    return 0;
}

/* Core reader: fills `dst` (cap floats) if non-NULL, else allocates. */
static float *bmp_read_core(const char *path, float *dst, size_t cap,
                            int *out_w, int *out_h, int *err) {
    *err = 0;
    FILE *f = fopen(path, "rb");
    if (!f) { *err = 1; return NULL; }
    uint8_t hdr[54];
    if (fread(hdr, 1, 54, f) != 54) { fclose(f); *err = 2; return NULL; }
    if (hdr[0] != 'B' || hdr[1] != 'M') { fclose(f); *err = 3; return NULL; }

    uint32_t off    = rd_u32(hdr + 10);
    int32_t  w      = (int32_t)rd_u32(hdr + 18);
    int32_t  h_raw  = (int32_t)rd_u32(hdr + 22);
    uint16_t bpp    = rd_u16(hdr + 28);
    uint32_t comp   = rd_u32(hdr + 30);
    if (comp != 0) { fclose(f); *err = 4; return NULL; }       /* uncompressed only */
    if (bpp != 8 && bpp != 24) { fclose(f); *err = 5; return NULL; }

    int bottom_up = (h_raw > 0);
    int h = bottom_up ? h_raw : -h_raw;
    if (w <= 0 || h <= 0) { fclose(f); *err = 6; return NULL; }
    int Bpp = bpp / 8;                                          /* 1 or 3 */
    int stride = ((bpp * w + 31) / 32) * 4;                     /* padded row size */

    size_t need = (size_t)w * h;
    float *img = dst;
    if (!img) {
        img = (float *)malloc(need * sizeof(float));
        if (!img) { fclose(f); *err = 7; return NULL; }
    } else if (cap < need) {
        fclose(f); *err = 8; return NULL;
    }

    /* Optional 8-bit palette read (identity assumed for grayscale, but honor it). */
    uint8_t pal[256][4];
    int have_pal = 0;
    if (bpp == 8) {
        long palpos = 54; /* after BITMAPINFOHEADER (assumes biSize==40) */
        if (rd_u32(hdr + 14) != 40) palpos = 14 + (long)rd_u32(hdr + 14);
        if (fseek(f, palpos, SEEK_SET) == 0 &&
            fread(pal, 1, sizeof(pal), f) == sizeof(pal)) {
            have_pal = 1;
        }
    }

    uint8_t *row = (uint8_t *)malloc((size_t)stride);
    if (!row) { if (!dst) free(img); fclose(f); *err = 9; return NULL; }

    for (int r = 0; r < h; ++r) {
        if (fseek(f, (long)off + (long)r * stride, SEEK_SET) != 0 ||
            fread(row, 1, (size_t)stride, f) != (size_t)stride) {
            free(row); if (!dst) free(img); fclose(f); *err = 10; return NULL;
        }
        int dstrow = bottom_up ? (h - 1 - r) : r;
        for (int c = 0; c < w; ++c) {
            float g;
            if (Bpp == 1) {
                uint8_t idx = row[c];
                if (have_pal) {
                    /* luma from palette entry (B,G,R,0) */
                    g = (pal[idx][2] * 299 + pal[idx][1] * 587 + pal[idx][0] * 114) / 1000.0f;
                } else {
                    g = (float)idx;
                }
            } else {
                /* 24-bit BGR -> luma */
                g = (row[c*3+2] * 299 + row[c*3+1] * 587 + row[c*3+0] * 114) / 1000.0f;
            }
            img[(size_t)dstrow * w + c] = g;
        }
    }
    free(row);
    fclose(f);
    if (out_w) *out_w = w;
    if (out_h) *out_h = h;
    return img;
}

float *bmp_read_gray(const char *path, int *out_w, int *out_h) {
    int err = 0;
    return bmp_read_core(path, NULL, 0, out_w, out_h, &err);
}

int bmp_read_gray_into(const char *path, float *dst, size_t cap,
                       int *out_w, int *out_h) {
    int err = 0;
    float *r = bmp_read_core(path, dst, cap, out_w, out_h, &err);
    return (r != NULL) ? 0 : (err ? err : 1);
}

int bmp_write_gray8(const char *path, const uint8_t *img, int w, int h) {
    if (w <= 0 || h <= 0) return 1;
    FILE *f = fopen(path, "wb");
    if (!f) return 2;

    int stride = ((8 * w + 31) / 32) * 4;
    uint32_t pixoff = 54 + 256 * 4;           /* 1078 */
    uint32_t imgsize = (uint32_t)stride * h;
    uint32_t filesize = pixoff + imgsize;

    uint8_t fh[14];
    memset(fh, 0, sizeof(fh));
    fh[0] = 'B'; fh[1] = 'M';
    wr_u32(fh + 2, filesize);
    wr_u32(fh + 10, pixoff);

    uint8_t ih[40];
    memset(ih, 0, sizeof(ih));
    wr_u32(ih + 0, 40);
    wr_u32(ih + 4, (uint32_t)w);
    wr_u32(ih + 8, (uint32_t)h);                /* positive => bottom-up */
    wr_u16(ih + 12, 1);                         /* planes */
    wr_u16(ih + 14, 8);                         /* bpp */
    wr_u32(ih + 16, 0);                         /* BI_RGB */
    wr_u32(ih + 20, imgsize);
    wr_u32(ih + 24, 2835);                      /* ~72 DPI x */
    wr_u32(ih + 28, 2835);                      /* ~72 DPI y */
    wr_u32(ih + 32, 256);                       /* clrUsed */
    wr_u32(ih + 36, 0);

    if (fwrite(fh, 1, 14, f) != 14) { fclose(f); return 3; }
    if (fwrite(ih, 1, 40, f) != 40) { fclose(f); return 4; }

    /* identity grayscale palette: B=G=R=i, reserved=0 */
    uint8_t pal[4];
    for (int i = 0; i < 256; ++i) {
        pal[0] = pal[1] = pal[2] = (uint8_t)i; pal[3] = 0;
        if (fwrite(pal, 1, 4, f) != 4) { fclose(f); return 5; }
    }

    uint8_t *row = (uint8_t *)calloc((size_t)stride, 1);
    if (!row) { fclose(f); return 6; }
    for (int r = 0; r < h; ++r) {
        int srcrow = h - 1 - r;                 /* bottom-up */
        for (int c = 0; c < w; ++c) row[c] = img[(size_t)srcrow * w + c];
        /* padding bytes already zero */
        if (fwrite(row, 1, (size_t)stride, f) != (size_t)stride) {
            free(row); fclose(f); return 7;
        }
    }
    free(row);
    fclose(f);
    return 0;
}

int bmp_selftest(void) {
    const char *tmp = "._bmp_selftest.bmp";
    const int w = 7, h = 5;                     /* odd width exercises padding */
    uint8_t src[35];
    for (int i = 0; i < w * h; ++i) src[i] = (uint8_t)((i * 7) & 0xFF);
    if (bmp_write_gray8(tmp, src, w, h) != 0) return 1;

    int rw = 0, rh = 0;
    float *back = bmp_read_gray(tmp, &rw, &rh);
    int rc = 0;
    if (!back) rc = 2;
    else if (rw != w || rh != h) rc = 3;
    else {
        for (int i = 0; i < w * h; ++i) {
            if ((uint8_t)(back[i] + 0.5f) != src[i]) { rc = 4; break; }
        }
    }
    free(back);
    remove(tmp);
    return rc;
}
