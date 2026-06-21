/* bmp.h -- hand-rolled BMP reader/writer (no external image libs).
 *
 * Supports uncompressed (BI_RGB) 8-bit grayscale (palette-aware) and 24-bit
 * BGR. Handles bottom-up vs top-down (sign of biHeight) and 4-byte row
 * padding. See ARCHITECTURE.md and research/07 PART A for the byte layout.
 *
 * The reader returns a malloc'd row-major (top-left origin) grayscale buffer.
 * For the real-time path, callers pre-allocate once and reuse via the _into
 * variant to avoid per-frame allocation.
 */
#ifndef BMP_H
#define BMP_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Read a BMP into a freshly malloc'd grayscale buffer (top-left origin,
 * row-major, one float per pixel, values 0..255). Returns the buffer and sets
 * *out_w,*out_h; returns NULL on error (bad magic, compressed, I/O).
 * Caller frees with free().
 *
 * TODO(impl): explicit little-endian header reads, stride formula
 * ((bpp*w+31)/32)*4, row flip for bottom-up, 8-bit palette / 24-bit BGR luma.
 */
float *bmp_read_gray(const char *path, int *out_w, int *out_h);

/* Read a BMP into a caller-provided buffer of capacity >= w*h floats. The
 * caller must know (or first query) the dimensions. Returns 0 on success and
 * sets *out_w,*out_h; non-zero if the file dims exceed capacity or on error.
 * No allocation -- suitable for the hot loop.
 *
 * TODO(impl): as bmp_read_gray but write into `dst`. */
int bmp_read_gray_into(const char *path, float *dst, size_t cap,
                       int *out_w, int *out_h);

/* Query BMP dimensions without reading pixels. Returns 0 on success. */
int bmp_dims(const char *path, int *out_w, int *out_h);

/* Write an 8-bit grayscale BMP with an identity palette (bfOffBits=1078),
 * bottom-up rows, 4-byte padded. `img` is row-major top-left origin, w*h
 * bytes (0..255). Returns 0 on success.
 *
 * TODO(impl): emit 14-byte file header + 40-byte info header + 256*4 palette,
 * pad rows, write bottom-up. */
int bmp_write_gray8(const char *path, const uint8_t *img, int w, int h);

/* Self-test: synthesize a small image, write, read back, compare.
 * Returns 0 if the roundtrip matches. Called by main.c --selftest. */
int bmp_selftest(void);

#ifdef __cplusplus
}
#endif

#endif /* BMP_H */
