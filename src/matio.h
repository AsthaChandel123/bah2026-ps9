/* matio.h -- AOMX self-describing binary matrix file format (reader/writer).
 *
 * Format (little-endian, header 32 bytes, then row-major payload) -- this MUST
 * byte-match aokit/matio.py. See ARCHITECTURE.md S4.2.
 *
 *   Offset Size Type    Field
 *   ------ ---- ------- ----------------------------------------------------
 *    0      4   char[4] magic   = 'A','O','M','X'
 *    4      4   uint32  version = 1
 *    8      4   uint32  rows
 *   12      4   uint32  cols    (vector => cols = 1)
 *   16      4   uint32  dtype   0 = float32, 1 = float64
 *   20      4   uint32  layout  0 = row-major (only value defined in v1)
 *   24      4   uint32  flags   bit0: 1 if semantically a vector
 *   28      4   uint32  checksum additive sum of payload bytes mod 2^32 (0=skip)
 *   32   R*C*sz data    payload element(i,j) at i*cols + j
 */
#ifndef MATIO_H
#define MATIO_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define AOMX_MAGIC0 'A'
#define AOMX_MAGIC1 'O'
#define AOMX_MAGIC2 'M'
#define AOMX_MAGIC3 'X'
#define AOMX_VERSION    1u
#define AOMX_HEADER_BYTES 32u

enum { AOMX_DTYPE_F32 = 0, AOMX_DTYPE_F64 = 1 };
enum { AOMX_LAYOUT_ROWMAJOR = 0 };
enum { AOMX_FLAG_VECTOR = 1u };

/* A loaded matrix. Data is owned by the struct (free with aomx_free).
 * Data is always returned as float32 in `f32` for the real-time path; if the
 * file was float64 it is converted on load (offline matrices may keep f64 via
 * the _raw variant). */
typedef struct {
    uint32_t rows;
    uint32_t cols;
    uint32_t dtype;   /* original on-disk dtype */
    float   *f32;     /* row-major, rows*cols floats (always populated) */
} AOMatrix;

/* Read an AOMX file. On success returns 0 and fills *out (caller frees with
 * aomx_free). Converts float64 payloads to float32 in out->f32.
 * Returns non-zero on I/O error, bad magic, or unsupported version/layout.
 *
 * TODO(impl): parse header with explicit little-endian reads, validate magic,
 * allocate out->f32, load+convert payload, verify checksum if non-zero. */
int aomx_read(const char *path, AOMatrix *out);

/* Free a matrix loaded by aomx_read. Safe on a zeroed struct. */
void aomx_free(AOMatrix *m);

/* Write a row-major float32 buffer to an AOMX file (dtype=float32).
 * Returns 0 on success.
 *
 * TODO(impl): write 32-byte header (explicit LE), compute checksum, write
 * payload. */
int aomx_write_f32(const char *path, uint32_t rows, uint32_t cols,
                   const float *data);

/* Low-level header read (does not load payload). Returns 0 on success.
 * Useful for sizing buffers before a streaming read. */
int aomx_read_header(const char *path, uint32_t *rows, uint32_t *cols,
                     uint32_t *dtype);

/* Run a self-contained write->read->compare roundtrip on a small synthetic
 * matrix. Returns 0 if the roundtrip is bit/value exact, non-zero otherwise.
 * Called by main.c --selftest and `make test`. */
int aomx_selftest(void);

#ifdef __cplusplus
}
#endif

#endif /* MATIO_H */
