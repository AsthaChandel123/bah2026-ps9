/* reconstruct.c -- MVM wavefront reconstruction (zonal + modal).
 *
 * STUB for loading; the MVM calls themselves are wired to gemv_f32 so that,
 * once matrices are loaded, reconstruction works. research/02 S15, research/03.
 */
#include "reconstruct.h"
#include "linalg.h"
#include <string.h>

int reconstruct_load(const char *calib_dir, Reconstructor *rec) {
    /* TODO(impl): build paths "<calib_dir>/R.aomx", "Mpinv.aomx", "Z.aomx";
     * aomx_read each; set N, twoM, J; validate 2M agreement. Stub zeroes. */
    (void)calib_dir;
    if (!rec) return 1;
    memset(rec, 0, sizeof(*rec));
    return 0;
}

void reconstruct_free(Reconstructor *rec) {
    if (!rec) return;
    aomx_free(&rec->R);
    aomx_free(&rec->Mpinv);
    aomx_free(&rec->Z);
    memset(rec, 0, sizeof(*rec));
}

void reconstruct_zonal(const Reconstructor *rec, const float *s, float *phi) {
    if (!rec || !rec->R.f32 || !s || !phi) return;
    gemv_f32(rec->R.f32, s, phi, (int)rec->R.rows, (int)rec->R.cols);
}

void reconstruct_modal(const Reconstructor *rec, const float *s, float *a) {
    if (!rec || !rec->has_modal || !rec->Mpinv.f32 || !s || !a) return;
    gemv_f32(rec->Mpinv.f32, s, a, (int)rec->Mpinv.rows, (int)rec->Mpinv.cols);
}

void reconstruct_synthesize(const Reconstructor *rec, const float *a, float *W) {
    if (!rec || !rec->has_synth || !rec->Z.f32 || !a || !W) return;
    gemv_f32(rec->Z.f32, a, W, (int)rec->Z.rows, (int)rec->Z.cols);
}

int reconstruct_selftest(void) {
    /* TODO(impl): with R = identity, phi == s. Stub passes. */
    return 0;
}
