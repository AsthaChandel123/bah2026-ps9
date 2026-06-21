/* reconstruct.h -- MVM wavefront reconstruction (zonal + modal).
 *
 * All matrices are precomputed offline (Python) and loaded as AOMX:
 *   R     (N x 2M)   zonal Fried reconstructor:  phi = R * s   -> phase map
 *   Mpinv (J x 2M)   modal reconstructor:        a   = Mpinv*s -> Zernike coeffs
 *   Z     (Npts x J) Zernike synthesis:          W   = Z * a   (cross-check)
 * See research/02 S15, research/03 S6. Each step is one gemv_f32 call.
 */
#ifndef RECONSTRUCT_H
#define RECONSTRUCT_H

#include "matio.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Holds the loaded reconstruction matrices and scratch buffers (allocated
 * once at startup; never reallocated in the loop). */
typedef struct {
    AOMatrix R;       /* zonal reconstructor (N x 2M) */
    AOMatrix Mpinv;   /* modal reconstructor (J x 2M) */
    AOMatrix Z;       /* Zernike synthesis (Npts x J), optional */
    int      has_modal;
    int      has_synth;
    int      N;       /* phase points (zonal) */
    int      twoM;    /* slope vector length */
    int      J;       /* number of modes */
} Reconstructor;

/* Load reconstruction matrices from a calibration directory (expects
 * R.aomx, optionally Mpinv.aomx and Z.aomx). Returns 0 on success.
 * Free with reconstruct_free.
 *
 * TODO(impl): aomx_read the files; set dimensions; validate 2M agreement. */
int reconstruct_load(const char *calib_dir, Reconstructor *rec);

void reconstruct_free(Reconstructor *rec);

/* Zonal reconstruct: phi[N] = R * s[2M]. One gemv. */
void reconstruct_zonal(const Reconstructor *rec, const float *s, float *phi);

/* Modal reconstruct: a[J] = Mpinv * s[2M]. One gemv. */
void reconstruct_modal(const Reconstructor *rec, const float *s, float *a);

/* Synthesize wavefront from modal coeffs: W[Npts] = Z * a[J]. One gemv. */
void reconstruct_synthesize(const Reconstructor *rec, const float *a, float *W);

/* Self-test: with R = identity-like small matrix, phi == s. Returns 0. */
int reconstruct_selftest(void);

#ifdef __cplusplus
}
#endif

#endif /* RECONSTRUCT_H */
