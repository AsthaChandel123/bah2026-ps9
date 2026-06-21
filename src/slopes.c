/* slopes.c -- centroid -> slope conversion.
 *
 * The only place pixel pitch and MLA focal length enter:
 *   s = (centroid - reference) * pixel_size_m / focal_length_m   [rad of tilt]
 * research/01 S1.1, research/02 S1.
 *
 * Slope vector layout (matches the reconstructor matrices R, M+):
 *   s[0   .. M-1]  = x-slopes for the M valid sub-apertures (packed in order)
 *   s[M   .. 2M-1] = y-slopes
 * where M is the number of valid sub-apertures emitted (== n_sub if no mask).
 */
#include "slopes.h"
#include <math.h>
#include <stddef.h>

int slopes_from_centroids(const float *cents, const int *valid, int n_sub,
                          const float *ref_centroids_xy,
                          double pixel_size_m, double focal_length_m,
                          const float *ref_slopes,
                          float *s) {
    if (!cents || !s || n_sub <= 0) return 0;
    double scale = (focal_length_m != 0.0) ? (pixel_size_m / focal_length_m) : 0.0;

    /* First pass: count valid sub-apertures so we know the half-stride M. */
    int m = 0;
    for (int k = 0; k < n_sub; ++k) {
        if (valid && !valid[k]) continue;
        ++m;
    }
    if (m == 0) return 0;

    /* Second pass: pack x-slopes into s[0..m-1], y-slopes into s[m..2m-1].
     * cents and ref_centroids_xy are indexed by the *raw* sub-aperture k
     * (x at [k], y at [n_sub + k]); the packed output is indexed by the
     * compacted slot p. */
    int p = 0;
    for (int k = 0; k < n_sub; ++k) {
        if (valid && !valid[k]) continue;
        float cx = cents[k];
        float cy = cents[n_sub + k];
        float rx = ref_centroids_xy ? ref_centroids_xy[k] : 0.0f;
        float ry = ref_centroids_xy ? ref_centroids_xy[n_sub + k] : 0.0f;
        s[p]     = (float)((double)(cx - rx) * scale); /* x-slope */
        s[m + p] = (float)((double)(cy - ry) * scale); /* y-slope */
        ++p;
    }

    /* Reference-slope subtraction (flat-wavefront common-mode cancellation).
     * ref_slopes has length 2M == 2m, same packing as s. */
    if (ref_slopes) {
        int two_m = 2 * m;
        for (int i = 0; i < two_m; ++i) s[i] -= ref_slopes[i];
    }
    return m;
}

/* ---- self-test: a known global tilt maps to a constant slope vector ---- */

int slopes_selftest(void) {
    const int n_sub = 4;
    /* Reference centroids = window centers; a uniform +1.5 px x-shift and
     * -0.5 px y-shift should produce a constant slope per axis. */
    float cents[8];          /* x at [0..3], y at [4..7] */
    float refs[8];
    const float dx = 1.5f, dy = -0.5f;
    for (int k = 0; k < n_sub; ++k) {
        refs[k]         = 10.0f + (float)k * 20.0f;  /* arbitrary ref x */
        refs[n_sub + k] = 5.0f  + (float)k * 20.0f;  /* arbitrary ref y */
        cents[k]         = refs[k] + dx;
        cents[n_sub + k] = refs[n_sub + k] + dy;
    }
    double pix = 5.5e-6, foc = 5.2e-3;
    double scale = pix / foc;
    float s[8];
    int m = slopes_from_centroids(cents, NULL, n_sub, refs, pix, foc, NULL, s);
    if (m != n_sub) return 1;

    float ex = (float)(dx * scale), ey = (float)(dy * scale);
    for (int k = 0; k < m; ++k) {
        if (fabsf(s[k]     - ex) > 1e-9f) return 2; /* x half */
        if (fabsf(s[m + k] - ey) > 1e-9f) return 3; /* y half */
    }

    /* Masking: drop sub-aperture 1; expect M==3 and correct half-stride. */
    int valid[4] = {1, 0, 1, 1};
    float s2[8];
    int m2 = slopes_from_centroids(cents, valid, n_sub, refs, pix, foc, NULL, s2);
    if (m2 != 3) return 4;
    for (int k = 0; k < m2; ++k) {
        if (fabsf(s2[k]      - ex) > 1e-9f) return 5;
        if (fabsf(s2[m2 + k] - ey) > 1e-9f) return 6;
    }

    /* Reference-slope subtraction zeroes a slope vector equal to itself. */
    float s3[8];
    int m3 = slopes_from_centroids(cents, NULL, n_sub, refs, pix, foc, s, s3);
    if (m3 != n_sub) return 7;
    for (int i = 0; i < 2 * m3; ++i)
        if (fabsf(s3[i]) > 1e-9f) return 8;

    return 0;
}
