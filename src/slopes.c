/* slopes.c -- centroid -> slope conversion.
 *
 * STUB. s = (centroid - reference) * pixel_size_m / focal_length_m  [rad].
 * Slope layout: s[0..M-1] = x-slopes, s[M..2M-1] = y-slopes.
 */
#include "slopes.h"

int slopes_from_centroids(const float *cents, const int *valid, int n_sub,
                          const float *ref_centroids_xy,
                          double pixel_size_m, double focal_length_m,
                          const float *ref_slopes,
                          float *s) {
    /* TODO(impl): pack only valid sub-apertures; compute slopes; subtract
     * ref_slopes if provided. Below is a minimal correct-shape implementation
     * that emits all sub-apertures (M == n_sub) without reference subtraction;
     * implementers add masking + reference handling. */
    if (!cents || !s) return 0;
    double scale = (focal_length_m != 0.0) ? (pixel_size_m / focal_length_m) : 0.0;
    int m = 0;
    for (int k = 0; k < n_sub; ++k) {
        if (valid && !valid[k]) continue;
        float cx = cents[k];
        float cy = cents[n_sub + k];
        float rx = ref_centroids_xy ? ref_centroids_xy[k] : 0.0f;
        float ry = ref_centroids_xy ? ref_centroids_xy[n_sub + k] : 0.0f;
        s[m]      = (float)((cx - rx) * scale);  /* x-slope (temporary packing) */
        s[n_sub + m] = (float)((cy - ry) * scale); /* y-slope (see note) */
        ++m;
    }
    /* NOTE(impl): the x/y halves should be packed at [0..M-1] and [M..2M-1]
     * with the SAME M; the placeholder above uses n_sub as the half-stride.
     * Replace with two-pass packing once M (valid count) is known. */
    if (ref_slopes) {
        for (int i = 0; i < 2 * m; ++i) s[i] -= ref_slopes[i];
    }
    return m;
}

int slopes_selftest(void) {
    /* TODO(impl): a known global tilt maps to a constant slope. Stub passes. */
    return 0;
}
