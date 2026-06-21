/* slopes.h -- centroid -> slope conversion.
 *
 * The only place pixel pitch and MLA focal length enter:
 *   s = (centroid - reference) * pixel_size_m / focal_length_m   [rad of tilt]
 * See research/01 S1.1 and research/02 S1.
 *
 * Slope vector layout (matches the reconstructor matrices R, M+):
 *   s[0 .. M-1]     = x-slopes for the M valid sub-apertures
 *   s[M .. 2M-1]    = y-slopes
 */
#ifndef SLOPES_H
#define SLOPES_H

#ifdef __cplusplus
extern "C" {
#endif

/* Convert centroid arrays (absolute px, length n_sub each axis, as produced
 * by centroid_frame) into the 2M slope vector, subtracting reference slopes.
 *
 * `cents` layout: cents[0..n_sub-1]=x, cents[n_sub..2n_sub-1]=y (absolute px).
 * `ref_slopes` (length 2M, from refslopes.aomx) is subtracted to cancel
 * common-mode bias; pass NULL to skip (if references already subtracted at
 * the centroid stage).
 * `valid` (length n_sub, may be NULL) selects which sub-apertures are emitted;
 * when non-NULL, only valid ones are packed (in order) into `s`.
 * Returns M (number of valid sub-apertures emitted).
 *
 * TODO(impl): for each valid sub-ap k: s_x = (cx_k - refx_k)*scale;
 * s_y = (cy_k - refy_k)*scale; scale = pixel_size_m / focal_length_m. Then
 * subtract ref_slopes if provided. */
int slopes_from_centroids(const float *cents, const int *valid, int n_sub,
                          const float *ref_centroids_xy, /* 2*n_sub or NULL */
                          double pixel_size_m, double focal_length_m,
                          const float *ref_slopes, /* 2M or NULL */
                          float *s);

/* Self-test: a known global tilt must map to a constant slope vector.
 * Returns 0 on success. */
int slopes_selftest(void);

#ifdef __cplusplus
}
#endif

#endif /* SLOPES_H */
