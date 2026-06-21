/* centroid.h -- TWCoG centroiding over a sub-aperture grid (real-time path).
 *
 * Primary estimator: Thresholded + Windowed Weighted Center-of-Gravity
 * (TWCoG), O(1) per sub-aperture. The plain CoG kernel is exposed as the
 * baseline and as the inner kernel. See research/01 S12.
 *
 * All per-lenslet geometry (window origin/size), the Gaussian weight LUT,
 * the i*W / j*W index tables, the WCoG gain, and the reference centroids are
 * PRECOMPUTED (loaded from AOMX / built once) so the hot loop is table-driven
 * arithmetic with no transcendentals.
 */
#ifndef CENTROID_H
#define CENTROID_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* One sub-aperture's fixed geometry (from flat-wavefront calibration). */
typedef struct {
    int   x0, y0;     /* window top-left origin in the frame (px) */
    int   w,  h;      /* window size (px) */
    float ref_x;      /* reference centroid x (px, absolute frame coords) */
    float ref_y;      /* reference centroid y (px) */
    int   valid;      /* 1 if illuminated/active (from flux mask) */
} SubAperture;

/* Centroiding configuration / precomputed tables. */
typedef struct {
    const SubAperture *subs;  /* array of n_sub sub-apertures */
    int    n_sub;
    float  thresh_frac;       /* T in I_T = max(T*Imax, m*sigma_read) */
    float  thresh_sigma;      /* m*sigma_read absolute floor */
    float  wcog_gain;         /* gamma: WCoG linearization gain */
    const float *weight_lut;  /* optional Gaussian weights, concatenated per sub-ap (or NULL => CoG) */
    int    min_pixels;        /* validity gate: >= this many px above threshold */
} CentroidCfg;

/* Compute one sub-aperture centroid with plain CoG (inner kernel / baseline).
 * `img` is the full frame (row-major, w*h floats); window given by `sa`.
 * Writes sub-pixel centroid (absolute frame coords) to *cx,*cy. Returns 1 if
 * valid, 0 if the window had insufficient signal.
 *
 * TODO(impl): single pass Sum(i*I), Sum(j*I), Sum(I) over the window. */
int centroid_cog(const float *img, int frame_w, int frame_h,
                 const SubAperture *sa, float *cx, float *cy);

/* Compute one sub-aperture centroid with TWCoG (threshold + window + Gaussian
 * weight). `weights` may be NULL (=> reduces to thresholded CoG). Returns
 * validity. This is the hot-path primitive.
 *
 * TODO(impl): threshold = max(T*Imax, m*sigma); clip; gate on min_pixels;
 * weighted moments via precomputed i*W/j*W; apply gain gamma. */
int centroid_twcog(const float *img, int frame_w, int frame_h,
                   const SubAperture *sa, const float *weights,
                   float thresh_frac, float thresh_sigma,
                   float wcog_gain, int min_pixels,
                   float *cx, float *cy);

/* Centroid all sub-apertures of a frame. Writes 2*n_sub interleaved-by-axis
 * values: cents[0..n_sub-1] = x, cents[n_sub..2n_sub-1] = y (absolute px).
 * `valid_out` (optional, may be NULL) receives per-sub validity flags.
 * Returns the number of valid sub-apertures.
 *
 * TODO(impl): loop sub-apertures calling centroid_twcog; this is the
 * per-frame centroiding stage. */
int centroid_frame(const float *img, int frame_w, int frame_h,
                   const CentroidCfg *cfg, float *cents, int *valid_out);

/* Self-test: synthesize a Gaussian spot at a known sub-pixel position and
 * assert recovery within tolerance. Returns 0 on success. */
int centroid_selftest(void);

#ifdef __cplusplus
}
#endif

#endif /* CENTROID_H */
