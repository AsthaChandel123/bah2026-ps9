/* centroid.c -- TWCoG centroiding over a sub-aperture grid.
 *
 * STUB: documented signatures with minimal compilable bodies. The primary
 * estimator is TWCoG (research/01 S12); centroid_cog is the inner kernel.
 * Implementers replace the TODO(impl) bodies.
 */
#include "centroid.h"
#include <math.h>

int centroid_cog(const float *img, int frame_w, int frame_h,
                 const SubAperture *sa, float *cx, float *cy) {
    /* TODO(impl): single pass over the window computing
     *   sx = Sum(i*I), sy = Sum(j*I), s = Sum(I)
     * then cx = x0 + sx/s, cy = y0 + sy/s. Validity if s > 0. */
    (void)img; (void)frame_w; (void)frame_h;
    if (cx) *cx = sa ? (sa->x0 + 0.5f * sa->w) : 0.0f;   /* placeholder: window center */
    if (cy) *cy = sa ? (sa->y0 + 0.5f * sa->h) : 0.0f;
    return 0; /* not yet valid */
}

int centroid_twcog(const float *img, int frame_w, int frame_h,
                   const SubAperture *sa, const float *weights,
                   float thresh_frac, float thresh_sigma,
                   float wcog_gain, int min_pixels,
                   float *cx, float *cy) {
    /* TODO(impl): threshold I_T = max(thresh_frac*Imax, thresh_sigma); clip
     * negatives; gate on min_pixels; weighted moments via precomputed weights;
     * apply gain wcog_gain. */
    (void)weights; (void)thresh_frac; (void)thresh_sigma;
    (void)wcog_gain; (void)min_pixels;
    return centroid_cog(img, frame_w, frame_h, sa, cx, cy);
}

int centroid_frame(const float *img, int frame_w, int frame_h,
                   const CentroidCfg *cfg, float *cents, int *valid_out) {
    /* TODO(impl): loop sub-apertures; for each call centroid_twcog and pack
     * cents[k]=cx, cents[n_sub+k]=cy; count valid. */
    if (!cfg) return 0;
    int nvalid = 0;
    for (int k = 0; k < cfg->n_sub; ++k) {
        float cx = 0.0f, cy = 0.0f;
        const float *w = cfg->weight_lut; /* per-sub offset TODO(impl) */
        int v = centroid_twcog(img, frame_w, frame_h, &cfg->subs[k], w,
                               cfg->thresh_frac, cfg->thresh_sigma,
                               cfg->wcog_gain, cfg->min_pixels, &cx, &cy);
        if (cents) { cents[k] = cx; cents[cfg->n_sub + k] = cy; }
        if (valid_out) valid_out[k] = v;
        nvalid += (v != 0);
    }
    return nvalid;
}

int centroid_selftest(void) {
    /* TODO(impl): synthesize a Gaussian spot at a known sub-pixel position and
     * assert recovery. Stub returns 0 (pass) so the project builds/runs. */
    return 0;
}
