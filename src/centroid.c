/* centroid.c -- TWCoG centroiding over a sub-aperture grid.
 *
 * Primary estimator: Thresholded + Windowed Weighted Center-of-Gravity
 * (TWCoG), O(1) per sub-aperture (work proportional to the fixed window area,
 * independent of frame size). research/01 S12.
 *
 *   1. slice the precomputed per-lenslet window,
 *   2. threshold  I_T = max(thresh_frac * Imax, thresh_sigma); clip negatives,
 *   3. enforce the >= min_pixels-above-threshold validity gate,
 *   4. weighted first moments via an optional per-sub-aperture Gaussian weight
 *      LUT (NULL => plain thresholded CoG),
 *   5. apply the WCoG linearization gain about the reference position.
 *
 * centroid_cog is the inner kernel / always-on baseline (TWCoG with W=1, T=0).
 * No runtime transcendentals: weights are precomputed.
 */
#include "centroid.h"
#include <math.h>

/* Clamp a window to the frame so we never read out of bounds. The clamped
 * extent is returned via *ix0,*iy0,*ix1,*iy1 (half-open in x1,y1). Returns 0 if
 * the window is entirely outside the frame (empty intersection). */
static int clamp_window(const SubAperture *sa, int frame_w, int frame_h,
                        int *ix0, int *iy0, int *ix1, int *iy1) {
    int x0 = sa->x0, y0 = sa->y0;
    int x1 = sa->x0 + sa->w; /* half-open */
    int y1 = sa->y0 + sa->h;
    if (x0 < 0) x0 = 0;
    if (y0 < 0) y0 = 0;
    if (x1 > frame_w) x1 = frame_w;
    if (y1 > frame_h) y1 = frame_h;
    *ix0 = x0; *iy0 = y0; *ix1 = x1; *iy1 = y1;
    return (x1 > x0 && y1 > y0);
}

int centroid_cog(const float *img, int frame_w, int frame_h,
                 const SubAperture *sa, float *cx, float *cy) {
    if (!img || !sa) { if (cx) *cx = 0.0f; if (cy) *cy = 0.0f; return 0; }
    int x0, y0, x1, y1;
    if (!clamp_window(sa, frame_w, frame_h, &x0, &y0, &x1, &y1)) {
        if (cx) *cx = (float)sa->x0 + 0.5f * (float)sa->w;
        if (cy) *cy = (float)sa->y0 + 0.5f * (float)sa->h;
        return 0;
    }
    /* Single pass: sx = Sum(i*I), sy = Sum(j*I), s = Sum(I). i,j are absolute
     * frame coordinates so the result is already in absolute px. */
    double sx = 0.0, sy = 0.0, s = 0.0;
    for (int j = y0; j < y1; ++j) {
        const float *row = img + (size_t)j * frame_w;
        for (int i = x0; i < x1; ++i) {
            float v = row[i];
            s  += v;
            sx += (double)i * v;
            sy += (double)j * v;
        }
    }
    if (s <= 0.0) {
        if (cx) *cx = (float)sa->x0 + 0.5f * (float)sa->w;
        if (cy) *cy = (float)sa->y0 + 0.5f * (float)sa->h;
        return 0;
    }
    if (cx) *cx = (float)(sx / s);
    if (cy) *cy = (float)(sy / s);
    return 1;
}

int centroid_twcog(const float *img, int frame_w, int frame_h,
                   const SubAperture *sa, const float *weights,
                   float thresh_frac, float thresh_sigma,
                   float wcog_gain, int min_pixels,
                   float *cx, float *cy) {
    /* Fall back to the window center so callers always see a defined value. */
    float fallback_x = sa ? ((float)sa->x0 + 0.5f * (float)sa->w) : 0.0f;
    float fallback_y = sa ? ((float)sa->y0 + 0.5f * (float)sa->h) : 0.0f;
    if (cx) *cx = fallback_x;
    if (cy) *cy = fallback_y;
    if (!img || !sa) return 0;

    int x0, y0, x1, y1;
    if (!clamp_window(sa, frame_w, frame_h, &x0, &y0, &x1, &y1)) return 0;

    /* Pass 1: window maximum (for the fractional threshold). */
    float imax = -INFINITY;
    for (int j = y0; j < y1; ++j) {
        const float *row = img + (size_t)j * frame_w;
        for (int i = x0; i < x1; ++i)
            if (row[i] > imax) imax = row[i];
    }
    if (!(imax > 0.0f)) return 0; /* empty / all non-positive window */

    /* Threshold = max(T * Imax, m * sigma_read). thresh_frac in [0,1]. */
    float thr = thresh_frac * imax;
    if (thresh_sigma > thr) thr = thresh_sigma;

    /* Pass 2: thresholded weighted first moments. The weight LUT (if given)
     * is laid out per-sub-aperture, w*h floats in window-row-major order using
     * the *nominal* (unclamped) window stride so a clamped edge window still
     * indexes the correct weights. */
    int win_w = sa->w;
    double sx = 0.0, sy = 0.0, sden = 0.0;
    int n_above = 0;
    for (int j = y0; j < y1; ++j) {
        const float *row = img + (size_t)j * frame_w;
        int wj = j - sa->y0; /* index into nominal window rows */
        for (int i = x0; i < x1; ++i) {
            float v = row[i] - thr;
            if (v <= 0.0f) continue;
            ++n_above;
            float wgt = 1.0f;
            if (weights) {
                int wi = i - sa->x0;
                wgt = weights[(size_t)wj * win_w + wi];
            }
            float wv = wgt * v;
            sx += (double)i * wv;
            sy += (double)j * wv;
            sden += wv;
        }
    }

    /* Validity gate: need enough illuminated pixels and positive denominator. */
    if (n_above < (min_pixels > 0 ? min_pixels : 1) || sden <= 0.0) return 0;

    float mx = (float)(sx / sden);
    float my = (float)(sy / sden);

    /* WCoG linearization gain gamma about the reference (window center if no
     * explicit reference): cx = ref + gamma*(measured - ref). gamma<=0 means
     * "unset" -> treat as 1 (no gain). */
    float g = (wcog_gain > 0.0f) ? wcog_gain : 1.0f;
    if (g != 1.0f) {
        float rx = sa->ref_x, ry = sa->ref_y;
        mx = rx + g * (mx - rx);
        my = ry + g * (my - ry);
    }

    if (cx) *cx = mx;
    if (cy) *cy = my;
    return 1;
}

int centroid_frame(const float *img, int frame_w, int frame_h,
                   const CentroidCfg *cfg, float *cents, int *valid_out) {
    if (!cfg || !cfg->subs) return 0;
    int nvalid = 0;
    size_t woff = 0; /* running offset into the concatenated weight LUT */
    for (int k = 0; k < cfg->n_sub; ++k) {
        const SubAperture *sa = &cfg->subs[k];
        float cx = (float)sa->x0 + 0.5f * (float)sa->w;
        float cy = (float)sa->y0 + 0.5f * (float)sa->h;
        int v = 0;
        if (sa->valid) {
            const float *w = NULL;
            if (cfg->weight_lut) { w = cfg->weight_lut + woff; }
            v = centroid_twcog(img, frame_w, frame_h, sa, w,
                               cfg->thresh_frac, cfg->thresh_sigma,
                               cfg->wcog_gain, cfg->min_pixels, &cx, &cy);
        }
        if (cfg->weight_lut) woff += (size_t)sa->w * sa->h;
        if (cents) { cents[k] = cx; cents[cfg->n_sub + k] = cy; }
        if (valid_out) valid_out[k] = v;
        nvalid += (v != 0);
    }
    return nvalid;
}

/* ---- self-test: recover a known sub-pixel Gaussian spot position ---- */

int centroid_selftest(void) {
    const int W = 16, H = 16;
    float img[16 * 16];
    const float true_x = 7.3f, true_y = 8.6f;
    const float sigma = 1.8f;
    for (int j = 0; j < H; ++j) {
        for (int i = 0; i < W; ++i) {
            float dx = (float)i - true_x, dy = (float)j - true_y;
            float val = 200.0f * expf(-(dx * dx + dy * dy) / (2.0f * sigma * sigma));
            img[j * W + i] = val + 2.0f; /* small flat background */
        }
    }
    SubAperture sa = { 0, 0, W, H, true_x, true_y, 1 };

    /* Plain CoG (background biases it slightly toward center -> just sanity). */
    float cx = 0, cy = 0;
    if (!centroid_cog(img, W, H, &sa, &cx, &cy)) return 1;
    if (fabsf(cx - true_x) > 1.0f || fabsf(cy - true_y) > 1.0f) return 2;

    /* TWCoG with a fraction-of-max threshold should kill the background and
     * recover the position tightly. */
    cx = cy = 0;
    if (!centroid_twcog(img, W, H, &sa, NULL, 0.10f, 0.0f, 1.0f, 3, &cx, &cy))
        return 3;
    if (fabsf(cx - true_x) > 0.15f || fabsf(cy - true_y) > 0.15f) return 4;

    /* Zero-flux window must report invalid, not crash. */
    float zero[16 * 16] = {0};
    cx = cy = -1.0f;
    if (centroid_twcog(zero, W, H, &sa, NULL, 0.10f, 0.0f, 1.0f, 3, &cx, &cy))
        return 5;

    /* frame-level pass with a tiny CentroidCfg. */
    SubAperture subs[1] = { sa };
    CentroidCfg cfg = { subs, 1, 0.10f, 0.0f, 1.0f, NULL, 3 };
    float cents[2]; int valid[1];
    int nv = centroid_frame(img, W, H, &cfg, cents, valid);
    if (nv != 1 || !valid[0]) return 6;
    if (fabsf(cents[0] - true_x) > 0.15f || fabsf(cents[1] - true_y) > 0.15f)
        return 7;

    return 0;
}
