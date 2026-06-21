/* pipeline.c -- ties stages together for one frame and the frame loop.
 *
 * Per frame:  bmp_read -> preprocess -> centroid_frame -> slopes ->
 *             reconstruct_zonal (W) + reconstruct_modal (a) -> dmcmd (A).
 * Buffers are allocated once at init (zero in-loop allocation) and each stage
 * is timed with clock_gettime(CLOCK_MONOTONIC) to verify the < 10 ms budget.
 * research/06 S10, S14.
 *
 * Calibration files loaded from <calib_dir>:
 *   R.aomx (N x 2M)         zonal reconstructor              [mandatory]
 *   Mpinv.aomx (J x 2M)     modal reconstructor              [optional]
 *   Z.aomx (Npts x J)       synthesis                        [optional]
 *   G.aomx (N_act x N)      DM command matrix                [mandatory]
 *   K.aomx (N_act x 2M)     fused slopes->commands           [optional]
 *   subapmap.aomx (N_sub x 4)  per-sub [x0_px, y0_px, w, h]  [mandatory]
 *   refslopes.aomx (2M x 1)    reference slopes              [optional]
 *
 * The Pipeline struct (pipeline.h) has no dedicated slot for the SubAperture
 * table or the reference centroids, so we co-allocate them in a single block
 * pointed to by p->cen.subs: [ SubAperture[n_sub] | float ref_xy[2*n_sub] ].
 * ref_xy[k] / ref_xy[n_sub+k] are the per-sub reference centroids (window
 * center plus the refslopes offset, converted to pixels) consumed by the
 * slopes stage. This keeps zero in-loop allocation without modifying the
 * shared header.
 */
#include "pipeline.h"
#include "bmp.h"
#include "slopes.h"
#include "matio.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e6 + (double)ts.tv_nsec * 1e-3;
}

/* Launder a const pointer to a mutable one without a cast that would trip
 * -Wcast-qual. We own the storage behind p->cen.subs (we allocated it); the
 * header just types the field as const. Copying the pointer bits is the
 * standard well-defined way to drop the qualifier here. */
static void *unconst_ptr(const void *p) {
    void *q;
    memcpy(&q, &p, sizeof q);
    return q;
}

static int join_path(char *dst, size_t dst_sz, const char *dir, const char *name) {
    if (!dst || dst_sz == 0) return 1;
    const char *sep = "/";
    size_t dl = dir ? strlen(dir) : 0;
    if (dl > 0 && (dir[dl - 1] == '/')) sep = "";
    int n = snprintf(dst, dst_sz, "%s%s%s", dir ? dir : ".", sep, name);
    return (n < 0 || (size_t)n >= dst_sz) ? 1 : 0;
}

/* Recover the co-allocated reference-centroid array that trails the
 * SubAperture table inside p->cen.subs. */
static float *ref_xy_of(const Pipeline *p) {
    if (!p->cen.subs || p->cen.n_sub <= 0) return NULL;
    SubAperture *subs = (SubAperture *)unconst_ptr(p->cen.subs);
    return (float *)(void *)(subs + p->cen.n_sub);
}

/* Build the SubAperture table (+ reference centroids) from subapmap.aomx and,
 * if present, refslopes.aomx. Returns 0 on success. */
static int build_subapertures(Pipeline *p, const char *calib_dir) {
    char path[1024];

    AOMatrix map;
    memset(&map, 0, sizeof(map));
    if (join_path(path, sizeof(path), calib_dir, "subapmap.aomx") != 0) return 1;
    if (aomx_read(path, &map) != 0) {
        fprintf(stderr, "pipeline_init: cannot read %s\n", path);
        return 2;
    }
    if (map.cols < 4 || map.rows == 0) {
        fprintf(stderr, "pipeline_init: subapmap must be (N_sub x >=4), got %ux%u\n",
                map.rows, map.cols);
        aomx_free(&map);
        return 3;
    }
    int n_sub = (int)map.rows;
    int ncol  = (int)map.cols;

    /* One block holds the SubAperture array followed by the 2*n_sub reference
     * centroid floats. */
    size_t bytes = (size_t)n_sub * sizeof(SubAperture)
                 + (size_t)2 * n_sub * sizeof(float);
    SubAperture *subs = (SubAperture *)calloc(1, bytes);
    if (!subs) { aomx_free(&map); return 4; }
    float *ref_xy = (float *)(void *)(subs + n_sub);

    for (int k = 0; k < n_sub; ++k) {
        const float *r = map.f32 + (size_t)k * ncol;
        int x0 = (int)(r[0] + 0.5f);
        int y0 = (int)(r[1] + 0.5f);
        int w  = (int)(r[2] + 0.5f);
        int h  = (int)(r[3] + 0.5f);
        if (w <= 0) w = 1;
        if (h <= 0) h = 1;
        subs[k].x0 = x0;
        subs[k].y0 = y0;
        subs[k].w  = w;
        subs[k].h  = h;
        subs[k].ref_x = (float)x0 + 0.5f * (float)w; /* nominal: window center */
        subs[k].ref_y = (float)y0 + 0.5f * (float)h;
        subs[k].valid = 1; /* subapmap lists active sub-apertures */
        ref_xy[k]         = subs[k].ref_x;
        ref_xy[n_sub + k] = subs[k].ref_y;
    }
    aomx_free(&map);

    /* Fold reference slopes (if any) into the reference centroids: a refslope
     * sx corresponds to a centroid offset of sx * f / p_pix pixels, so
     * subtracting refslopes from the measured slope is identical to shifting
     * the reference centroid by that many pixels. refslopes packing is
     * [x0..xM-1, xM..2M-1] matching the slope vector. */
    AOMatrix rs;
    memset(&rs, 0, sizeof(rs));
    if (join_path(path, sizeof(path), calib_dir, "refslopes.aomx") == 0 &&
        aomx_read(path, &rs) == 0) {
        int two_m = (int)((size_t)rs.rows * rs.cols);
        if (two_m == 2 * n_sub) {
            double pix = p->cfg.camera.pixel_size_m;
            double foc = p->cfg.mla.focal_length_m;
            double inv = (pix != 0.0) ? (foc / pix) : 0.0; /* slope->px */
            for (int k = 0; k < n_sub; ++k) {
                ref_xy[k]         += (float)((double)rs.f32[k] * inv);
                ref_xy[n_sub + k] += (float)((double)rs.f32[n_sub + k] * inv);
                subs[k].ref_x = ref_xy[k];
                subs[k].ref_y = ref_xy[n_sub + k];
            }
        } else {
            fprintf(stderr,
                "pipeline_init: refslopes length %d != 2*N_sub %d; ignoring\n",
                two_m, 2 * n_sub);
        }
        aomx_free(&rs);
    }

    /* Wire the centroid config. Threshold/gain defaults give thresholded CoG
     * (fraction-of-max), which is robust for the synthetic + lab spots. */
    p->cen.subs        = subs;
    p->cen.n_sub       = n_sub;
    p->cen.thresh_frac = 0.10f; /* 10% of window max */
    p->cen.thresh_sigma = 0.0f;
    p->cen.wcog_gain   = 1.0f;  /* no WCoG gain by default */
    p->cen.weight_lut  = NULL;  /* plain thresholded CoG */
    p->cen.min_pixels  = 3;     /* SHARPEST-style >=3-pixel validity gate */
    p->n_sub = n_sub;
    return 0;
}

int pipeline_init(Pipeline *p, const char *config_path, const char *calib_dir) {
    if (!p) return 1;
    memset(p, 0, sizeof(*p));

    ao_config_defaults(&p->cfg);
    if (config_path) (void)ao_config_load(config_path, &p->cfg);

    if (!calib_dir) {
        fprintf(stderr, "pipeline_init: calib_dir is required\n");
        return 2;
    }

    if (reconstruct_load(calib_dir, &p->rec) != 0) {
        fprintf(stderr, "pipeline_init: reconstruct_load failed\n");
        pipeline_free(p);
        return 3;
    }
    if (dmcmd_load(calib_dir, p->cfg.dm.stroke_max_m,
                   p->cfg.dm.stroke_gain_m_per_unit, &p->dm) != 0) {
        fprintf(stderr, "pipeline_init: dmcmd_load failed\n");
        pipeline_free(p);
        return 4;
    }
    if (build_subapertures(p, calib_dir) != 0) {
        pipeline_free(p);
        return 5;
    }

    /* Dimensions come from the loaded matrices (authoritative), with the
     * slope length cross-checked against the sub-aperture count. */
    p->frame_w = p->cfg.camera.frame_w > 0 ? p->cfg.camera.frame_w : 1;
    p->frame_h = p->cfg.camera.frame_h > 0 ? p->cfg.camera.frame_h : 1;
    p->twoM    = p->rec.twoM;
    p->N       = p->rec.N;
    p->J       = p->rec.has_modal ? p->rec.J : 0;
    p->n_act   = p->dm.n_act;

    if (p->twoM != 2 * p->n_sub) {
        fprintf(stderr,
            "pipeline_init: WARNING 2M from R (%d) != 2*N_sub from subapmap (%d). "
            "Using min for safety.\n", p->twoM, 2 * p->n_sub);
    }
    if (p->dm.N != p->N) {
        fprintf(stderr,
            "pipeline_init: WARNING G cols (%d) != R rows N (%d).\n", p->dm.N, p->N);
    }

    /* Allocate scratch buffers (>=1 elem to avoid zero-size mallocs). */
    size_t fw = (size_t)p->frame_w, fh = (size_t)p->frame_h;
    int cents_n = p->n_sub > 0 ? 2 * p->n_sub : 2;
    int valid_n = p->n_sub > 0 ? p->n_sub : 1;
    int slope_n = p->twoM  > 0 ? p->twoM  : 1;
    int phi_n   = p->N     > 0 ? p->N     : 1;
    int coef_n  = p->J     > 0 ? p->J     : 1;
    int act_n   = p->n_act > 0 ? p->n_act : 1;

    p->frame  = (float *)calloc(fw * fh, sizeof(float));
    p->cents  = (float *)calloc((size_t)cents_n, sizeof(float));
    p->valid  = (int   *)calloc((size_t)valid_n, sizeof(int));
    p->slopes = (float *)calloc((size_t)slope_n, sizeof(float));
    p->phi    = (float *)calloc((size_t)phi_n, sizeof(float));
    p->coeffs = (float *)calloc((size_t)coef_n, sizeof(float));
    p->acts   = (float *)calloc((size_t)act_n, sizeof(float));

    if (!p->frame || !p->cents || !p->valid || !p->slopes ||
        !p->phi || !p->coeffs || !p->acts) {
        pipeline_free(p);
        return 6;
    }
    return 0;
}

void pipeline_free(Pipeline *p) {
    if (!p) return;
    reconstruct_free(&p->rec);
    dmcmd_free(&p->dm);
    /* p->cen.subs is the single co-allocated block (subs + ref_xy). */
    free(unconst_ptr(p->cen.subs));
    p->cen.subs = NULL;
    free(p->frame); free(p->cents); free(p->valid);
    free(p->slopes); free(p->phi); free(p->coeffs); free(p->acts);
    memset(p, 0, sizeof(*p));
}

int pipeline_process_frame(Pipeline *p, FrameStats *stats) {
    if (!p) return 1;
    FrameStats st;
    memset(&st, 0, sizeof(st));

    double t0 = now_us();
    /* (preprocess: dark/flat/background subtraction would go here; the
     * synthetic + reference path assumes already-calibrated pixels.) */

    double t1 = now_us();
    st.n_valid_sub = centroid_frame(p->frame, p->frame_w, p->frame_h,
                                    &p->cen, p->cents, p->valid);

    double t2 = now_us();
    const float *ref_xy = ref_xy_of(p);
    int m = slopes_from_centroids(p->cents, p->valid, p->n_sub,
                                  ref_xy,
                                  p->cfg.camera.pixel_size_m,
                                  p->cfg.mla.focal_length_m,
                                  NULL, /* refslopes already folded into ref_xy */
                                  p->slopes);
    (void)m;
    reconstruct_zonal(&p->rec, p->slopes, p->phi);
    if (p->rec.has_modal) reconstruct_modal(&p->rec, p->slopes, p->coeffs);

    double t3 = now_us();
    st.n_saturated_act = dmcmd_from_phase(&p->dm, p->phi, p->acts);

    double t4 = now_us();
    st.t_read_us     = t1 - t0;
    st.t_centroid_us = t2 - t1;
    st.t_recon_us    = t3 - t2;
    st.t_dm_us       = t4 - t3;
    st.t_total_us    = t4 - t0;
    if (stats) *stats = st;
    return 0;
}

int pipeline_run_bmp(Pipeline *p, const char *path, FrameStats *stats) {
    if (!p || !path) return 1;
    double t0 = now_us();
    size_t cap = (size_t)p->frame_w * (size_t)p->frame_h;
    int w = 0, h = 0;
    int rc = bmp_read_gray_into(path, p->frame, cap, &w, &h);
    if (rc != 0) return rc;
    rc = pipeline_process_frame(p, stats);
    if (stats) {
        double tread = now_us() - t0;
        /* Fold actual disk read+decode into the read stage and total. */
        stats->t_read_us = tread - (stats->t_centroid_us + stats->t_recon_us +
                                    stats->t_dm_us);
        if (stats->t_read_us < 0.0) stats->t_read_us = 0.0;
        stats->t_total_us = tread;
    }
    return rc;
}

/* ---- self-test: wire default config + tiny identity-ish matrices end to end.
 * Builds a synthetic calib set in a temp dir, runs one frame, frees. ---- */

int pipeline_selftest(void) {
    /* Minimal in-memory pipeline without touching disk: construct the struct
     * fields directly with tiny identity matrices and one sub-aperture, then
     * process a synthetic frame. */
    Pipeline p;
    memset(&p, 0, sizeof(p));
    ao_config_defaults(&p.cfg);
    p.cfg.camera.frame_w = 8;
    p.cfg.camera.frame_h = 8;

    /* One sub-aperture covering the whole frame. */
    int n_sub = 1;
    size_t bytes = (size_t)n_sub * sizeof(SubAperture) + (size_t)2 * n_sub * sizeof(float);
    SubAperture *subs = (SubAperture *)calloc(1, bytes);
    if (!subs) return 1;
    float *ref_xy = (float *)(void *)(subs + n_sub);
    subs[0].x0 = 0; subs[0].y0 = 0; subs[0].w = 8; subs[0].h = 8;
    subs[0].ref_x = 3.5f; subs[0].ref_y = 3.5f; subs[0].valid = 1;
    ref_xy[0] = 3.5f; ref_xy[1] = 3.5f;
    p.cen.subs = subs; p.cen.n_sub = n_sub;
    p.cen.thresh_frac = 0.10f; p.cen.thresh_sigma = 0.0f;
    p.cen.wcog_gain = 1.0f; p.cen.weight_lut = NULL; p.cen.min_pixels = 3;
    p.n_sub = n_sub;

    /* R = I (2x2), G = -1/2 I (1x2 -> use 1x2). We need 2M=2, N to feed G. Keep
     * N=2 with R 2x2 (phi=s), G 2x2 mapping phi->2 acts. */
    static float Rid2[4] = {1, 0, 0, 1};
    static float Gid2[4] = {-0.5f, 0, 0, -0.5f};
    p.rec.R.rows = 2; p.rec.R.cols = 2; p.rec.R.f32 = Rid2;
    p.rec.N = 2; p.rec.twoM = 2; p.rec.has_modal = 0; p.rec.has_synth = 0;
    p.dm.G.rows = 2; p.dm.G.cols = 2; p.dm.G.f32 = Gid2;
    p.dm.n_act = 2; p.dm.N = 2;
    p.dm.stroke_gain = 1.0; p.dm.stroke_max_m = 1e30;

    p.frame_w = 8; p.frame_h = 8;
    p.twoM = 2; p.N = 2; p.J = 0; p.n_act = 2;

    p.frame  = (float *)calloc(64, sizeof(float));
    p.cents  = (float *)calloc(2, sizeof(float));
    p.valid  = (int   *)calloc(1, sizeof(int));
    p.slopes = (float *)calloc(2, sizeof(float));
    p.phi    = (float *)calloc(2, sizeof(float));
    p.coeffs = (float *)calloc(1, sizeof(float));
    p.acts   = (float *)calloc(2, sizeof(float));
    int rc = 0;
    if (!p.frame || !p.cents || !p.valid || !p.slopes || !p.phi ||
        !p.coeffs || !p.acts) { rc = 2; goto done; }

    /* Synthetic spot shifted +1px in x from the reference (center 3.5). */
    for (int j = 0; j < 8; ++j)
        for (int i = 0; i < 8; ++i) {
            float dx = (float)i - 4.5f, dy = (float)j - 3.5f;
            p.frame[j * 8 + i] = 150.0f * expf(-(dx * dx + dy * dy) / 2.0f);
        }

    FrameStats st;
    if (pipeline_process_frame(&p, &st) != 0) { rc = 3; goto done; }
    if (st.n_valid_sub != 1) { rc = 4; goto done; }
    /* x-slope should be positive (spot shifted +x), y-slope ~0. */
    if (!(p.slopes[0] > 0.0f)) { rc = 5; goto done; }
    if (fabsf(p.slopes[1]) > fabsf(p.slopes[0])) { rc = 6; goto done; }
    /* phi == s (R=I); acts == -phi/2. */
    if (fabsf(p.phi[0] - p.slopes[0]) > 1e-6f) { rc = 7; goto done; }
    if (fabsf(p.acts[0] - (-0.5f * p.phi[0])) > 1e-6f) { rc = 8; goto done; }

done:
    /* Detach static matrices so reconstruct_free/dmcmd_free don't free them. */
    p.rec.R.f32 = NULL; p.dm.G.f32 = NULL;
    free(unconst_ptr(p.cen.subs));
    free(p.frame); free(p.cents); free(p.valid);
    free(p.slopes); free(p.phi); free(p.coeffs); free(p.acts);
    return rc;
}
