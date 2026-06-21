/* pipeline.c -- ties stages together for one frame and the frame loop.
 *
 * STUB. Per frame: read -> preprocess -> centroid -> slopes -> reconstruct
 * (zonal + modal) -> dmcmd. Buffers are allocated once (no in-loop alloc).
 * research/06 S10, S14.
 */
#include "pipeline.h"
#include "bmp.h"
#include "slopes.h"
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec * 1e-3;
}

int pipeline_init(Pipeline *p, const char *config_path, const char *calib_dir) {
    if (!p) return 1;
    memset(p, 0, sizeof(*p));

    /* TODO(impl): ao_config_load(config_path,&p->cfg); on failure fall back to
     * defaults. Then reconstruct_load, dmcmd_load, build SubAperture table from
     * subapmap.aomx + refslopes.aomx, and allocate all scratch buffers sized
     * from the loaded dimensions. Below sets up enough to not crash. */
    ao_config_defaults(&p->cfg);
    if (config_path) (void)ao_config_load(config_path, &p->cfg);

    (void)reconstruct_load(calib_dir, &p->rec);
    (void)dmcmd_load(calib_dir,
                     p->cfg.dm.stroke_max_m,
                     p->cfg.dm.stroke_gain_m_per_unit,
                     &p->dm);

    p->frame_w = p->cfg.camera.frame_w;
    p->frame_h = p->cfg.camera.frame_h;
    p->n_sub   = ao_nominal_nsub(&p->cfg);
    p->twoM    = 2 * p->n_sub;
    p->N       = p->rec.N;
    p->J       = p->rec.J;
    p->n_act   = ao_nominal_nact(&p->cfg);

    /* Allocate scratch buffers (guard against zero sizes). */
    size_t fw = (size_t)(p->frame_w > 0 ? p->frame_w : 1);
    size_t fh = (size_t)(p->frame_h > 0 ? p->frame_h : 1);
    p->frame  = (float *)calloc(fw * fh, sizeof(float));
    p->cents  = (float *)calloc((size_t)(p->n_sub > 0 ? 2 * p->n_sub : 2), sizeof(float));
    p->valid  = (int   *)calloc((size_t)(p->n_sub > 0 ? p->n_sub : 1), sizeof(int));
    p->slopes = (float *)calloc((size_t)(p->twoM > 0 ? p->twoM : 1), sizeof(float));
    p->phi    = (float *)calloc((size_t)(p->N > 0 ? p->N : 1), sizeof(float));
    p->coeffs = (float *)calloc((size_t)(p->J > 0 ? p->J : 1), sizeof(float));
    p->acts   = (float *)calloc((size_t)(p->n_act > 0 ? p->n_act : 1), sizeof(float));

    if (!p->frame || !p->cents || !p->valid || !p->slopes ||
        !p->phi || !p->coeffs || !p->acts) {
        pipeline_free(p);
        return 2;
    }
    return 0;
}

void pipeline_free(Pipeline *p) {
    if (!p) return;
    reconstruct_free(&p->rec);
    dmcmd_free(&p->dm);
    free(p->frame); free(p->cents); free(p->valid);
    free(p->slopes); free(p->phi); free(p->coeffs); free(p->acts);
    memset(p, 0, sizeof(*p));
}

int pipeline_process_frame(Pipeline *p, FrameStats *stats) {
    if (!p) return 1;
    FrameStats st; memset(&st, 0, sizeof(st));

    double t0 = now_us();
    /* TODO(impl): preprocess (dark/flat/background). */

    double t1 = now_us();
    st.n_valid_sub = centroid_frame(p->frame, p->frame_w, p->frame_h,
                                    &p->cen, p->cents, p->valid);

    double t2 = now_us();
    int m = slopes_from_centroids(p->cents, p->valid, p->n_sub,
                                  NULL, p->cfg.camera.pixel_size_m,
                                  p->cfg.mla.focal_length_m, NULL, p->slopes);
    (void)m;
    reconstruct_zonal(&p->rec, p->slopes, p->phi);
    reconstruct_modal(&p->rec, p->slopes, p->coeffs);

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
    if (!p) return 1;
    double t0 = now_us();
    size_t cap = (size_t)p->frame_w * p->frame_h;
    int w = 0, h = 0;
    int rc = bmp_read_gray_into(path, p->frame, cap, &w, &h);
    if (rc != 0) return rc;
    rc = pipeline_process_frame(p, stats);
    if (stats) stats->t_read_us += (now_us() - t0); /* include disk read */
    return rc;
}

int pipeline_selftest(void) {
    /* TODO(impl): wire default config + identity-ish matrices end-to-end.
     * Stub passes so the project builds/runs. */
    return 0;
}
