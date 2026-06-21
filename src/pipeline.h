/* pipeline.h -- ties stages together for one frame and the frame loop.
 *
 * Per frame:  bmp_read -> preprocess -> centroid_frame -> slopes ->
 *             reconstruct_zonal (W) + reconstruct_modal (a) -> dmcmd (A).
 * The loop reuses pre-allocated buffers (zero in-loop allocation) and times
 * each frame to verify the < 10 ms budget (research/06 S10, S14).
 */
#ifndef PIPELINE_H
#define PIPELINE_H

#include "aoconfig.h"
#include "centroid.h"
#include "reconstruct.h"
#include "dmcmd.h"

#ifdef __cplusplus
extern "C" {
#endif

/* All state and pre-sized buffers for the real-time loop. Allocate once with
 * pipeline_init, free with pipeline_free. */
typedef struct {
    AOConfig       cfg;
    CentroidCfg    cen;
    Reconstructor  rec;
    DmCommander    dm;

    /* scratch buffers (sized at init; never reallocated in the loop) */
    float *frame;     /* frame_w*frame_h floats */
    float *cents;     /* 2*n_sub */
    int   *valid;     /* n_sub */
    float *slopes;    /* 2M */
    float *phi;       /* N (phase map) */
    float *coeffs;    /* J (Zernike) */
    float *acts;      /* N_act (actuator map, stroke units) */

    int    frame_w, frame_h;
    int    n_sub, twoM, N, J, n_act;
} Pipeline;

/* Per-frame timing/diagnostics. */
typedef struct {
    double t_read_us;
    double t_centroid_us;
    double t_recon_us;
    double t_dm_us;
    double t_total_us;
    int    n_valid_sub;
    int    n_saturated_act;
} FrameStats;

/* Initialize the pipeline from a config and a calibration directory.
 * Loads all AOMX matrices, builds the sub-aperture table, and allocates all
 * scratch buffers. Returns 0 on success.
 *
 * TODO(impl): ao_config_load, reconstruct_load, dmcmd_load, build SubAperture
 * table from subapmap.aomx + refslopes.aomx, allocate buffers. */
int pipeline_init(Pipeline *p, const char *config_path, const char *calib_dir);

void pipeline_free(Pipeline *p);

/* Process a single frame already loaded into p->frame (frame_w*frame_h).
 * Fills p->phi, p->coeffs, p->acts and *stats. Returns 0 on success.
 *
 * TODO(impl): centroid_frame -> slopes_from_centroids -> reconstruct_zonal +
 * reconstruct_modal -> dmcmd_from_phase; fill timings. */
int pipeline_process_frame(Pipeline *p, FrameStats *stats);

/* Read a BMP from `path` into p->frame then process it. Returns 0 on success.
 * TODO(impl): bmp_read_gray_into(p->frame...) then pipeline_process_frame. */
int pipeline_run_bmp(Pipeline *p, const char *path, FrameStats *stats);

/* Self-test of the wired pipeline with default config + identity-ish matrices.
 * Returns 0 on success. (Stub returns 0.) */
int pipeline_selftest(void);

#ifdef __cplusplus
}
#endif

#endif /* PIPELINE_H */
