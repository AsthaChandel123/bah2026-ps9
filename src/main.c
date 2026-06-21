/* main.c -- CLI entry point for the PS9 real-time core (wfs_rt).
 *
 * Usage:
 *   wfs_rt --selftest
 *   wfs_rt --config <cfg> --calib <dir> --frames "<glob-or-list>" --out <dir>
 *
 * --selftest runs the built-in roundtrips (AOMX, BMP, GEMV) + stage self-tests
 * and is what `make test` invokes. The normal path loads config + AOMX
 * matrices, then runs the pipeline over the listed frames, writing outputs and
 * timing. Frame globbing is intentionally minimal here (stub): pass an
 * explicit space-separated list after --frames, or extend with glob(3).
 *
 * Also defines ao_config_defaults / ao_config_load (declared in aoconfig.h).
 * The C core avoids a JSON dependency: ao_config_load currently falls back to
 * defaults; implementers wire a flat key=value sidecar emitted by the Python
 * calibration step (ARCHITECTURE.md S4.1, src/aoconfig.h).
 */
#include "aoconfig.h"
#include "matio.h"
#include "bmp.h"
#include "linalg.h"
#include "centroid.h"
#include "slopes.h"
#include "reconstruct.h"
#include "dmcmd.h"
#include "pipeline.h"

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

/* ----- config (defined here; declared in aoconfig.h) ----- */

void ao_config_defaults(AOConfig *cfg) {
    if (!cfg) return;
    memset(cfg, 0, sizeof(*cfg));
    cfg->schema_version = 1;
    cfg->camera.pixel_size_m = 5.5e-6;
    cfg->camera.frame_w = 256;
    cfg->camera.frame_h = 256;
    cfg->camera.bit_depth = 8;
    cfg->mla.n_lenslets_x = 10;
    cfg->mla.n_lenslets_y = 10;
    cfg->mla.pitch_m = 1.5e-4;
    cfg->mla.focal_length_m = 5.2e-3;
    cfg->pupil.diameter_m = 1.5e-3;
    cfg->pupil.center_x_px = 128.0;
    cfg->pupil.center_y_px = 128.0;
    cfg->dm.n_act_x = 11;
    cfg->dm.n_act_y = 11;
    cfg->dm.pitch_m = 1.5e-4;
    cfg->dm.coupling_coeff = 0.15;
    cfg->dm.stroke_max_m = 3.5e-6;
    cfg->dm.influence_model = AO_IF_GAUSSIAN;
    cfg->dm.influence_alpha = 2.0;
    cfg->dm.stroke_gain_m_per_unit = 1.0e-6;
    cfg->sys.geometry_fried = 1;
    cfg->sys.rotation_deg = 0.0;
    cfg->sys.flip_y = 0;
    cfg->sys.dt_s = 2.0e-3;
    cfg->sys.wavelength_m = 6.33e-7;
}

int ao_config_load(const char *path, AOConfig *cfg) {
    /* TODO(impl): parse a flat key=value sidecar (or restricted-JSON subset)
     * written by scripts/build_calibration.py. For now, populate defaults so
     * the binary runs end-to-end; return 0 (success). */
    (void)path;
    ao_config_defaults(cfg);
    return 0;
}

/* ----- self-test ----- */

static int run_selftest(void) {
    int rc = 0, t;
    printf("wfs_rt selftest:\n");

    t = linalg_selftest();     printf("  linalg     : %s (%d)\n", t ? "FAIL" : "ok", t); rc |= (t != 0);
    t = aomx_selftest();       printf("  matio/AOMX : %s (%d)\n", t ? "FAIL" : "ok", t); rc |= (t != 0);
    t = bmp_selftest();        printf("  bmp        : %s (%d)\n", t ? "FAIL" : "ok", t); rc |= (t != 0);
    t = centroid_selftest();   printf("  centroid   : %s (%d)\n", t ? "FAIL" : "ok", t); rc |= (t != 0);
    t = slopes_selftest();     printf("  slopes     : %s (%d)\n", t ? "FAIL" : "ok", t); rc |= (t != 0);
    t = reconstruct_selftest();printf("  reconstruct: %s (%d)\n", t ? "FAIL" : "ok", t); rc |= (t != 0);
    t = dmcmd_selftest();      printf("  dmcmd      : %s (%d)\n", t ? "FAIL" : "ok", t); rc |= (t != 0);
    t = pipeline_selftest();   printf("  pipeline   : %s (%d)\n", t ? "FAIL" : "ok", t); rc |= (t != 0);

    printf("selftest %s\n", rc ? "FAILED" : "PASSED");
    return rc;
}

static void usage(const char *prog) {
    fprintf(stderr,
        "Usage:\n"
        "  %s --selftest\n"
        "  %s --config <cfg> --calib <dir> --frames <f1> [f2 ...] --out <dir>\n",
        prog, prog);
}

int main(int argc, char **argv) {
    if (argc < 2) { usage(argv[0]); return 2; }

    const char *config_path = NULL;
    const char *calib_dir = NULL;
    const char *out_dir = ".";
    int frames_start = -1;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--selftest") == 0) {
            return run_selftest();
        } else if (strcmp(argv[i], "--config") == 0 && i + 1 < argc) {
            config_path = argv[++i];
        } else if (strcmp(argv[i], "--calib") == 0 && i + 1 < argc) {
            calib_dir = argv[++i];
        } else if (strcmp(argv[i], "--out") == 0 && i + 1 < argc) {
            out_dir = argv[++i];
        } else if (strcmp(argv[i], "--frames") == 0) {
            frames_start = i + 1;   /* remaining args (until next flag) are frames */
        } else if (strncmp(argv[i], "--", 2) == 0) {
            /* stop frame collection at the next flag */
            if (frames_start >= 0 && i > frames_start) { /* already collected */ }
        }
    }

    if (!calib_dir) {
        fprintf(stderr, "error: --calib <dir> is required (or use --selftest)\n");
        usage(argv[0]);
        return 2;
    }

    Pipeline p;
    if (pipeline_init(&p, config_path, calib_dir) != 0) {
        fprintf(stderr, "error: pipeline_init failed\n");
        return 1;
    }

    printf("wfs_rt: config=%s calib=%s out=%s\n",
           config_path ? config_path : "(defaults)", calib_dir, out_dir);
    printf("  frame=%dx%d n_sub=%d 2M=%d N=%d J=%d n_act=%d\n",
           p.frame_w, p.frame_h, p.n_sub, p.twoM, p.N, p.J, p.n_act);

    /* Process listed frames. TODO(impl): glob expansion; write phase maps,
     * Zernike CSV, actuator maps, and a timing summary into out_dir. */
    int nframes = 0;
    double tsum = 0.0, tmax = 0.0;
    if (frames_start >= 0) {
        for (int i = frames_start; i < argc; ++i) {
            if (strncmp(argv[i], "--", 2) == 0) break;
            FrameStats st;
            int rc = pipeline_run_bmp(&p, argv[i], &st);
            if (rc != 0) {
                fprintf(stderr, "  warn: frame '%s' failed (rc=%d)\n", argv[i], rc);
                continue;
            }
            ++nframes;
            tsum += st.t_total_us;
            if (st.t_total_us > tmax) tmax = st.t_total_us;
        }
    }

    if (nframes > 0) {
        printf("  processed %d frames: mean %.1f us, max %.1f us (budget 10000 us)\n",
               nframes, tsum / nframes, tmax);
    } else {
        printf("  no frames processed (provide --frames f1 [f2 ...]).\n");
    }

    pipeline_free(&p);
    return 0;
}
