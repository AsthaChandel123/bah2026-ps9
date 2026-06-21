/* main.c -- CLI entry point for the PS9 real-time core (wfs_rt).
 *
 * Usage:
 *   wfs_rt --selftest
 *   wfs_rt --config <cfg> --calib <dir> --frames <glob|f1 [f2 ...]> --out <dir>
 *
 * --selftest runs the built-in roundtrips (AOMX, BMP, GEMV) + stage self-tests
 * and is what `make test` invokes. The normal path loads config + AOMX
 * matrices, then runs the pipeline over the listed/globbed frames, writing:
 *   <out>/phase_<NNNN>.aomx    reconstructed wavefront (N x 1, radians)
 *   <out>/zernike_coeffs.csv   frame_idx,j2,...,jJ  (Noll order, radians)
 *   <out>/actuators.csv        frame_idx,a0,...,a(N_act-1)  (stroke meters)
 *   <out>/actuators_<NNNN>.csv act_idx,stroke_m  (per-frame, Deliverable 3)
 *   <out>/slopes.csv           frame_idx,s0,...,s(2M-1)  (for turbulence)
 *   <out>/timing_report.txt    per-stage min/mean/max us + FPS
 *
 * Frame arguments after --frames may be an explicit space-separated list or a
 * single shell-style glob (e.g. "data/run01/frame_*.bmp"); the glob is
 * expanded with glob(3). Robust to missing matrices/frames: prints a helpful
 * error and exits non-zero.
 *
 * Also defines ao_config_defaults / ao_config_load (declared in aoconfig.h).
 * The C core avoids a JSON dependency: ao_config_load parses a flat key=value
 * sidecar if present (config.txt-style), else falls back to defaults.
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
#include <sys/stat.h>
#include <sys/types.h>
#include <glob.h>

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

/* Trim leading/trailing whitespace in place; returns the trimmed start. */
static char *trim(char *s) {
    while (*s == ' ' || *s == '\t' || *s == '\r' || *s == '\n') ++s;
    char *e = s + strlen(s);
    while (e > s && (e[-1] == ' ' || e[-1] == '\t' || e[-1] == '\r' || e[-1] == '\n'))
        *--e = '\0';
    return s;
}

/* Apply a single flat "key value" (or "key=value") pair to the config. Keys use
 * a flat dotted namespace (camera.pixel_size_m, mla.focal_length_m, ...) so the
 * Python calibration step can emit a sidecar with no JSON parser needed. */
static void apply_kv(AOConfig *c, const char *key, const char *val) {
    double d = atof(val);
    int    i = atoi(val);
    if      (!strcmp(key, "camera.pixel_size_m")) c->camera.pixel_size_m = d;
    else if (!strcmp(key, "camera.frame_w"))      c->camera.frame_w = i;
    else if (!strcmp(key, "camera.frame_h"))      c->camera.frame_h = i;
    else if (!strcmp(key, "camera.bit_depth"))    c->camera.bit_depth = i;
    else if (!strcmp(key, "mla.n_lenslets_x"))    c->mla.n_lenslets_x = i;
    else if (!strcmp(key, "mla.n_lenslets_y"))    c->mla.n_lenslets_y = i;
    else if (!strcmp(key, "mla.pitch_m"))         c->mla.pitch_m = d;
    else if (!strcmp(key, "mla.focal_length_m"))  c->mla.focal_length_m = d;
    else if (!strcmp(key, "pupil.diameter_m"))    c->pupil.diameter_m = d;
    else if (!strcmp(key, "pupil.center_x_px"))   c->pupil.center_x_px = d;
    else if (!strcmp(key, "pupil.center_y_px"))   c->pupil.center_y_px = d;
    else if (!strcmp(key, "dm.n_act_x"))          c->dm.n_act_x = i;
    else if (!strcmp(key, "dm.n_act_y"))          c->dm.n_act_y = i;
    else if (!strcmp(key, "dm.pitch_m"))          c->dm.pitch_m = d;
    else if (!strcmp(key, "dm.coupling_coeff"))   c->dm.coupling_coeff = d;
    else if (!strcmp(key, "dm.stroke_max_m"))     c->dm.stroke_max_m = d;
    else if (!strcmp(key, "dm.influence_model"))  c->dm.influence_model = i;
    else if (!strcmp(key, "dm.influence_alpha"))  c->dm.influence_alpha = d;
    else if (!strcmp(key, "dm.stroke_gain_m_per_unit")) c->dm.stroke_gain_m_per_unit = d;
    else if (!strcmp(key, "geometry.fried"))      c->sys.geometry_fried = i;
    else if (!strcmp(key, "geometry.rotation_deg")) c->sys.rotation_deg = d;
    else if (!strcmp(key, "geometry.flip_y"))     c->sys.flip_y = i;
    else if (!strcmp(key, "cadence.dt_s"))        c->sys.dt_s = d;
    else if (!strcmp(key, "wavelength_m"))        c->sys.wavelength_m = d;
    /* unknown keys are ignored (forward-compatible) */
}

int ao_config_load(const char *path, AOConfig *cfg) {
    if (!cfg) return 1;
    ao_config_defaults(cfg); /* start from defaults; sidecar overrides */
    if (!path) return 0;

    FILE *f = fopen(path, "r");
    if (!f) {
        /* Not fatal: caller may pass a JSON path the C core can't parse; we
         * keep defaults and let the operator know. */
        fprintf(stderr, "ao_config_load: cannot open '%s'; using defaults\n", path);
        return 0;
    }
    char line[512];
    while (fgets(line, sizeof(line), f)) {
        char *s = trim(line);
        if (*s == '\0' || *s == '#' || *s == '/' || *s == '{' || *s == '}') continue;
        /* accept "key=value" or "key value" */
        char *sep = strpbrk(s, "=\t ");
        if (!sep) continue;
        char saved = *sep;
        *sep = '\0';
        char *key = trim(s);
        char *val = trim(sep + 1);
        (void)saved;
        if (*key && *val) apply_kv(cfg, key, val);
    }
    fclose(f);
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
        "  %s --config <cfg> --calib <dir> --frames <glob|f1 [f2 ...]> --out <dir>\n"
        "\n"
        "  --config  flat key=value sidecar (optional; defaults used otherwise)\n"
        "  --calib   directory with R.aomx G.aomx subapmap.aomx [Mpinv.aomx Z.aomx\n"
        "            refslopes.aomx K.aomx] (required)\n"
        "  --frames  one shell glob or an explicit list of .bmp frames (required)\n"
        "  --out     output directory (default '.')\n",
        prog, prog);
}

/* ----- per-stage timing accumulator ----- */

typedef struct {
    double minv, maxv, sum;
    long   n;
} Stat;

static void stat_init(Stat *s) { s->minv = 1e300; s->maxv = 0.0; s->sum = 0.0; s->n = 0; }
static void stat_add(Stat *s, double v) {
    if (v < s->minv) s->minv = v;
    if (v > s->maxv) s->maxv = v;
    s->sum += v; s->n++;
}
static double stat_mean(const Stat *s) { return s->n ? s->sum / (double)s->n : 0.0; }

/* ----- frame list assembly (glob or explicit args) ----- */

typedef struct { char **paths; int count; glob_t gl; int used_glob; } FrameList;

static int build_frame_list(FrameList *fl, char **argv, int start, int argc) {
    memset(fl, 0, sizeof(*fl));
    int n = 0;
    for (int i = start; i < argc; ++i) {
        if (!strncmp(argv[i], "--", 2)) break;
        ++n;
    }
    if (n == 0) return 1;

    /* If exactly one argument and it contains glob metacharacters, expand it. */
    if (n == 1 && strpbrk(argv[start], "*?[")) {
        if (glob(argv[start], 0, NULL, &fl->gl) != 0 || fl->gl.gl_pathc == 0) {
            globfree(&fl->gl);
            return 2; /* no match */
        }
        fl->paths = fl->gl.gl_pathv;
        fl->count = (int)fl->gl.gl_pathc;
        fl->used_glob = 1;
        return 0;
    }
    fl->paths = &argv[start];
    fl->count = n;
    fl->used_glob = 0;
    return 0;
}

static void free_frame_list(FrameList *fl) {
    if (fl->used_glob) globfree(&fl->gl);
}

/* ----- output writers ----- */

static int write_csv_row_open(FILE **fp, const char *path, const char *header) {
    FILE *f = fopen(path, "w");
    if (!f) { fprintf(stderr, "error: cannot open %s for writing\n", path); return 1; }
    if (header) fputs(header, f);
    *fp = f;
    return 0;
}

static void write_csv_row(FILE *f, int frame_idx, const float *v, int n) {
    fprintf(f, "%d", frame_idx);
    for (int i = 0; i < n; ++i) fprintf(f, ",%.8g", (double)v[i]);
    fputc('\n', f);
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
            frames_start = i + 1;
        }
    }

    if (!calib_dir) {
        fprintf(stderr, "error: --calib <dir> is required (or use --selftest)\n");
        usage(argv[0]);
        return 2;
    }
    if (frames_start < 0) {
        fprintf(stderr, "error: --frames <glob|f1 ...> is required\n");
        usage(argv[0]);
        return 2;
    }

    FrameList fl;
    int frc = build_frame_list(&fl, argv, frames_start, argc);
    if (frc != 0) {
        fprintf(stderr, "error: no frames matched after --frames\n");
        usage(argv[0]);
        return 2;
    }

    /* Ensure output directory exists (ignore EEXIST). */
    if (mkdir(out_dir, 0755) != 0) {
        /* mkdir failing is fine if it already exists; we detect real failures
         * lazily when opening output files below. */
    }

    Pipeline p;
    if (pipeline_init(&p, config_path, calib_dir) != 0) {
        fprintf(stderr, "error: pipeline_init failed (check --calib '%s' has "
                "R.aomx, G.aomx, subapmap.aomx)\n", calib_dir);
        free_frame_list(&fl);
        return 1;
    }

    printf("wfs_rt: config=%s calib=%s out=%s\n",
           config_path ? config_path : "(defaults)", calib_dir, out_dir);
    printf("  frame=%dx%d n_sub=%d 2M=%d N=%d J=%d n_act=%d  (frames=%d)\n",
           p.frame_w, p.frame_h, p.n_sub, p.twoM, p.N, p.J, p.n_act, fl.count);

    /* Open aggregate CSVs. */
    char path[1024];
    FILE *fz = NULL, *fa = NULL, *fs = NULL;
    int csv_ok = 1;

    snprintf(path, sizeof(path), "%s/zernike_coeffs.csv", out_dir);
    {
        char hdr[1024]; int off = snprintf(hdr, sizeof(hdr), "frame_idx");
        for (int j = 0; j < p.J && off < (int)sizeof(hdr) - 16; ++j)
            off += snprintf(hdr + off, sizeof(hdr) - off, ",j%d", j + 2); /* Noll j>=2 */
        snprintf(hdr + off, sizeof(hdr) - off, "\n");
        if (write_csv_row_open(&fz, path, p.J > 0 ? hdr : "frame_idx\n") != 0) csv_ok = 0;
    }
    snprintf(path, sizeof(path), "%s/actuators.csv", out_dir);
    {
        char hdr[2048]; int off = snprintf(hdr, sizeof(hdr), "frame_idx");
        for (int a = 0; a < p.n_act && off < (int)sizeof(hdr) - 16; ++a)
            off += snprintf(hdr + off, sizeof(hdr) - off, ",a%d", a);
        snprintf(hdr + off, sizeof(hdr) - off, "\n");
        if (write_csv_row_open(&fa, path, hdr) != 0) csv_ok = 0;
    }
    snprintf(path, sizeof(path), "%s/slopes.csv", out_dir);
    {
        char hdr[2048]; int off = snprintf(hdr, sizeof(hdr), "frame_idx");
        for (int s = 0; s < p.twoM && off < (int)sizeof(hdr) - 16; ++s)
            off += snprintf(hdr + off, sizeof(hdr) - off, ",s%d", s);
        snprintf(hdr + off, sizeof(hdr) - off, "\n");
        if (write_csv_row_open(&fs, path, hdr) != 0) csv_ok = 0;
    }
    if (!csv_ok) {
        if (fz) fclose(fz);
        if (fa) fclose(fa);
        if (fs) fclose(fs);
        pipeline_free(&p);
        free_frame_list(&fl);
        return 1;
    }

    Stat s_read, s_cen, s_rec, s_dm, s_tot;
    stat_init(&s_read); stat_init(&s_cen); stat_init(&s_rec);
    stat_init(&s_dm); stat_init(&s_tot);

    int nframes = 0, nfail = 0;
    long total_sat = 0, total_valid = 0;

    for (int k = 0; k < fl.count; ++k) {
        FrameStats st;
        int rc = pipeline_run_bmp(&p, fl.paths[k], &st);
        if (rc != 0) {
            fprintf(stderr, "  warn: frame '%s' failed (rc=%d)\n", fl.paths[k], rc);
            ++nfail;
            continue;
        }

        /* Per-frame phase map (Deliverable 1). */
        snprintf(path, sizeof(path), "%s/phase_%04d.aomx", out_dir, nframes);
        if (aomx_write_f32(path, (uint32_t)p.N, 1u, p.phi) != 0)
            fprintf(stderr, "  warn: could not write %s\n", path);

        /* Per-frame actuator map CSV (Deliverable 3, stroke meters). */
        snprintf(path, sizeof(path), "%s/actuators_%04d.csv", out_dir, nframes);
        FILE *fac = fopen(path, "w");
        if (fac) {
            fputs("act_idx,stroke_m\n", fac);
            for (int a = 0; a < p.n_act; ++a)
                fprintf(fac, "%d,%.8g\n", a, (double)p.acts[a]);
            fclose(fac);
        }

        /* Aggregate CSVs. */
        if (p.J > 0) write_csv_row(fz, nframes, p.coeffs, p.J);
        else         fprintf(fz, "%d\n", nframes);
        write_csv_row(fa, nframes, p.acts, p.n_act);
        write_csv_row(fs, nframes, p.slopes, p.twoM);

        stat_add(&s_read, st.t_read_us);
        stat_add(&s_cen,  st.t_centroid_us);
        stat_add(&s_rec,  st.t_recon_us);
        stat_add(&s_dm,   st.t_dm_us);
        stat_add(&s_tot,  st.t_total_us);
        total_sat   += st.n_saturated_act;
        total_valid += st.n_valid_sub;
        ++nframes;
    }

    fclose(fz); fclose(fa); fclose(fs);

    /* Timing report. */
    if (nframes > 0) {
        double mean_tot = stat_mean(&s_tot);
        double fps = (mean_tot > 0.0) ? 1e6 / mean_tot : 0.0;

        char rpt[2048];
        int o = 0;
        o += snprintf(rpt + o, sizeof(rpt) - o,
            "wfs_rt timing report\n"
            "====================\n"
            "frames processed : %d (failed %d)\n"
            "grid             : frame %dx%d, n_sub=%d, 2M=%d, N=%d, J=%d, n_act=%d\n"
            "mean valid subaps: %.1f / %d   mean saturated acts: %.1f / %d\n"
            "\n"
            "per-stage latency (microseconds)   min      mean      max\n"
            "  read+decode                   %8.2f %8.2f %8.2f\n"
            "  centroid (TWCoG)              %8.2f %8.2f %8.2f\n"
            "  reconstruct (R*s, M+*s)       %8.2f %8.2f %8.2f\n"
            "  dm command (G*phi)            %8.2f %8.2f %8.2f\n"
            "  TOTAL per frame               %8.2f %8.2f %8.2f\n"
            "\n"
            "throughput       : %.0f FPS (mean total %.2f us)\n"
            "budget           : 10000 us  -> %.0fx margin (mean), %.0fx (worst)\n",
            nframes, nfail,
            p.frame_w, p.frame_h, p.n_sub, p.twoM, p.N, p.J, p.n_act,
            (double)total_valid / nframes, p.n_sub,
            (double)total_sat / nframes, p.n_act,
            s_read.minv, stat_mean(&s_read), s_read.maxv,
            s_cen.minv,  stat_mean(&s_cen),  s_cen.maxv,
            s_rec.minv,  stat_mean(&s_rec),  s_rec.maxv,
            s_dm.minv,   stat_mean(&s_dm),   s_dm.maxv,
            s_tot.minv,  stat_mean(&s_tot),  s_tot.maxv,
            fps, mean_tot,
            mean_tot > 0 ? 10000.0 / mean_tot : 0.0,
            s_tot.maxv > 0 ? 10000.0 / s_tot.maxv : 0.0);

        fputs(rpt, stdout);

        snprintf(path, sizeof(path), "%s/timing_report.txt", out_dir);
        FILE *ft = fopen(path, "w");
        if (ft) { fputs(rpt, ft); fclose(ft); }
    } else {
        printf("  no frames processed successfully.\n");
    }

    pipeline_free(&p);
    free_frame_list(&fl);
    return (nframes > 0) ? 0 : 1;
}
