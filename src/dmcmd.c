/* dmcmd.c -- DM actuator-map computation (real-time path).
 *
 * The actuator map is the coupling-DECONVOLVED command, NOT a naive sampling
 * of the conjugate wavefront:
 *   a_dm = G * phi     where  G = H^+ * (-1/2)   (coupling + reflection baked in)
 * The factor -1/2 is the reflection (a surface displaced by z changes the OPD
 * by 2z). research/05 S3, S9, S11; ARCHITECTURE.md S3.6.
 *
 * Output is in actuator-stroke-LENGTH units (meters):
 *   1. a_dm = G * phi                       (gemv: command units)
 *   2. a_dm *= stroke_gain                  (meters of surface per unit command)
 *   3. a_dm = clip(a_dm, +/- stroke_max_m)  (saturation in meters)
 * Scaling precedes clipping so the saturation limit (a length, stroke_max_m)
 * is applied in the same units as the value being clipped.
 *
 * The optional fused matrix K = G*R (N_act x 2M) maps slopes directly to
 * commands in a single gemv (used when the phase map is not also required).
 */
#include "dmcmd.h"
#include "linalg.h"
#include <math.h>
#include <stdio.h>
#include <string.h>

static int join_path(char *dst, size_t dst_sz, const char *dir, const char *name) {
    if (!dst || dst_sz == 0) return 1;
    const char *sep = "/";
    size_t dl = dir ? strlen(dir) : 0;
    if (dl > 0 && (dir[dl - 1] == '/')) sep = "";
    int n = snprintf(dst, dst_sz, "%s%s%s", dir ? dir : ".", sep, name);
    return (n < 0 || (size_t)n >= dst_sz) ? 1 : 0;
}

/* Scale by gain into stroke-length units, then clip to +/- stroke_max.
 * Returns the number of saturated actuators. */
static int scale_and_clip(float *a, int n, double gain, double stroke_max) {
    float g = (float)((gain != 0.0) ? gain : 1.0);
    if (g != 1.0f) {
        for (int i = 0; i < n; ++i) a[i] *= g;
    }
    if (!(stroke_max > 0.0)) return 0; /* no/invalid limit -> no clipping */
    float lim = (float)stroke_max;
    int clipped = 0;
    for (int i = 0; i < n; ++i) {
        if (a[i] > lim)       { a[i] =  lim; ++clipped; }
        else if (a[i] < -lim) { a[i] = -lim; ++clipped; }
    }
    return clipped;
}

int dmcmd_load(const char *calib_dir, double stroke_max_m, double stroke_gain,
               DmCommander *dm) {
    if (!dm) return 1;
    memset(dm, 0, sizeof(*dm));
    dm->stroke_max_m = stroke_max_m;
    dm->stroke_gain  = stroke_gain;

    char path[1024];

    /* G is mandatory for the phase->commands path. */
    if (join_path(path, sizeof(path), calib_dir, "G.aomx") != 0) return 2;
    if (aomx_read(path, &dm->G) != 0) {
        fprintf(stderr, "dmcmd_load: cannot read %s\n", path);
        return 3;
    }
    dm->n_act = (int)dm->G.rows;
    dm->N     = (int)dm->G.cols;

    /* K is optional (fused slopes->commands). */
    if (join_path(path, sizeof(path), calib_dir, "K.aomx") == 0 &&
        aomx_read(path, &dm->K) == 0) {
        if ((int)dm->K.rows != dm->n_act) {
            fprintf(stderr,
                "dmcmd_load: K rows (%u) != N_act (%d); ignoring fused path\n",
                dm->K.rows, dm->n_act);
            aomx_free(&dm->K);
            dm->has_fused = 0;
        } else {
            dm->has_fused = 1;
        }
    }
    return 0;
}

void dmcmd_free(DmCommander *dm) {
    if (!dm) return;
    aomx_free(&dm->G);
    aomx_free(&dm->K);
    memset(dm, 0, sizeof(*dm));
}

int dmcmd_from_phase(const DmCommander *dm, const float *phi, float *a_dm) {
    if (!dm || !dm->G.f32 || !phi || !a_dm) return 0;
    gemv_f32(dm->G.f32, phi, a_dm, (int)dm->G.rows, (int)dm->G.cols);
    return scale_and_clip(a_dm, (int)dm->G.rows, dm->stroke_gain, dm->stroke_max_m);
}

int dmcmd_from_slopes(const DmCommander *dm, const float *s, float *a_dm) {
    if (!dm || !dm->has_fused || !dm->K.f32 || !s || !a_dm) return 0;
    gemv_f32(dm->K.f32, s, a_dm, (int)dm->K.rows, (int)dm->K.cols);
    return scale_and_clip(a_dm, (int)dm->K.rows, dm->stroke_gain, dm->stroke_max_m);
}

/* ---- self-test ----
 * With G = -1/2 * I and H = I (so H*(G*phi) == -phi/2), a unit gain and a
 * generous stroke limit, the produced commands must equal -phi/2; with a
 * tight limit, the large entry must saturate. */
int dmcmd_selftest(void) {
    DmCommander dm;
    memset(&dm, 0, sizeof(dm));
    static float Gid[9] = {-0.5f, 0, 0, 0, -0.5f, 0, 0, 0, -0.5f};
    dm.G.rows = 3; dm.G.cols = 3; dm.G.f32 = Gid;
    dm.n_act = 3; dm.N = 3;
    dm.stroke_gain = 1.0;
    dm.stroke_max_m = 1e30; /* effectively no clip */

    float phi[3] = {2.0f, -4.0f, 1.0f};
    float a[3]   = {0, 0, 0};
    int sat = dmcmd_from_phase(&dm, phi, a);
    if (sat != 0) { dm.G.f32 = NULL; return 1; }
    for (int i = 0; i < 3; ++i)
        if (fabsf(a[i] - (-0.5f * phi[i])) > 1e-6f) { dm.G.f32 = NULL; return 2; }

    /* Tight clip: limit 1.0 -> phi[1] = -4 maps to +2.0, saturates to +1.0. */
    dm.stroke_max_m = 1.0;
    sat = dmcmd_from_phase(&dm, phi, a);
    if (sat != 1) { dm.G.f32 = NULL; return 3; }
    if (fabsf(a[1] - 1.0f) > 1e-6f) { dm.G.f32 = NULL; return 4; }
    /* unsaturated entries unchanged */
    if (fabsf(a[0] - (-1.0f)) > 1e-6f) { dm.G.f32 = NULL; return 5; }

    /* Gain scaling: gain 2.0, generous limit -> a == 2*(-phi/2) == -phi. */
    dm.stroke_gain = 2.0;
    dm.stroke_max_m = 1e30;
    sat = dmcmd_from_phase(&dm, phi, a);
    if (sat != 0) { dm.G.f32 = NULL; return 6; }
    for (int i = 0; i < 3; ++i)
        if (fabsf(a[i] - (-phi[i])) > 1e-6f) { dm.G.f32 = NULL; return 7; }

    dm.G.f32 = NULL; /* static buffer; do not free */
    return 0;
}
