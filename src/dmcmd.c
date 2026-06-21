/* dmcmd.c -- DM actuator-map computation.
 *
 * STUB for loading; the command MVM + clip + gain are wired so that once G is
 * loaded the actuator map is produced. G = H^+ * (-1/2) (coupling baked in,
 * reflection factor included). research/05 S3, S9, S11.
 */
#include "dmcmd.h"
#include "linalg.h"
#include <string.h>

int dmcmd_load(const char *calib_dir, double stroke_max_m, double stroke_gain,
               DmCommander *dm) {
    /* TODO(impl): aomx_read "<calib_dir>/G.aomx" (and optional "K.aomx");
     * set n_act, N; copy stroke params. Stub zeroes + stores params. */
    (void)calib_dir;
    if (!dm) return 1;
    memset(dm, 0, sizeof(*dm));
    dm->stroke_max_m = stroke_max_m;
    dm->stroke_gain = stroke_gain;
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
    /* clip to +/- stroke_max (in command units == stroke units pre-gain) */
    float lim = (float)(dm->stroke_max_m != 0.0 ? dm->stroke_max_m : 1e30);
    int clipped = 0;
    for (int i = 0; i < (int)dm->G.rows; ++i) {
        if (a_dm[i] > lim) { a_dm[i] = lim; ++clipped; }
        else if (a_dm[i] < -lim) { a_dm[i] = -lim; ++clipped; }
    }
    /* scale to stroke-length units */
    if (dm->stroke_gain != 0.0 && dm->stroke_gain != 1.0) {
        float g = (float)dm->stroke_gain;
        for (int i = 0; i < (int)dm->G.rows; ++i) a_dm[i] *= g;
    }
    return clipped;
}

int dmcmd_from_slopes(const DmCommander *dm, const float *s, float *a_dm) {
    if (!dm || !dm->has_fused || !dm->K.f32 || !s || !a_dm) return 0;
    gemv_f32(dm->K.f32, s, a_dm, (int)dm->K.rows, (int)dm->K.cols);
    /* TODO(impl): clip + scale as in dmcmd_from_phase. */
    return 0;
}

int dmcmd_selftest(void) {
    /* TODO(impl): H * (G * phi) ~ -phi/2. Stub passes. */
    return 0;
}
