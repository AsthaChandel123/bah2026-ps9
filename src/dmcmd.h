/* dmcmd.h -- DM actuator-map computation (real-time path).
 *
 * The actuator map is the coupling-DECONVOLVED command:
 *   a_dm = G * phi   where  G = H^+ * (-1/2)   (built offline, coupling baked in)
 * NOT a naive sampling of the conjugate wavefront. The factor -1/2 is the
 * reflection (surface displaced by z changes OPD by 2z). Output is then
 * clipped to +/- stroke_max and scaled to stroke-length units by the gain g.
 * See research/05 S3, S9, S11.
 *
 * Optionally a fused matrix K = G*R (N_act x 2M) maps slopes directly to
 * commands in a single gemv (a_dm = K * s); used when the phase map output is
 * not needed on that path.
 */
#ifndef DMCMD_H
#define DMCMD_H

#include "matio.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    AOMatrix G;            /* DM command matrix (N_act x N), = H^+ * (-1/2) */
    AOMatrix K;            /* fused slopes->commands (N_act x 2M), optional */
    int      has_fused;
    int      n_act;        /* number of actuators */
    int      N;            /* phase points (input to G) */
    double   stroke_max_m; /* clip limit a_max (m) */
    double   stroke_gain;  /* g: meters of surface per unit command */
} DmCommander;

/* Load DM command matrix/matrices from a calibration directory (expects
 * G.aomx, optionally K.aomx) and stroke params from the config. Returns 0.
 *
 * TODO(impl): aomx_read G (and K if present); copy stroke_max/gain from cfg. */
int dmcmd_load(const char *calib_dir, double stroke_max_m, double stroke_gain,
               DmCommander *dm);

void dmcmd_free(DmCommander *dm);

/* Compute actuator map from a reconstructed phase: a_dm[N_act] = G * phi[N],
 * then clip to +/- stroke_max and scale by gain into stroke-length units.
 * (G already encodes the -1/2 reflection and the coupling deconvolution.)
 * Returns the number of actuators that saturated (clipped).
 *
 * TODO(impl): gemv (G, phi) -> a_dm; clip_f32; multiply by gain; count clips. */
int dmcmd_from_phase(const DmCommander *dm, const float *phi, float *a_dm);

/* Fused path: a_dm[N_act] = K * s[2M], then clip + scale. Requires has_fused.
 * TODO(impl). */
int dmcmd_from_slopes(const DmCommander *dm, const float *s, float *a_dm);

/* Self-test: a target phase recovered through G should produce commands whose
 * re-applied surface (H * a) approximates -phi/2. Returns 0. (Stub returns 0.) */
int dmcmd_selftest(void);

#ifdef __cplusplus
}
#endif

#endif /* DMCMD_H */
