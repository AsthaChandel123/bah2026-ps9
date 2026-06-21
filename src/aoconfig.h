/* aoconfig.h -- shared configuration struct and loader interface for the
 * PS9 self-contained C real-time core.
 *
 * Mirrors the JSON schema in ARCHITECTURE.md S4.1 and aokit/config.py.
 * The loader is intentionally minimal: a tiny key/value / hard-coded params
 * reader so the C core needs NO external JSON library (zero deps except libm).
 *
 * Implementers: fill in ao_config_load() to parse either a flat key=value
 * sidecar (recommended for the C core) or a restricted-JSON subset. The
 * Python side (scripts/build_calibration.py) can emit the flat sidecar
 * alongside the AOMX matrices so the C core never needs a JSON parser.
 */
#ifndef AOCONFIG_H
#define AOCONFIG_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Camera / detector parameters. */
typedef struct {
    double pixel_size_m;   /* detector pixel pitch (m) */
    int    frame_w;        /* frame width  (px) */
    int    frame_h;        /* frame height (px) */
    int    bit_depth;      /* 8 or 24 */
} AOCameraCfg;

/* Microlens array parameters. */
typedef struct {
    int    n_lenslets_x;   /* lenslets across */
    int    n_lenslets_y;
    double pitch_m;        /* lenslet pitch on the pupil grid (m) */
    double focal_length_m; /* f_MLA (m) */
} AOMlaCfg;

/* Pupil (turbulated beam) parameters. */
typedef struct {
    double diameter_m;     /* D (m) */
    double center_x_px;    /* pupil center on detector (px) */
    double center_y_px;
} AOPupilCfg;

/* Deformable mirror parameters. */
typedef struct {
    int    n_act_x;            /* Fried: n_lenslets + 1 */
    int    n_act_y;
    double pitch_m;            /* actuator pitch on pupil (m) */
    double coupling_coeff;     /* inter-actuator coupling c (fraction) */
    double stroke_max_m;       /* |stroke| limit a_max (m) */
    int    influence_model;    /* 0=gaussian, 1=power_law, 2=measured */
    double influence_alpha;    /* power index (gaussian => 2.0) */
    double stroke_gain_m_per_unit; /* g: meters of surface per unit command */
} AODmCfg;

/* Geometry / cadence. */
typedef struct {
    int    geometry_fried;   /* 1 if Fried geometry (actuators at sub-ap corners) */
    double rotation_deg;     /* MLA-to-detector clocking */
    int    flip_y;           /* row-order / handedness flag */
    double dt_s;             /* inter-frame interval (s); drives tau0 */
    double wavelength_m;     /* sensing wavelength (m) */
} AOSysCfg;

/* Full configuration. */
typedef struct {
    int         schema_version;
    AOCameraCfg camera;
    AOMlaCfg    mla;
    AOPupilCfg  pupil;
    AODmCfg     dm;
    AOSysCfg    sys;
} AOConfig;

/* Influence-model enum values (match dm.influence_model). */
enum { AO_IF_GAUSSIAN = 0, AO_IF_POWER_LAW = 1, AO_IF_MEASURED = 2 };

/* Load a configuration from a flat key/value sidecar (or restricted JSON).
 * Returns 0 on success, non-zero on error. On error *cfg is left unspecified.
 *
 * TODO(impl): parse `path`; for the stub we zero the struct and return 0 so
 * the project links and runs. */
int ao_config_load(const char *path, AOConfig *cfg);

/* Initialize a config with the example/default values (matches
 * config/example_config.json). Useful for --selftest and as a fallback. */
void ao_config_defaults(AOConfig *cfg);

/* Convenience: number of valid slope measurements is 2 * N_sub; the actual
 * valid-sub-aperture count comes from the sub-aperture map (subapmap.aomx),
 * not from the nominal lenslet grid. These helpers give NOMINAL counts. */
static inline int ao_nominal_nsub(const AOConfig *cfg) {
    return cfg->mla.n_lenslets_x * cfg->mla.n_lenslets_y;
}
static inline int ao_nominal_nact(const AOConfig *cfg) {
    return cfg->dm.n_act_x * cfg->dm.n_act_y;
}

#ifdef __cplusplus
}
#endif

#endif /* AOCONFIG_H */
