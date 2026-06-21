/* reconstruct.c -- MVM wavefront reconstruction (zonal + modal).
 *
 * All matrices are precomputed offline (Python) and loaded as AOMX:
 *   R     (N x 2M)   zonal Fried reconstructor:  phi = R * s   -> phase map
 *   Mpinv (J x 2M)   modal reconstructor:        a   = Mpinv*s -> Zernike coeffs
 *   Z     (Npts x J) Zernike synthesis:          W   = Z * a   (cross-check)
 * research/02 S15, research/03 S6. Each apply step is one gemv_f32 call (the
 * O(1)/actuator real-time step).
 */
#include "reconstruct.h"
#include "linalg.h"
#include <math.h>
#include <stdio.h>
#include <string.h>

/* Join a directory and a filename into dst (dst_sz bytes). Tolerates a
 * trailing '/' on dir or not. Returns 0 on success. */
static int join_path(char *dst, size_t dst_sz, const char *dir, const char *name) {
    if (!dst || dst_sz == 0) return 1;
    const char *sep = "/";
    size_t dl = dir ? strlen(dir) : 0;
    if (dl > 0 && (dir[dl - 1] == '/')) sep = "";
    int n = snprintf(dst, dst_sz, "%s%s%s", dir ? dir : ".", sep, name);
    return (n < 0 || (size_t)n >= dst_sz) ? 1 : 0;
}

int reconstruct_load(const char *calib_dir, Reconstructor *rec) {
    if (!rec) return 1;
    memset(rec, 0, sizeof(*rec));

    char path[1024];

    /* R is mandatory (it produces Deliverable 1, the phase map). */
    if (join_path(path, sizeof(path), calib_dir, "R.aomx") != 0) return 2;
    if (aomx_read(path, &rec->R) != 0) {
        fprintf(stderr, "reconstruct_load: cannot read %s\n", path);
        return 3;
    }
    rec->N    = (int)rec->R.rows;
    rec->twoM = (int)rec->R.cols;

    /* Mpinv is optional (modal Zernike coefficients). */
    if (join_path(path, sizeof(path), calib_dir, "Mpinv.aomx") == 0 &&
        aomx_read(path, &rec->Mpinv) == 0) {
        if ((int)rec->Mpinv.cols != rec->twoM) {
            fprintf(stderr,
                "reconstruct_load: Mpinv cols (%u) != 2M (%d); ignoring modal\n",
                rec->Mpinv.cols, rec->twoM);
            aomx_free(&rec->Mpinv);
            rec->has_modal = 0;
        } else {
            rec->J = (int)rec->Mpinv.rows;
            rec->has_modal = 1;
        }
    }

    /* Z is optional (synthesis cross-check). Requires modal J to agree. */
    if (join_path(path, sizeof(path), calib_dir, "Z.aomx") == 0 &&
        aomx_read(path, &rec->Z) == 0) {
        if (rec->has_modal && (int)rec->Z.cols != rec->J) {
            fprintf(stderr,
                "reconstruct_load: Z cols (%u) != J (%d); ignoring synthesis\n",
                rec->Z.cols, rec->J);
            aomx_free(&rec->Z);
            rec->has_synth = 0;
        } else {
            if (!rec->has_modal) rec->J = (int)rec->Z.cols;
            rec->has_synth = 1;
        }
    }

    return 0;
}

void reconstruct_free(Reconstructor *rec) {
    if (!rec) return;
    aomx_free(&rec->R);
    aomx_free(&rec->Mpinv);
    aomx_free(&rec->Z);
    memset(rec, 0, sizeof(*rec));
}

void reconstruct_zonal(const Reconstructor *rec, const float *s, float *phi) {
    if (!rec || !rec->R.f32 || !s || !phi) return;
    gemv_f32(rec->R.f32, s, phi, (int)rec->R.rows, (int)rec->R.cols);
}

void reconstruct_modal(const Reconstructor *rec, const float *s, float *a) {
    if (!rec || !rec->has_modal || !rec->Mpinv.f32 || !s || !a) return;
    gemv_f32(rec->Mpinv.f32, s, a, (int)rec->Mpinv.rows, (int)rec->Mpinv.cols);
}

void reconstruct_synthesize(const Reconstructor *rec, const float *a, float *W) {
    if (!rec || !rec->has_synth || !rec->Z.f32 || !a || !W) return;
    gemv_f32(rec->Z.f32, a, W, (int)rec->Z.rows, (int)rec->Z.cols);
}

/* ---- self-test: with R = identity, phi == s ---- */

int reconstruct_selftest(void) {
    /* Build a tiny identity R (3x3) in-memory and run the MVM directly. */
    Reconstructor rec;
    memset(&rec, 0, sizeof(rec));
    static float Rid[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    rec.R.rows = 3; rec.R.cols = 3; rec.R.f32 = Rid;
    rec.N = 3; rec.twoM = 3;

    float s[3]   = {1.5f, -2.0f, 3.25f};
    float phi[3] = {0, 0, 0};
    reconstruct_zonal(&rec, s, phi);
    for (int i = 0; i < 3; ++i)
        if (fabsf(phi[i] - s[i]) > 1e-6f) { rec.R.f32 = NULL; return 1; }

    /* Modal: Mpinv = 2*I (2x3 -> use 3x3 scaled) so a == 2*s. */
    static float Md[9] = {2, 0, 0, 0, 2, 0, 0, 0, 2};
    rec.Mpinv.rows = 3; rec.Mpinv.cols = 3; rec.Mpinv.f32 = Md;
    rec.has_modal = 1; rec.J = 3;
    float a[3] = {0, 0, 0};
    reconstruct_modal(&rec, s, a);
    for (int i = 0; i < 3; ++i)
        if (fabsf(a[i] - 2.0f * s[i]) > 1e-6f) { rec.R.f32 = NULL; rec.Mpinv.f32 = NULL; return 2; }

    /* Detach static buffers so reconstruct_free is not called (we never free
     * these). The struct goes out of scope; nothing was malloc'd. */
    rec.R.f32 = NULL; rec.Mpinv.f32 = NULL;
    return 0;
}
