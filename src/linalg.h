/* linalg.h -- portable dense linear algebra for the real-time path.
 *
 * Self-contained (no BLAS). The workhorse is a row-major single-precision
 * matrix-vector multiply (GEMV): y = A * x, with A being rows x cols stored
 * row-major. An #ifdef __AVX2__ FMA path accelerates the inner dot products;
 * an #ifdef _OPENMP path parallelizes over rows. Falls back to portable
 * scalar C otherwise. See ARCHITECTURE.md S3.7 and research/06.
 */
#ifndef LINALG_H
#define LINALG_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* y[rows] = A[rows*cols] (row-major) * x[cols].
 * The hot-path primitive: phi = R*s, a = Mpinv*s, a_dm = G*phi.
 * Buffers must be distinct (y must not alias A or x).
 *
 * TODO(impl): scalar triple-free inner loop by default; #ifdef __AVX2__ use
 * _mm256_fmadd_ps over 8-wide chunks + horizontal sum + scalar tail; #ifdef
 * _OPENMP add `#pragma omp parallel for` over rows. */
void gemv_f32(const float *A, const float *x, float *y,
              int rows, int cols);

/* General form y = alpha*A*x + beta*y (row-major). gemv_f32 == gemv_axpy with
 * alpha=1, beta=0. Useful for fused/accumulated steps.
 *
 * TODO(impl). */
void gemv_axpy_f32(float alpha, const float *A, const float *x,
                   float beta, float *y, int rows, int cols);

/* Dot product of two float vectors of length n.
 * TODO(impl): scalar by default; AVX2 FMA reduction when available. */
float dot_f32(const float *a, const float *b, int n);

/* In-place clip: v[i] = clamp(v[i], lo, hi). Used for stroke saturation.
 * TODO(impl). */
void clip_f32(float *v, int n, float lo, float hi);

/* AXPY: y[i] += alpha * x[i].  TODO(impl). */
void axpy_f32(float alpha, const float *x, float *y, int n);

/* Self-test: verify gemv_f32 / dot_f32 against a tiny hand-checked case.
 * Returns 0 on success. Called by main.c --selftest. */
int linalg_selftest(void);

#ifdef __cplusplus
}
#endif

#endif /* LINALG_H */
