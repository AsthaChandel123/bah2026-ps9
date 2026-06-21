/* linalg.c -- portable dense GEMV/dot for the real-time path (no BLAS).
 *
 * This module is implemented (not stubbed): it is tiny, self-contained, and
 * the real-time budget depends on it. An #ifdef __AVX2__ FMA path accelerates
 * the inner dot product; an #ifdef _OPENMP path parallelizes over rows. The
 * portable scalar fallback always exists.
 */
#include "linalg.h"
#include <math.h>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

#if defined(_OPENMP)
#include <omp.h>
#endif

float dot_f32(const float *a, const float *b, int n) {
    int i = 0;
    float acc = 0.0f;
#if defined(__AVX2__)
    __m256 vacc = _mm256_setzero_ps();
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        vacc = _mm256_fmadd_ps(va, vb, vacc);
    }
    /* horizontal sum of vacc */
    __m128 lo = _mm256_castps256_ps128(vacc);
    __m128 hi = _mm256_extractf128_ps(vacc, 1);
    __m128 s  = _mm_add_ps(lo, hi);
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    acc = _mm_cvtss_f32(s);
#endif
    for (; i < n; ++i) acc += a[i] * b[i];
    return acc;
}

void gemv_f32(const float *A, const float *x, float *y, int rows, int cols) {
    /* y[r] = dot(A[r,:], x) for each row r. Row-major => stride-1 inner loop. */
#if defined(_OPENMP)
    #pragma omp parallel for schedule(static) if (rows > 256)
#endif
    for (int r = 0; r < rows; ++r) {
        y[r] = dot_f32(A + (size_t)r * cols, x, cols);
    }
}

void gemv_axpy_f32(float alpha, const float *A, const float *x,
                   float beta, float *y, int rows, int cols) {
#if defined(_OPENMP)
    #pragma omp parallel for schedule(static) if (rows > 256)
#endif
    for (int r = 0; r < rows; ++r) {
        float d = dot_f32(A + (size_t)r * cols, x, cols);
        y[r] = alpha * d + beta * y[r];
    }
}

void clip_f32(float *v, int n, float lo, float hi) {
    for (int i = 0; i < n; ++i) {
        float t = v[i];
        if (t < lo) t = lo;
        else if (t > hi) t = hi;
        v[i] = t;
    }
}

void axpy_f32(float alpha, const float *x, float *y, int n) {
    for (int i = 0; i < n; ++i) y[i] += alpha * x[i];
}

int linalg_selftest(void) {
    /* A = [[1,2,3],[4,5,6]], x = [1,1,1] => y = [6,15]. dot([1,2],[3,4])=11. */
    const float A[6] = {1, 2, 3, 4, 5, 6};
    const float x[3] = {1, 1, 1};
    float y[2] = {0, 0};
    gemv_f32(A, x, y, 2, 3);
    if (fabsf(y[0] - 6.0f) > 1e-5f || fabsf(y[1] - 15.0f) > 1e-5f) return 1;

    const float a[2] = {1, 2}, b[2] = {3, 4};
    if (fabsf(dot_f32(a, b, 2) - 11.0f) > 1e-5f) return 2;

    float v[4] = {-2, 0.5f, 3, 1};
    clip_f32(v, 4, -1.0f, 2.0f);
    if (v[0] != -1.0f || v[2] != 2.0f) return 3;

    return 0;
}
