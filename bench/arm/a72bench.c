/* Cortex-A72 (no dot-product) matrix-multiply ceiling benchmark.  Step 0 of the low-level speed plan (2026-10-27).
 *
 *   gcc -O3 -mcpu=cortex-a72 -pthread -o a72bench a72bench.c && taskset -c 0,1 ./a72bench
 *
 * Measures, in this order:
 *   1. peak instruction rates with no memory traffic (what the chip could do at best): int8 widening multiply-add, the pair-add into int32, float32 multiply-add;
 *   2. memory read bandwidth with 1 and 2 threads (what limits one-token decode: every weight is read once);
 *   3. real matrix multiplies of the model's shapes with several kernel designs, checked against a plain C reference, with 1 and 2 threads:
 *        i8   : int8 x int8 -> int32, widening multiply (vmull_s8 + vmlal_s8 + vpadalq_s16).  Same integers as the product's MatMulInteger.
 *        f32  : float32 activations x int8 weights (converted to float in registers) with fused multiply-add by lane.
 *   Shapes are the Qwen2.5-0.5B matrices: 896x4864 (gate / up), 4864x896 (down), 896x896 (attention q / o); M = tokens per call. */
#define _GNU_SOURCE
#include <arm_neon.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static void *amalloc(size_t n) { void *p = NULL; if (posix_memalign(&p, 64, n)) exit(1); return p; }

/* ---------- 1. peak rates ---------- */
#define UNROLL8(x) x(0) x(1) x(2) x(3) x(4) x(5) x(6) x(7)
static double peak_mull(long iters) {                       /* 8 independent chains of vmlal_s8 : 8 MAC per instruction */
    int8x8_t a = vdup_n_s8(3), b = vdup_n_s8(5); int16x8_t acc[8]; for (int i = 0; i < 8; i++) acc[i] = vdupq_n_s16(i);
    double t0 = now();
    for (long n = 0; n < iters; n++) {
#define M(i) acc[i] = vmlal_s8(acc[i], a, b);
        UNROLL8(M)
#undef M
        a = vadd_s8(a, vdup_n_s8(1));
    }
    double t = now() - t0; int s = 0; for (int i = 0; i < 8; i++) s += vgetq_lane_s16(acc[i], 0); if (s == 12345) puts("");
    return iters * 8 * 8 * 2.0 / t / 1e9;                    /* GOPS (multiply + add counted as 2) */
}
static double peak_padal(long iters) {                      /* pair-add of int16 into int32 : 8 adds per instruction */
    int16x8_t x = vdupq_n_s16(7); int32x4_t acc[8]; for (int i = 0; i < 8; i++) acc[i] = vdupq_n_s32(i);
    double t0 = now();
    for (long n = 0; n < iters; n++) {
#define M(i) acc[i] = vpadalq_s16(acc[i], x);
        UNROLL8(M)
#undef M
        x = vaddq_s16(x, vdupq_n_s16(1));
    }
    double t = now() - t0; int s = 0; for (int i = 0; i < 8; i++) s += vgetq_lane_s32(acc[i], 0); if (s == 12345) puts("");
    return iters * 8 * 8 * 2.0 / t / 1e9;
}
static double peak_mull_padal(long iters) {                 /* the real int8 inner loop: 2 widening MACs + 1 pair-add per 16 MACs */
    int8x16_t a = vdupq_n_s8(3), b = vdupq_n_s8(5); int32x4_t acc[8]; for (int i = 0; i < 8; i++) acc[i] = vdupq_n_s32(i);
    double t0 = now();
    for (long n = 0; n < iters; n++) {
#define M(i) { int16x8_t t = vmull_s8(vget_low_s8(a), vget_low_s8(b)); t = vmlal_s8(t, vget_high_s8(a), vget_high_s8(b)); acc[i] = vpadalq_s16(acc[i], t); }
        UNROLL8(M)
#undef M
        a = vaddq_s8(a, vdupq_n_s8(1));
    }
    double t = now() - t0; int s = 0; for (int i = 0; i < 8; i++) s += vgetq_lane_s32(acc[i], 0); if (s == 12345) puts("");
    return iters * 8 * 16 * 2.0 / t / 1e9;
}
static double peak_fmla(long iters) {                       /* 12 independent chains of fused multiply-add on 4 floats */
    float32x4_t a = vdupq_n_f32(1.0001f), b = vdupq_n_f32(0.9999f), acc[12]; for (int i = 0; i < 12; i++) acc[i] = vdupq_n_f32(i);
    double t0 = now();
    for (long n = 0; n < iters; n++) {
#define M(i) acc[i] = vfmaq_f32(acc[i], a, b);
        UNROLL8(M) M(8) M(9) M(10) M(11)
#undef M
    }
    double t = now() - t0; float s = 0; for (int i = 0; i < 12; i++) s += vgetq_lane_f32(acc[i], 0); if (s == 12345.f) puts("");
    return iters * 12 * 4 * 2.0 / t / 1e9;
}

/* ---------- 2. memory bandwidth ---------- */
typedef struct { const uint8_t *p; size_t n; int reps; double sink; } bw_t;
static void *bw_run(void *v) {
    bw_t *w = v; int32x4_t acc = vdupq_n_s32(0);
    for (int r = 0; r < w->reps; r++)
        for (size_t i = 0; i < w->n; i += 64) {
            uint8x16_t a = vld1q_u8(w->p + i), b = vld1q_u8(w->p + i + 16), c = vld1q_u8(w->p + i + 32), d = vld1q_u8(w->p + i + 48);
            acc = vaddq_s32(acc, vreinterpretq_s32_u8(veorq_u8(veorq_u8(a, b), veorq_u8(c, d))));
        }
    w->sink = vgetq_lane_s32(acc, 0); return NULL;
}
static double bandwidth(int threads, size_t bytes) {
    uint8_t *buf = amalloc(bytes); memset(buf, 1, bytes); pthread_t th[4]; bw_t w[4]; size_t per = bytes / threads / 64 * 64;
    for (int t = 0; t < threads; t++) w[t] = (bw_t){buf + t * per, per, 3, 0};
    double t0 = now();
    for (int t = 0; t < threads; t++) pthread_create(&th[t], NULL, bw_run, &w[t]);
    for (int t = 0; t < threads; t++) pthread_join(th[t], NULL);
    double dt = now() - t0; free(buf); return per * threads * 3.0 / dt / 1e9;
}

/* ---------- 3. matrix multiplies ----------  C[m][n] = sum_k A[m][k] * B[n][k]   (B = the weights, one output channel per row) */
typedef struct { int M, N, K; const int8_t *A, *B; int32_t *C; const float *Af; float *Cf; const int8_t *Bp; int n0, n1; } job_t;

static void ref_i8(int M, int N, int K, const int8_t *A, const int8_t *B, int32_t *C) {
    for (int m = 0; m < M; m++) for (int n = 0; n < N; n++) { int32_t s = 0; for (int k = 0; k < K; k++) s += A[m * K + k] * B[n * K + k]; C[m * N + n] = s; }
}

/* i8 kernel: tile of MR rows of A x 4 rows of B; B tile stays in L1 while all M rows of A pass over it (weights are read from memory once) */
#define I8_TILE(MR) \
static inline void i8_tile_##MR(int K, const int8_t *A, const int8_t *B, int32_t *C, int N) { \
    int32x4_t acc[MR][4]; for (int i = 0; i < MR; i++) for (int j = 0; j < 4; j++) acc[i][j] = vdupq_n_s32(0); \
    for (int k = 0; k < K; k += 16) { \
        int8x16_t b[4]; for (int j = 0; j < 4; j++) b[j] = vld1q_s8(B + (size_t)j * K + k); \
        _Pragma("GCC unroll 8") for (int i = 0; i < MR; i++) { int8x16_t a = vld1q_s8(A + (size_t)i * K + k); \
            _Pragma("GCC unroll 4") for (int j = 0; j < 4; j++) { int16x8_t t = vmull_s8(vget_low_s8(a), vget_low_s8(b[j])); t = vmlal_s8(t, vget_high_s8(a), vget_high_s8(b[j])); acc[i][j] = vpadalq_s16(acc[i][j], t); } } } \
    for (int i = 0; i < MR; i++) for (int j = 0; j < 4; j++) C[(size_t)i * N + j] = vaddvq_s32(acc[i][j]); }
I8_TILE(1) I8_TILE(2) I8_TILE(4) I8_TILE(8)

static void *i8_run(void *v) {
    job_t *J = v; int K = J->K, N = J->N, M = J->M;
    for (int n = J->n0; n < J->n1; n += 4) {
        int m = 0;
        for (; getenv("MR8") && m + 8 <= M; m += 8) i8_tile_8(K, J->A + (size_t)m * K, J->B + (size_t)n * K, J->C + (size_t)m * N + n, N);
        for (; m + 4 <= M; m += 4) i8_tile_4(K, J->A + (size_t)m * K, J->B + (size_t)n * K, J->C + (size_t)m * N + n, N);
        for (; m + 2 <= M; m += 2) i8_tile_2(K, J->A + (size_t)m * K, J->B + (size_t)n * K, J->C + (size_t)m * N + n, N);
        for (; m < M; m++) i8_tile_1(K, J->A + (size_t)m * K, J->B + (size_t)n * K, J->C + (size_t)m * N + n, N);
    }
    return NULL;
}

/* f32 kernel: activations are float32, weights int8 packed as [N/8][K][8]; per k the 8 weights become two float vectors, then MR fused multiply-adds by lane.
 * B is re-packed so that one load gives 16 weights = 2 values of k for 8 outputs. */
#define F32_TILE(MR) \
static inline void f32_tile_##MR(int K, const float *A, const int8_t *Bp, float *C, int N) { \
    float32x4_t acc[MR][2]; for (int i = 0; i < MR; i++) { acc[i][0] = vdupq_n_f32(0); acc[i][1] = vdupq_n_f32(0); } \
    for (int k = 0; k < K; k += 4) { \
        float32x4_t w[4][2]; \
        for (int q = 0; q < 2; q++) { int8x16_t raw = vld1q_s8(Bp + (size_t)k * 8 + q * 16); \
            int16x8_t lo = vmovl_s8(vget_low_s8(raw)), hi = vmovl_s8(vget_high_s8(raw)); \
            w[2 * q][0] = vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo))); w[2 * q][1] = vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo))); \
            w[2 * q + 1][0] = vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi))); w[2 * q + 1][1] = vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi))); } \
        _Pragma("GCC unroll 8") for (int i = 0; i < MR; i++) { float32x4_t a = vld1q_f32(A + (size_t)i * K + k); \
            acc[i][0] = vfmaq_laneq_f32(acc[i][0], w[0][0], a, 0); acc[i][1] = vfmaq_laneq_f32(acc[i][1], w[0][1], a, 0); \
            acc[i][0] = vfmaq_laneq_f32(acc[i][0], w[1][0], a, 1); acc[i][1] = vfmaq_laneq_f32(acc[i][1], w[1][1], a, 1); \
            acc[i][0] = vfmaq_laneq_f32(acc[i][0], w[2][0], a, 2); acc[i][1] = vfmaq_laneq_f32(acc[i][1], w[2][1], a, 2); \
            acc[i][0] = vfmaq_laneq_f32(acc[i][0], w[3][0], a, 3); acc[i][1] = vfmaq_laneq_f32(acc[i][1], w[3][1], a, 3); } } \
    for (int i = 0; i < MR; i++) { vst1q_f32(C + (size_t)i * N, acc[i][0]); vst1q_f32(C + (size_t)i * N + 4, acc[i][1]); } }
F32_TILE(1) F32_TILE(2) F32_TILE(4) F32_TILE(6)

static void *f32_run(void *v) {
    job_t *J = v; int K = J->K, N = J->N, M = J->M;
    for (int n = J->n0; n < J->n1; n += 8) {
        int m = 0; const int8_t *Bp = J->Bp + (size_t)(n / 8) * K * 8;
        for (; m + 6 <= M; m += 6) f32_tile_6(K, J->Af + (size_t)m * K, Bp, J->Cf + (size_t)m * N + n, N);
        for (; m + 4 <= M; m += 4) f32_tile_4(K, J->Af + (size_t)m * K, Bp, J->Cf + (size_t)m * N + n, N);
        for (; m + 2 <= M; m += 2) f32_tile_2(K, J->Af + (size_t)m * K, Bp, J->Cf + (size_t)m * N + n, N);
        for (; m < M; m++) f32_tile_1(K, J->Af + (size_t)m * K, Bp, J->Cf + (size_t)m * N + n, N);
    }
    return NULL;
}

static void run_threads(void *(*fn)(void *), job_t *base, int threads, int step) {
    pthread_t th[4]; job_t j[4]; int per = (base->N / threads) / step * step;
    for (int t = 0; t < threads; t++) { j[t] = *base; j[t].n0 = t * per; j[t].n1 = (t == threads - 1) ? base->N : (t + 1) * per; }
    if (threads == 1) { fn(&j[0]); return; }
    for (int t = 0; t < threads; t++) pthread_create(&th[t], NULL, fn, &j[t]);
    for (int t = 0; t < threads; t++) pthread_join(th[t], NULL);
}

static void bench_shape(int K, int N, const char *name) {
    int Ms[] = {1, 2, 4, 8, 16, 32, 48};
    int8_t *B = amalloc((size_t)N * K); for (size_t i = 0; i < (size_t)N * K; i++) B[i] = (int8_t)((rand() % 255) - 127);
    int8_t *Bp = amalloc((size_t)N * K);                    /* [N/8][K][8] for the float kernel */
    for (int nb = 0; nb < N / 8; nb++) for (int k = 0; k < K; k++) for (int c = 0; c < 8; c++) Bp[((size_t)nb * K + k) * 8 + c] = B[(size_t)(nb * 8 + c) * K + k];
    for (int mi = 0; mi < 7; mi++) {
        int M = Ms[mi];
        int8_t *A = amalloc((size_t)M * K); for (int i = 0; i < M * K; i++) A[i] = (int8_t)((rand() % 255) - 127);
        float *Af = amalloc((size_t)M * K * 4); for (int i = 0; i < M * K; i++) Af[i] = A[i] * 0.01f;
        int32_t *C = amalloc((size_t)M * N * 4), *R = amalloc((size_t)M * N * 4); float *Cf = amalloc((size_t)M * N * 4);
        if (mi < 3) ref_i8(M, N, K, A, B, R);
        for (int th = 1; th <= 2; th++) {
            job_t J = {M, N, K, A, B, C, Af, Cf, Bp, 0, N}; double best_i = 1e9, best_f = 1e9; int reps = (M >= 16 ? 5 : 10);
            for (int r = 0; r < reps; r++) { double t0 = now(); run_threads(i8_run, &J, th, 4); double dt = now() - t0; if (dt < best_i) best_i = dt; }
            for (int r = 0; r < reps; r++) { double t0 = now(); run_threads(f32_run, &J, th, 8); double dt = now() - t0; if (dt < best_f) best_f = dt; }
            int ok = 1; if (mi < 3) for (long i = 0; i < (long)M * N; i++) if (C[i] != R[i]) { ok = 0; break; }
            double ops = 2.0 * M * K * N; double wb = (double)N * K;
            printf("%-10s M=%-3d threads=%d | i8  %7.2f ms %6.1f GOPS %s | f32 %7.2f ms %6.1f GOPS | weights %.2f GB/s (i8)\n", name, M, th, best_i * 1e3, ops / best_i / 1e9, mi < 3 ? (ok ? "exact" : "WRONG") : "     ", best_f * 1e3, ops / best_f / 1e9, wb / best_i / 1e9);
            fflush(stdout);
        }
        free(A); free(Af); free(C); free(R); free(Cf);
    }
    free(B); free(Bp);
}

int main(int argc, char **argv) {
    (void)argc; (void)argv; srand(1);
    printf("== 1. peak instruction rates, one core, no memory traffic (GOPS = multiply+add counted as 2) ==\n");
    long it = 20000000;
    printf("int8 widening multiply-add (vmlal_s8)      : %6.1f GOPS\n", peak_mull(it));
    printf("int16 -> int32 pair-add (vpadalq_s16)      : %6.1f GOPS-equivalent\n", peak_padal(it));
    printf("real int8 inner loop (2 widening + 1 padal): %6.1f GOPS\n", peak_mull_padal(it));
    printf("float32 fused multiply-add (vfmaq_f32)     : %6.1f GFLOPS\n", peak_fmla(it));
    printf("== 2. memory read bandwidth (GB/s) ==\n");
    printf("1 thread : %.2f\n2 threads: %.2f\n", bandwidth(1, 400u << 20), bandwidth(2, 400u << 20));
    printf("== 3. matrix multiplies, model shapes ==\n");
    bench_shape(896, 4864, "896x4864");
    bench_shape(4864, 896, "4864x896");
    bench_shape(896, 896, "896x896");
    return 0;
}
