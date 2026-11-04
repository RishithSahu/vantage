/* Hand-written assembly int8 matrix multiply for the Cortex-A72 and its benchmark (Step 1 of the low-level speed plan, 2026-10-27).
 *   gcc -O3 -mcpu=cortex-a72 -pthread -o a72kernel a72kernel.c && taskset -c 0,1 ./a72kernel
 * Weights are packed once, in the order the kernel reads them; the activations are packed per call (M rows, a few KB).  Result is checked bit for bit against a plain C reference. */
#define _GNU_SOURCE
#include <arm_neon.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "k4x4.h"

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static void *amalloc(size_t n) { void *p = NULL; if (posix_memalign(&p, 64, n)) exit(1); return p; }

/* pack rows [r0, r0+4) of a row-major R x K int8 matrix into [kb][4][16]; rows >= R are zero */
static void pack_tile(const int8_t *X, int R, int K, int r0, int8_t *out) {
    for (int kb = 0; kb < K / 16; kb++) for (int r = 0; r < 4; r++) {
        if (r0 + r < R) memcpy(out + ((size_t)kb * 4 + r) * 16, X + (size_t)(r0 + r) * K + kb * 16, 16); else memset(out + ((size_t)kb * 4 + r) * 16, 0, 16);
    }
}
typedef struct { int M, N, K; const int8_t *Apk; const int8_t *Bpk; int32_t *C; int nb0, nb1; } job_t;   /* Apk: [M/4][K*4], Bpk: [N/4][K*4] */

static void *run(void *v) {
    job_t *J = v; int K = J->K, N = J->N, MT = (J->M + 3) / 4; size_t tile = (size_t)K * 4; int32_t tmp[16];
    for (int nb = J->nb0; nb < J->nb1; nb++) {
        const int8_t *b = J->Bpk + nb * tile;                  /* this weight tile stays in cache while every activation tile passes over it */
        for (int mt = 0; mt < MT; mt++) {
            if (mt * 4 + 4 <= J->M && nb * 4 + 4 <= N) k4x4(J->Apk + mt * tile, b, K / 16, J->C + (size_t)mt * 4 * N + nb * 4, (long)N * 4);
            else { k4x4(J->Apk + mt * tile, b, K / 16, tmp, 16); for (int r = 0; r < 4 && mt * 4 + r < J->M; r++) for (int c = 0; c < 4; c++) J->C[(size_t)(mt * 4 + r) * N + nb * 4 + c] = tmp[r * 4 + c]; }
        }
    }
    return NULL;
}
static void matmul(int M, int N, int K, const int8_t *Apk, const int8_t *Bpk, int32_t *C, int threads) {
    int NB = N / 4; pthread_t th[4]; job_t j[4]; int per = NB / threads;
    for (int t = 0; t < threads; t++) j[t] = (job_t){M, N, K, Apk, Bpk, C, t * per, (t == threads - 1) ? NB : (t + 1) * per};
    if (threads == 1) { run(&j[0]); return; }
    for (int t = 0; t < threads; t++) pthread_create(&th[t], NULL, run, &j[t]);
    for (int t = 0; t < threads; t++) pthread_join(th[t], NULL);
}
static void ref(int M, int N, int K, const int8_t *A, const int8_t *B, int32_t *C) {
    for (int m = 0; m < M; m++) for (int n = 0; n < N; n++) { int32_t s = 0; for (int k = 0; k < K; k++) s += A[m * K + k] * B[(size_t)n * K + k]; C[(size_t)m * N + n] = s; }
}
static void shape(int K, int N, const char *name) {
    int Ms[] = {1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48};
    int8_t *B = amalloc((size_t)N * K); for (size_t i = 0; i < (size_t)N * K; i++) B[i] = (int8_t)((rand() % 255) - 127);   /* weights in [-127, 127]: the pair of products always fits int16 */
    int8_t *Bpk = amalloc((size_t)N * K); for (int nb = 0; nb < N / 4; nb++) pack_tile(B, N, K, nb * 4, Bpk + (size_t)nb * K * 4);
    for (int mi = 0; mi < 11; mi++) {
        int M = Ms[mi]; int8_t *A = amalloc((size_t)M * K); for (int i = 0; i < M * K; i++) A[i] = (int8_t)((rand() % 255) - 127);
        int MT = (M + 3) / 4; int8_t *Apk = amalloc((size_t)MT * K * 4); int32_t *C = amalloc((size_t)M * N * 4), *R = amalloc((size_t)M * N * 4);
        ref(M, N, K, A, B, R);
        for (int th = 1; th <= 2; th++) {
            double best = 1e9, bestp = 1e9;
            for (int r = 0; r < 12; r++) { double t0 = now(); for (int mt = 0; mt < MT; mt++) pack_tile(A, M, K, mt * 4, Apk + (size_t)mt * K * 4); double tp = now() - t0; matmul(M, N, K, Apk, Bpk, C, th); double dt = now() - t0; if (dt < best) best = dt; if (tp < bestp) bestp = tp; }
            int ok = 1; for (long i = 0; i < (long)M * N; i++) if (C[i] != R[i]) { ok = 0; break; }
            printf("%-9s M=%-3d threads=%d | %7.3f ms (pack %.3f) %6.1f GOPS  weights %.2f GB/s  %s\n", name, M, th, best * 1e3, bestp * 1e3, 2.0 * M * K * N / best / 1e9, (double)N * K / best / 1e9, ok ? "exact" : "WRONG"); fflush(stdout);
        }
        free(A); free(Apk); free(C); free(R);
    }
    free(B); free(Bpk);
}
int main(void) { srand(1); shape(896, 4864, "896x4864"); shape(4864, 896, "4864x896"); shape(896, 896, "896x896"); return 0; }
