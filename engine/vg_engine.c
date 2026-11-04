/* The Vantage inference engine: the whole Qwen2.5-0.5B forward pass (the same arithmetic as tools/build_ort_model.py's ONNX graph) in one C file.
 *
 *   - int8 weights packed once into the order the multiply kernel reads them; per-token int8 activations; exact int32 sums; float rescale (identical integers to ONNX Runtime's MatMulInteger path);
 *   - quantise + multiply + rescale + bias in one pass, norm / rotary / attention / SiLU fused: no per-operator overhead (an ONNX decode pass runs about 1,500 operators);
 *   - AArch64 assembly kernels for the Cortex-A72 (no dot product), AVX2 and plain C for other CPUs: integer sums are exact, so every kernel gives the same logits;
 *   - a two-thread spinning pool inside one forward call; no threads spinning between calls;
 *   - several sequences (cars) in one call: their rows share every weight read, which is what makes a server scale (vg_forward takes a list of sequences).
 *
 *   build:  gcc -O3 -mcpu=cortex-a72 -ffp-contract=off -shared -fPIC -pthread -o libvg.so vg_engine.c -lm        (Pi)
 *           gcc -O3 -mavx2 -ffp-contract=off -shared -o libvg.dll vg_engine.c -lm                                (x86)
 * Weight file: tools/export_vgw.py.  Python side: vgpt/meaning/native_backend.py. */
#define _GNU_SOURCE
#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#if defined(__aarch64__)
#include <arm_neon.h>
#include "k_aarch64.h"
#elif defined(__AVX2__)
#include <immintrin.h>
#endif
#ifdef _WIN32
#include <windows.h>
#else
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#endif

#define H 896
#define NL 24
#define NH 14
#define NKV 2
#define HD 64
#define FF 4864
#define QKV 1152          /* 896 + 128 + 128 */
#define GU 9728           /* 4864 * 2 */
#define EPS 1e-6f
#define ROPE_N 4096
#define MAXT 16

#ifdef _MSC_VER
#define VG_API __declspec(dllexport)
#else
#define VG_API __attribute__((visibility("default")))
#endif

/* ------------------------------------------------------------------ time */
static double now_ms(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e3 + t.tv_nsec * 1e-6; }

/* ------------------------------------------------------------------ integer kernel: c[mr][4] = sum_k a[row][k] * b[col][k] */
#if defined(__aarch64__)
static inline void ktile(int mr, const int8_t *a, const int8_t *b, long kb, int32_t *c) {
    if (mr == 1) k1x4(a, b, kb, c); else if (mr == 2) k2x4(a, b, kb, c); else k4x4(a, b, kb, c);
}
#elif defined(__AVX2__)
static inline void ktile(int mr, const int8_t *a, const int8_t *b, long kb, int32_t *c) {
    __m256i acc[4][4]; for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) acc[i][j] = _mm256_setzero_si256();
    for (long k = 0; k < kb; k++) {
        __m256i bw[4], aw;
        for (int j = 0; j < 4; j++) bw[j] = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *)(b + (k * 4 + j) * 16)));
        for (int i = 0; i < mr; i++) { aw = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *)(a + (k * mr + i) * 16)));
            for (int j = 0; j < 4; j++) acc[i][j] = _mm256_add_epi32(acc[i][j], _mm256_madd_epi16(aw, bw[j])); }
    }
    for (int i = 0; i < mr; i++) for (int j = 0; j < 4; j++) {
        __m128i s = _mm_add_epi32(_mm256_castsi256_si128(acc[i][j]), _mm256_extracti128_si256(acc[i][j], 1)); s = _mm_hadd_epi32(s, s); s = _mm_hadd_epi32(s, s); c[i * 4 + j] = _mm_cvtsi128_si32(s);
    }
}
#else
static inline void ktile(int mr, const int8_t *a, const int8_t *b, long kb, int32_t *c) {
    for (int i = 0; i < mr; i++) for (int j = 0; j < 4; j++) { int32_t s = 0; for (long k = 0; k < kb; k++) for (int t = 0; t < 16; t++) s += a[(k * mr + i) * 16 + t] * b[(k * 4 + j) * 16 + t]; c[i * 4 + j] = s; }
}
#endif

/* ------------------------------------------------------------------ thread pool (spins only while a forward call is running) */
typedef void (*job_fn)(void *ctx, int tid, int nt);
static struct { pthread_t th[MAXT]; int n; atomic_int gen, done, active, quit, inloop, epoch; job_fn fn; void *ctx; pthread_mutex_t mu; pthread_cond_t cv; int started; } P;
static inline void relax(void) {
#if defined(__aarch64__)
    __asm__ volatile("yield");
#elif defined(__x86_64__)
    __builtin_ia32_pause();
#endif
}
static void *worker(void *arg) {
    int tid = (int)(intptr_t)arg, myepoch = 0;
    for (;;) {
        pthread_mutex_lock(&P.mu); while (atomic_load(&P.epoch) == myepoch && !atomic_load(&P.quit)) pthread_cond_wait(&P.cv, &P.mu); myepoch = atomic_load(&P.epoch); pthread_mutex_unlock(&P.mu);
        if (atomic_load(&P.quit)) return NULL;
        int seen = atomic_load(&P.gen); atomic_fetch_add(&P.inloop, 1);
        while (atomic_load(&P.active)) {
            int g = atomic_load_explicit(&P.gen, memory_order_acquire);
            if (g != seen) { seen = g; P.fn(P.ctx, tid, P.n); atomic_fetch_add(&P.done, 1); } else relax();
        }
        atomic_fetch_sub(&P.inloop, 1);
    }
}
/* the workers spin only between pool_begin and pool_end (one forward call); in between they sleep */
static void pool_begin(void) {
    if (P.n <= 1) return;
    pthread_mutex_lock(&P.mu); atomic_store(&P.active, 1); atomic_fetch_add(&P.epoch, 1); pthread_cond_broadcast(&P.cv); pthread_mutex_unlock(&P.mu);
    while (atomic_load(&P.inloop) < P.n - 1) relax();                       /* every worker has joined: none can miss a job */
}
static void pool_end(void) { if (P.n <= 1) return; atomic_store(&P.active, 0); while (atomic_load(&P.inloop) > 0) relax(); }
static void run(job_fn fn, void *ctx) {
    if (P.n <= 1) { fn(ctx, 0, 1); return; }
    P.fn = fn; P.ctx = ctx; atomic_store(&P.done, 0); atomic_fetch_add_explicit(&P.gen, 1, memory_order_release);
    fn(ctx, 0, P.n); while (atomic_load(&P.done) < P.n - 1) relax();
}

/* ------------------------------------------------------------------ model */
typedef struct {
    const float *attn_norm, *ffn_norm; const int8_t *wqkv, *wo, *wgu, *wd; const float *sqkv, *so, *sgu, *sd, *bqkv;
} layer_t;
typedef struct {
    layer_t L[NL]; const int8_t *emb; const float *embs; const float *out_norm; const float *cosv, *sinv; void *map; size_t maplen; int threads; int vocab;
    /* scratch */
    int cap; float *x, *y, *gu, *act, *sc; int8_t *apk; float *srow; float *ctx; int8_t *hq; float *hs; double prof[10];
} engine_t;

typedef struct {
    const int64_t *ids; int T; int pos0; int P; const float *const *kv_in; float *const *kv_out; int nkeep; float *logits;
} vg_seq;

#ifdef _WIN32
static void *amalloc(size_t n) { return _aligned_malloc(n ? n : 64, 64); }
#else
static void *amalloc(size_t n) { void *p = NULL; if (posix_memalign(&p, 64, n ? n : 64)) return NULL; return p; }
#endif

VG_API void *vg_load(const char *path, int threads) {
    engine_t *E = calloc(1, sizeof *E); if (!E) return NULL;
#ifdef _WIN32
    FILE *f = fopen(path, "rb"); if (!f) { free(E); return NULL; } _fseeki64(f, 0, SEEK_END); size_t len = (size_t)_ftelli64(f); _fseeki64(f, 0, SEEK_SET);
    char *buf = amalloc(len); if (!buf || fread(buf, 1, len, f) != len) { fclose(f); free(E); return NULL; } fclose(f); E->map = buf; E->maplen = len;
#else
    int fd = open(path, O_RDONLY); if (fd < 0) { free(E); return NULL; } struct stat st; fstat(fd, &st); size_t len = st.st_size;
    void *buf = mmap(NULL, len, PROT_READ, MAP_PRIVATE, fd, 0); close(fd); if (buf == MAP_FAILED) { free(E); return NULL; } E->map = buf; E->maplen = len;
    madvise(buf, len, MADV_WILLNEED);
#endif
    const char *p = E->map;
    if (len < 64 || memcmp(p, "VGW1", 4)) goto bad_file;                 /* too short for the header, or not a model file */
    int32_t hdr[8]; memcpy(hdr, p + 4, sizeof hdr); if (hdr[0] != NL || hdr[1] != H || hdr[2] != FF || hdr[3] != ROPE_N) goto bad_file;
    size_t off = 64; uint64_t nb;
    /* every block is a 64-byte header holding its byte length, then the data padded to 64: a truncated or corrupt file must be refused, never read past its end */
#define NEXT(ptr, T) do { if (off > len || len - off < 64) goto bad_file; memcpy(&nb, p + off, 8); off += 64; if (nb > len - off) goto bad_file; ptr = (const T *)(p + off); off += (size_t)((nb + 63) / 64 * 64); } while (0)
    /* each block must also have exactly the size the arithmetic below reads: a block that is too short would be read past its end */
#define NEXTX(ptr, T, want) do { NEXT(ptr, T); if (nb != (uint64_t)(want)) goto bad_file; } while (0)
    NEXT(E->emb, int8_t); if (nb == 0 || nb % H || nb / H > 4000000) goto bad_file; E->vocab = (int)(nb / H);
    NEXTX(E->embs, float, (uint64_t)E->vocab * 4); NEXTX(E->out_norm, float, (uint64_t)H * 4); NEXTX(E->cosv, float, (uint64_t)ROPE_N * HD * 4); NEXTX(E->sinv, float, (uint64_t)ROPE_N * HD * 4);
    for (int l = 0; l < NL; l++) { layer_t *y = &E->L[l];
        NEXTX(y->attn_norm, float, (uint64_t)H * 4); NEXTX(y->wqkv, int8_t, (uint64_t)QKV * H); NEXTX(y->sqkv, float, (uint64_t)QKV * 4); NEXTX(y->bqkv, float, (uint64_t)QKV * 4);
        NEXTX(y->wo, int8_t, (uint64_t)H * H); NEXTX(y->so, float, (uint64_t)H * 4); NEXTX(y->ffn_norm, float, (uint64_t)H * 4);
        NEXTX(y->wgu, int8_t, (uint64_t)GU * H); NEXTX(y->sgu, float, (uint64_t)GU * 4); NEXTX(y->wd, int8_t, (uint64_t)H * FF); NEXTX(y->sd, float, (uint64_t)H * 4); }
    E->threads = threads < 1 ? 1 : (threads > MAXT ? MAXT : threads);
#ifdef __linux__
    { cpu_set_t cs; if (sched_getaffinity(0, sizeof cs, &cs) == 0) { int n = CPU_COUNT(&cs); if (n >= 1 && E->threads > n) E->threads = n; } }       /* spinning threads on fewer cores than threads would stall every barrier */
#endif
    if (!P.started) {
        P.started = 1; P.n = E->threads; pthread_mutex_init(&P.mu, NULL); pthread_cond_init(&P.cv, NULL);
        for (int t = 1; t < P.n; t++) pthread_create(&P.th[t], NULL, worker, (void *)(intptr_t)t);
    }
    return E;
bad_file:                                                               /* refused: give back the file image and the engine, report failure */
#ifdef _WIN32
    _aligned_free(E->map);
#else
    munmap(E->map, E->maplen);
#endif
    free(E); return NULL;
}
VG_API void vg_free(void *e) { (void)e; /* the pool and the mapping live as long as the process */ }

#ifdef _WIN32
#define afree(p) _aligned_free(p)
#else
#define afree(p) free(p)
#endif
static void ensure(engine_t *E, int R, int maxS) {
    (void)maxS;
    if (R <= E->cap) return;
    int c = R < 64 ? 64 : R; afree(E->x); afree(E->y); afree(E->gu); afree(E->act); afree(E->apk); afree(E->srow); afree(E->ctx); afree(E->hq); afree(E->hs);
    E->x = amalloc((size_t)c * H * 4); E->y = amalloc((size_t)c * QKV * 4); E->gu = amalloc((size_t)c * GU * 4); E->act = amalloc((size_t)c * FF * 4);
    E->apk = amalloc((size_t)(c + 4) * FF); E->srow = amalloc((size_t)(c + 4) * 4); E->ctx = amalloc((size_t)c * H * 4); E->hq = amalloc((size_t)(c + 4) * H); E->hs = amalloc((size_t)(c + 4) * 4); E->cap = c;
}

/* ------------------------------------------------------------------ elementwise jobs */
typedef struct { engine_t *E; int R; } rj_t;

/* quantise one row of floats (K values) to int8 with its own scale and write it into the packed activation layout ([tile][kb][mr][16], tiles of 4 rows, then the remainder tile) */
static inline void quant_row(const float *x, int K, int r, int R, int8_t *apk, float *srow) {
    int full = R / 4, rem = R - full * 4, remmr = rem == 3 ? 4 : rem;
    float amax = 0.f; for (int k = 0; k < K; k++) { float a = fabsf(x[k]); if (a > amax) amax = a; }
    float s = (amax > 1e-8f ? amax : 1e-8f) / 127.0f; srow[r] = s;
    size_t base; int ri, mr; if (r < full * 4) { base = (size_t)(r / 4) * 4 * K; ri = r % 4; mr = 4; } else { base = (size_t)full * 4 * K; ri = r - full * 4; mr = remmr; }
    int8_t *dst = apk + base;
    for (int k = 0; k < K; k++) { float v = rintf(x[k] / s); if (v > 127.f) v = 127.f; if (v < -128.f) v = -128.f; dst[((size_t)(k >> 4) * mr + ri) * 16 + (k & 15)] = (int8_t)v; }
}
typedef struct { const float *x; int R, K; int8_t *apk; float *srow; } qj_t;
static void quant_job(void *c, int tid, int nt) { qj_t *q = c; for (int r = tid; r < q->R; r += nt) quant_row(q->x + (size_t)r * q->K, q->K, r, q->R, q->apk, q->srow); }
static void quant_pack(engine_t *E, const float *x, int R, int K, int8_t *apk, float *srow) {
    int rem = R % 4; if (rem == 3) memset(apk + (size_t)(R / 4) * 4 * K, 0, (size_t)4 * K);          /* the padded 4th row of a 3-row remainder tile */
    qj_t q = {x, R, K, apk, srow}; run(quant_job, &q); (void)E;
}

typedef struct { const int8_t *apk; const float *srow; int R, N, K; const int8_t *bpk; const float *ws, *bias; float *Y; } mm_t;
static void mm_job(void *c, int tid, int nt) {
    mm_t *m = c; int NB = m->N / 4, b0 = NB * tid / nt, b1 = NB * (tid + 1) / nt; long kb = m->K / 16; size_t tile = (size_t)m->K * 4;
    for (int nb = b0; nb < b1; nb++) {
        const int8_t *b = m->bpk + nb * tile, *a = m->apk; int r = 0;
        while (r < m->R) {
            int left = m->R - r, mr = left >= 4 ? 4 : (left == 3 ? 4 : left), rows = left < mr ? left : mr; int32_t t[16];
            ktile(mr, a, b, kb, t);
            for (int i = 0; i < rows; i++) { float s = m->srow[r + i]; float *y = m->Y + (size_t)(r + i) * m->N + nb * 4;
                for (int j = 0; j < 4; j++) { float v = ((float)t[i * 4 + j] * s) * m->ws[nb * 4 + j]; if (m->bias) v += m->bias[nb * 4 + j]; y[j] = v; } }
            a += (size_t)mr * m->K; r += rows;
        }
    }
}
static void matmul(engine_t *E, const int8_t *apk, const float *srow, int R, const int8_t *bpk, const float *ws, const float *bias, int N, int K, float *Y) {
    mm_t m = {apk, srow, R, N, K, bpk, ws, bias, Y}; run(mm_job, &m); (void)E;
}

/* RMS norm of x rows (w: weight): h = x / sqrt(mean(x^2) + eps) * w ; optionally quantised and packed straight away while the row is still in cache */
typedef struct { const float *x, *w; float *h; int R; int8_t *apk; float *srow; } rms_t;
static inline void rms_row(const float *x, const float *w, float *h) {
    float s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    for (int k = 0; k < H; k += 4) { s0 += x[k] * x[k]; s1 += x[k + 1] * x[k + 1]; s2 += x[k + 2] * x[k + 2]; s3 += x[k + 3] * x[k + 3]; }
    float ms = ((s0 + s1) + (s2 + s3)) / (float)H, d = sqrtf(ms + EPS);
    for (int k = 0; k < H; k++) h[k] = (x[k] / d) * w[k];
}
static void rms_job(void *c, int tid, int nt) {
    rms_t *m = c; float tmp[H];
    for (int r = tid; r < m->R; r += nt) { if (m->apk) { rms_row(m->x + (size_t)r * H, m->w, tmp); quant_row(tmp, H, r, m->R, m->apk, m->srow); } else rms_row(m->x + (size_t)r * H, m->w, m->h + (size_t)r * H); }
}
static void rms(engine_t *E, const float *x, const float *w, float *h, int R) { rms_t m = {x, w, h, R, NULL, NULL}; run(rms_job, &m); (void)E; }
static void rms_quant(engine_t *E, const float *x, const float *w, int R, int8_t *apk, float *srow) {
    int rem = R % 4; if (rem == 3) memset(apk + (size_t)(R / 4) * 4 * H, 0, (size_t)4 * H);
    rms_t m = {x, w, NULL, R, apk, srow}; run(rms_job, &m); (void)E;
}

/* ------------------------------------------------------------------ attention */
typedef struct { engine_t *E; const vg_seq *seq; int nseq; const int *rowseq; const int *rowi; int R; int layer; const float *qkv; float *ctx; const float *cosv, *sinv; } att_t;
static void rope_head(float *t, const float *c, const float *s) {            /* t = t*cos + rotate(t)*sin ; rotate(a, b) = (-b, a) */
    float a[HD / 2], b[HD / 2]; memcpy(a, t, sizeof a); memcpy(b, t + HD / 2, sizeof b);
    for (int i = 0; i < HD / 2; i++) { t[i] = a[i] * c[i] + (-b[i]) * s[i]; t[HD / 2 + i] = b[i] * c[HD / 2 + i] + a[i] * s[HD / 2 + i]; }
}
static void kv_job(void *c, int tid, int nt) {                                  /* rotary on q and k (in place), then write k and v of the new rows into the output cache */
    att_t *a = c; const int l = a->layer;
    for (int r = tid; r < a->R; r += nt) {
        const vg_seq *sq = &a->seq[a->rowseq[r]]; int i = a->rowi[r]; int pos = sq->pos0 + i; float *row = (float *)a->qkv + (size_t)r * QKV;
        const float *cs = a->cosv + (size_t)pos * HD, *sn = a->sinv + (size_t)pos * HD;
        for (int h = 0; h < NH; h++) rope_head(row + h * HD, cs, sn);
        for (int h = 0; h < NKV; h++) rope_head(row + H + h * HD, cs, sn);
        int S = sq->P + sq->T;
        for (int h = 0; h < NKV; h++) { memcpy(sq->kv_out[2 * l] + ((size_t)h * S + sq->P + i) * HD, row + H + h * HD, HD * 4); memcpy(sq->kv_out[2 * l + 1] + ((size_t)h * S + sq->P + i) * HD, row + H + NKV * HD + h * HD, HD * 4); }
    }
}
static void att_job(void *c, int tid, int nt) {
    att_t *a = c; const int l = a->layer; float p[1024]; float *sc = p;
    for (int u = tid; u < a->R * NH; u += nt) {
        int r = u / NH, h = u % NH; const vg_seq *sq = &a->seq[a->rowseq[r]]; int i = a->rowi[r], S = sq->P + sq->T, n = sq->P + i + 1, kvh = h / (NH / NKV);
        const float *q = a->qkv + (size_t)r * QKV + h * HD, *K = sq->kv_out[2 * l] + (size_t)kvh * S * HD, *V = sq->kv_out[2 * l + 1] + (size_t)kvh * S * HD; float mx = -1e30f;
        if (n > 1024) sc = malloc((size_t)n * 4); else sc = p;
        for (int j = 0; j < n; j++) { const float *k = K + (size_t)j * HD; float d0 = 0, d1 = 0, d2 = 0, d3 = 0; for (int t = 0; t < HD; t += 4) { d0 += q[t] * k[t]; d1 += q[t + 1] * k[t + 1]; d2 += q[t + 2] * k[t + 2]; d3 += q[t + 3] * k[t + 3]; }
            float v = ((d0 + d1) + (d2 + d3)) * 0.125f; sc[j] = v; if (v > mx) mx = v; }
        float sum = 0; for (int j = 0; j < n; j++) { sc[j] = expf(sc[j] - mx); sum += sc[j]; }
        float o[HD]; memset(o, 0, sizeof o);
        for (int j = 0; j < n; j++) { float pj = sc[j] / sum; const float *v = V + (size_t)j * HD; for (int t = 0; t < HD; t++) o[t] += pj * v[t]; }
        memcpy(a->ctx + (size_t)r * H + h * HD, o, HD * 4); if (sc != p) free(sc);
    }
}

/* ------------------------------------------------------------------ SiLU(gate) * up, residual add */
#if defined(__aarch64__)
/* exp(x) for 4 floats: 2^n * polynomial(r), relative error about 2e-7 */
static inline float32x4_t vexpq(float32x4_t x) {
    x = vminq_f32(vmaxq_f32(x, vdupq_n_f32(-87.0f)), vdupq_n_f32(88.0f));
    float32x4_t n = vrndnq_f32(vmulq_f32(x, vdupq_n_f32(1.44269504089f)));
    float32x4_t r = vfmsq_f32(x, n, vdupq_n_f32(0.693359375f)); r = vfmsq_f32(r, n, vdupq_n_f32(-2.12194440e-4f));
    float32x4_t p = vdupq_n_f32(1.9875691500e-4f); p = vfmaq_f32(vdupq_n_f32(1.3981999507e-3f), p, r); p = vfmaq_f32(vdupq_n_f32(8.3334519073e-3f), p, r);
    p = vfmaq_f32(vdupq_n_f32(4.1665795894e-2f), p, r); p = vfmaq_f32(vdupq_n_f32(1.6666665459e-1f), p, r); p = vfmaq_f32(vdupq_n_f32(5.0000001201e-1f), p, r);
    float32x4_t y = vfmaq_f32(vaddq_f32(r, vdupq_n_f32(1.0f)), vmulq_f32(r, r), p);
    int32x4_t e = vshlq_n_s32(vaddq_s32(vcvtq_s32_f32(n), vdupq_n_s32(127)), 23); return vmulq_f32(y, vreinterpretq_f32_s32(e));
}
#endif
typedef struct { const float *gu; int R; int8_t *apk; float *srow; } sj_t;
static void silu_job(void *c, int tid, int nt) {                   /* act = silu(gate) * up, quantised and packed for the down projection while it is in cache */
    sj_t *m = c; float tmp[FF];
    for (int r = tid; r < m->R; r += nt) {
        const float *g = m->gu + (size_t)r * GU, *u = g + FF; int k = 0;
#if defined(__aarch64__)
        for (; k + 4 <= FF; k += 4) { float32x4_t x = vld1q_f32(g + k); float32x4_t sg = vdivq_f32(vdupq_n_f32(1.0f), vaddq_f32(vdupq_n_f32(1.0f), vexpq(vnegq_f32(x)))); vst1q_f32(tmp + k, vmulq_f32(vmulq_f32(x, sg), vld1q_f32(u + k))); }
#endif
        for (; k < FF; k++) { float x = g[k]; tmp[k] = (x * (1.0f / (1.0f + expf(-x)))) * u[k]; }
        quant_row(tmp, FF, r, m->R, m->apk, m->srow);
    }
}
typedef struct { float *x; const float *y; size_t n; } ad_t;
static void add_job(void *c, int tid, int nt) { ad_t *m = c; size_t a = m->n * tid / nt, b = m->n * (tid + 1) / nt; for (size_t i = a; i < b; i++) m->x[i] += m->y[i]; }

/* ------------------------------------------------------------------ the output head: only the candidate token rows */
typedef struct { const engine_t *E; const int8_t *hq; const float *hs; int R; const int64_t *cand; int C; float *out; } hd_t;
static inline int32_t dot8(const int8_t *a, const int8_t *b, int K) {
#if defined(__aarch64__)
    int32x4_t acc = vdupq_n_s32(0);
    for (int k = 0; k < K; k += 16) { int8x16_t x = vld1q_s8(a + k), y = vld1q_s8(b + k); int16x8_t t = vmull_s8(vget_low_s8(x), vget_low_s8(y)); t = vmlal_s8(t, vget_high_s8(x), vget_high_s8(y)); acc = vpadalq_s16(acc, t); }
    return vaddvq_s32(acc);
#elif defined(__AVX2__)
    __m256i acc = _mm256_setzero_si256();
    for (int k = 0; k < K; k += 16) acc = _mm256_add_epi32(acc, _mm256_madd_epi16(_mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *)(a + k))), _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *)(b + k)))));
    __m128i s = _mm_add_epi32(_mm256_castsi256_si128(acc), _mm256_extracti128_si256(acc, 1)); s = _mm_hadd_epi32(s, s); s = _mm_hadd_epi32(s, s); return _mm_cvtsi128_si32(s);
#else
    int32_t s = 0; for (int k = 0; k < K; k++) s += a[k] * b[k]; return s;
#endif
}
static void head_job(void *c, int tid, int nt) {
    hd_t *m = c;
    for (int ci = tid; ci < m->C; ci += nt) { int64_t id = m->cand[ci]; const int8_t *w = m->E->emb + (size_t)id * H; float sc = m->E->embs[id];
        for (int r = 0; r < m->R; r++) m->out[(size_t)r * m->C + ci] = ((float)dot8(m->hq + (size_t)r * H, w, H) * m->hs[r]) * sc; }
}

/* ------------------------------------------------------------------ forward */
VG_API int vg_forward(void *eh, int nseq, vg_seq *seq, const int64_t *cand, int C) {
    engine_t *E = eh; int R = 0, nkeepTot = 0;
    /* refuse what would read or write outside the model's tables or the caller's buffers: -1, nothing computed */
    if (!E || !seq || !cand || nseq < 1 || nseq > 256 || C < 1) return -1;
    for (int ci = 0; ci < C; ci++) if (cand[ci] < 0 || cand[ci] >= E->vocab) return -1;
    { long long tot = 0, keep = 0;
      for (int s = 0; s < nseq; s++) { const vg_seq *q = &seq[s];
          if (!q->ids || !q->kv_in || !q->kv_out || !q->logits || q->T < 1 || q->P < 0 || q->pos0 < 0 || q->nkeep < 0 || q->nkeep > q->T || (long long)q->pos0 + q->T > ROPE_N || (long long)q->P + q->T > (1 << 24)) return -1;
          for (int i = 0; i < q->T; i++) if (q->ids[i] < 0 || q->ids[i] >= E->vocab) return -1;
          tot += q->T; keep += q->nkeep; }
      if (tot > (1 << 20) || keep > (1 << 20)) return -1; }
    for (int s = 0; s < nseq; s++) { R += seq[s].T; nkeepTot += seq[s].nkeep; }
    ensure(E, R > nkeepTot ? R : nkeepTot, 0);
    int *rowseq = malloc(sizeof(int) * R * 2), *rowi = rowseq + R; if (!rowseq) return -1; { int r = 0; for (int s = 0; s < nseq; s++) for (int i = 0; i < seq[s].T; i++) { rowseq[r] = s; rowi[r] = i; r++; } }
    double t0 = now_ms(), t1; pool_begin();
    /* the system prompt and earlier turns: copy the old cache into the output cache once, then rows are appended in place */
    for (int s = 0; s < nseq; s++) { const vg_seq *q = &seq[s]; int S = q->P + q->T;
        for (int l = 0; l < 2 * NL; l++) for (int h = 0; h < NKV; h++) memcpy(q->kv_out[l] + (size_t)h * S * HD, q->kv_in[l] + (size_t)h * q->P * HD, (size_t)q->P * HD * 4); }
    { int r = 0; for (int s = 0; s < nseq; s++) for (int i = 0; i < seq[s].T; i++, r++) { int64_t id = seq[s].ids[i]; float sc = E->embs[id]; const int8_t *e = E->emb + (size_t)id * H; float *x = E->x + (size_t)r * H; for (int k = 0; k < H; k++) x[k] = (float)e[k] * sc; } }
    t1 = now_ms(); E->prof[0] += t1 - t0; t0 = t1;
    att_t A = {E, seq, nseq, rowseq, rowi, R, 0, E->y, E->ctx, E->cosv, E->sinv};
    for (int l = 0; l < NL; l++) {
        const layer_t *y = &E->L[l]; A.layer = l;
        rms_quant(E, E->x, y->attn_norm, R, E->apk, E->srow); t1 = now_ms(); E->prof[1] += t1 - t0; t0 = t1;
        matmul(E, E->apk, E->srow, R, y->wqkv, y->sqkv, y->bqkv, QKV, H, E->y); t1 = now_ms(); E->prof[2] += t1 - t0; t0 = t1;
        run(kv_job, &A); run(att_job, &A); t1 = now_ms(); E->prof[3] += t1 - t0; t0 = t1;
        quant_pack(E, E->ctx, R, H, E->apk, E->srow); matmul(E, E->apk, E->srow, R, y->wo, y->so, NULL, H, H, E->y); { ad_t ad = {E->x, E->y, (size_t)R * H}; run(add_job, &ad); } t1 = now_ms(); E->prof[4] += t1 - t0; t0 = t1;
        rms_quant(E, E->x, y->ffn_norm, R, E->apk, E->srow); t1 = now_ms(); E->prof[1] += t1 - t0; t0 = t1;
        matmul(E, E->apk, E->srow, R, y->wgu, y->sgu, NULL, GU, H, E->gu); t1 = now_ms(); E->prof[5] += t1 - t0; t0 = t1;
        { int rem = R % 4; if (rem == 3) memset(E->apk + (size_t)(R / 4) * 4 * FF, 0, (size_t)4 * FF); sj_t sj = {E->gu, R, E->apk, E->srow}; run(silu_job, &sj); } t1 = now_ms(); E->prof[6] += t1 - t0; t0 = t1;
        matmul(E, E->apk, E->srow, R, y->wd, y->sd, NULL, H, FF, E->y); { ad_t ad = {E->x, E->y, (size_t)R * H}; run(add_job, &ad); } t1 = now_ms(); E->prof[7] += t1 - t0; t0 = t1;
    }
    /* final norm and head on the kept rows of every sequence */
    { int kr = 0, r0 = 0; float *hf = E->ctx;
      for (int s = 0; s < nseq; s++) { for (int j = 0; j < seq[s].nkeep; j++) memcpy(E->y + (size_t)kr++ * H, E->x + (size_t)(r0 + seq[s].T - seq[s].nkeep + j) * H, H * 4); r0 += seq[s].T; }
      rms(E, E->y, E->out_norm, hf, kr);
      for (int r = 0; r < kr; r++) { const float *x = hf + (size_t)r * H; float amax = 0; for (int k = 0; k < H; k++) { float a = fabsf(x[k]); if (a > amax) amax = a; } float sc = (amax > 1e-8f ? amax : 1e-8f) / 127.0f; E->hs[r] = sc;
          for (int k = 0; k < H; k++) { float v = rintf(x[k] / sc); if (v > 127.f) v = 127.f; if (v < -128.f) v = -128.f; E->hq[(size_t)r * H + k] = (int8_t)v; } }
      float *lg = malloc((size_t)(kr ? kr : 1) * C * 4); if (!lg) { pool_end(); free(rowseq); return -1; } hd_t hd = {E, E->hq, E->hs, kr, cand, C, lg}; run(head_job, &hd);
      kr = 0; for (int s = 0; s < nseq; s++) { memcpy(seq[s].logits, lg + (size_t)kr * C, (size_t)seq[s].nkeep * C * 4); kr += seq[s].nkeep; } free(lg); }
    t1 = now_ms(); E->prof[8] += t1 - t0; pool_end(); free(rowseq); return 0;
}

VG_API const float *vg_hidden(void *eh) { return ((engine_t *)eh)->ctx; }          /* the normed hidden state of the kept rows of the last call (for tests) */
VG_API void vg_profile(void *eh, double *out, int reset) { engine_t *E = eh; memcpy(out, E->prof, sizeof E->prof); if (reset) memset(E->prof, 0, sizeof E->prof); }
VG_API const char *vg_kernel(void) {
#if defined(__aarch64__)
    return "aarch64-asm";
#elif defined(__AVX2__)
    return "avx2";
#else
    return "c";
#endif
}
