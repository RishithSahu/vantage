/* A stand-in for a vehicle's safety function, used to MEASURE that the assistant does not disturb it (docs/SAFETY_CASE.md, freedom from interference).
 *
 * It does what a control task does: wake up on an exact period, do a fixed amount of computation, and be done before the next period.  It records how late each wake-up was (jitter) and how long each cycle
 * took, and counts every deadline miss.  Run it on its own core(s) with a real-time priority while the assistant is idle, busy, crashing or attacking, and compare the numbers.
 *
 *   gcc -O2 -o rt_task tools/rt_task.c
 *   ./rt_task [period_us=10000] [seconds=60] [work_us=200] [deadline_us=1000]  > result.json
 *
 * Output: one JSON line: cycles, wake-up lateness percentiles (us), worst execution time, number of cycles later than deadline_us, and the highest lateness seen. */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdint.h>

static inline int64_t ns(struct timespec t) { return (int64_t)t.tv_sec * 1000000000LL + t.tv_nsec; }
static int cmp(const void *a, const void *b) { int64_t x = *(const int64_t *)a, y = *(const int64_t *)b; return (x > y) - (x < y); }

int main(int argc, char **argv) {
    int period_us = argc > 1 ? atoi(argv[1]) : 10000, seconds = argc > 2 ? atoi(argv[2]) : 60, work_us = argc > 3 ? atoi(argv[3]) : 200, deadline_us = argc > 4 ? atoi(argv[4]) : 1000;
    long n = (long)seconds * 1000000L / period_us; int64_t *late = malloc(sizeof(int64_t) * n), *exec = malloc(sizeof(int64_t) * n);
    if (!late || !exec) return 2;
    struct timespec next, now, a, b; clock_gettime(CLOCK_MONOTONIC, &next); next.tv_sec += 1; next.tv_nsec = 0;           /* start on a whole second */
    long misses = 0; int64_t worst = 0, worst_exec = 0; volatile double x = 1.0;
    for (long i = 0; i < n; i++) {
        clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &next, NULL);
        clock_gettime(CLOCK_MONOTONIC, &now); int64_t l = ns(now) - ns(next); late[i] = l;
        if (l > (int64_t)deadline_us * 1000) misses++;
        if (l > worst) worst = l;
        a = now; do { for (int k = 0; k < 200; k++) x = x * 1.0000001 + 0.0000001; clock_gettime(CLOCK_MONOTONIC, &b); } while (ns(b) - ns(a) < (int64_t)work_us * 1000);   /* the "control computation" */
        exec[i] = ns(b) - ns(a); if (exec[i] > worst_exec) worst_exec = exec[i];
        next.tv_nsec += (long)period_us * 1000L; while (next.tv_nsec >= 1000000000L) { next.tv_nsec -= 1000000000L; next.tv_sec++; }
        clock_gettime(CLOCK_MONOTONIC, &now); if (ns(now) > ns(next)) { misses += (ns(now) - ns(next)) / ((int64_t)period_us * 1000); }   /* whole periods lost */
    }
    qsort(late, n, sizeof(int64_t), cmp);
    printf("{\"cycles\":%ld,\"period_us\":%d,\"work_us\":%d,\"deadline_us\":%d,\"late_p50_us\":%.1f,\"late_p99_us\":%.1f,\"late_p999_us\":%.1f,\"late_p9999_us\":%.1f,\"late_max_us\":%.1f,\"exec_max_us\":%.1f,\"deadline_misses\":%ld}\n",
           n, period_us, work_us, deadline_us, late[n / 2] / 1e3, late[(long)(n * 0.99)] / 1e3, late[(long)(n * 0.999)] / 1e3, late[(long)(n * 0.9999)] / 1e3, worst / 1e3, worst_exec / 1e3, misses);
    return 0;
}
