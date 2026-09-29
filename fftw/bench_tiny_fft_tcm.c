/*
 * Tiny power-of-2 complex DFT hand kernel × TCM A/B (SpaceMiT K1 / X60).
 *
 * Not FFTW — a minimal in-place radix-2 Cooley–Tukey used to ask whether
 * /dev/tcm helps when the whole working set fits (same lesson as ime GEMM):
 *   DRAM           — work + twiddles in malloc; signal filled in-place each rep
 *   TCM offline    — work + twiddles resident in TCM; signal filled in TCM
 *   TCM staged     — fill in DRAM scratch, memcpy into TCM, then FFT
 *
 * Usage: taskset -c 0 ./tiny-fft-tcm-bench [N] [reps]
 *   N must be power of 2; working set ≈ 24·N bytes (work + twiddles; signal
 *   is regenerated each rep — no resident src buffer).
 *   Default N=4096 (~96 KiB). N=16384 (~384 KiB) still fits in 512 KiB TCM.
 *
 * SPDX-License-Identifier: MIT
 */
#define _POSIX_C_SOURCE 200809L
#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "tcm.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

typedef struct {
    double re, im;
} c64;

static inline int is_pow2(int n) { return n > 0 && (n & (n - 1)) == 0; }

static inline double ns_now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e9 + (double)ts.tv_nsec;
}

/* Twiddles W[k] = exp(-2πi k / N), k = 0 .. N/2 - 1 */
static void make_twiddles(c64 *W, int n)
{
    const int half = n / 2;
    for (int k = 0; k < half; ++k) {
        const double a = -2.0 * M_PI * (double)k / (double)n;
        W[k].re = cos(a);
        W[k].im = sin(a);
    }
}

static void bit_reverse_permute(c64 *a, int n)
{
    for (int i = 0, j = 0; i < n; ++i) {
        if (i < j) {
            const c64 t = a[i];
            a[i] = a[j];
            a[j] = t;
        }
        int m = n >> 1;
        while (j & m) {
            j ^= m;
            m >>= 1;
        }
        j |= m;
    }
}

/* In-place forward DFT (unitary scale omitted — match naive DFT). */
static void fft_radix2_inplace(c64 *a, int n, const c64 *W)
{
    bit_reverse_permute(a, n);
    for (int len = 2; len <= n; len <<= 1) {
        const int half = len >> 1;
        const int step = n / len;
        for (int i = 0; i < n; i += len) {
            for (int k = 0; k < half; ++k) {
                const c64 w = W[k * step];
                const c64 u = a[i + k];
                const c64 t = a[i + k + half];
                const c64 v = {
                    t.re * w.re - t.im * w.im,
                    t.re * w.im + t.im * w.re,
                };
                a[i + k].re = u.re + v.re;
                a[i + k].im = u.im + v.im;
                a[i + k + half].re = u.re - v.re;
                a[i + k + half].im = u.im - v.im;
            }
        }
    }
}

/* O(N²) reference for small N. */
static void dft_naive(const c64 *in, c64 *out, int n)
{
    for (int k = 0; k < n; ++k) {
        double re = 0.0, im = 0.0;
        for (int t = 0; t < n; ++t) {
            const double a = -2.0 * M_PI * (double)k * (double)t / (double)n;
            const double c = cos(a), s = sin(a);
            re += in[t].re * c - in[t].im * s;
            im += in[t].re * s + in[t].im * c;
        }
        out[k].re = re;
        out[k].im = im;
    }
}

static void fill_signal(c64 *a, int n, unsigned seed)
{
    /* Deterministic mix: impulse + a few tones (finite, non-trivial). */
    memset(a, 0, (size_t)n * sizeof(c64));
    a[0].re = 1.0;
    a[1 % n].re = 0.5;
    a[(n / 3) % n].im = -0.25;
    a[(7 * n / 8) % n].re = 0.125;
    (void)seed;
}

static double max_abs_err(const c64 *a, const c64 *b, int n)
{
    double m = 0.0;
    for (int i = 0; i < n; ++i) {
        const double er = fabs(a[i].re - b[i].re);
        const double ei = fabs(a[i].im - b[i].im);
        if (er > m) {
            m = er;
        }
        if (ei > m) {
            m = ei;
        }
    }
    return m;
}

/* Classic complex FFT flop estimate: 5 N log2 N. */
static double fft_flops(int n)
{
    int lg = 0;
    for (int t = n; t > 1; t >>= 1) {
        ++lg;
    }
    return 5.0 * (double)n * (double)lg;
}

typedef struct {
    const char *name;
    c64 *work;
    c64 *W;
    int n;
} fft_bufs;

static void run_timed(const fft_bufs *b, int reps, double *med_ns, double *mflops)
{
    double *samples = calloc((size_t)reps, sizeof(double));
    if (!samples) {
        abort();
    }

    /* Warmup */
    fill_signal(b->work, b->n, 2);
    fft_radix2_inplace(b->work, b->n, b->W);

    /* Batch enough work that each sample is ≥ ~50 µs (host clock / small N). */
    int batch = 1;
    {
        const double t0 = ns_now();
        for (int i = 0; i < 8; ++i) {
            fill_signal(b->work, b->n, 2);
            fft_radix2_inplace(b->work, b->n, b->W);
        }
        const double dt = ns_now() - t0;
        if (dt > 0.0) {
            batch = (int)(50e3 * 8.0 / dt);
            if (batch < 1) {
                batch = 1;
            }
            if (batch > 100000) {
                batch = 100000;
            }
        }
    }

    for (int r = 0; r < reps; ++r) {
        const double t0 = ns_now();
        for (int i = 0; i < batch; ++i) {
            fill_signal(b->work, b->n, 2);
            fft_radix2_inplace(b->work, b->n, b->W);
        }
        samples[r] = (ns_now() - t0) / (double)batch;
    }

    for (int i = 0; i < reps; ++i) {
        for (int j = i + 1; j < reps; ++j) {
            if (samples[j] < samples[i]) {
                const double t = samples[i];
                samples[i] = samples[j];
                samples[j] = t;
            }
        }
    }
    *med_ns = samples[reps / 2];
    *mflops = (*med_ns > 0.0) ? (fft_flops(b->n) / (*med_ns) * 1e3) : 0.0;
    free(samples);
}

static void *xmalloc(size_t n)
{
    void *p = malloc(n);
    if (!p) {
        fprintf(stderr, "malloc %zu failed\n", n);
        exit(2);
    }
    return p;
}

int main(int argc, char **argv)
{
    const int n = (argc > 1) ? atoi(argv[1]) : 4096;
    const int reps = (argc > 2) ? atoi(argv[2]) : 50;

    if (!is_pow2(n) || n < 2) {
        fprintf(stderr, "N must be power of 2 >= 2 (got %d)\n", n);
        return 2;
    }

    const size_t bytes = (size_t)n * sizeof(c64);
    const size_t wbytes = (size_t)(n / 2) * sizeof(c64);
    const size_t working = bytes + wbytes; /* work + W (signal filled in-place) */

    printf("tiny-fft-tcm  N=%d  reps=%d  working≈%zu KiB  (work+W)\n",
           n, reps, working / 1024);

    /* ---- correctness vs naive (small N only) ---- */
    {
        const int nc = (n <= 512) ? n : 64;
        c64 *in = xmalloc((size_t)nc * sizeof(c64));
        c64 *ref = xmalloc((size_t)nc * sizeof(c64));
        c64 *got = xmalloc((size_t)nc * sizeof(c64));
        c64 *W = xmalloc((size_t)(nc / 2) * sizeof(c64));
        fill_signal(in, nc, 1);
        make_twiddles(W, nc);
        dft_naive(in, ref, nc);
        memcpy(got, in, (size_t)nc * sizeof(c64));
        fft_radix2_inplace(got, nc, W);
        const double err = max_abs_err(got, ref, nc);
        printf("check radix2 vs naive N=%d  max|Δ|=%.3e  %s\n",
               nc, err, err < 1e-9 * (double)nc ? "ok" : "FAIL");
        if (err >= 1e-9 * (double)nc) {
            return 3;
        }
        free(in);
        free(ref);
        free(got);
        free(W);
    }

    /* ---- DRAM ---- */
    fft_bufs dram = {
        .name = "DRAM",
        .work = xmalloc(bytes),
        .W = xmalloc(wbytes),
        .n = n,
    };
    make_twiddles(dram.W, n);

    double dram_ns = 0, dram_mf = 0;
    run_timed(&dram, reps, &dram_ns, &dram_mf);
    printf("%-12s  med=%.0f ns  %.1f MFLOPS\n", dram.name, dram_ns, dram_mf);

    c64 *dram_out = xmalloc(bytes);
    fill_signal(dram.work, n, 2);
    fft_radix2_inplace(dram.work, n, dram.W);
    memcpy(dram_out, dram.work, bytes);

    /* Scratch for staged path (fill in DRAM, copy to TCM). */
    c64 *dram_scratch = xmalloc(bytes);

    /* ---- TCM ---- */
    if (tcm_init(0) != 0) {
        fprintf(stderr, "tcm_init failed (need /dev/tcm + group tcm, cluster 0)\n");
        printf("TCM SKIP\n");
        free(dram.work);
        free(dram.W);
        free(dram_out);
        free(dram_scratch);
        return 0;
    }

    printf("tcm block=%zu B  ready=%d\n", tcm_block_size(), tcm_is_ready());

    if (working > 512u * 1024u) {
        printf("TCM SKIP (working set %zu KiB > 512 KiB)\n", working / 1024);
        tcm_shutdown();
        free(dram.work);
        free(dram.W);
        free(dram_out);
        free(dram_scratch);
        return 0;
    }

    c64 *tcm_work = tcm_malloc(bytes);
    c64 *tcm_W = tcm_malloc(wbytes);
    if (!tcm_work || !tcm_W) {
        fprintf(stderr, "tcm_malloc failed (need ~%zu KiB)\n", working / 1024);
        printf("TCM SKIP\n");
        if (tcm_work) {
            tcm_free(tcm_work);
        }
        if (tcm_W) {
            tcm_free(tcm_W);
        }
        tcm_shutdown();
        free(dram.work);
        free(dram.W);
        free(dram_out);
        free(dram_scratch);
        return 0;
    }

    make_twiddles(tcm_W, n);

    fft_bufs offline = {
        .name = "TCM offline",
        .work = tcm_work,
        .W = tcm_W,
        .n = n,
    };
    double off_ns = 0, off_mf = 0;
    run_timed(&offline, reps, &off_ns, &off_mf);
    printf("%-12s  med=%.0f ns  %.1f MFLOPS  (vs DRAM %+.1f%%)\n",
           offline.name, off_ns, off_mf,
           100.0 * (dram_ns / off_ns - 1.0));

    {
        fill_signal(tcm_work, n, 2);
        fft_radix2_inplace(tcm_work, n, tcm_W);
        const double err = max_abs_err(tcm_work, dram_out, n);
        printf("TCM offline vs DRAM max|Δ|=%.3e  %s\n",
               err, err < 1e-12 ? "ok" : "FAIL");
    }

    /* staged: fill in DRAM scratch → memcpy into TCM → FFT */
    {
        int batch = 1;
        {
            const double t0 = ns_now();
            for (int i = 0; i < 8; ++i) {
                fill_signal(dram_scratch, n, 2);
                memcpy(tcm_work, dram_scratch, bytes);
                fft_radix2_inplace(tcm_work, n, tcm_W);
            }
            const double dt = ns_now() - t0;
            if (dt > 0.0) {
                batch = (int)(50e3 * 8.0 / dt);
                if (batch < 1) {
                    batch = 1;
                }
                if (batch > 100000) {
                    batch = 100000;
                }
            }
        }
        double *samples = calloc((size_t)reps, sizeof(double));
        if (!samples) {
            abort();
        }
        for (int r = 0; r < reps; ++r) {
            const double t0 = ns_now();
            for (int i = 0; i < batch; ++i) {
                fill_signal(dram_scratch, n, 2);
                memcpy(tcm_work, dram_scratch, bytes);
                fft_radix2_inplace(tcm_work, n, tcm_W);
            }
            samples[r] = (ns_now() - t0) / (double)batch;
        }
        for (int i = 0; i < reps; ++i) {
            for (int j = i + 1; j < reps; ++j) {
                if (samples[j] < samples[i]) {
                    const double t = samples[i];
                    samples[i] = samples[j];
                    samples[j] = t;
                }
            }
        }
        const double st_ns = samples[reps / 2];
        const double st_mf = (st_ns > 0.0) ? (fft_flops(n) / st_ns * 1e3) : 0.0;
        printf("%-12s  med=%.0f ns  %.1f MFLOPS  (vs DRAM %+.1f%%)\n",
               "TCM staged", st_ns, st_mf, 100.0 * (dram_ns / st_ns - 1.0));
        free(samples);
    }

    tcm_free(tcm_work);
    tcm_free(tcm_W);
    tcm_shutdown();

    free(dram.work);
    free(dram.W);
    free(dram_out);
    free(dram_scratch);
    return 0;
}
