/*
 * ime2_bench.c — A100 IME2 (smt.vmadot) vs RVV vs scalar, s8s8s32.
 *
 * On A100 (VLEN=1024, lambda=4) one smt.vmadot is an 8x8x16 tile:
 *   C[8x8 int32] += A[8x16 int8] * B[8x16 int8]^T
 * with both operands row-major K-vectors (same contract as the X60 kernel,
 * which is 4x4x8 at VLEN=256). The instruction word is the X60 encoding;
 * the tile shape is a property of the core, not of the opcode.
 *
 * C = A * B^T, A is MxK, B is NxK, both row-major int8.
 * Dims must be multiples of 16.
 *
 * Build: gcc -O2 -march=rv64gcv_zvl256b -o ime2-bench ime2_bench.c
 * Run on an A100 hart (set_ai_thread before exec): ./ime2-bench M N K [reps]
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <riscv_vector.h>

#define SMT_VMADOT(vd, vs1, vs2)                                                \
    ".insn 4, 0xe200302b|((" #vd "/2)<<8)|((" #vs1 ")<<15)|((" #vs2 ")<<20)\n\t"

static double secs(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

static unsigned long rd_vlenb(void)
{
    unsigned long v;
    __asm__ volatile("csrr %0, vlenb" : "=r"(v));
    return v;
}

static void gemm_ref(const int8_t *A, const int8_t *B, int32_t *C, int M, int N, int K)
{
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            int32_t acc = 0;
            const int8_t *a = A + (size_t)i * K;
            const int8_t *b = B + (size_t)j * K;
            for (int k = 0; k < K; k++)
                acc += (int32_t)a[k] * (int32_t)b[k];
            C[(size_t)i * N + j] = acc;
        }
    }
}

static void gemm_rvv(const int8_t *A, const int8_t *B, int32_t *C, int M, int N, int K)
{
    for (int i = 0; i < M; i++) {
        const int8_t *arow = A + (size_t)i * K;
        for (int j = 0; j < N; j++) {
            const int8_t *brow = B + (size_t)j * K;
            vint32m1_t acc = __riscv_vmv_v_x_i32m1(0, 1);
            for (size_t k = 0, rem = (size_t)K; rem > 0;) {
                size_t vl = __riscv_vsetvl_e8m1(rem);
                vint8m1_t va = __riscv_vle8_v_i8m1(arow + k, vl);
                vint8m1_t vb = __riscv_vle8_v_i8m1(brow + k, vl);
                vint16m2_t vp = __riscv_vwmul_vv_i16m2(va, vb, vl);
                acc = __riscv_vwredsum_vs_i16m2_i32m1(vp, acc, vl);
                k += vl;
                rem -= vl;
            }
            C[(size_t)i * N + j] = __riscv_vmv_x_s_i32m1_i32(acc);
        }
    }
}

/* Pack row-major MxK into 8x16 tiles, order [rowtile][ktile][8][16]. */
static void pack8x16(const int8_t *S, int8_t *D, int rows, int K)
{
    const int RT = rows / 8;
    const int KT = K / 16;
    for (int rt = 0; rt < RT; rt++) {
        for (int kt = 0; kt < KT; kt++) {
            int8_t *d = D + ((size_t)rt * KT + kt) * 128;
            for (int r = 0; r < 8; r++)
                memcpy(d + r * 16, S + ((size_t)rt * 8 + r) * K + kt * 16, 16);
        }
    }
}

/* One 16x16 output block. Ap tiles are 8 rows; ar0 is the even tile. */
static void ime2_16x16(const int8_t *a0, const int8_t *a1, const int8_t *b0,
                       const int8_t *b1, int KT, int32_t out[4][64])
{
    __asm__ volatile(
        "vsetvli t0, zero, e32, m8, ta, ma\n\t"
        "vmv.v.i v16, 0\n\t"
        "vsetvli t0, zero, e8, m1, ta, ma\n\t"
        "beqz %[kt], 2f\n\t"
        "1:\n\t"
        "vle8.v v0, (%[a0])\n\t"
        "addi %[a0], %[a0], 128\n\t"
        "vle8.v v2, (%[a1])\n\t"
        "addi %[a1], %[a1], 128\n\t"
        "vle8.v v4, (%[b0])\n\t"
        "addi %[b0], %[b0], 128\n\t"
        "vle8.v v6, (%[b1])\n\t"
        "addi %[b1], %[b1], 128\n\t"
        SMT_VMADOT(16, 0, 4)
        SMT_VMADOT(18, 0, 6)
        SMT_VMADOT(20, 2, 4)
        SMT_VMADOT(22, 2, 6)
        "addi %[kt], %[kt], -1\n\t"
        "bnez %[kt], 1b\n\t"
        "2:\n\t"
        "vsetvli t0, zero, e32, m2, ta, ma\n\t"
        "vse32.v v16, (%[o0])\n\t"
        "vse32.v v18, (%[o1])\n\t"
        "vse32.v v20, (%[o2])\n\t"
        "vse32.v v22, (%[o3])\n\t"
        : [a0] "+r"(a0), [a1] "+r"(a1), [b0] "+r"(b0), [b1] "+r"(b1), [kt] "+r"(KT)
        : [o0] "r"(out[0]), [o1] "r"(out[1]), [o2] "r"(out[2]), [o3] "r"(out[3])
        : "t0", "memory", "v0", "v2", "v4", "v6", "v16", "v17", "v18", "v19", "v20",
          "v21", "v22", "v23");
}

static void scatter8(const int32_t *t, int32_t *C, int ldc, int row, int col)
{
    for (int r = 0; r < 8; r++)
        memcpy(C + ((size_t)row + r) * ldc + col, t + r * 8, 8 * sizeof(int32_t));
}

static void gemm_ime2(const int8_t *Ap, const int8_t *Bp, int32_t *C, int M, int N, int K)
{
    const int KT = K / 16;
    const int a_stride = (size_t)KT * 128; /* bytes between consecutive 8-row tiles, same k */
    for (int sr = 0; sr < M / 16; sr++) {
        for (int sc = 0; sc < N / 16; sc++) {
            const int8_t *a0 = Ap + ((size_t)(2 * sr) * KT) * 128;
            const int8_t *a1 = a0 + a_stride;
            const int8_t *b0 = Bp + ((size_t)(2 * sc) * KT) * 128;
            const int8_t *b1 = b0 + a_stride;
            int32_t out[4][64];
            ime2_16x16(a0, a1, b0, b1, KT, out);
            scatter8(out[0], C, N, sr * 16, sc * 16);
            scatter8(out[1], C, N, sr * 16, sc * 16 + 8);
            scatter8(out[2], C, N, sr * 16 + 8, sc * 16);
            scatter8(out[3], C, N, sr * 16 + 8, sc * 16 + 8);
        }
    }
}

static int check(const char *name, const int32_t *got, const int32_t *ref, int M, int N)
{
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            size_t p = (size_t)i * N + j;
            if (got[p] != ref[p]) {
                printf("  %s FAIL at %d,%d: got %d want %d\n", name, i, j, got[p], ref[p]);
                return 1;
            }
        }
    }
    printf("  %s ok\n", name);
    return 0;
}

static int cmp_d(const void *a, const void *b)
{
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

static double median_gops(double *t, int n, double ops)
{
    qsort(t, n, sizeof(double), cmp_d);
    double mid = t[n / 2];
    return ops / mid / 1e9;
}

/* One instruction, known tile. Returns 0 on bit-exact match. */
static int tile_selftest(void)
{
    int8_t A[128], B[128];
    int32_t got[64], want[64];
    for (int i = 0; i < 128; i++) {
        A[i] = (int8_t)(i * 3 - 40);
        B[i] = (int8_t)(20 - (i % 17));
    }
    memset(want, 0, sizeof want);
    for (int i = 0; i < 8; i++)
        for (int j = 0; j < 8; j++) {
            int32_t s = 0;
            for (int q = 0; q < 16; q++)
                s += (int32_t)A[i * 16 + q] * (int32_t)B[j * 16 + q];
            want[i * 8 + j] = s;
        }
    const int8_t *a = A, *b = B;
    int32_t *c = got;
    memset(got, 0x5a, sizeof got);
    __asm__ volatile(
        "vsetvli t0, zero, e32, m2, ta, ma\n\t"
        "vmv.v.i v16, 0\n\t"
        "vsetvli t0, zero, e8, m1, ta, ma\n\t"
        "vle8.v v0, (%[a])\n\t"
        "vle8.v v2, (%[b])\n\t"
        SMT_VMADOT(16, 0, 2)
        "vsetvli t0, zero, e32, m2, ta, ma\n\t"
        "vse32.v v16, (%[c])\n\t"
        :
        : [a] "r"(a), [b] "r"(b), [c] "r"(c)
        : "t0", "memory", "v0", "v2", "v16", "v17");
    for (int i = 0; i < 64; i++) {
        if (got[i] != want[i]) {
            printf("tile FAIL p=%d got %d want %d\n", i, got[i], want[i]);
            printf("got row0:");
            for (int j = 0; j < 8; j++)
                printf(" %d", got[j]);
            printf("\nwant row0:");
            for (int j = 0; j < 8; j++)
                printf(" %d", want[j]);
            printf("\n");
            return 1;
        }
    }
    printf("tile ok\n");
    return 0;
}

int main(int argc, char **argv)
{
    int M = argc > 1 ? atoi(argv[1]) : 512;
    int N = argc > 2 ? atoi(argv[2]) : 512;
    int K = argc > 3 ? atoi(argv[3]) : 512;
    int reps = argc > 4 ? atoi(argv[4]) : 10;
    int do_ime = argc > 5 ? atoi(argv[5]) : 1;
    if (M % 16 || N % 16 || K % 16 || reps < 1) {
        printf("need M,N,K %% 16 == 0 and reps>=1\n");
        return 1;
    }
    printf("vlenb=%lu  M=%d N=%d K=%d reps=%d ime=%d\n", rd_vlenb(), M, N, K, reps, do_ime);

    if (do_ime && tile_selftest())
        return 1;

    int8_t *A = NULL, *B = NULL, *Ap = NULL, *Bp = NULL;
    int32_t *Cref = NULL, *C = NULL;
    if (posix_memalign((void **)&A, 64, (size_t)M * K) ||
        posix_memalign((void **)&B, 64, (size_t)N * K) ||
        posix_memalign((void **)&Ap, 64, (size_t)M * K) ||
        posix_memalign((void **)&Bp, 64, (size_t)N * K) ||
        posix_memalign((void **)&Cref, 64, (size_t)M * N * 4) ||
        posix_memalign((void **)&C, 64, (size_t)M * N * 4)) {
        printf("alloc fail\n");
        return 1;
    }
    uint32_t s = 0x12345678u;
    for (size_t i = 0; i < (size_t)M * K; i++) {
        A[i] = (int8_t)(s >> 24);
        s = s * 1664525u + 1013904223u;
    }
    for (size_t i = 0; i < (size_t)N * K; i++) {
        B[i] = (int8_t)(s >> 24);
        s = s * 1664525u + 1013904223u;
    }

    double t0 = secs();
    gemm_ref(A, B, Cref, M, N, K);
    double tref = secs() - t0;
    const double ops = 2.0 * M * N * K;
    printf("  scalar  %6.2f GOP/s\n", ops / tref / 1e9);

    double *ts = calloc((size_t)reps, sizeof(double));
    if (!ts)
        return 1;
    for (int r = 0; r < reps; r++) {
        t0 = secs();
        gemm_rvv(A, B, C, M, N, K);
        ts[r] = secs() - t0;
    }
    if (check("rvv", C, Cref, M, N))
        return 1;
    printf("  rvv     %6.2f GOP/s\n", median_gops(ts, reps, ops));

    if (do_ime) {
        t0 = secs();
        pack8x16(A, Ap, M, K);
        pack8x16(B, Bp, N, K);
        double tpack = secs() - t0;
        for (int r = 0; r < reps; r++) {
            t0 = secs();
            gemm_ime2(Ap, Bp, C, M, N, K);
            ts[r] = secs() - t0;
        }
        if (check("ime2", C, Cref, M, N))
            return 1;
        double g_comp = median_gops(ts, reps, ops);
        for (int r = 0; r < reps; r++) {
            t0 = secs();
            pack8x16(A, Ap, M, K);
            pack8x16(B, Bp, N, K);
            gemm_ime2(Ap, Bp, C, M, N, K);
            ts[r] = secs() - t0;
        }
        double g_full = median_gops(ts, reps, ops);
        printf("  ime2-compute %6.2f GOP/s   (pack once %.3f s)\n", g_comp, tpack);
        printf("  ime2-full    %6.2f GOP/s\n", g_full);
    }
    return 0;
}
