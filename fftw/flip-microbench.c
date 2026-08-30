/* Isolated FLIP_RI microbench: gather vs slide+merge vs store-shuffle.
 * Build: gcc -O3 -march=rv64imafdcv_zvl256b flip-microbench.c -o flip-mb -lm
 */
#include <riscv_vector.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + 1e-9 * ts.tv_nsec;
}

static inline vfloat64m1_t flip_gather(vfloat64m1_t x) {
  vuint64m1_t idx = __riscv_vid_v_u64m1(4);
  vuint64m1_t hidx = __riscv_vxor_vx_u64m1(idx, 1, 4);
  return __riscv_vrgather_vv_f64m1(x, hidx, 4);
}

static inline vfloat64m1_t flip_slide(vfloat64m1_t x) {
  vfloat64m1_t down = __riscv_vslidedown_vx_f64m1(x, 1, 4);
  vfloat64m1_t up = __riscv_vslideup_vx_f64m1(x, x, 1, 4);
  vuint64m1_t idx = __riscv_vid_v_u64m1(4);
  vuint64m1_t odd = __riscv_vand_vx_u64m1(idx, 1, 4);
  vbool64_t even = __riscv_vmseq_vx_u64m1_b64(odd, 0, 4);
  return __riscv_vmerge_vvm_f64m1(up, down, even, 4);
}

static inline vfloat64m1_t flip_store(vfloat64m1_t x) {
  double a[4] __attribute__((aligned(32)));
  __riscv_vse64_v_f64m1(a, x, 4);
  double t0 = a[0]; a[0] = a[1]; a[1] = t0;
  double t2 = a[2]; a[2] = a[3]; a[3] = t2;
  return __riscv_vle64_v_f64m1(a, 4);
}

static int check_eq(vfloat64m1_t a, vfloat64m1_t b, const char *tag) {
  double aa[4], bb[4];
  __riscv_vse64_v_f64m1(aa, a, 4);
  __riscv_vse64_v_f64m1(bb, b, 4);
  for (int i = 0; i < 4; i++) {
    if (aa[i] != bb[i]) {
      printf("FAIL %s i=%d %g vs %g\n", tag, i, aa[i], bb[i]);
      return 0;
    }
  }
  return 1;
}

#define REPS 20000000

static double bench(vfloat64m1_t (*fn)(vfloat64m1_t), vfloat64m1_t x0, const char *name) {
  vfloat64m1_t x = x0;
  /* warmup */
  for (int i = 0; i < 1000; i++) x = fn(x);
  double t0 = now();
  for (int i = 0; i < REPS; i++) x = fn(x);
  double t = now() - t0;
  double a[4];
  __riscv_vse64_v_f64m1(a, x, 4);
  printf("%-12s  %.4fs  (sink %g)  ~%.1f ns/call\n", name, t, a[0]+a[1], 1e9*t/REPS);
  return t;
}

int main(void) {
  double init[4] = {1.0, 2.0, 3.0, 4.0};
  vfloat64m1_t x = __riscv_vle64_v_f64m1(init, 4);
  vfloat64m1_t g = flip_gather(x);
  if (!check_eq(g, flip_slide(x), "slide") || !check_eq(g, flip_store(x), "store"))
    return 1;
  printf("correctness OK (1,2,3,4)->");
  double o[4]; __riscv_vse64_v_f64m1(o, g, 4);
  printf("(%g,%g,%g,%g)\n", o[0], o[1], o[2], o[3]);
  double tg = bench(flip_gather, x, "gather");
  double ts = bench(flip_slide, x, "slide");
  double tt = bench(flip_store, x, "store");
  printf("slide/gather=%.3f  store/gather=%.3f\n", ts/tg, tt/tg);
  return 0;
}
