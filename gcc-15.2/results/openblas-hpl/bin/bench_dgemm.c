#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

extern void dgemm_(const char*, const char*, const int*, const int*, const int*,
                   const double*, const double*, const int*, const double*, const int*,
                   const double*, double*, const int*);

static double now_raw(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC_RAW, &t);
  return t.tv_sec + t.tv_nsec * 1e-9;
}

int main(int argc, char **argv) {
  int n = argc > 1 ? atoi(argv[1]) : 1024;
  int reps = argc > 2 ? atoi(argv[2]) : 5;
  if (n <= 0 || reps <= 0) return 2;
  size_t nn = (size_t)n * (size_t)n;
  double *A = aligned_alloc(64, nn * sizeof(double));
  double *B = aligned_alloc(64, nn * sizeof(double));
  double *C = aligned_alloc(64, nn * sizeof(double));
  if (!A || !B || !C) { fprintf(stderr, "alloc fail\n"); return 1; }
  for (size_t i = 0; i < nn; i++) {
    A[i] = (i % 7) * 0.1;
    B[i] = (i % 5) * 0.2;
    C[i] = 0.0;
  }
  double alpha = 1.0, beta = 0.0;
  /* warmup */
  dgemm_("N", "N", &n, &n, &n, &alpha, A, &n, B, &n, &beta, C, &n);

  double best = 1e300, sum = 0.0;
  for (int r = 0; r < reps; r++) {
    double t0 = now_raw();
    dgemm_("N", "N", &n, &n, &n, &alpha, A, &n, B, &n, &beta, C, &n);
    double dt = now_raw() - t0;
    if (dt < best) best = dt;
    sum += dt;
  }
  double flops = 2.0 * (double)n * (double)n * (double)n;
  double best_g = flops / best / 1e9;
  double mean_g = flops / (sum / reps) / 1e9;
  /* simple checksum */
  double cs = 0.0;
  for (size_t i = 0; i < nn; i += (size_t)n + 1) cs += C[i];
  printf("N=%d reps=%d best_gflops=%.6f mean_gflops=%.6f checksum=%.6e best_s=%.6f mean_s=%.6f\n",
         n, reps, best_g, mean_g, cs, best, sum / reps);
  free(A); free(B); free(C);
  return 0;
}
