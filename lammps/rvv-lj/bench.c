/* Correctness + timing A/B: scalar vs RVV LJ/cut CSR pair (force-on-i). */
#include "lj_pair.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double wall_s(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

/* Simple cubic lattice + random jitter; Verlet list with cutoff. */
static void build_system(int n, double box, double cutoff, double **x_out, int **off_out,
                         int **neighs_out, int *nneigh_out) {
  double *x = aligned_alloc(64, (size_t)n * 3 * sizeof(double));
  int nside = (int)cbrt((double)n);
  while (nside * nside * nside < n)
    nside++;
  const double a = box / (double)nside;
  int idx = 0;
  for (int iz = 0; iz < nside && idx < n; iz++)
    for (int iy = 0; iy < nside && idx < n; iy++)
      for (int ix = 0; ix < nside && idx < n; ix++, idx++) {
        x[3 * idx + 0] = (ix + 0.5) * a + 0.01 * ((idx * 17) % 10);
        x[3 * idx + 1] = (iy + 0.5) * a + 0.01 * ((idx * 29) % 10);
        x[3 * idx + 2] = (iz + 0.5) * a + 0.01 * ((idx * 43) % 10);
      }

  const double cutsq = cutoff * cutoff;
  /* count */
  size_t nnz = 0;
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      if (i == j)
        continue;
      double dx = x[3 * i] - x[3 * j];
      double dy = x[3 * i + 1] - x[3 * j + 1];
      double dz = x[3 * i + 2] - x[3 * j + 2];
      /* min-image */
      if (dx > 0.5 * box)
        dx -= box;
      if (dx < -0.5 * box)
        dx += box;
      if (dy > 0.5 * box)
        dy -= box;
      if (dy < -0.5 * box)
        dy += box;
      if (dz > 0.5 * box)
        dz -= box;
      if (dz < -0.5 * box)
        dz += box;
      if (dx * dx + dy * dy + dz * dz < cutsq)
        nnz++;
    }
  }

  int *off = malloc((size_t)(n + 1) * sizeof(int));
  int *neighs = malloc(nnz * sizeof(int));
  int k = 0;
  off[0] = 0;
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      if (i == j)
        continue;
      double dx = x[3 * i] - x[3 * j];
      double dy = x[3 * i + 1] - x[3 * j + 1];
      double dz = x[3 * i + 2] - x[3 * j + 2];
      if (dx > 0.5 * box)
        dx -= box;
      if (dx < -0.5 * box)
        dx += box;
      if (dy > 0.5 * box)
        dy -= box;
      if (dy < -0.5 * box)
        dy += box;
      if (dz > 0.5 * box)
        dz -= box;
      if (dz < -0.5 * box)
        dz += box;
      /* store unwrapped delta via shifting j coord into neighbor of i's image:
         for the microkernel we pass positions as-is; use a ghost trick:
         encode by temporarily using minimum-image positions in a scratch —
         simpler: build list only for pairs within cut using raw positions in
         a larger effective cutoff without PBC in the kernel.
         → Recompute positions with PBC images duplicated is heavy.
         For this bench: NO PBC in force kernel; only include pairs with
         Euclidean distance < cut in the primary box (no wrap). */
      dx = x[3 * i] - x[3 * j];
      dy = x[3 * i + 1] - x[3 * j + 1];
      dz = x[3 * i + 2] - x[3 * j + 2];
      if (dx * dx + dy * dy + dz * dz < cutsq)
        neighs[k++] = j;
    }
    off[i + 1] = k;
  }

  *x_out = x;
  *off_out = off;
  *neighs_out = neighs;
  *nneigh_out = k;
  (void)box;
}

static double max_abs_diff(const double *a, const double *b, int n) {
  double m = 0.0;
  for (int i = 0; i < n; i++) {
    double d = fabs(a[i] - b[i]);
    if (d > m)
      m = d;
  }
  return m;
}

int main(int argc, char **argv) {
  int n = 2048;
  int rounds = 20;
  if (argc > 1)
    n = atoi(argv[1]);
  if (argc > 2)
    rounds = atoi(argv[2]);

  /* LJ units: epsilon=1, sigma=1 → lj1=48, lj2=24 (LAMMPS lj/cut). cut=2.5 */
  LjParams params = {.lj1 = 48.0, .lj2 = 24.0, .cutsq = 2.5 * 2.5};
  const double box = cbrt((double)n / 0.84); /* ~liquid density */

  double *x;
  int *off, *neighs, nnz;
  build_system(n, box, 2.5, &x, &off, &neighs, &nnz);
  printf("n=%d box=%.3f nnz=%d avg_neigh=%.1f rounds=%d\n", n, box, nnz,
         (double)nnz / (double)n, rounds);

  double *fs = aligned_alloc(64, (size_t)n * 3 * sizeof(double));
  double *fv = aligned_alloc(64, (size_t)n * 3 * sizeof(double));

  memset(fs, 0, (size_t)n * 3 * sizeof(double));
  lj_pair_scalar_csr(n, x, off, neighs, &params, fs);

  memset(fv, 0, (size_t)n * 3 * sizeof(double));
  lj_pair_rvv_csr(n, x, off, neighs, &params, fv);

  const double err = max_abs_diff(fs, fv, n * 3);
  printf("max|f_scalar - f_rvv| = %.3e %s\n", err, err < 1e-8 ? "OK" : "FAIL");

  /* timing */
  for (int w = 0; w < 3; w++) {
    memset(fs, 0, (size_t)n * 3 * sizeof(double));
    lj_pair_scalar_csr(n, x, off, neighs, &params, fs);
  }
  double t0 = wall_s();
  for (int r = 0; r < rounds; r++) {
    memset(fs, 0, (size_t)n * 3 * sizeof(double));
    lj_pair_scalar_csr(n, x, off, neighs, &params, fs);
  }
  double ts = wall_s() - t0;

  for (int w = 0; w < 3; w++) {
    memset(fv, 0, (size_t)n * 3 * sizeof(double));
    lj_pair_rvv_csr(n, x, off, neighs, &params, fv);
  }
  t0 = wall_s();
  for (int r = 0; r < rounds; r++) {
    memset(fv, 0, (size_t)n * 3 * sizeof(double));
    lj_pair_rvv_csr(n, x, off, neighs, &params, fv);
  }
  double tv = wall_s() - t0;

  const double pairs = (double)nnz * (double)rounds;
  printf("scalar: %.4f s  (%.3f ns/pair)\n", ts, 1e9 * ts / pairs);
  printf("rvv:    %.4f s  (%.3f ns/pair)  speedup %.2fx\n", tv, 1e9 * tv / pairs, ts / tv);

  free(x);
  free(off);
  free(neighs);
  free(fs);
  free(fv);
  return err < 1e-8 ? 0 : 1;
}
