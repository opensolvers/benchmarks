/* scafacos_bench.c — timed ScaFaCoS P3M (and optional FMM) solve.
 *
 * Lattice of alternating charges; reports wall time + potential checksum.
 * P3M path uses FFTW → A/B via LD_PRELOAD of libfftw3.so.3.
 *
 * Usage: mpirun -np N ./scafacos_bench [method] [N_side] [reps]
 *   method: p3m (default) or fmm
 *   N_side: particles = N_side^3 (default 16 → 4096)
 *   reps:   timed fcs_run repetitions after tune (default 5)
 *
 * SPDX-License-Identifier: MIT
 */
#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <fcs.h>

static void die_fcs(const char *where, FCSResult r) {
  if (r == FCS_RESULT_SUCCESS)
    return;
  fprintf(stderr, "FCS error in %s: [%s] %s\n", where,
          fcs_result_get_function(r), fcs_result_get_message(r));
  fcs_result_destroy(r);
  MPI_Abort(MPI_COMM_WORLD, 1);
}

static double now_s(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + 1e-9 * ts.tv_nsec;
}

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  int rank = 0, nproc = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);

  const char *method = (argc > 1) ? argv[1] : "p3m";
  int n_side = (argc > 2) ? atoi(argv[2]) : 16;
  int reps = (argc > 3) ? atoi(argv[3]) : 5;
  if (n_side < 2)
    n_side = 2;
  if (reps < 1)
    reps = 1;

  fcs_int total = (fcs_int)n_side * n_side * n_side;
  /* distribute particles across ranks */
  fcs_int base = total / nproc;
  fcs_int rem = total % nproc;
  fcs_int local_n = base + (rank < rem ? 1 : 0);
  fcs_int start = rank * base + (rank < rem ? rank : rem);

  const fcs_float box = 1.0;
  const fcs_float spacing = box / (fcs_float)n_side;

  fcs_float *pos = calloc((size_t)local_n * 3, sizeof(fcs_float));
  fcs_float *q = calloc((size_t)local_n, sizeof(fcs_float));
  fcs_float *field = calloc((size_t)local_n * 3, sizeof(fcs_float));
  fcs_float *pot = calloc((size_t)local_n, sizeof(fcs_float));
  if (!pos || !q || !field || !pot) {
    fprintf(stderr, "OOM\n");
    MPI_Abort(MPI_COMM_WORLD, 2);
  }

  for (fcs_int i = 0; i < local_n; ++i) {
    fcs_int g = start + i;
    int iz = (int)(g % n_side);
    int iy = (int)((g / n_side) % n_side);
    int ix = (int)(g / (n_side * n_side));
    pos[3 * i + 0] = (ix + 0.5) * spacing;
    pos[3 * i + 1] = (iy + 0.5) * spacing;
    pos[3 * i + 2] = (iz + 0.5) * spacing;
    q[i] = ((ix + iy + iz) & 1) ? 1.0 : -1.0;
  }

  FCS handle = FCS_NULL;
  die_fcs("init", fcs_init(&handle, method, MPI_COMM_WORLD));

  fcs_float a[3] = {box, 0, 0}, b[3] = {0, box, 0}, c[3] = {0, 0, box};
  fcs_float origin[3] = {0, 0, 0};
  fcs_int period[3] = {1, 1, 1};
  die_fcs("set_common",
          fcs_set_common(handle, 1 /* near field */, a, b, c, origin, period,
                         total));

  if (strcmp(method, "p3m") == 0) {
    die_fcs("p3m_tol", fcs_p3m_set_tolerance_field(handle, 1e-3));
  }

  die_fcs("tune", fcs_tune(handle, local_n, pos, q));

  /* warmup */
  die_fcs("warmup", fcs_run(handle, local_n, pos, q, field, pot));

  double t0 = now_s();
  for (int r = 0; r < reps; ++r)
    die_fcs("run", fcs_run(handle, local_n, pos, q, field, pot));
  double wall = now_s() - t0;

  double local_psum = 0.0, local_fsum = 0.0;
  for (fcs_int i = 0; i < local_n; ++i) {
    local_psum += pot[i] * q[i];
    local_fsum += fabs(field[3 * i]) + fabs(field[3 * i + 1]) +
                  fabs(field[3 * i + 2]);
  }
  double psum = 0.0, fsum = 0.0;
  MPI_Reduce(&local_psum, &psum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(&local_fsum, &fsum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    printf("ScaFaCoS method=%s N=%d np=%d reps=%d "
           "wall=%.4fs per_run=%.4fs E≈%.6f |F|_sum=%.6f\n",
           method, (int)total, nproc, reps, wall, wall / reps, 0.5 * psum,
           fsum);
  }

  fcs_destroy(handle);
  free(pos);
  free(q);
  free(field);
  free(pot);
  MPI_Finalize();
  return 0;
}
