/* fftw-est2meas-interposer.c — LD_PRELOAD for apps that hardcode FFTW_ESTIMATE.
 *
 * Binds plan/import/export to one libfftw3 via dlopen(RTLD_DEEPBIND) so a second
 * copy (EESSI RPATH) cannot corrupt wisdom hashes. Remaps ESTIMATE ->
 * MEASURE|PATIENT|EXHAUSTIVE ($FFTW_EST2MEAS_FLAGS).
 *
 *   FFTW_R5V_SO=/path/to/libfftw3.so.3.6.10   # required under MPI RPATH binaries
 *   FFTW_EST2MEAS_FLAGS=patient               # default: measure
 *   FFTW_WISDOM_FILE=w.fftw LD_PRELOAD=./libfftw-est2meas.so[:r5v] app
 *
 * MPI: each rank writes $FFTW_WISDOM_OUT.rankN; merge with merge-fftw-wisdom.
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <fftw3.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef fftw_plan (*plan_dft_3d_fn)(int, int, int, fftw_complex *, fftw_complex *,
                                    int, unsigned);
typedef fftw_plan (*plan_many_dft_fn)(int, const int *, int,
                                      fftw_complex *, const int *, int, int,
                                      fftw_complex *, const int *, int, int,
                                      int, unsigned);
typedef int (*import_file_fn)(FILE *);
typedef void (*export_file_fn)(FILE *);

static void *fftw_h;
static plan_dft_3d_fn real_dft_3d;
static plan_many_dft_fn real_many_dft;
static import_file_fn real_import;
static export_file_fn real_export;
static int inited;
static const char *wisdom_out;
static unsigned target_flags = FFTW_MEASURE;
static const char *target_name = "MEASURE";

static unsigned remap(unsigned flags) {
  if (!(flags & FFTW_ESTIMATE))
    return flags;
  flags &= ~((unsigned)FFTW_ESTIMATE | (unsigned)FFTW_PATIENT |
             (unsigned)FFTW_EXHAUSTIVE);
  return flags | target_flags;
}

static int mpi_rank(void) {
  const char *s;
  if ((s = getenv("OMPI_COMM_WORLD_RANK")) && s[0]) return atoi(s);
  if ((s = getenv("PMI_RANK")) && s[0]) return atoi(s);
  if ((s = getenv("PMIX_RANK")) && s[0]) return atoi(s);
  return -1;
}

static void export_on_exit(void) {
  if (!wisdom_out || !wisdom_out[0] || !real_export) return;
  int rank = mpi_rank();
  char path[4096];
  if (rank >= 0)
    snprintf(path, sizeof path, "%s.rank%d", wisdom_out, rank);
  else
    snprintf(path, sizeof path, "%s", wisdom_out);
  FILE *f = fopen(path, "w");
  if (!f) {
    fprintf(stderr, "[fftw-est2meas] cannot write %s: %s\n", path, strerror(errno));
    return;
  }
  real_export(f);
  fclose(f);
  fprintf(stderr, "[fftw-est2meas] rank %d exported wisdom → %s\n", rank, path);
}

static void pick_target(void) {
  const char *s = getenv("FFTW_EST2MEAS_FLAGS");
  if (!s || !s[0] || !strcasecmp(s, "measure")) {
    target_flags = FFTW_MEASURE;
    target_name = "MEASURE";
  } else if (!strcasecmp(s, "patient")) {
    target_flags = FFTW_PATIENT;
    target_name = "PATIENT";
  } else if (!strcasecmp(s, "exhaustive")) {
    target_flags = FFTW_EXHAUSTIVE | FFTW_PATIENT;
    target_name = "EXHAUSTIVE";
  } else {
    fprintf(stderr,
            "[fftw-est2meas] unknown FFTW_EST2MEAS_FLAGS=%s; defaulting to MEASURE\n",
            s);
    target_flags = FFTW_MEASURE;
    target_name = "MEASURE";
  }
}

static void init_once(void) {
  if (inited) return;
  inited = 1;
  pick_target();

  const char *so = getenv("FFTW_R5V_SO");
  if (so && so[0]) {
    fftw_h = dlopen(so, RTLD_NOW | RTLD_DEEPBIND);
    if (!fftw_h)
      fprintf(stderr, "[fftw-est2meas] dlopen(%s): %s\n", so, dlerror());
  }
  if (fftw_h) {
    real_dft_3d = (plan_dft_3d_fn)dlsym(fftw_h, "fftw_plan_dft_3d");
    real_many_dft = (plan_many_dft_fn)dlsym(fftw_h, "fftw_plan_many_dft");
    real_import = (import_file_fn)dlsym(fftw_h, "fftw_import_wisdom_from_file");
    real_export = (export_file_fn)dlsym(fftw_h, "fftw_export_wisdom_to_file");
  } else {
    real_dft_3d = (plan_dft_3d_fn)dlsym(RTLD_NEXT, "fftw_plan_dft_3d");
    real_many_dft = (plan_many_dft_fn)dlsym(RTLD_NEXT, "fftw_plan_many_dft");
    real_import = (import_file_fn)dlsym(RTLD_NEXT, "fftw_import_wisdom_from_file");
    real_export = (export_file_fn)dlsym(RTLD_NEXT, "fftw_export_wisdom_to_file");
  }
  if (!real_dft_3d || !real_many_dft || !real_import || !real_export) {
    fprintf(stderr, "[fftw-est2meas] dlsym failed: %s\n", dlerror());
    abort();
  }

  const char *w = getenv("FFTW_WISDOM_FILE");
  if (w && w[0]) {
    FILE *f = fopen(w, "r");
    if (f) {
      int ok = real_import(f);
      fclose(f);
      if (mpi_rank() <= 0)
        fprintf(stderr, "[fftw-est2meas] wisdom %s: %s\n", w, ok ? "ok" : "FAIL");
    } else if (mpi_rank() <= 0) {
      fprintf(stderr, "[fftw-est2meas] cannot open wisdom %s\n", w);
    }
  }
  wisdom_out = getenv("FFTW_WISDOM_OUT");
  if (wisdom_out && wisdom_out[0])
    atexit(export_on_exit);
  if (mpi_rank() <= 0)
    fprintf(stderr, "[fftw-est2meas] ESTIMATE -> %s%s (lib=%s)\n", target_name,
            (wisdom_out && wisdom_out[0]) ? " (+export on exit)" : "",
            so && so[0] ? so : "RTLD_NEXT");
}

/* Mute OpenMP FFTW API — QE links libfftw3_omp against stock; avoid dual state. */
int fftw_init_threads(void) { return 1; }
void fftw_cleanup_threads(void) {}
void fftw_plan_with_nthreads(int nthreads) { (void)nthreads; }
void fftw_plan_with_nthreads_(int *nthreads) {
  if (nthreads) fftw_plan_with_nthreads(*nthreads);
}

fftw_plan fftw_plan_dft_3d(int nx, int ny, int nz,
                           fftw_complex *in, fftw_complex *out,
                           int sign, unsigned flags) {
  init_once();
  return real_dft_3d(nx, ny, nz, in, out, sign, remap(flags));
}

fftw_plan fftw_plan_many_dft(int rank, const int *n, int howmany,
                             fftw_complex *in, const int *inembed,
                             int istride, int idist,
                             fftw_complex *out, const int *onembed,
                             int ostride, int odist,
                             int sign, unsigned flags) {
  init_once();
  return real_many_dft(rank, n, howmany, in, inembed, istride, idist,
                       out, onembed, ostride, odist, sign, remap(flags));
}
