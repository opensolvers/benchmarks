/* fftw-wisdom-preload.c — import wisdom only (no flag remap).
 * Uses dlopen(RTLD_DEEPBIND) on $FFTW_R5V_SO when set so wisdom matches that lib.
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <fftw3.h>
#include <stdio.h>
#include <stdlib.h>

typedef int (*import_file_fn)(FILE *);

__attribute__((constructor))
static void import_wisdom(void) {
  const char *w = getenv("FFTW_WISDOM_FILE");
  if (!w || !w[0]) {
    fprintf(stderr, "[fftw-wisdom-preload] set FFTW_WISDOM_FILE\n");
    return;
  }
  import_file_fn imp = NULL;
  const char *so = getenv("FFTW_R5V_SO");
  void *h = NULL;
  if (so && so[0]) {
    h = dlopen(so, RTLD_NOW | RTLD_DEEPBIND);
    if (h) imp = (import_file_fn)dlsym(h, "fftw_import_wisdom_from_file");
  }
  if (!imp) imp = (import_file_fn)dlsym(RTLD_DEFAULT, "fftw_import_wisdom_from_file");
  if (!imp) {
    fprintf(stderr, "[fftw-wisdom-preload] no fftw_import_wisdom_from_file\n");
    return;
  }
  FILE *f = fopen(w, "r");
  if (!f) {
    fprintf(stderr, "[fftw-wisdom-preload] cannot open %s\n", w);
    return;
  }
  int ok = imp(f);
  fclose(f);
  fprintf(stderr, "[fftw-wisdom-preload] %s: %s\n", w, ok ? "imported" : "FAIL");
}
