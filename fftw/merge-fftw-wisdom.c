/* merge-fftw-wisdom.c — import many wisdom files, export one union.
 *   merge-fftw-wisdom out.fftw in1.fftw [in2.fftw ...]
 */
#include <fftw3.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
  if (argc < 3) {
    fprintf(stderr, "usage: %s out.fftw in1.fftw [in2...]\n", argv[0]);
    return 2;
  }
  int n_ok = 0;
  for (int i = 2; i < argc; i++) {
    FILE *f = fopen(argv[i], "r");
    if (!f) {
      fprintf(stderr, "skip %s: cannot open\n", argv[i]);
      continue;
    }
    int ok = fftw_import_wisdom_from_file(f);
    fclose(f);
    fprintf(stderr, "import %s: %s\n", argv[i], ok ? "ok" : "FAIL");
    if (ok) n_ok++;
  }
  if (!n_ok) {
    fprintf(stderr, "no wisdom imported\n");
    return 1;
  }
  FILE *out = fopen(argv[1], "w");
  if (!out) {
    perror(argv[1]);
    return 1;
  }
  fftw_export_wisdom_to_file(out);
  fclose(out);
  fprintf(stderr, "wrote %s (%d inputs ok)\n", argv[1], n_ok);
  return 0;
}
