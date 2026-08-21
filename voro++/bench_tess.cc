// Timed Voronoi tessellation for RVV auto-vec A/B.
// Builds cells for N random particles in a cube; reports wall time + volume checksum.
#include "voro++.hh"
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <vector>

using namespace voro;

static double rnd() { return double(rand()) / RAND_MAX; }

int main(int argc, char **argv) {
  const int n = (argc > 1) ? atoi(argv[1]) : 20000;
  const int reps = (argc > 2) ? atoi(argv[2]) : 3;
  const unsigned seed = (argc > 3) ? (unsigned)atoi(argv[3]) : 42u;

  const double lo = 0.0, hi = 1.0;
  const double cvol = (hi - lo) * (hi - lo) * (hi - lo);
  // ~n^{1/3} blocks; clamp to keep blocks ~2–4 particles each
  int nblk = (int)std::cbrt((double)n / 3.0);
  if (nblk < 4) nblk = 4;
  if (nblk > 64) nblk = 64;

  std::vector<double> xs(n), ys(n), zs(n);
  srand(seed);
  for (int i = 0; i < n; i++) {
    xs[i] = lo + rnd() * (hi - lo);
    ys[i] = lo + rnd() * (hi - lo);
    zs[i] = lo + rnd() * (hi - lo);
  }

  double best_ms = 1e300, last_vvol = 0.0;
  long long last_faces = 0;

  for (int r = 0; r < reps; r++) {
    container con(lo, hi, lo, hi, lo, hi, nblk, nblk, nblk, false, false, false, 8);
    for (int i = 0; i < n; i++) con.put(i, xs[i], ys[i], zs[i]);

    auto t0 = std::chrono::steady_clock::now();
    double vvol = 0.0;
    long long faces = 0;
    c_loop_all cl(con);
    voronoicell c;
    if (cl.start()) do {
      if (con.compute_cell(c, cl)) {
        vvol += c.volume();
        faces += c.number_of_faces();
      }
    } while (cl.inc());
    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    if (ms < best_ms) best_ms = ms;
    last_vvol = vvol;
    last_faces = faces;
    printf("rep=%d  n=%d  ms=%.3f  vvol=%.10g  dvol=%.3e  faces=%lld\n",
           r, n, ms, vvol, vvol - cvol, faces);
  }

  printf("BEST_MS=%.3f  N=%d  REPS=%d  VVOL=%.10g  FACES=%lld\n",
         best_ms, n, reps, last_vvol, last_faces);
  return 0;
}
