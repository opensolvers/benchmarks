// bench_arma.cpp — Armadillo DGEMM + eig_sym FlexiBLAS proxy (SpaceMiT X60).
// Mirrors numpy/bench_blas.py: dgemm GFLOP/s + symmetric eigensolve seconds + finite gate.
//
// Build (EESSI foss-2023b):
//   module load Armadillo/12.8.0-foss-2023b
//   g++ -O2 -std=c++17 bench_arma.cpp -o bench_arma -larmadillo
//
// Run:
//   OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 ./bench_arma [dgemm_N] [eig_N]
// A/B:
//   OPENBLAS_CORETYPE=RISCV64_GENERIC ./bench_arma
//   FLEXIBLAS=/path/to/libopenblas.so ./bench_arma
#include <armadillo>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>

using SteadyClock = std::chrono::steady_clock;

static double secs(SteadyClock::time_point a, SteadyClock::time_point b) {
  return std::chrono::duration<double>(b - a).count();
}

int main(int argc, char** argv) {
  const arma::uword n = (argc > 1) ? std::strtoul(argv[1], nullptr, 10) : 2048;
  const arma::uword m = (argc > 2) ? std::strtoul(argv[2], nullptr, 10) : 1024;

  arma::arma_rng::set_seed(0);
  arma::mat A = arma::randn(n, n);
  arma::mat B = arma::randn(n, n);
  arma::mat C = A * B;  // warmup
  (void)C;

  double best = 1e100;
  for (int r = 0; r < 3; ++r) {
    auto t0 = SteadyClock::now();
    C = A * B;
    best = std::min(best, secs(t0, SteadyClock::now()));
  }
  const double gflops = 2.0 * double(n) * double(n) * double(n) / best / 1e9;
  const bool gemm_ok = C.is_finite();
  std::printf("  DGEMM N=%llu: %6.2f s   %6.2f GFLOP/s   finite=%s\n",
              (unsigned long long)n, best, gflops, gemm_ok ? "True" : "False");

  arma::mat S = arma::randn(m, m);
  S = S * S.t();
  arma::vec eigval;
  auto t0 = SteadyClock::now();
  const bool ok = arma::eig_sym(eigval, S);
  const double te = secs(t0, SteadyClock::now());
  const bool eig_ok = ok && eigval.is_finite();
  std::printf("  EIG   N=%llu: %6.2f s   (LAPACK dsyev*)         finite=%s\n",
              (unsigned long long)m, te, eig_ok ? "True" : "False");

  return (gemm_ok && eig_ok) ? 0 : 2;
}
