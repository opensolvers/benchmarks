#!/usr/bin/env Rscript
# bench_r.R — FlexiBLAS-backed R BLAS/LAPACK proxy (%*% + eigen).
#
# Usage: Rscript bench_r.R [dgemm_N] [eig_N]
# A/B via FlexiBLAS env (same as numpy/):
#   OPENBLAS_CORETYPE=RISCV64_GENERIC Rscript bench_r.R
#   FLEXIBLAS=/path/to/libopenblas.so Rscript bench_r.R

args <- commandArgs(trailingOnly = TRUE)
n <- if (length(args) >= 1) as.integer(args[1]) else 2048L
m <- if (length(args) >= 2) as.integer(args[2]) else 1024L

set.seed(0L)
best <- function(fn, reps = 3L) {
  b <- Inf
  r <- NULL
  for (i in seq_len(reps)) {
    t0 <- proc.time()
    r <- fn()
    b <- min(b, (proc.time() - t0)[["elapsed"]])
  }
  list(time = b, result = r)
}

A <- matrix(rnorm(n * n), n, n)
B <- matrix(rnorm(n * n), n, n)
A %*% B  # warmup
g <- best(function() A %*% B, 3L)
C <- g$result
gflops <- 2 * n^3 / g$time / 1e9
cat(sprintf(
  "  GEMM  N=%d: %6.2f s   %6.2f GFLOP/s   finite=%s\n",
  n, g$time, gflops, all(is.finite(C))
))

S <- crossprod(matrix(rnorm(m * m), m, m))
e <- best(function() eigen(S, symmetric = TRUE, only.values = TRUE)$values, 1L)
w <- e$result
cat(sprintf(
  "  EIGEN N=%d: %6.2f s   (LAPACK dsyev*)         finite=%s  chk=%.6e\n",
  m, e$time, all(is.finite(w)), sum(w)
))

if (!all(is.finite(C)) || !all(is.finite(w))) {
  quit(status = 2L)
}
