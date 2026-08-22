#!/usr/bin/env python3
"""bench_sklearn.py — FlexiBLAS-backed scikit-learn proxy (PCA + Ridge).

Times two BLAS-heavy estimators and reports wall time + a residual checksum.

  PCA   : TruncatedSVD / PCA on dense X (N x D) — SVD / GEMM heavy
  Ridge : linear_model.Ridge fit — LAPACK/BLAS

A/B via FlexiBLAS env (same as numpy/):
  OPENBLAS_CORETYPE=RISCV64_GENERIC python3 bench_sklearn.py
  FLEXIBLAS=/path/to/libopenblas.so python3 bench_sklearn.py
"""
import sys
import time

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    d = int(sys.argv[2]) if len(sys.argv) > 2 else 512
    k = int(sys.argv[3]) if len(sys.argv) > 3 else 64
    rng = np.random.default_rng(0)
    X = rng.standard_normal((n, d))
    y = X @ rng.standard_normal(d) + 0.1 * rng.standard_normal(n)

    # PCA
    pca = PCA(n_components=k, svd_solver="full", random_state=0)
    t0 = time.perf_counter()
    Z = pca.fit_transform(X)
    tp = time.perf_counter() - t0
    chk_p = float(np.sum(Z) + np.sum(pca.explained_variance_))
    print(
        f"  PCA   N={n} D={d} K={k}: {tp:6.2f} s   "
        f"finite={bool(np.isfinite(Z).all())}  chk={chk_p:.6e}"
    )

    # Ridge
    ridge = Ridge(alpha=1.0, solver="cholesky")
    t0 = time.perf_counter()
    ridge.fit(X, y)
    tr = time.perf_counter() - t0
    pred = ridge.predict(X[:100])
    chk_r = float(np.sum(ridge.coef_) + np.sum(pred))
    print(
        f"  Ridge N={n} D={d}:      {tr:6.2f} s   "
        f"finite={bool(np.isfinite(ridge.coef_).all())}  chk={chk_r:.6e}"
    )


if __name__ == "__main__":
    main()
