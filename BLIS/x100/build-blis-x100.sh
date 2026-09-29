#!/bin/bash
# Apply X100 8x8 (d) / 2vx8 (s) kernels onto ~/blis → ~/blis-x100-install
GCC14=/cvmfs/dev.eessi.io/riscv/versions/2025.06-001/software/linux/riscv64/generic/software/GCCcore/14.3.0
export PATH="$GCC14/bin:$PATH"
export LD_LIBRARY_PATH="$GCC14/lib64:$GCC14/lib:${LD_LIBRARY_PATH:-}"
export LDFLAGS="-L$GCC14/lib -B$GCC14/lib -Wl,-rpath,$GCC14/lib ${LDFLAGS:-}"
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
SRC=${BLIS_SRC:-$HOME/blis}
PREFIX=${1:-$HOME/blis-x100-install}
cp "$HERE"/bli_sdgemm_rviv_asm_2vx8.h "$HERE"/bli_{s,d}gemm_rviv_{2vx8.c,asm_2vx8.S} \
   "$HERE"/bli_dgemm_rviv_8x8.c "$SRC/kernels/rviv/3/"
cp "$HERE"/bli_kernels_rviv.h "$SRC/kernels/rviv/"
cp "$HERE"/bli_cntx_init_rv64iv.c "$SRC/config/rv64iv/"
cd "$SRC"
make distclean >/dev/null 2>&1 || true
./configure --prefix="$PREFIX" --enable-cblas --enable-blas --enable-threading=openmp rv64iv
make -j"$(nproc)" && make install
echo "Installed to $PREFIX"
echo "Run with: OMP_PROC_BIND=close OMP_PLACES=cores BLIS_NUM_THREADS=<n> …"
