#!/bin/bash
set -euo pipefail
R5=/home/orangepi/fftwbuild/src-r5v
EESSI_SW=/cvmfs/dev.eessi.io/riscv/versions/2025.06-001/software/linux/riscv64/generic/software
export PATH=$EESSI_SW/GCCcore/14.3.0/bin:$PATH
export LIBRARY_PATH=$EESSI_SW/GCCcore/14.3.0/lib64
export LD_LIBRARY_PATH=$EESSI_SW/GCCcore/14.3.0/lib64
mkdir -p /tmp/t2test && cd /tmp/t2test

python3 <<'PY'
from pathlib import Path

raw = Path("/home/orangepi/fftwbuild/src-r5v/dft/simd/common/t2bv_8.c").read_text()
# Keep preamble + FMA branch only
start = raw.index("#if defined(ARCH_PREFERS_FMA)")
else_pos = raw.index("\n#else\n", start)
endif_pos = raw.index("\n#endif\n", else_pos)
fma_only = raw[:start] + raw[start:else_pos] + raw[endif_pos:]  # drop #else..#endif content, keep final #endif
fma_only = fma_only.replace("#if defined(ARCH_PREFERS_FMA) || defined(ISA_EXTENSION_PREFERS_FMA)\n", "")
fma_only = fma_only.replace("\n#endif\n", "\n", 1)  # remove matching endif
fma_only = fma_only.replace("static void t2bv_8(", "void stock_t2bv_8(")
fma_only = fma_only.replace("XSIMD(codelet_t2bv_8)", "stock_codelet_t2bv_8_UNUSED")
fma_only = fma_only.replace("(p, t2bv_8, &desc)", "(p, stock_t2bv_8, &desc)")
assert fma_only.count("void stock_t2bv_8") == 1
assert "VBYI" not in fma_only or fma_only.count("VFNMSI") >= 2
Path("stock_t2bv_8.c").write_text(
    '#define SIMD_HEADER "simd-support/simd-r5v256.h"\n' + fma_only
)
print("stock ok, VFNMSI", fma_only.count("VFNMSI"))

body = Path("/home/orangepi/fftw-wisdom-src/t2bv_8_r5v256_split.c").read_text()
body = body.replace('#include "../common/t2bv_8.c"', '#error no fallback\n')
body = body.replace("static void t2bv_8(", "void split_t2bv_8(")
body = body.replace("XSIMD(codelet_t2bv_8)", "split_codelet_t2bv_8_UNUSED")
body = body.replace("(p, t2bv_8, &desc)", "(p, split_t2bv_8, &desc)")
Path("split_t2bv_8.c").write_text(
    '#define SIMD_HEADER "simd-support/simd-r5v256.h"\n' + body
)
print("split ok")
PY

cat > drive.c <<'EOF'
#include <stdio.h>
#include <math.h>
typedef double R;
typedef int INT;
typedef INT stride;
void stock_t2bv_8(R *ri, R *ii, const R *W, stride rs, INT mb, INT me, INT ms);
void split_t2bv_8(R *ri, R *ii, const R *W, stride rs, INT mb, INT me, INT ms);
void fftw_kdft_dit_register(void *p, void *f, const void *d) { (void)p;(void)f;(void)d; }
char fftw_dft_t2bsimd_genus_r5v256;
enum { VL=2, TWVL=4, N=8 };
int main(void){
  R buf_s[N*2*VL], buf_k[N*2*VL], W[TWVL*14*8];
  INT rs=4, mb=0, me=VL, ms=2;
  for(int i=0;i<N*2*VL;i++){ buf_s[i]=sin(0.1*i)+0.01*i; buf_k[i]=buf_s[i]; }
  for(int i=0;i<(int)(sizeof(W)/sizeof(W[0]));i++) W[i]=cos(0.03*i)+0.001*i;
  stock_t2bv_8(buf_s+1, buf_s, W, rs, mb, me, ms);
  split_t2bv_8(buf_k+1, buf_k, W, rs, mb, me, ms);
  double maxe=0;
  for(int i=0;i<N*2*VL;i++){
    double e=fabs(buf_s[i]-buf_k[i]);
    if(e>maxe) maxe=e;
  }
  printf("max_abs_err=%.3e\n", maxe);
  for(int i=0;i<8;i++)
    printf(" %d: stock=%.8f split=%.8f\n", i, buf_s[i], buf_k[i]);
  return maxe>1e-6;
}
EOF

CFLAGS="-O3 -march=rv64imafdcv_zvl256b -DHAVE_CONFIG_H -I$R5 -I$R5/dft -I$R5/api -I$R5/kernel"
gcc $CFLAGS -c stock_t2bv_8.c -o stock_t2bv_8.o
gcc $CFLAGS -c split_t2bv_8.c -o split_t2bv_8.o
nm stock_t2bv_8.o | grep ' stock_t2bv'
gcc $CFLAGS drive.c stock_t2bv_8.o split_t2bv_8.o -o drive -lm
./drive
