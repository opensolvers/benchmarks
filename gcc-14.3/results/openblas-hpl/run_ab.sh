#!/usr/bin/env bash
set -eo pipefail
OUT=/home/orangepi/gcc-tune/measurements/results/rv2-gcc143-openblas-hpl-ab
ROOT=/home/orangepi/gcc-tune
GCC143=$ROOT/install/gcc-14.3.0-x60/bin/gcc
GCC143_LIB=$ROOT/install/gcc-14.3.0-x60/lib
AS_BINDIR=$ROOT/tools/bin
GFORTRAN_LIB=/cvmfs/dev.eessi.io/riscv/versions/2025.06-001/software/linux/riscv64/generic/software/GCCcore/14.3.0/lib64
LOG=$OUT/logs/bench.log
mkdir -p "$OUT"/{bin,raw,logs}

{
  echo "=== bench $(date -Iseconds) ==="
  echo "governor=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null) freq=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq 2>/dev/null)kHz"
  for c in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do echo performance | sudo -n tee "$c" >/dev/null 2>&1 || true; done
} | tee "$LOG"

OOO_LIB=$OUT/libs/ooo
X60_LIB=$OUT/libs/x60
test -f "$OOO_LIB/libopenblas.a"
test -f "$X60_LIB/libopenblas.a"

export PATH="$AS_BINDIR:$PATH"
export LD_LIBRARY_PATH="$GFORTRAN_LIB:$GCC143_LIB:${LD_LIBRARY_PATH:-}"
export LIBRARY_PATH="$GFORTRAN_LIB:$GCC143_LIB:${LIBRARY_PATH:-}"

link_one() {
  local name=$1 libdir=$2
  local a="$libdir/libopenblas.a"
  "$GCC143" -O2 -B"$AS_BINDIR/" \
    -o "$OUT/bin/bench-$name" "$OUT/bin/bench_dgemm.c" "$a" \
    -L"$GCC143_LIB" -Wl,-rpath,"$GCC143_LIB" \
    -L"$GFORTRAN_LIB" -Wl,-rpath,"$GFORTRAN_LIB" \
    -lgfortran -lm -lpthread -ldl -fopenmp
  echo "linked bench-$name" | tee -a "$LOG"
}
link_one ooo "$OOO_LIB"
link_one x60 "$X60_LIB"

run_sizes() {
  local name=$1
  for n in 512 1024 2048; do
    local reps=7
    [[ $n -ge 2048 ]] && reps=5
    echo "RUN $name N=$n" | tee -a "$LOG"
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
      taskset -c 0 "$OUT/bin/bench-$name" "$n" "$reps" \
      | tee "$OUT/raw/dgemm-${name}-N${n}-$(date +%s).txt" | tee -a "$LOG"
    sleep 2
  done
}

sleep 2
run_sizes ooo
sleep 3
run_sizes x60
sleep 3
run_sizes x60
sleep 3
run_sizes ooo

HPL_LOG=$OUT/logs/hpl.log
: > "$HPL_LOG"
{
  echo "=== HPL attempt $(date -Iseconds) ==="
  HPL_SRC=/home/orangepi/hpl-2.3
  if [[ ! -d "$HPL_SRC/src" ]]; then
    echo "HPL source missing; skip"
    exit 0
  fi
  set +u
  export EESSI_VERSION_OVERRIDE=2025.06-001
  if source /cvmfs/software.eessi.io/versions/2025.06/init/lmod/bash 2>/dev/null; then
    module load OpenMPI/5.0.8-GCC-14.3.0 2>/dev/null || module load OpenMPI 2>/dev/null || true
  fi
  set -u
  MPICC=$(command -v mpicc || true)
  MPIRUN=$(command -v mpirun || true)
  echo "MPICC=$MPICC MPIRUN=$MPIRUN"
  if [[ -z "$MPICC" || -z "$MPIRUN" ]]; then
    echo "MPI unavailable; skip HPL rebuild"
    exit 0
  fi

  build_xhpl() {
    local name=$1
    local lib="$OUT/libs/$name/libopenblas.a"
    local bdir="$OUT/hpl-build-$name"
    rm -rf "$bdir"
    mkdir -p "$bdir"
    local arch=Linux_RISCV64_gcc143_$name
    cat > "$HPL_SRC/Make.$arch" <<EOF
SHELL        = /bin/sh
CD           = cd
CP           = cp
LN_S         = ln -s
MKDIR        = mkdir
RM           = /bin/rm -f
TOUCH        = touch
ARCH         = $arch
TOPdir       = $HPL_SRC
INCdir       = \$(TOPdir)/include
BINdir       = $bdir
LIBdir       = $bdir
HPLlib       = \$(LIBdir)/libhpl.a
MPdir        =
MPinc        =
MPlib        =
LAdir        =
LAinc        =
LAlib        = $lib -L$GFORTRAN_LIB -lgfortran -lm -lpthread -ldl -fopenmp
F2CDEFS      =
HPL_OPTS     = -DHPL_DETAILED_TIMING -DHPL_PROGRESS_REPORT
HPL_INCLUDES = -I\$(INCdir) -I\$(INCdir)/\$(ARCH) \$(LAinc) \$(MPinc)
CC           = $MPICC
CCNOOPT      = \$(HPL_INCLUDES)
CCFLAGS      = \$(HPL_INCLUDES) -O3 -fomit-frame-pointer -DADD_
LINKER       = $MPICC
LINKFLAGS    = \$(CCFLAGS) -L$GCC143_LIB -Wl,-rpath,$GCC143_LIB -L$GFORTRAN_LIB -Wl,-rpath,$GFORTRAN_LIB
ARCHIVER     = ar
ARFLAGS      = r
RANLIB       = echo
EOF
    (cd "$HPL_SRC" && make arch="$arch" clean_arch_all >/dev/null 2>&1 || true)
    (cd "$HPL_SRC" && make -j"$(nproc)" arch="$arch") || return 1
    test -x "$bdir/xhpl"
    cp -a "$bdir/xhpl" "$OUT/bin/xhpl-$name"
    echo "built xhpl-$name"
  }

  export PATH="$AS_BINDIR:$PATH"
  export LD_LIBRARY_PATH="$GFORTRAN_LIB:$GCC143_LIB:${LD_LIBRARY_PATH:-}"
  build_xhpl ooo || echo "xhpl ooo build failed"
  build_xhpl x60 || echo "xhpl x60 build failed"

  cp -a "$OUT/HPL.dat" "$OUT/bin/HPL.dat"
  cd "$OUT/bin"
  for name in ooo x60; do
    if [[ -x ./xhpl-$name ]]; then
      echo "HPL RUN $name" | tee -a "$HPL_LOG"
      OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
        "$MPIRUN" -np 8 ./xhpl-$name 2>&1 | tee "$OUT/raw/hpl-$name.txt" | tee -a "$HPL_LOG"
      sleep 3
    fi
  done
} 2>&1 | tee -a "$HPL_LOG" || echo "HPL blocked or failed (see hpl.log)" | tee -a "$LOG"

python3 - <<'PY' | tee "$OUT/summary.md" | tee -a "$LOG"
from pathlib import Path
import re, statistics as st
out = Path("/home/orangepi/gcc-tune/measurements/results/rv2-gcc143-openblas-hpl-ab")
log = (out/"logs"/"bench.log").read_text()
data = {}
parts = re.split(r'^RUN (\w+) N=(\d+)$', log, flags=re.M)
for i in range(1, len(parts), 3):
    name, n, body = parts[i], int(parts[i+1]), parts[i+2]
    m = re.search(r'best_gflops=([\d.]+) mean_gflops=([\d.]+) checksum=([^\s]+)', body)
    if not m:
        continue
    data.setdefault((name,n), []).append({
        "best": float(m.group(1)),
        "mean": float(m.group(2)),
        "cs": m.group(3),
    })

lines = ["# OpenBLAS DGEMM mtune A/B (GCC 14.3-x60)", "",
         "OpenBLAS `RISCV64_ZVL256B`, `NO_SHARED=1`, `USE_OPENMP=1`, built with gcc-14.3.0-x60.",
         "Flags: `-O3 -march=rv64gcv_zba_zbb_zbc_zvl256b -mabi=lp64d` + `-mtune=generic-ooo` vs `-mtune=spacemit-x60`.",
         "Bench: `CLOCK_MONOTONIC_RAW`, `taskset -c 0`, `OPENBLAS_NUM_THREADS=1`.",
         "",
         "| N | ooo mean GF/s | x60 mean GF/s | Δ% (x60 vs ooo) | ooo best | x60 best |",
         "|---|---------------|---------------|-----------------|----------|----------|"]
csv = ["N,ooo_mean_gflops,x60_mean_gflops,delta_pct,ooo_best_gflops,x60_best_gflops,n_runs"]
for n in (512,1024,2048):
    oo = data.get(("ooo",n), [])
    xx = data.get(("x60",n), [])
    if not oo or not xx:
        lines.append(f"| {n} | missing | missing |  |  |  |")
        continue
    om = st.mean(r["mean"] for r in oo)
    xm = st.mean(r["mean"] for r in xx)
    ob = st.mean(r["best"] for r in oo)
    xb = st.mean(r["best"] for r in xx)
    dlt = 100.0*(xm-om)/om if om else float("nan")
    lines.append(f"| {n} | {om:.3f} | {xm:.3f} | {dlt:+.2f}% | {ob:.3f} | {xb:.3f} |")
    csv.append(f"{n},{om:.4f},{xm:.4f},{dlt:.3f},{ob:.4f},{xb:.4f},{len(oo)}")

hpl = out/"logs"/"hpl.log"
if hpl.exists() and hpl.stat().st_size > 0:
    lines += ["", "## HPL", ""]
    txt = hpl.read_text()
    for name in ("ooo","x60"):
        chunks = re.split(rf'HPL RUN {name}', txt)
        if len(chunks)<2:
            lines.append(f"- {name}: no run")
            continue
        body = chunks[-1]
        gm = re.search(r'WR\S+\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+([\d.]+)\s+([\d.eE+\-]+)', body)
        res = re.search(r'(PASSED|FAILED)', body)
        if gm:
            lines.append(f"- **{name}**: N={gm.group(1)} NB={gm.group(2)} P={gm.group(3)} Q={gm.group(4)} time={gm.group(5)}s Gflops={gm.group(6)} residual={res.group(1) if res else '?'}")
        else:
            lines.append(f"- {name}: parse failed / incomplete")

(out/"summary.md").write_text("\n".join(lines)+"\n")
(out/"summary.csv").write_text("\n".join(csv)+"\n")
print("\n".join(lines))
PY
echo "BENCH ALL DONE $(date -Iseconds)" | tee -a "$LOG"
