#!/bin/bash
set -euo pipefail
EESSI_SW=/cvmfs/dev.eessi.io/riscv/versions/2025.06-001/software/linux/riscv64/generic/software
export PATH=$EESSI_SW/GCCcore/14.3.0/bin:$PATH
export LIBRARY_PATH=$EESSI_SW/GCCcore/14.3.0/lib64
export LD_LIBRARY_PATH=$EESSI_SW/GCCcore/14.3.0/lib64
cd /home/orangepi/fftwbuild/src-r5v

restore_dft() {
  local d=$1
  local a
  a=$(ls "$d"/.libs/libdft_*_codelets.a 2>/dev/null | head -1 || true)
  [ -n "$a" ] || { echo "skip $d"; return; }
  local oc; oc=$(find "$d" -maxdepth 1 -name "*.o" | wc -l)
  if [ "$oc" -gt 100 ]; then echo "$d ok ($oc)"; return; fi
  echo "restore $d from $a"
  local tmp; tmp=$(mktemp -d)
  (cd "$tmp" && ar x "$OLDPWD/$a")
  mkdir -p "$d/.libs"
  cp -a "$tmp"/*.o "$d/.libs/"
  cp -a "$tmp"/*.o "$d/"
  for c in "$d"/*.c; do
    b=$(basename "$c" .c)
    touch "$d/$b.lo" "$d/$b.o" "$d/.libs/$b.o"
  done
  rm -rf "$tmp"
}

for d in dft/simd/r5v128 dft/simd/r5v512 dft/simd/r5v1024 dft/simd/r5v2048 \
         dft/simd/r5v4096 dft/simd/r5v8192 dft/simd/r5v16384; do
  restore_dft "$d"
done

for d in rdft/simd/r5v*; do
  [ -d "$d" ] || continue
  a=$(ls "$d"/.libs/librdft_*_codelets.a 2>/dev/null | head -1 || true)
  [ -n "$a" ] || continue
  oc=$(find "$d" -maxdepth 1 -name "*.o" 2>/dev/null | wc -l)
  if [ "$oc" -gt 20 ]; then continue; fi
  echo "restore $d"
  tmp=$(mktemp -d)
  (cd "$tmp" && ar x "$OLDPWD/$a")
  mkdir -p "$d/.libs"
  cp -a "$tmp"/*.o "$d/.libs/"
  cp -a "$tmp"/*.o "$d/"
  for c in "$d"/*.c; do b=$(basename "$c" .c); touch "$d/$b.lo" "$d/$b.o" "$d/.libs/$b.o"; done
  rm -rf "$tmp"
done

echo "ensure r5v256 archive current"
make -j4 -C dft/simd/r5v256
make -j4 -C rdft/simd/r5v256 2>/dev/null || true

for d in dft/simd/r5v* rdft/simd/r5v*; do
  [ -d "$d" ] || continue
  make -t -C "$d" >/dev/null 2>&1 || true
done

echo "relink libfftw3"
rm -f .libs/libfftw3.so .libs/libfftw3.so.3 .libs/libfftw3.so.3.6.10
make -j4
ls -la .libs/libfftw3.so.3.6.10

python3 - <<'PY'
import subprocess, re
so = "/home/orangepi/fftwbuild/src-r5v/.libs/libfftw3.so.3.6.10"
out = subprocess.check_output(["objdump", "-t", so], text=True, errors="replace")
addrs = []
for line in out.splitlines():
    if re.search(r"\bt2bv_8$", line):
        addrs.append(line.split()[0])
print("t2bv_8 addrs", addrs)
for a in addrs:
    if a.startswith("000000000020"):
        start = int(a, 16)
        end = start + 0x300
        dump = subprocess.check_output(
            ["objdump", "-d", f"--start-address={start}", f"--stop-address={end}", so],
            text=True, errors="replace")
        print(a, "gather", dump.count("vrgather"),
              "spill", len(re.findall(r"sd\t.*\(sp\)", dump)))
        open("/tmp/t2bv8_patched.s", "w").write(dump)
        break
PY

bash /home/orangepi/fftw-wisdom-src/bench-codelet-hot.sh
cp -a "$(ls -t /home/orangepi/logs/fftw-codelet-hot-r5v-*.log | head -1)" \
  /home/orangepi/logs/fftw-codelet-hot-noshuffle.log
echo ALL_DONE
