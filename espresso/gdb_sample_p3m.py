#!/usr/bin/env python3
"""
Statistical stack sampler for Espresso (no kernel perf required).
Attaches via gdb periodically during the timed MD window and histograms frames.
"""
from __future__ import annotations

import os
import re
import signal
import subprocess
import time
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
CASE = os.environ.get("ESP_CASE", "dense_large")
STEPS = os.environ.get("ESP_STEPS", "300")
SAMPLES = int(os.environ.get("ESP_SAMPLES", "80"))
INTERVAL = float(os.environ.get("ESP_SAMPLE_INTERVAL", "0.15"))
LOG = os.environ.get("LOG", os.path.expanduser("~/logs/espresso-gdb-sample.log"))


def env_for_case():
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["OPENBLAS_CORETYPE"] = "RISCV64_GENERIC"
    scalar = env.get(
        "SCALAR", os.path.expanduser("~/fftwbuild/src-scalar/.libs/libfftw3.so.3.6.10")
    )
    env["LD_PRELOAD"] = scalar
    if CASE == "lattice512":
        env.update(
            ESP_N="512",
            ESP_BOX="20",
            ESP_STEPS=STEPS,
            ESP_ACCURACY="1e-3",
            ESP_MESH="18",
            ESP_CAO="5",
            ESP_RCUT="4.4896",
            ESP_ALPHA="0.53745",
            ESP_TUNE="0",
        )
        script = os.path.join(HERE, "p3m_lj.py")
    else:
        env.update(
            ESP_BOX="24",
            ESP_DENSITY="0.4",
            ESP_STEPS=STEPS,
            ESP_ACCURACY="1e-3",
            ESP_MESH="32",
            ESP_CAO="6",
            ESP_RCUT="3.4847",
            ESP_ALPHA="0.751886",
            ESP_TUNE="0",
        )
        script = os.path.join(HERE, "mpi_dense_large.py")
    return env, script


def interesting(frame: str) -> str | None:
    f = frame.strip()
    # gdb frames look like: #0  0xaddr in name(args) at file:line
    m = re.search(r"\bin\s+([^\s(]+)", f)
    if not m:
        m = re.search(r"^#\d+\s+0x[0-9a-f]+\s+([^\s(]+)", f)
    if not m:
        return None
    name = m.group(1)
    # keep native-looking / espresso / fftw / math
    low = name.lower()
    if any(
        k in low
        for k in (
            "p3m",
            "fft",
            "coulomb",
            "assign",
            "lj",
            "lennard",
            "verlet",
            "force",
            "integrate",
            "erfc",
            "ewald",
            "add_non_bonded",
            "calc_",
            "kernel",
            "short_range",
            "thermostat",
            "langevin",
            "espresso",
        )
    ):
        return name
    if name.startswith("_Z") or "fftw" in low:
        return name
    return name  # keep all leaf frames; filter later


def sample_once(pid: int) -> list[str]:
    cmd = [
        "gdb",
        "-p",
        str(pid),
        "-batch",
        "-ex",
        "set pagination off",
        "-ex",
        "thread apply all bt 16",
    ]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, timeout=20)
    except Exception:
        return []
    text = out.decode("utf-8", "replace")
    frames = []
    for line in text.splitlines():
        if line.startswith("#"):
            frames.append(line)
    return frames


def main():
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    env, script = env_for_case()
    # ensure module env already loaded by wrapper
    pypresso = env.get("PYPRESSO", "pypresso")
    cmd = [pypresso, script]
    with open(LOG, "w") as log:
        log.write(f"START case={CASE} steps={STEPS} samples={SAMPLES}\n")
        log.flush()
        proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
        # let setup / steepest descent / P3M init finish
        time.sleep(float(os.environ.get("ESP_SAMPLE_DELAY", "8")))
        leaf = Counter()
        anyframe = Counter()
        ok = 0
        for i in range(SAMPLES):
            if proc.poll() is not None:
                break
            frames = sample_once(proc.pid)
            if not frames:
                time.sleep(INTERVAL)
                continue
            ok += 1
            # leaf = #0
            for fr in frames:
                if fr.startswith("#0"):
                    name = interesting(fr) or fr
                    leaf[name] += 1
                name = interesting(fr)
                if name:
                    anyframe[name] += 1
            time.sleep(INTERVAL)
        if proc.poll() is None:
            # let it finish timed region; kill if stuck
            try:
                proc.wait(timeout=120)
            except subprocess.TimeoutExpired:
                proc.send_signal(signal.SIGTERM)
                proc.wait(timeout=10)
        log.write(f"\nSAMPLES_OK={ok}/{SAMPLES} RC={proc.returncode}\n")
        log.write("\n===== LEAF (#0) =====\n")
        for name, c in leaf.most_common(40):
            log.write(f"{c:5d}  {name}\n")
        log.write("\n===== ANY FRAME =====\n")
        for name, c in anyframe.most_common(50):
            log.write(f"{c:5d}  {name}\n")
        log.write("DONE\n")
    print(f"LOG={LOG}")


if __name__ == "__main__":
    main()
