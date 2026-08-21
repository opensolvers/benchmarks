#!/usr/bin/env python3
"""Charged LJ fluid with P3M — soft-matter FFT probe for ESPResSo."""
import os
import time
import numpy as np
import espressomd
from espressomd import electrostatics

N = int(os.environ.get("ESP_N", "512"))
STEPS = int(os.environ.get("ESP_STEPS", "200"))
BOX = float(os.environ.get("ESP_BOX", "20.0"))
SEED = 42

# Simple cubic lattice so particles do not start overlapped.
n_side = int(np.ceil(N ** (1.0 / 3.0)))
assert n_side ** 3 >= N
spacing = BOX / n_side
coords = []
for i in range(n_side):
    for j in range(n_side):
        for k in range(n_side):
            if len(coords) >= N:
                break
            coords.append([(i + 0.5) * spacing, (j + 0.5) * spacing, (k + 0.5) * spacing])
        if len(coords) >= N:
            break
    if len(coords) >= N:
        break
pos = np.array(coords, dtype=float)
# tiny jitter
rng = np.random.default_rng(SEED)
pos += rng.uniform(-0.05 * spacing, 0.05 * spacing, size=pos.shape)
pos %= BOX

system = espressomd.System(box_l=[BOX, BOX, BOX])
system.time_step = 0.005
system.cell_system.skin = 0.4

system.part.add(
    pos=pos,
    q=np.tile([1.0, -1.0], N // 2),
    type=np.zeros(N, dtype=int),
)
system.non_bonded_inter[0, 0].lennard_jones.set_params(
    epsilon=1.0, sigma=1.0, cutoff=2 ** (1.0 / 6.0), shift="auto"
)
system.integrator.set_vv()
system.thermostat.set_langevin(kT=1.0, gamma=1.0, seed=SEED)

p3m = electrostatics.P3M(prefactor=1.0, accuracy=1e-3)
system.actors.add(p3m)
print(f"P3M params: cao={p3m.cao} r_cut={p3m.r_cut:.4f} mesh={p3m.mesh}", flush=True)

# gentle warmup
system.integrator.run(50)
t0 = time.perf_counter()
system.integrator.run(STEPS)
wall = time.perf_counter() - t0
e = system.analysis.energy()
etot = float(e["total"])
ecoul = float(e.get("coulomb", float("nan")))
print(
    f"N={N} steps={STEPS} box={BOX} wall={wall:.3f}s "
    f"ns_per_day={STEPS * 1e-6 / wall * 86400:.2f} "
    f"E_tot={etot:.6f} E_coul={ecoul:.6f}",
    flush=True,
)
