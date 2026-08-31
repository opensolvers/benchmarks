#!/usr/bin/env python3
"""Tune + bench P3M at one fixed cao for dense_large. Run once per cao (one System)."""
import os
import sys
import time

import numpy as np
import espressomd
from espressomd import electrostatics

espressomd.assert_features(["P3M", "WCA"])

SEED = 42
STEPS = int(os.environ.get("ESP_STEPS", "200"))
BOX = float(os.environ.get("ESP_BOX", "24.0"))
DENSITY = float(os.environ.get("ESP_DENSITY", "0.4"))
ACCURACY = float(os.environ.get("ESP_ACCURACY", "1e-3"))
SKIN = float(os.environ.get("ESP_SKIN", "0.4"))
CAO = int(os.environ["ESP_CAO"])


n_part = 2 * int(0.5 * BOX**3 * DENSITY)
system = espressomd.System(box_l=[BOX] * 3)
np.random.seed(SEED)
system.time_step = 0.005
system.cell_system.skin = SKIN
system.part.add(
    pos=np.random.random((n_part, 3)) * BOX,
    q=np.resize([1.0, -1.0], n_part),
)
system.non_bonded_inter[0, 0].wca.set_params(epsilon=1.0, sigma=1.0)
system.integrator.set_steepest_descent(f_max=0, gamma=1e-3, max_displacement=0.01)
system.integrator.run(200)
system.integrator.set_vv()
system.thermostat.set_langevin(kT=1.0, gamma=1.0, seed=SEED)

p3m = electrostatics.P3M(
    prefactor=1.0,
    accuracy=ACCURACY,
    cao=CAO,
    tune=True,
)
system.actors.add(p3m)
mesh = list(p3m.mesh)
print(
    f"TUNED cao={p3m.cao} mesh={mesh} r_cut={p3m.r_cut:.4f} "
    f"alpha={p3m.alpha:.6g} N={n_part} box={BOX}",
    flush=True,
)
system.integrator.run(30)
t0 = time.perf_counter()
system.integrator.run(STEPS)
wall = time.perf_counter() - t0
e = system.analysis.energy()
print(
    f"BENCH cao={CAO} wall={wall:.3f}s ns_per_day={STEPS * 1e-6 / wall * 86400:.2f} "
    f"E_tot={float(e['total']):.6f} E_coul={float(e.get('coulomb', float('nan'))):.6f}",
    flush=True,
)
print(
    f"PIN cao={int(p3m.cao)} mesh={mesh[0]} "
    f"rcut={float(p3m.r_cut):.4f} alpha={float(p3m.alpha):.6g} wall={wall:.3f}",
    flush=True,
)
