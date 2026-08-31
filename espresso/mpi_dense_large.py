#!/usr/bin/env python3
"""Dense charged WCA + P3M for MPI scaling (mirrors hotpath dense_large)."""
import os
import time

import numpy as np
import espressomd
from espressomd import electrostatics

espressomd.assert_features(["P3M", "WCA"])

SEED = 42
STEPS = int(os.environ.get("ESP_STEPS", os.environ.get("ESP_HOT_STEPS", "200")))
BOX = float(os.environ.get("ESP_BOX", "24.0"))
DENSITY = float(os.environ.get("ESP_DENSITY", "0.4"))
ACCURACY = float(os.environ.get("ESP_ACCURACY", "1e-3"))

n_part = 2 * int(0.5 * BOX**3 * DENSITY)
system = espressomd.System(box_l=[BOX] * 3)
np.random.seed(SEED)
system.time_step = 0.005
system.cell_system.skin = float(os.environ.get("ESP_SKIN", "0.4"))
system.part.add(
    pos=np.random.random((n_part, 3)) * BOX,
    q=np.resize([1.0, -1.0], n_part),
)
system.non_bonded_inter[0, 0].wca.set_params(epsilon=1.0, sigma=1.0)
system.integrator.set_steepest_descent(f_max=0, gamma=1e-3, max_displacement=0.01)
system.integrator.run(200)
system.integrator.set_vv()
system.thermostat.set_langevin(kT=1.0, gamma=1.0, seed=SEED)

# Pin if provided; else tune once (same on all ranks via Espresso collective).
kw = dict(prefactor=1.0, accuracy=ACCURACY)
if os.environ.get("ESP_TUNE", "1") == "0":
    m = int(os.environ["ESP_MESH"])
    kw.update(
        mesh=[m, m, m],
        cao=int(os.environ["ESP_CAO"]),
        r_cut=float(os.environ["ESP_RCUT"]),
        alpha=float(os.environ["ESP_ALPHA"]),
        tune=False,
    )
p3m = electrostatics.P3M(**kw)
system.actors.add(p3m)

# Only rank 0 prints (Espresso may still print tune on all — keep short).
comm = system.cell_system.get_state().get("n_nodes", None)
try:
    from mpi4py import MPI

    rank = MPI.COMM_WORLD.rank
    n_nodes = MPI.COMM_WORLD.size
except Exception:
    rank, n_nodes = 0, 1

if rank == 0:
    print(
        f"P3M params: cao={p3m.cao} r_cut={p3m.r_cut:.4f} mesh={p3m.mesh} "
        f"n_part={n_part} box={BOX} skin={system.cell_system.skin} mpi={n_nodes}",
        flush=True,
    )

system.integrator.run(30)
try:
    from mpi4py import MPI

    MPI.COMM_WORLD.Barrier()
except Exception:
    pass
t0 = time.perf_counter()
system.integrator.run(STEPS)
wall = time.perf_counter() - t0
if rank == 0:
    e = system.analysis.energy()
    print(
        f"N={n_part} steps={STEPS} box={BOX} np={n_nodes} wall={wall:.3f}s "
        f"ns_per_day={STEPS * 1e-6 / wall * 86400:.2f} "
        f"E_tot={float(e['total']):.6f} E_coul={float(e.get('coulomb', float('nan'))):.6f}",
        flush=True,
    )
