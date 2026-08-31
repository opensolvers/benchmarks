#!/usr/bin/env python3
"""One mesh (ESP_MESH) at fixed cao/r_cut/alpha; compare to no-Coulomb baseline."""
from __future__ import annotations

import os
import time

import numpy as np
import espressomd
from espressomd import electrostatics

espressomd.assert_features(["P3M", "WCA"])

N = int(os.environ.get("ESP_N", "512"))
BOX = float(os.environ.get("ESP_BOX", "20"))
STEPS = int(os.environ.get("ESP_HOT_STEPS", "300"))
CAO = int(os.environ.get("ESP_CAO", "5"))
RCUT = float(os.environ.get("ESP_RCUT", "4.4896"))
ALPHA = float(os.environ.get("ESP_ALPHA", "0.53745"))
MESH = int(os.environ["ESP_MESH"])


def build():
    system = espressomd.System(box_l=[BOX] * 3)
    system.time_step = 0.005
    system.cell_system.skin = 0.4
    n_side = int(np.ceil(N ** (1.0 / 3.0)))
    spacing = BOX / n_side
    coords = [
        [(i + 0.5) * spacing, (j + 0.5) * spacing, (k + 0.5) * spacing]
        for i in range(n_side)
        for j in range(n_side)
        for k in range(n_side)
    ][:N]
    rng = np.random.default_rng(42)
    pos = np.asarray(coords, float)
    pos += rng.uniform(-0.05 * spacing, 0.05 * spacing, size=pos.shape)
    pos %= BOX
    system.part.add(pos=pos, q=np.tile([1.0, -1.0], N // 2), type=np.zeros(N, int))
    system.non_bonded_inter[0, 0].wca.set_params(epsilon=1.0, sigma=1.0)
    system.integrator.set_vv()
    system.thermostat.set_langevin(kT=1.0, gamma=1.0, seed=42)
    return system


def time_md(system, steps):
    system.integrator.run(max(10, steps // 10))
    t0 = time.perf_counter()
    system.integrator.run(steps)
    return time.perf_counter() - t0


def main():
    print(f"N={N} BOX={BOX} STEPS={STEPS} mesh={MESH} cao={CAO} r_cut={RCUT} alpha={ALPHA}")
    system = build()
    t_nocoul = time_md(system, STEPS)
    # cannot rebuild System — add P3M on same system after baseline
    p3m = electrostatics.P3M(
        prefactor=1.0,
        accuracy=1e-3,
        mesh=[MESH, MESH, MESH],
        cao=CAO,
        r_cut=RCUT,
        alpha=ALPHA,
        tune=False,
    )
    system.actors.add(p3m)
    t_full = time_md(system, STEPS)
    extra = t_full - t_nocoul
    share = extra / t_full if t_full > 0 else 0.0
    print(
        f"RESULT mesh={MESH} nocoul={t_nocoul:.3f}s full={t_full:.3f}s "
        f"coul_extra={extra:.3f}s coul_share={share:.1%}",
        flush=True,
    )


if __name__ == "__main__":
    main()
