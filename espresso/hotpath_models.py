#!/usr/bin/env python3
"""
Hot-path survey across ESPResSo model classes that use P3M.

For each model:
  1) auto-tune P3M, record mesh/cao/r_cut
  2) time full MD (WCA/LJ + thermostat + P3M)
  3) time MD with Coulomb actor removed
  4) time force-only recalcs (recalc_forces=True, steps=0) with P3M

Coulomb wall share ≈ (t_full - t_nocoul) / t_full.
FFT is only a slice of that (spread/gather + real-space + k-space).

Models mirror official 4.2.2 samples / our bench harness:
  lattice512     — espresso/p3m_lj.py style
  sample_p3m     — samples/p3m.py (box=10, dens=0.3)
  dense_wca      — samples/minimal-charged-particles.py (dens=0.7)
  salt_box50     — samples/grand_canonical.py salt density in box=50
  electrophoresis — samples/electrophoresis.py (dilute, box=100)
  dense_large    — denser charged fluid, larger N (FFT stress candidate)
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import espressomd
from espressomd import electrostatics, interactions, polymer

espressomd.assert_features(["P3M", "WCA"])

SEED = 42
STEPS = int(os.environ.get("ESP_HOT_STEPS", "200"))
FORCE_LOOPS = int(os.environ.get("ESP_HOT_FORCE_LOOPS", "50"))


def _time_md(system, steps: int) -> float:
    system.integrator.run(max(10, steps // 10))  # warm
    t0 = time.perf_counter()
    system.integrator.run(steps)
    return time.perf_counter() - t0


def _time_forces(system, loops: int) -> float:
    system.integrator.run(0, recalc_forces=True)
    t0 = time.perf_counter()
    for _ in range(loops):
        system.integrator.run(0, recalc_forces=True)
    return time.perf_counter() - t0


def _add_p3m(system, accuracy: float, prefactor: float = 1.0):
    p3m = electrostatics.P3M(prefactor=prefactor, accuracy=accuracy)
    system.actors.add(p3m)
    return p3m


def _wca(system, types=(0,)):
    for t in types:
        for u in types:
            if u < t:
                continue
            system.non_bonded_inter[t, u].wca.set_params(epsilon=1.0, sigma=1.0)


def model_lattice512():
    """Our A/B harness: cubic lattice, N=512, box=20."""
    N, BOX = 512, 20.0
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
    rng = np.random.default_rng(SEED)
    pos = np.asarray(coords, float)
    pos += rng.uniform(-0.05 * spacing, 0.05 * spacing, size=pos.shape)
    pos %= BOX
    system.part.add(pos=pos, q=np.tile([1.0, -1.0], N // 2), type=np.zeros(N, int))
    _wca(system)
    system.integrator.set_vv()
    system.thermostat.set_langevin(kT=1.0, gamma=1.0, seed=SEED)
    return system, dict(accuracy=1e-3, prefactor=1.0, note="p3m_lj harness")


def model_sample_p3m():
    """Official samples/p3m.py defaults."""
    box_l, density = 10.0, 0.3
    n_part = int(box_l**3 * density)
    n_part -= n_part % 2
    system = espressomd.System(box_l=[box_l] * 3)
    np.random.seed(SEED)
    system.time_step = 0.01
    system.cell_system.skin = 0.4
    system.part.add(
        pos=np.random.random((n_part, 3)) * box_l,
        q=np.resize([1.0, -1.0], n_part),
    )
    _wca(system)
    # short steepest-descent so P3M tune is not on overlaps
    system.integrator.set_steepest_descent(f_max=0, gamma=1e-3, max_displacement=0.01)
    system.integrator.run(200)
    system.integrator.set_vv()
    system.thermostat.set_langevin(kT=1.0, gamma=1.0, seed=SEED)
    return system, dict(accuracy=1e-3, prefactor=1.0, note="samples/p3m.py")


def model_dense_wca():
    """Official samples/minimal-charged-particles.py."""
    box_l, density = 10.7437, 0.7
    n_part = 2 * int(0.5 * box_l**3 * density)
    system = espressomd.System(box_l=[box_l] * 3)
    np.random.seed(SEED)
    system.time_step = 0.01
    system.cell_system.skin = 0.4
    system.part.add(
        pos=np.random.random((n_part, 3)) * box_l,
        q=np.resize([1.0, -1.0], n_part),
    )
    _wca(system)
    system.integrator.set_steepest_descent(f_max=0, gamma=1e-3, max_displacement=0.01)
    system.integrator.run(400)
    system.integrator.set_vv()
    system.thermostat.set_langevin(kT=1.0, gamma=1.0, seed=SEED)
    return system, dict(accuracy=1e-2, prefactor=1.0, note="samples/minimal-charged")


def model_salt_box50():
    """Grand-canonical salt box without reactions (MD-only slice)."""
    box_l, cs_bulk = 50.0, 1e-3
    n_pairs = max(2, int(cs_bulk * box_l**3))
    system = espressomd.System(box_l=[box_l] * 3)
    np.random.seed(SEED)
    system.time_step = 0.01
    system.cell_system.skin = 0.4
    pos = np.random.random((2 * n_pairs, 3)) * box_l
    q = np.resize([1.0, -1.0], 2 * n_pairs)
    types = np.resize([1, 2], 2 * n_pairs)
    system.part.add(pos=pos, q=q, type=types)
    _wca(system, types=(1, 2))
    system.integrator.set_vv()
    system.thermostat.set_langevin(kT=1.0, gamma=1.0, seed=SEED)
    return system, dict(accuracy=1e-3, prefactor=2.0, note="samples/grand_canonical salt")


def model_electrophoresis():
    """Dilute polyelectrolyte + salt in large box (samples/electrophoresis.py)."""
    espressomd.assert_features(["EXTERNAL_FORCES"])
    N_MONOMERS, N_IONS = 20, 100
    system = espressomd.System(box_l=[100.0] * 3)
    np.random.seed(SEED)
    system.time_step = 0.01
    system.cell_system.skin = 0.4
    for a, b in [(0, 0), (0, 1), (0, 2), (1, 2)]:
        system.non_bonded_inter[a, b].wca.set_params(epsilon=1, sigma=1)
    hb = interactions.HarmonicBond(k=10, r_0=2)
    system.bonded_inter.add(hb)
    init = polymer.linear_polymer_positions(
        n_polymers=1,
        beads_per_chain=N_MONOMERS,
        bond_length=2.0,
        seed=2,
        bond_angle=np.pi,
        min_distance=1.8,
        start_positions=np.array([system.box_l / 2.0]),
    )
    monomers = system.part.add(pos=init[0], q=-np.ones(N_MONOMERS), type=np.zeros(N_MONOMERS, int))
    prev = None
    for part in monomers:
        if prev is not None:
            part.add_bond((hb, prev))
        prev = part
    system.part.add(
        pos=np.random.random((N_MONOMERS, 3)) * system.box_l,
        q=np.ones(N_MONOMERS),
        type=np.ones(N_MONOMERS, int),
    )
    system.part.add(
        pos=np.random.random((N_IONS, 3)) * system.box_l,
        q=np.resize([1.0, -1.0], N_IONS),
        type=np.resize([1, 2], N_IONS),
    )
    system.integrator.set_steepest_descent(f_max=0, gamma=1e-3, max_displacement=0.01)
    system.integrator.run(200)
    system.integrator.set_vv()
    system.thermostat.set_langevin(kT=1.0, gamma=1.0, seed=SEED)
    return system, dict(accuracy=1e-2, prefactor=1.0, note="samples/electrophoresis")


def model_dense_large():
    """Medium-large dense ± fluid (keeps P3M tune tractable on RV2)."""
    box_l, density = 24.0, 0.4
    n_part = 2 * int(0.5 * box_l**3 * density)
    system = espressomd.System(box_l=[box_l] * 3)
    np.random.seed(SEED)
    system.time_step = 0.005
    system.cell_system.skin = 0.4
    system.part.add(
        pos=np.random.random((n_part, 3)) * box_l,
        q=np.resize([1.0, -1.0], n_part),
    )
    _wca(system)
    system.integrator.set_steepest_descent(f_max=0, gamma=1e-3, max_displacement=0.01)
    system.integrator.run(400)
    system.integrator.set_vv()
    system.thermostat.set_langevin(kT=1.0, gamma=1.0, seed=SEED)
    return system, dict(accuracy=1e-3, prefactor=1.0, note="dense N~box^3*0.4 acc=1e-3")


MODELS = {
    "lattice512": model_lattice512,
    "sample_p3m": model_sample_p3m,
    "dense_wca": model_dense_wca,
    "salt_box50": model_salt_box50,
    "electrophoresis": model_electrophoresis,
    "dense_large": model_dense_large,
}


def run_one(name: str) -> dict:
    build = MODELS[name]
    system, cfg = build()
    n = len(system.part)
    box = float(system.box_l[0])
    print(f"\n===== {name} N={n} box={box} acc={cfg['accuracy']} ({cfg['note']}) =====", flush=True)

    # Baseline BEFORE P3M (actors.clear() is unreliable / can skew dilute cases).
    t_nocoul = _time_md(system, STEPS)
    t_force_nocoul = _time_forces(system, FORCE_LOOPS)

    p3m = _add_p3m(system, accuracy=cfg["accuracy"], prefactor=cfg["prefactor"])
    mesh = list(p3m.mesh)
    cao = int(p3m.cao)
    r_cut = float(p3m.r_cut)
    alpha = float(p3m.alpha)
    print(
        f"P3M tuned: mesh={mesh} cao={cao} r_cut={r_cut:.4f} alpha={alpha:.6g}",
        flush=True,
    )

    t_full = _time_md(system, STEPS)
    t_force = _time_forces(system, FORCE_LOOPS)

    coul_share = max(0.0, (t_full - t_nocoul) / t_full) if t_full > 0 else 0.0
    force_coul_share = (
        max(0.0, (t_force - t_force_nocoul) / t_force) if t_force > 0 else 0.0
    )
    row = dict(
        model=name,
        N=n,
        box=box,
        mesh=mesh[0],
        cao=cao,
        r_cut=r_cut,
        t_full=t_full,
        t_nocoul=t_nocoul,
        coul_share=coul_share,
        t_force=t_force,
        t_force_nocoul=t_force_nocoul,
        force_coul_share=force_coul_share,
        ns_per_day=STEPS * 1e-6 / t_full * 86400 if t_full > 0 else 0.0,
    )
    print(
        f"MD  full={t_full:.3f}s nocoul={t_nocoul:.3f}s coul_share={coul_share:.1%}  "
        f"force_loops={FORCE_LOOPS} force_coul_share={force_coul_share:.1%}",
        flush=True,
    )
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "models",
        nargs="*",
        default=list(MODELS),
        help=f"subset of {list(MODELS)}",
    )
    args = ap.parse_args()
    unknown = [m for m in args.models if m not in MODELS]
    if unknown:
        raise SystemExit(f"unknown models: {unknown}")

    print(
        f"ESP_HOT_STEPS={STEPS} ESP_HOT_FORCE_LOOPS={FORCE_LOOPS} "
        f"OMP={os.environ.get('OMP_NUM_THREADS', '?')}",
        flush=True,
    )
    rows = []
    for name in args.models:
        try:
            rows.append(run_one(name))
        except Exception as exc:  # keep surveying other models
            print(f"FAIL {name}: {type(exc).__name__}: {exc}", flush=True)

    print("\n===== SUMMARY =====", flush=True)
    print(
        f"{'model':18s} {'N':>6s} {'box':>6s} {'mesh':>5s} "
        f"{'t_full':>8s} {'coul%':>7s} {'f_coul%':>8s}",
        flush=True,
    )
    for r in rows:
        print(
            f"{r['model']:18s} {r['N']:6d} {r['box']:6.1f} {r['mesh']:5d} "
            f"{r['t_full']:8.3f} {100 * r['coul_share']:6.1f}% "
            f"{100 * r['force_coul_share']:7.1f}%",
            flush=True,
        )


if __name__ == "__main__":
    main()
