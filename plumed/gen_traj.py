#!/usr/bin/env python3
"""Generate a cubic LJ-like XYZ trajectory for plumed driver."""
from __future__ import annotations

import argparse
import math
import random


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--output", default="traj.xyz")
    ap.add_argument("-n", type=int, default=200, help="number of atoms")
    ap.add_argument("-f", type=int, default=20, help="number of frames")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--density", type=float, default=0.8, help="N/L^3")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    L = (args.n / args.density) ** (1.0 / 3.0)
    # start on a slightly jittered simple-cubic lattice
    g = max(1, int(math.ceil(args.n ** (1.0 / 3.0))))
    spacing = L / g
    base = []
    for i in range(g):
        for j in range(g):
            for k in range(g):
                if len(base) >= args.n:
                    break
                base.append(
                    [
                        (i + 0.5) * spacing + rng.uniform(-0.05, 0.05) * spacing,
                        (j + 0.5) * spacing + rng.uniform(-0.05, 0.05) * spacing,
                        (k + 0.5) * spacing + rng.uniform(-0.05, 0.05) * spacing,
                    ]
                )
            if len(base) >= args.n:
                break
        if len(base) >= args.n:
            break

    with open(args.output, "w") as out:
        for fr in range(args.f):
            # PLUMED driver expects the box on the comment line as Lattice="..."
            out.write(f"{args.n}\n")
            out.write(
                f' Lattice="{L:.10f} 0.0 0.0 0.0 {L:.10f} 0.0 0.0 0.0 {L:.10f}"\n'
            )
            amp = 0.02 * spacing
            for x, y, z in base:
                out.write(
                    "Ar "
                    f"{(x + rng.uniform(-amp, amp) + 0.001 * fr) % L:.8f} "
                    f"{(y + rng.uniform(-amp, amp)) % L:.8f} "
                    f"{(z + rng.uniform(-amp, amp)) % L:.8f}\n"
                )
    print(f"wrote {args.output}: N={args.n} frames={args.f} L={L:.4f}")


if __name__ == "__main__":
    main()
