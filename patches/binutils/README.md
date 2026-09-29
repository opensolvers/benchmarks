# Binutils SpacemiT IME (`xsmtvdot` / `xsmtvdotii`)

Shared with [`opensolvers/spacemit-x60-gcc-tune`](https://github.com/opensolvers/spacemit-x60-gcc-tune)
`patches/binutils/`. Use with GCC 14.3 or 15.2 SpacemiT patches (both already
expose `-march=…_xsmtvdotii`).

| File | Role |
|------|------|
| `binutils-2.46.1_add-spacemit-xsmtvdot.patch` | IME1: `xsmtvdot` + `smt.vmadot*` (X60) |
| `binutils-2.46.1_add-spacemit-xsmtvdotii.patch` | IME2 follow-on: `xsmtvdotii` + `smt.vfwmadot*`, `smt.vpack.vv`, … (A100) |

Apply **in order** on stock **binutils 2.46.1** (`patch -p1`, `--fuzz=0`).

See `gcc-14.3/EASYBUILD-NOTE.md` and `gcc-15.2/EASYBUILD-NOTE.md`.
