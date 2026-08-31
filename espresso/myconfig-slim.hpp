/*
 * Slim myconfig for Coulomb/P3M + WCA/LJ benchmarks on Orange Pi RV2.
 * Cuts dead pair-potential / NPT / dipoles branches out of
 * add_non_bonded_pair_force / calc_non_bonded_pair_force.
 *
 * DO NOT use as a general-purpose Espresso build.
 */
#pragma once

/* Thermostat / integrator pieces we need */
#define EXTERNAL_FORCES

/* Electrostatics + P3M (P3M also needs FFTW at CMake time → #define FFTW) */
#define ELECTROSTATICS

/* Non-bonded: only what our benches use */
#define WCA
#define LENNARD_JONES

/* Explicitly NOT enabling (present in default myconfig, expensive in pair loop):
 * NPT, DIPOLES, DPD, ROTATION, MASS, LJCOS*, GAUSSIAN, HAT, TABULATED,
 * LENNARD_JONES_GENERIC, COLLISION_DETECTION, VIRTUAL_SITES*, THOLE, ENGINE, …
 */
