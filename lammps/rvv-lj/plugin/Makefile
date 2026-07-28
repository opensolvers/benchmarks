# Build lj/cut/rvv LAMMPS plugin against the board overlay install.
#
#   export EESSI_VERSION_OVERRIDE=2025.06-001
#   source /cvmfs/software.eessi.io/versions/2025.06/init/lmod/bash
#   module load foss/2025b
#   make
#
# Override paths if needed:
#   make LAMMPS_PREFIX=... LAMMPS_SRC=...

LAMMPS_PREFIX ?= $(HOME)/eessi-overlay/versions/2025.06-001/software/linux/riscv64/generic/software/LAMMPS/22Jul2025_update4-foss-2025b-kokkos
LAMMPS_SRC ?= $(HOME)/eb-work/easybuild/build/LAMMPS/22Jul2025_update4/foss-2025b-kokkos/lammps-stable_22Jul2025_update4

CXX ?= mpicxx
ARCH ?= -march=rv64imafdcv_zba_zbb_zbs
CXXFLAGS ?= -O3 -fPIC -fopenmp -fno-math-errno $(ARCH) \
	-I$(LAMMPS_PREFIX)/include/lammps \
	-I$(LAMMPS_SRC)/src
LDFLAGS ?= -shared -rdynamic -fopenmp -L$(LAMMPS_PREFIX)/lib -Wl,-rpath,$(LAMMPS_PREFIX)/lib -llammps

PLUGIN = ljcutrvvplugin.so

.PHONY: all clean

all: $(PLUGIN)

$(PLUGIN): ljcutrvvplugin.o pair_lj_cut_rvv.o
	$(CXX) -o $@ $^ $(LDFLAGS)
	@echo "built: $@ with $$($(CXX) --version | head -1)"

ljcutrvvplugin.o: ljcutrvvplugin.cpp pair_lj_cut_rvv.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

pair_lj_cut_rvv.o: pair_lj_cut_rvv.cpp pair_lj_cut_rvv.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f *.o $(PLUGIN)
