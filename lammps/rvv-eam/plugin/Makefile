# EAM RVV plugin — build on RV2 with EESSI foss/2025b
LAMMPS_PREFIX ?= $(HOME)/eessi-overlay/versions/2025.06-001/software/linux/riscv64/generic/software/LAMMPS/22Jul2025_update4-foss-2025b-kokkos
LAMMPS_SRC ?= $(HOME)/eb-work/easybuild/build/LAMMPS/22Jul2025_update4/foss-2025b-kokkos/lammps-stable_22Jul2025_update4

CXX ?= mpicxx
ARCH ?= -march=rv64imafdcv_zba_zbb_zbs
CXXFLAGS ?= -O3 -fPIC -fopenmp -fno-math-errno $(ARCH) \
	-I$(LAMMPS_PREFIX)/include/lammps \
	-I$(LAMMPS_SRC)/src \
	-I$(LAMMPS_SRC)/src/MANYBODY
LDFLAGS ?= -shared -rdynamic -fopenmp -L$(LAMMPS_PREFIX)/lib -Wl,-rpath,$(LAMMPS_PREFIX)/lib -llammps

PLUGIN = eamrvvplugin.so

.PHONY: all clean

all: $(PLUGIN)

$(PLUGIN): eamrvvplugin.o pair_eam_rvv.o
	$(CXX) -o $@ $^ $(LDFLAGS)
	@echo "built: $@ with $$($(CXX) --version | head -1)"

eamrvvplugin.o: eamrvvplugin.cpp pair_eam_rvv.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

pair_eam_rvv.o: pair_eam_rvv.cpp pair_eam_rvv.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f *.o $(PLUGIN)
